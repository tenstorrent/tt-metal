// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <cabling_generator/cabling_generator.hpp>
#include "protobuf/factory_system_descriptor.pb.h"

// Exercises CablingGenerator::apply_instance_filter against a 4-superpod x 4-node cluster
// (16_n300_lb_cluster): host_id i maps to superpod{i/4+1}/node{i%4+1}, superpods are internally
// K4-connected (6 inter-node edges each) plus 6 inter-superpod edges (30 cross-host edges total).
namespace tt::scaleout_tools {
namespace {

using ::testing::_;
using ::testing::Each;
using ::testing::ElementsAre;

constexpr std::string_view kCabling = "tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto";
constexpr std::string_view kDeploy = "tools/tests/scaleout/deployment_descriptors/16_lb_deployment.textproto";

CablingGenerator make_gen() { return CablingGenerator(std::string(kCabling), std::string(kDeploy)); }

// FSD instance_path segments for each host, indexed by host_id.
std::vector<std::vector<std::string>> instance_paths(const CablingGenerator& gen) {
    auto fsd = gen.generate_factory_system_descriptor();
    std::vector<std::vector<std::string>> paths;
    for (const auto& host : fsd.hosts()) {
        paths.emplace_back(host.instance_path().begin(), host.instance_path().end());
    }
    return paths;
}

std::vector<std::string> hostnames(const CablingGenerator& gen) {
    std::vector<std::string> names;
    for (const auto& host : gen.get_deployment_hosts()) {
        names.push_back(host.hostname);
    }
    return names;
}

// Distinct cross-host (endpoint_a != endpoint_b) host_id pairs among the FSD eth connections;
// channel-level connections collapse to one entry per connected host pair.
std::set<std::pair<uint32_t, uint32_t>> cross_host_pairs(const CablingGenerator& gen) {
    auto fsd = gen.generate_factory_system_descriptor();
    std::set<std::pair<uint32_t, uint32_t>> pairs;
    for (const auto& conn : fsd.eth_connections().connection()) {
        uint32_t a = conn.endpoint_a().host_id();
        uint32_t b = conn.endpoint_b().host_id();
        if (a != b) {
            pairs.insert(a < b ? std::make_pair(a, b) : std::make_pair(b, a));
        }
    }
    return pairs;
}

// Highest host_id referenced by any board location or connection endpoint (catches dangling refs).
uint32_t max_referenced_host_id(const CablingGenerator& gen) {
    auto fsd = gen.generate_factory_system_descriptor();
    uint32_t max_id = 0;
    for (const auto& board : fsd.board_types().board_locations()) {
        max_id = std::max(max_id, board.host_id());
    }
    for (const auto& conn : fsd.eth_connections().connection()) {
        max_id = std::max({max_id, conn.endpoint_a().host_id(), conn.endpoint_b().host_id()});
    }
    return max_id;
}

}  // namespace

// ---- Baseline (instance paths before filtering) ----

TEST(InstanceFilterTest, BaselineUnfiltered) {
    auto gen = make_gen();
    EXPECT_EQ(gen.get_deployment_hosts().size(), 16u);
    auto paths = instance_paths(gen);
    ASSERT_EQ(paths.size(), 16u);
    EXPECT_THAT(paths.front(), ElementsAre("n300_lb_cluster", "superpod1", "node1"));
    EXPECT_THAT(paths.back(), ElementsAre("n300_lb_cluster", "superpod4", "node4"));
    EXPECT_EQ(cross_host_pairs(gen).size(), 30u);
    EXPECT_EQ(max_referenced_host_id(gen), 15u);
}

TEST(InstanceFilterTest, EmptyFiltersAreNoOp) {
    auto gen = make_gen();
    gen.apply_instance_filter({}, {});
    EXPECT_EQ(gen.get_deployment_hosts().size(), 16u);
    EXPECT_EQ(cross_host_pairs(gen).size(), 30u);
}

// ---- Subtree matching ----

TEST(InstanceFilterTest, IncludeSubtreeSelectsAllDescendants) {
    auto gen = make_gen();
    gen.apply_instance_filter({{"superpod1"}}, {});
    EXPECT_THAT(
        instance_paths(gen),
        ElementsAre(
            ElementsAre("n300_lb_cluster", "superpod1", "node1"),
            ElementsAre("n300_lb_cluster", "superpod1", "node2"),
            ElementsAre("n300_lb_cluster", "superpod1", "node3"),
            ElementsAre("n300_lb_cluster", "superpod1", "node4")));
    EXPECT_EQ(cross_host_pairs(gen).size(), 6u);  // K4 intra-superpod; inter-superpod edges dropped
}

TEST(InstanceFilterTest, IncludeExactLeafPathSelectsOneNode) {
    auto gen = make_gen();
    gen.apply_instance_filter({{"superpod2", "node3"}}, {});
    EXPECT_THAT(instance_paths(gen), ElementsAre(ElementsAre("n300_lb_cluster", "superpod2", "node3")));
    EXPECT_TRUE(cross_host_pairs(gen).empty());  // no surviving inter-node edges
}

// ---- Suffix / relative matching ----

TEST(InstanceFilterTest, RelativeExcludeDropsNameUnderEveryParent) {
    auto gen = make_gen();
    gen.apply_instance_filter({}, {{"node1"}});
    auto paths = instance_paths(gen);
    EXPECT_EQ(paths.size(), 12u);  // node1 removed from each of the 4 superpods
    for (const auto& path : paths) {
        ASSERT_FALSE(path.empty());
        EXPECT_NE(path.back(), "node1");
    }
    EXPECT_EQ(cross_host_pairs(gen).size(), 15u);
    EXPECT_EQ(max_referenced_host_id(gen), 11u);
}

TEST(InstanceFilterTest, RelativeIncludeSelectsNameUnderEveryParent) {
    auto gen = make_gen();
    gen.apply_instance_filter({{"node1"}}, {});
    auto paths = instance_paths(gen);
    EXPECT_EQ(paths.size(), 4u);
    EXPECT_THAT(paths, Each(ElementsAre("n300_lb_cluster", _, "node1")));
    // Only the node1<->node1 inter-superpod edge has both endpoints in the selection.
    EXPECT_EQ(cross_host_pairs(gen).size(), 1u);
}

TEST(InstanceFilterTest, RootLevelNameDoesNotMatchDeeperInstances) {
    // "superpod1" is a suffix only of the top-level instance, never of a deeper node path.
    auto gen = make_gen();
    gen.apply_instance_filter({{"superpod1"}}, {});
    EXPECT_EQ(gen.get_deployment_hosts().size(), 4u);
    EXPECT_THAT(instance_paths(gen), Each(ElementsAre("n300_lb_cluster", "superpod1", _)));
}

// ---- Include / exclude precedence ----

TEST(InstanceFilterTest, ExcludeOverridesInclude) {
    auto gen = make_gen();
    gen.apply_instance_filter({{"superpod1"}, {"superpod2"}}, {{"superpod2"}});
    EXPECT_EQ(gen.get_deployment_hosts().size(), 4u);
    EXPECT_THAT(instance_paths(gen), Each(ElementsAre("n300_lb_cluster", "superpod1", _)));
    EXPECT_EQ(cross_host_pairs(gen).size(), 6u);
}

// ---- Dense host-id remap + connection remap ----

TEST(InstanceFilterTest, DenseHostIdRemapAndDeploymentSourcing) {
    auto gen = make_gen();
    auto baseline_names = hostnames(gen);
    ASSERT_EQ(baseline_names.size(), 16u);

    gen.apply_instance_filter({{"superpod3"}}, {});  // original host_ids 8..11

    // Survivors re-index to a dense 0..3 space, each sourced from its original deployment host.
    EXPECT_THAT(
        hostnames(gen), ElementsAre(baseline_names[8], baseline_names[9], baseline_names[10], baseline_names[11]));
    EXPECT_THAT(
        instance_paths(gen),
        ElementsAre(
            ElementsAre("n300_lb_cluster", "superpod3", "node1"),
            ElementsAre("n300_lb_cluster", "superpod3", "node2"),
            ElementsAre("n300_lb_cluster", "superpod3", "node3"),
            ElementsAre("n300_lb_cluster", "superpod3", "node4")));
}

TEST(InstanceFilterTest, ConnectionsRemappedIntoDenseRange) {
    auto gen = make_gen();
    gen.apply_instance_filter({{"superpod1"}}, {});

    // Every board location and connection endpoint must reference a dense 0..3 host_id.
    auto fsd = gen.generate_factory_system_descriptor();
    for (const auto& board : fsd.board_types().board_locations()) {
        EXPECT_LT(board.host_id(), 4u);
    }
    for (const auto& conn : fsd.eth_connections().connection()) {
        EXPECT_LT(conn.endpoint_a().host_id(), 4u);
        EXPECT_LT(conn.endpoint_b().host_id(), 4u);
    }
    for (const auto& [a, b] : gen.get_chip_connections()) {
        EXPECT_LT(a.host_id.get(), 4u);
        EXPECT_LT(b.host_id.get(), 4u);
    }
    EXPECT_EQ(cross_host_pairs(gen).size(), 6u);
}

// ---- Unmatched / empty filters ----

TEST(InstanceFilterTest, UnmatchedIncludeThrows) {
    auto gen = make_gen();
    EXPECT_THROW(gen.apply_instance_filter({{"nonexistent"}}, {}), std::exception);
}

TEST(InstanceFilterTest, UnmatchedExcludeThrows) {
    auto gen = make_gen();
    EXPECT_THROW(gen.apply_instance_filter({}, {{"nonexistent"}}), std::exception);
}

TEST(InstanceFilterTest, ExcludingEverythingThrows) {
    auto gen = make_gen();
    EXPECT_THROW(
        gen.apply_instance_filter({}, {{"superpod1"}, {"superpod2"}, {"superpod3"}, {"superpod4"}}), std::exception);
}

TEST(InstanceFilterTest, EmptyPathThrows) {
    auto gen = make_gen();
    EXPECT_THROW(gen.apply_instance_filter({{}}, {}), std::exception);
}

TEST(InstanceFilterTest, EmptyPathSegmentThrows) {
    auto gen = make_gen();
    EXPECT_THROW(gen.apply_instance_filter({{"superpod1", ""}}, {}), std::exception);
}

}  // namespace tt::scaleout_tools
