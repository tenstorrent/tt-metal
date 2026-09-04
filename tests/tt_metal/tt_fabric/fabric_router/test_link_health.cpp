// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LinkHealth against a real mapper. The expected descriptor is the one the mapper was built on; the
// live descriptor is a second, independently discovered copy that each test mutates. That is the
// whole shape of the feature: the factory descriptor says what should be cabled, the live one says
// what is, and the difference is a downed link.
//
// Requires the T3K mock cluster descriptor (TT_METAL_MOCK_CLUSTER_DESC_PATH), like the topology
// mapper tests it is modelled on. T3K is a single mesh, so intermesh downed links are not covered
// here -- that needs a multi-mesh mock and is tracked in the FSD plan.

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/cluster.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/link_health.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/physical_node_id.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/fabric/physical_system_discovery.hpp"

namespace tt::tt_fabric {
namespace {

using tt::tt_metal::AsicID;
using tt::tt_metal::EthConnection;
using tt::tt_metal::PhysicalSystemDescriptor;

// A cable, as the descriptor stores it: two ASICs and the channel each one leaves through.
struct CableRef {
    AsicID src{0};
    AsicID dst{0};
    uint8_t src_chan = 0;
    uint8_t dst_chan = 0;
};

// The first cable in the descriptor, whatever it is. Which one does not matter -- what matters is
// that it exists in both descriptors before the test removes it from one.
std::optional<CableRef> first_cable(const PhysicalSystemDescriptor& descriptor) {
    for (const auto& [host, topology] : descriptor.get_system_graph().asic_connectivity_graph) {
        for (const auto& [src, edges] : topology) {
            for (const auto& [dst, connections] : edges) {
                if (!connections.empty()) {
                    return CableRef{src, dst, connections.front().src_chan, connections.front().dst_chan};
                }
            }
        }
    }
    return std::nullopt;
}

// Unplug one cable: remove the connection from both of the directed records the descriptor keeps for
// it, which is what discovery would have produced had the cable not been there.
void unplug(PhysicalSystemDescriptor& descriptor, const CableRef& cable) {
    auto remove_one = [&descriptor](AsicID from, AsicID to, uint8_t chan) {
        for (auto& [host, topology] : descriptor.get_system_graph().asic_connectivity_graph) {
            const auto entry = topology.find(from);
            if (entry == topology.end()) {
                continue;
            }
            for (auto& [peer, connections] : entry->second) {
                if (peer != to) {
                    continue;
                }
                connections.erase(
                    std::remove_if(
                        connections.begin(),
                        connections.end(),
                        [chan](const EthConnection& connection) { return connection.src_chan == chan; }),
                    connections.end());
            }
        }
    };
    remove_one(cable.src, cable.dst, cable.src_chan);
    remove_one(cable.dst, cable.src, cable.dst_chan);
}

class LinkHealthTest : public ::testing::Test {
protected:
    void SetUp() override {
        setenv("TT_METAL_OPERATION_TIMEOUT_SECONDS", "10", 1);

        expected_ = discover();
        live_ = discover();

        const std::filesystem::path mesh_graph_desc_path =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
            "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
        mesh_graph_ = std::make_unique<MeshGraph>(tt::tt_metal::ClusterType::T3K, mesh_graph_desc_path.string());

        LocalMeshBinding local_mesh_binding;
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{0};

        mapper_ = std::make_unique<TopologyMapper>(
            tt::tt_metal::MetalContext::instance().get_cluster(),
            tt::tt_metal::MetalContext::instance().global_distributed_context(),
            *mesh_graph_,
            *expected_,
            local_mesh_binding);
    }

    void TearDown() override {
        mapper_.reset();
        mesh_graph_.reset();
        live_.reset();
        expected_.reset();
    }

    static std::unique_ptr<PhysicalSystemDescriptor> discover() {
        auto distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        return std::make_unique<PhysicalSystemDescriptor>(tt::tt_metal::run_physical_system_discovery(
            *cluster.get_cluster_desc(), distributed_context, rtoptions.get_target_device()));
    }

    std::unique_ptr<PhysicalSystemDescriptor> expected_;
    std::unique_ptr<PhysicalSystemDescriptor> live_;
    std::unique_ptr<MeshGraph> mesh_graph_;
    std::unique_ptr<TopologyMapper> mapper_;
};

TEST_F(LinkHealthTest, MatchingDescriptorsHaveNoDownedLinks) {
    LinkHealth health(*mapper_, *live_);

    EXPECT_TRUE(health.get_downed_links().empty());
    EXPECT_FALSE(health.has_downed_links());
    EXPECT_FALSE(health.fsd_rerouting_active());
    EXPECT_TRUE(health.get_unused_downed_links().empty());
    EXPECT_GT(health.fsd_expected_count(), 0u);
}

TEST_F(LinkHealthTest, UnpluggingACableReportsBothDirections) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);

    ASSERT_EQ(health.get_downed_links().size(), 2u);
    EXPECT_TRUE(health.has_downed_links());
    EXPECT_TRUE(health.fsd_rerouting_active());

    // One record per end, and each names the other end.
    const auto& records = health.get_downed_links();
    EXPECT_NE(records[0].src_chan, records[1].src_chan);
    EXPECT_EQ(records[0].src_chan, records[1].dst_chan);
    EXPECT_EQ(records[0].dst_chan, records[1].src_chan);
}

// The physical side of a record describes what should have been there, so it comes from the expected
// descriptor -- the live one, by definition, has nothing to say about a cable it does not have.
TEST_F(LinkHealthTest, PhysicalFieldsComeFromTheExpectedDescriptor) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);
    ASSERT_FALSE(health.get_downed_links().empty());

    const auto& record = health.get_downed_links().front();
    const auto& expected_descriptors = expected_->get_asic_descriptors();
    const auto expected_src = expected_descriptors.at(record.src_asic);

    EXPECT_EQ(record.src_host_id, tt::tt_metal::canonical_host_for_node_id(expected_src.host_name));
    EXPECT_EQ(record.src_tray, expected_src.tray_id);
    EXPECT_EQ(record.src_loc, expected_src.asic_location);
    // The medium is the expected edge's port type.
    EXPECT_EQ(record.medium, expected_->get_eth_connections(record.src_asic, record.dst_asic).front().port_type);
}

TEST_F(LinkHealthTest, IntraMeshRecordsCarryBothDirectionsFromTheMeshGraph) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);
    ASSERT_FALSE(health.get_downed_links().empty());

    for (const auto& record : health.get_downed_links()) {
        ASSERT_TRUE(record.logical_resolved);
        ASSERT_TRUE(record.is_intramesh());
        EXPECT_EQ(record.src_mesh(), record.dst_mesh());
        // The two ends of a cable face each other, so their directions are never the same one.
        EXPECT_NE(record.src_direction, RoutingDirection::NONE);
        EXPECT_NE(record.dst_direction, RoutingDirection::NONE);
        EXPECT_NE(record.src_direction, record.dst_direction);
    }
}

// Presence against the live descriptor, and nothing else. This is why the API works on a mock: no
// call asks the cluster whether ethernet came up.
TEST_F(LinkHealthTest, HealthIsPresenceInTheLiveDescriptor) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);
    ASSERT_FALSE(health.get_downed_links().empty());
    const auto& record = health.get_downed_links().front();

    EXPECT_FALSE(health.is_link_healthy(record.src_node, record.src_chan));
    EXPECT_FALSE(health.is_link_healthy(record.src_asic, record.src_chan));
    EXPECT_FALSE(health.is_link_healthy(record.src_host_id, record.src_tray, record.src_loc, record.src_chan));

    // A channel the factory descriptor never declared has no health to report, which is different
    // from being healthy.
    EXPECT_THROW(health.is_link_healthy(record.src_node, 200), std::out_of_range);
}

TEST_F(LinkHealthTest, SurvivingCablesStayHealthy) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);
    const auto downed = health.get_downed_links();

    // Every other expected endpoint on the same chip is still present.
    const auto& record = downed.front();
    for (const auto& connection : expected_->get_eth_connections(record.src_asic, record.dst_asic)) {
        if (connection.src_chan == record.src_chan) {
            continue;
        }
        EXPECT_TRUE(health.is_link_healthy(record.src_node, connection.src_chan));
    }
}

// The factory descriptor is golden: the live cluster having more than it declares is not a fault.
TEST_F(LinkHealthTest, ExtraLiveCablesAreNotDowned) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    // Give the live descriptor a cable the expected one does not have, on channels nothing uses.
    for (auto& [host, topology] : live_->get_system_graph().asic_connectivity_graph) {
        const auto entry = topology.find(cable->src);
        if (entry == topology.end()) {
            continue;
        }
        entry->second.emplace_back(
            cable->dst, std::vector<EthConnection>{EthConnection{200, 201, true, tt::tt_metal::PortType::TRACE}});
        break;
    }

    LinkHealth health(*mapper_, *live_);

    EXPECT_TRUE(health.get_downed_links().empty());
    EXPECT_FALSE(health.fsd_rerouting_active());
}

TEST_F(LinkHealthTest, PerNodeAndPerDirectionQueriesAgree) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);
    ASSERT_FALSE(health.get_downed_links().empty());
    const auto record = health.get_downed_links().front();

    EXPECT_EQ(health.get_downed_links(record.src_node).size(), 1u);
    EXPECT_EQ(health.get_downed_eth_chans(record.src_node), std::vector<chan_id_t>{record.src_chan});
    EXPECT_TRUE(health.has_downed_link_in_direction(record.src_node, record.src_direction));
    EXPECT_EQ(
        health.get_downed_eth_chans_in_direction(record.src_node, record.src_direction),
        std::vector<chan_id_t>{record.src_chan});
    // Lost capacity is the count of active records on that (node, direction).
    EXPECT_EQ(health.get_num_downed_routing_planes_in_direction(record.src_node, record.src_direction), 1u);

    const auto found = health.find_downed_link(record.src_node, record.src_chan);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->dst_node, record.dst_node);
    EXPECT_FALSE(health.find_downed_link(record.src_node, 200).has_value());
}

TEST_F(LinkHealthTest, ScopeQueriesPartitionTheResolvedRecords) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);

    const auto intra = health.get_downed_intramesh_links();
    const auto inter = health.get_downed_intermesh_links();
    EXPECT_EQ(intra.size() + inter.size(), health.get_downed_links().size());
    EXPECT_EQ(intra.size(), health.get_downed_links(LinkScope::IntraMesh).size());
    EXPECT_EQ(inter.size(), health.get_downed_links(LinkScope::InterMesh).size());
    // Unknown is not a bucket -- an unresolved record has no logical view to classify.
    EXPECT_TRUE(health.get_downed_links(LinkScope::Unknown).empty());
}

TEST_F(LinkHealthTest, PhysicalQueriesFindTheCable) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);
    ASSERT_FALSE(health.get_downed_links().empty());
    const auto record = health.get_downed_links().front();

    EXPECT_FALSE(health.get_downed_links_for_host(record.src_host_id).empty());
    EXPECT_FALSE(health.get_downed_links_for_asic(record.src_asic).empty());
    // Host ids are canonicalized on the way in, so any spelling of the same host works.
    auto shouted = record.src_host_id;
    std::transform(shouted.begin(), shouted.end(), shouted.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    EXPECT_EQ(
        health.get_downed_links_for_host(shouted).size(), health.get_downed_links_for_host(record.src_host_id).size());

    const auto between = health.get_downed_links_between_hosts(record.src_host_id, record.dst_host_id);
    EXPECT_FALSE(between.empty());
}

TEST_F(LinkHealthTest, RouteQueriesFindTheCable) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);
    ASSERT_FALSE(health.get_downed_links().empty());
    const auto record = health.get_downed_links().front();

    EXPECT_EQ(health.get_downed_links_between(record.src_node, record.dst_node).size(), 1u);
    EXPECT_EQ(
        health.get_downed_forwarding_eth_chans_to_chip(record.src_node, record.dst_node),
        std::vector<chan_id_t>{record.src_chan});
}

// A hole on a routing plane fabric already downgraded away is a real unplugged cable that fabric
// will never route over, so it must not drive rerouting -- but it stays documented.
TEST_F(LinkHealthTest, HolesOnDowngradedPlanesMoveToTheUnusedSet) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);
    ASSERT_EQ(health.get_downed_links().size(), 2u);
    const auto records = health.get_downed_links();

    // Say fabric ended up routing on fewer planes than the mesh graph asked for, in the direction
    // one of the two records faces.
    RoutingPlaneSnapshot snapshot;
    snapshot.expected_planes[records.front().src_node][records.front().src_direction] = 4;
    snapshot.live_planes[records.front().src_node][records.front().src_direction] = 2;

    health.classify_unused_from_routing_planes(snapshot);

    EXPECT_EQ(health.get_unused_downed_links().size(), 1u);
    EXPECT_EQ(health.get_downed_links().size(), 1u);
    // Lost capacity ignores them: fabric already stopped routing there.
    EXPECT_EQ(
        health.get_num_downed_routing_planes_in_direction(records.front().src_node, records.front().src_direction), 0u);
    // The other half of the cable faced a direction nothing was said about, so it stays active.
    EXPECT_TRUE(health.fsd_rerouting_active());
}

TEST_F(LinkHealthTest, UndowngradedHolesStayActive) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);
    const auto records = health.get_downed_links();
    ASSERT_EQ(records.size(), 2u);

    // Fabric is routing on everything the mesh graph asked for, so the hole sits on a live plane.
    RoutingPlaneSnapshot snapshot;
    for (const auto& record : records) {
        snapshot.expected_planes[record.src_node][record.src_direction] = 4;
        snapshot.live_planes[record.src_node][record.src_direction] = 4;
    }

    health.classify_unused_from_routing_planes(snapshot);

    EXPECT_TRUE(health.get_unused_downed_links().empty());
    EXPECT_EQ(health.get_downed_links().size(), 2u);
}

TEST_F(LinkHealthTest, RefreshIsIdempotentAndCanRebind) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);
    const auto first = health.get_downed_links().size();
    ASSERT_EQ(first, 2u);

    health.refresh();
    EXPECT_EQ(health.get_downed_links().size(), first);

    // Rebinding to an unmutated descriptor clears the set: nothing is missing from it.
    auto pristine = discover();
    health.refresh(nullptr, pristine.get());
    EXPECT_TRUE(health.get_downed_links().empty());
    EXPECT_FALSE(health.fsd_rerouting_active());
}

// The indexes hand out pointers into the stored set, so they must point at the real records rather
// than at copies that could go stale.
TEST_F(LinkHealthTest, IndexesPointIntoTheStoredSet) {
    const auto cable = first_cable(*live_);
    ASSERT_TRUE(cable.has_value());
    unplug(*live_, *cable);

    LinkHealth health(*mapper_, *live_);
    const auto& records = health.get_downed_links();
    ASSERT_FALSE(records.empty());

    for (const auto& record : records) {
        const auto found = health.find_downed_link(record.src_node, record.src_chan);
        ASSERT_TRUE(found.has_value());
        EXPECT_EQ(found->src_chan, record.src_chan);
        EXPECT_EQ(found->dst_chan, record.dst_chan);
    }
}

TEST_F(LinkHealthTest, TheComparisonHasNoMeaningWithoutBothSides) {
    static_assert(!std::is_default_constructible_v<LinkHealth>);
    static_assert(!std::is_copy_constructible_v<LinkHealth>);
    static_assert(!std::is_move_constructible_v<LinkHealth>);
    static_assert(!std::is_copy_assignable_v<LinkHealth>);
    static_assert(!std::is_move_assignable_v<LinkHealth>);
}

}  // namespace
}  // namespace tt::tt_fabric
