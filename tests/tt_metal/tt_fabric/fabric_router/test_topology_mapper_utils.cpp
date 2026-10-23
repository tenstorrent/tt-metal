// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdlib>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <llrt/tt_cluster.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include "impl/context/metal_context.hpp"
#include "mock_psd_builder.hpp"
#include "tt_metal/fabric/physical_system_discovery.hpp"

using namespace tt::tt_fabric::test;

namespace tt::tt_metal::experimental::tt_fabric {
namespace {

class TopologyMapperUtilsTest : public ::testing::Test {
protected:
    static void verify_bidirectional_consistency(const TopologyMappingResult& result) {
        for (const auto& [node, asic] : result.fabric_node_to_asic) {
            ASSERT_TRUE(result.asic_to_fabric_node.contains(asic))
                << "ASIC " << asic.get() << " not found in reverse mapping";
            EXPECT_EQ(result.asic_to_fabric_node.at(asic), node)
                << "Bidirectional mapping inconsistent for ASIC " << asic.get();
        }
        for (const auto& [asic, node] : result.asic_to_fabric_node) {
            ASSERT_TRUE(result.fabric_node_to_asic.contains(node)) << "Node not found in forward mapping";
            EXPECT_EQ(result.fabric_node_to_asic.at(node), asic) << "Bidirectional mapping inconsistent for node";
        }
    }
};

void fill_hosts_from_psd(TopologyMappingConfig& config, const tt::tt_metal::PhysicalSystemDescriptor& psd) {
    for (const auto& [asic_id, desc] : psd.get_asic_descriptors()) {
        config.hostname_to_asics[desc.host_name].insert(asic_id);
    }
}

std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> unset_asic_ranks(
    const tt::tt_metal::PhysicalSystemDescriptor& psd, MeshId mesh = MeshId{0}) {
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> ranks;
    for (const auto& [asic_id, _] : psd.get_asic_descriptors()) {
        ranks[mesh][asic_id] = ::tt::tt_fabric::MESH_HOST_RANK_UNSET;
    }
    return ranks;
}

std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> fabric_ranks_for_host_grid(
    MeshId mesh, uint32_t device_rows, uint32_t device_cols, uint32_t host_rows, uint32_t host_cols) {
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> ranks;
    const uint32_t chips_per_host_row = device_rows / host_rows;
    const uint32_t chips_per_host_col = device_cols / host_cols;
    for (uint32_t r = 0; r < device_rows; ++r) {
        for (uint32_t c = 0; c < device_cols; ++c) {
            const uint32_t chip = (r * device_cols) + c;
            const uint32_t rank = ((r / chips_per_host_row) * host_cols) + (c / chips_per_host_col);
            ranks[mesh][FabricNodeId(mesh, chip)] = MeshHostRankId{rank};
        }
    }
    return ranks;
}

void verify_each_rank_on_one_host(
    const TopologyMappingResult& result,
    const std::map<FabricNodeId, MeshHostRankId>& fabric_node_id_to_mesh_rank,
    const std::map<std::string, std::set<tt::tt_metal::AsicID>>& hostname_to_asics) {
    std::map<tt::tt_metal::AsicID, std::string> asic_to_host;
    for (const auto& [hostname, asics] : hostname_to_asics) {
        for (const auto& asic : asics) {
            asic_to_host[asic] = hostname;
        }
    }
    std::map<MeshHostRankId, std::set<std::string>> hosts_per_rank;
    for (const auto& [node, rank] : fabric_node_id_to_mesh_rank) {
        auto node_it = result.fabric_node_to_asic.find(node);
        ASSERT_NE(node_it, result.fabric_node_to_asic.end())
            << "Fabric node (mesh=" << node.mesh_id.get() << ", chip=" << node.chip_id << ") was not mapped";
        auto host_it = asic_to_host.find(node_it->second);
        ASSERT_NE(host_it, asic_to_host.end()) << "Mapped ASIC " << node_it->second.get() << " has no host";
        hosts_per_rank[rank].insert(host_it->second);
    }
    for (const auto& [rank, hosts] : hosts_per_rank) {
        EXPECT_EQ(hosts.size(), 1u) << "Mesh host rank " << rank.get() << " straddles " << hosts.size()
                                    << " physical hosts";
    }
}

::tt::tt_fabric::PhysicalGroupingDescriptor unspecified_line_1x2_pgd() {
    return ::tt::tt_fabric::PhysicalGroupingDescriptor{std::string(R"delimiter(
groupings {
  name: "1x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 2] }
}

groupings {
  name: "machine_host"
  preset_type: HOSTS
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [1, 4] }
}
)delimiter")};
}

::tt::tt_fabric::MeshGraphDescriptor two_1x2_meshes_mgd() {
    return ::tt::tt_fabric::MeshGraphDescriptor{std::string(R"delimiter(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
  device_topology { dims: [ 1, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

        graph_descriptors {
          name: "G0"
          type: "FABRIC"
          instances { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
          instances { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
    channels { count: 2 policy: RELAXED }
          }
        }

        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter")};
}

std::vector<std::set<uint64_t>> mapped_asic_footprints(const TopologyMappingResult& mapping) {
    std::map<MeshId, std::set<uint64_t>> per_mesh;
    for (const auto& [fabric_node, asic] : mapping.fabric_node_to_asic) {
        per_mesh[fabric_node.mesh_id].insert(*asic);
    }
    std::vector<std::set<uint64_t>> footprints;
    footprints.reserve(per_mesh.size());
    for (auto& [_, asics] : per_mesh) {
        footprints.push_back(std::move(asics));
    }
    return footprints;
}

tt::tt_metal::PhysicalSystemDescriptor create_psd_from_mock_cluster() {
    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        throw std::runtime_error("TT_METAL_MOCK_CLUSTER_DESC_PATH must be set for PSD tests");
    }

    auto distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    return tt::tt_metal::run_physical_system_discovery(
        *cluster.get_cluster_desc(), distributed_context, rtoptions.get_target_device());
}

std::size_t chip_count(const ::tt::tt_fabric::MeshGraph& mesh_graph) {
    std::size_t n = 0;
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        n += mesh_graph.get_chip_ids(mesh_id).size();
    }
    return n;
}

std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> fabric_ranks_from_mesh_graph(
    const ::tt::tt_fabric::MeshGraph& mesh_graph) {
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> ranks;
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        for (const auto& [coord, chip_id] : mesh_graph.get_chip_ids(mesh_id)) {
            (void)coord;
            auto rank = mesh_graph.get_host_rank_for_chip(mesh_id, chip_id);
            if (rank.has_value()) {
                ranks[mesh_id][FabricNodeId(mesh_id, chip_id)] = rank.value();
            }
        }
    }
    return ranks;
}

TopologyMappingResult map_sp4_blitz_pipeline(const std::filesystem::path& mgd_path) {
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    EXPECT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";
    const std::filesystem::path pgd_path = std::filesystem::path(tt_metal_home) /
                                           "tests/tt_metal/tt_fabric/physical_groupings/"
                                           "bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    EXPECT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;
    EXPECT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    ::tt::tt_fabric::PhysicalGroupingDescriptor pgd{pgd_path};
    ::tt::tt_fabric::MeshGraphDescriptor mgd{mgd_path};
    ::tt::tt_fabric::MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, mgd_path.string());

    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = false;
    fill_hosts_from_psd(config, psd);
    for (const auto& [_, groups] : mgd.get_pinnings()) {
        for (const auto& group : groups) {
            config.pinnings.push_back({group.fabric_nodes, group.asic_positions});
        }
    }
    if (!config.pinnings.empty()) {
        for (const auto& [asic_id, unused] : psd.get_asic_descriptors()) {
            (void)unused;
            config.asic_positions[asic_id] = std::make_pair(psd.get_tray_id(asic_id), psd.get_asic_location(asic_id));
        }
    }
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        config.mesh_validation_modes[mesh_id] = mesh_graph.is_intra_mesh_policy_relaxed(mesh_id)
                                                    ? ::tt::tt_fabric::ConnectionValidationMode::RELAXED
                                                    : ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    }
    config.inter_mesh_validation_mode = mesh_graph.is_inter_mesh_policy_relaxed()
                                            ? ::tt::tt_fabric::ConnectionValidationMode::RELAXED
                                            : ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    return map_multi_mesh_to_physical(
        psd, pgd, mgd, config, /*pinnings=*/{}, /*asic_id_to_mesh_rank=*/{}, fabric_ranks_from_mesh_graph(mesh_graph));
}

}  // namespace

// Two linked 1x2 meshes on a 4-chip line: the public PSD+PGD+MGD mapper must seat the
// only disjoint adjacent pairing.
TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_TwoLinked1x2Meshes_OnFourAsicLine) {
    using namespace ::tt::tt_fabric;

    auto pgd = unspecified_line_1x2_pgd();
    auto mgd = two_1x2_meshes_mgd();
    auto psd = build_mock_psd(std::vector<std::string>(4, "host0"), line_edges(4));

    TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    const auto mapping = map_multi_mesh_to_physical(psd, pgd, mgd, config);
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 4u);
    EXPECT_THAT(
        mapped_asic_footprints(mapping),
        ::testing::UnorderedElementsAre(std::set<uint64_t>{100, 101}, std::set<uint64_t>{102, 103}));
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_IncompatibleTopology_Fails) {
    using namespace ::tt::tt_fabric;

    auto psd = build_grid_mock_psd(2, 2, {"host0", "host1", "host0", "host1"});
    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } }
  ]
  row_major_mesh { dims: [2, 2] }
}
)")};
    MeshGraphDescriptor mgd{std::string(R"(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    EXPECT_THROW(map_multi_mesh_to_physical(psd, pgd, mgd, config), std::exception);
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_ImpossibleIntraMeshConstraints_2x2OnLine_Fails) {
    using namespace ::tt::tt_fabric;

    auto psd = build_mock_psd(std::vector<std::string>(4, "host0"), line_edges(4));
    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 1 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 2 location { asic_location: ASIC_LOCATION_UNSPECIFIED } },
    { id: 3 location { asic_location: ASIC_LOCATION_UNSPECIFIED } }
  ]
  row_major_mesh { dims: [2, 2] }
}
)")};
    MeshGraphDescriptor mgd{std::string(R"(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
  device_topology { dims: [ 2, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 1 ] }
            channels { count: 2 policy: STRICT }
          }

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    EXPECT_THROW(map_multi_mesh_to_physical(psd, pgd, mgd, config), std::exception);
}

// Rank containment goes through the PSD+PGD mapper, not the graph-only topology solver.
TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_NoHostRankAssigned_2x2) {
    using namespace ::tt::tt_fabric;

    auto psd = build_grid_mock_psd(2, 2, {"host0", "host1", "host0", "host1"});
    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x2_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } }
  ]
  row_major_mesh { dims: [2, 2] }
}

groupings { name: "2x2_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } } ]
  row_major_mesh { dims: [2, 1] } }
groupings { name: "2x2_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 1] } }
)")};
    MeshGraphDescriptor mgd{std::string(R"(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
  device_topology { dims: [ 2, 2 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    TopologyMappingConfig config;
    config.disable_rank_bindings = false;
    fill_hosts_from_psd(config, psd);
    const auto fabric_ranks = fabric_ranks_for_host_grid(MeshId{0}, 2, 2, 1, 2);
    const auto mapping =
        map_multi_mesh_to_physical(psd, pgd, mgd, config, /*pinnings=*/{}, unset_asic_ranks(psd), fabric_ranks);
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 4u);
    verify_bidirectional_consistency(mapping);
    verify_each_rank_on_one_host(mapping, fabric_ranks.at(MeshId{0}), config.hostname_to_asics);
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_Sp4Glx_Blitz2x4) {
    using namespace ::tt::tt_fabric;
    if (getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH") == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr);
    const std::filesystem::path mgd_path = std::filesystem::path(tt_metal_home) /
                                           "tests/tt_metal/tt_fabric/custom_mesh_descriptors/"
                                           "bh_glx_10stage_4x2_pipeline.textproto";
    MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, mgd_path.string());
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    const auto mapping_result = map_sp4_blitz_pipeline(mgd_path);
    ASSERT_TRUE(mapping_result.success) << mapping_result.error_message;
    EXPECT_EQ(mapping_result.fabric_node_to_asic.size(), chip_count(mesh_graph));
    std::set<std::string> hosts;
    for (const auto& [_, asic_id] : mapping_result.fabric_node_to_asic) {
        hosts.insert(psd.get_host_name_for_asic(asic_id));
    }
    EXPECT_GE(hosts.size(), 1u);
    EXPECT_LE(hosts.size(), 4u);
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_Sp4Glx_Blitz2x4_11Stage) {
    using namespace ::tt::tt_fabric;
    if (getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH") == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr);
    const std::filesystem::path mgd_path = std::filesystem::path(tt_metal_home) /
                                           "tests/tt_metal/tt_fabric/custom_mesh_descriptors/"
                                           "bh_glx_11stage_4x2_pipeline.textproto";
    MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, mgd_path.string());
    ASSERT_EQ(chip_count(mesh_graph), 11u * 8u);
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    const auto mapping_result = map_sp4_blitz_pipeline(mgd_path);
    ASSERT_TRUE(mapping_result.success) << mapping_result.error_message;
    EXPECT_EQ(mapping_result.fabric_node_to_asic.size(), chip_count(mesh_graph));
    std::set<std::string> hosts;
    for (const auto& [_, asic_id] : mapping_result.fabric_node_to_asic) {
        hosts.insert(psd.get_host_name_for_asic(asic_id));
    }
    EXPECT_GE(hosts.size(), 1u);
    EXPECT_LE(hosts.size(), 5u);
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_Sp4Glx_Blitz2x4_32Stage) {
    using namespace ::tt::tt_fabric;
    if (getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH") == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }
    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr);
    const std::filesystem::path mgd_path = std::filesystem::path(tt_metal_home) /
                                           "tests/tt_metal/tt_fabric/custom_mesh_descriptors/"
                                           "bh_glx_32stage_4x2_pipeline.textproto";
    MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, mgd_path.string());
    ASSERT_EQ(chip_count(mesh_graph), 32u * 8u);
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    const auto mapping_result = map_sp4_blitz_pipeline(mgd_path);
    ASSERT_TRUE(mapping_result.success) << mapping_result.error_message;
    EXPECT_EQ(mapping_result.fabric_node_to_asic.size(), chip_count(mesh_graph));
    std::set<std::string> hosts;
    for (const auto& [_, asic_id] : mapping_result.fabric_node_to_asic) {
        hosts.insert(psd.get_host_name_for_asic(asic_id));
    }
    EXPECT_EQ(hosts.size(), 8u);
}

TEST_F(TopologyMapperUtilsTest, SweepConsumer_SolutionSpansExpectedHosts) {
    const char* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP()
            << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run via the sweep / tt-run --mock-cluster-rank-binding";
    }

    std::size_t expected_hosts = 8;
    if (const char* env = getenv("SWEEP_EXPECTED_HOSTS"); env != nullptr && env[0] != '\0') {
        expected_hosts = static_cast<std::size_t>(std::stoul(env));
    }

    auto galaxy_tag = [](const std::string& h) -> std::string {
        auto pos = h.find("bh-glx-");
        if (pos == std::string::npos) {
            return h;
        }
        auto is_tag_char = [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-';
        };
        std::size_t end = pos;
        while (end < h.size() && is_tag_char(h[end])) {
            ++end;
        }
        return h.substr(pos, end - pos);
    };

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    std::set<std::string> hosts;
    for (const auto& [asic_id, desc] : psd.get_asic_descriptors()) {
        (void)asic_id;
        hosts.insert(galaxy_tag(desc.host_name));
    }
    EXPECT_EQ(hosts.size(), expected_hosts)
        << "each swept solution must span exactly " << expected_hosts << " distinct hosts; got " << hosts.size();
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_SingleBHGalaxy_2x4Pipeline) {
    using namespace ::tt::tt_fabric;
    if (getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH") == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }

    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr);
    const std::filesystem::path pgd_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/physical_groupings/wh_bh_rev_c_galaxy_physical_grouping_descriptor.textproto";
    const std::filesystem::path mgd_path = std::filesystem::path(tt_metal_home) /
                                           "tests/tt_metal/tt_fabric/custom_mesh_descriptors/"
                                           "bh_galaxy_2x4_pipeline.textproto";
    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;
    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();
    PhysicalGroupingDescriptor pgd{pgd_path};
    MeshGraphDescriptor mgd{mgd_path};
    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = true;
    const auto mapping_result = map_multi_mesh_to_physical(psd, pgd, mgd, config);
    ASSERT_TRUE(mapping_result.success) << mapping_result.error_message;
    EXPECT_EQ(mapping_result.fabric_node_to_asic.size(), 32u);
}

}  // namespace tt::tt_metal::experimental::tt_fabric
