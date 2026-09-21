// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <set>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <unordered_set>
#include <string>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include <tt-metalium/experimental/fabric/topology_solver.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/cluster.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "tt_metal/fabric/physical_system_discovery.hpp"
#include "tt_metal/fabric/serialization/physical_system_descriptor_serialization.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::experimental::tt_fabric {
namespace {

// =============================================================================
// Test Fixture with Helper Methods
// =============================================================================

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
}  // namespace

static tt::tt_metal::PhysicalSystemDescriptor load_mock_psd(const char* relative_path) {
    return tt::tt_metal::deserialize_physical_system_descriptor_from_text_proto_file(relative_path);
}

static void fill_hosts_from_psd(TopologyMappingConfig& config, const tt::tt_metal::PhysicalSystemDescriptor& psd) {
    for (const auto& [asic_id, desc] : psd.get_asic_descriptors()) {
        config.hostname_to_asics[desc.host_name].insert(asic_id);
    }
}

static std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> unset_asic_ranks(
    const tt::tt_metal::PhysicalSystemDescriptor& psd, MeshId mesh = MeshId{0}) {
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> ranks;
    for (const auto& [asic_id, _] : psd.get_asic_descriptors()) {
        ranks[mesh][asic_id] = ::tt::tt_fabric::MESH_HOST_RANK_UNSET;
    }
    return ranks;
}

static std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> fabric_ranks_for_host_grid(
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

static void verify_each_rank_on_one_host(
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

static ::tt::tt_fabric::PhysicalGroupingDescriptor unspecified_line_1x2_pgd() {
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

static ::tt::tt_fabric::MeshGraphDescriptor two_1x2_meshes_mgd(bool linked) {
    std::string body = R"delimiter(
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
)delimiter";
    if (linked) {
        body += R"delimiter(
          connections {
            nodes { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
    nodes { mesh { mesh_descriptor: "M0" mesh_id: 1 } }
    channels { count: 2 policy: RELAXED }
          }
)delimiter";
    }
    body += R"delimiter(
        }

        top_level_instance { graph { graph_descriptor: "G0" graph_id: 0 } }
)delimiter";
    return ::tt::tt_fabric::MeshGraphDescriptor{body};
}

static std::vector<std::set<uint64_t>> mapped_asic_footprints(const TopologyMappingResult& mapping) {
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

// Two linked 1x2 meshes on a 4-chip line: the public PSD+PGD+MGD mapper must seat the
// only disjoint adjacent pairing.
TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_TwoLinked1x2Meshes_OnFourAsicLine) {
    using namespace ::tt::tt_fabric;

    auto pgd = unspecified_line_1x2_pgd();
    auto mgd = two_1x2_meshes_mgd(/*linked=*/true);
    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_line.textproto");

    TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    const auto mapping = map_multi_mesh_to_physical(psd, pgd, mgd, config);
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 4u);
    EXPECT_THAT(
        mapped_asic_footprints(mapping),
        ::testing::UnorderedElementsAre(std::set<uint64_t>{100, 101}, std::set<uint64_t>{102, 103}));
}

// Same two 1x2 meshes, no intermesh edge: they still map onto the two isolated pairs.
TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_TwoMeshes_Succeeds) {
    using namespace ::tt::tt_fabric;

    auto pgd = unspecified_line_1x2_pgd();
    auto mgd = two_1x2_meshes_mgd(/*linked=*/false);
    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2mesh.textproto");

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

    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2x2_hosts_by_column.textproto");
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

    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_line.textproto");
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

TEST_F(TopologyMapperUtilsTest, Pinning_MapMultiMeshToPhysical_MeshLevelPinningsAppliedFirst) {
    using namespace ::tt::tt_fabric;

    auto pgd = unspecified_line_1x2_pgd();
    auto mgd = two_1x2_meshes_mgd(/*linked=*/false);
    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2mesh.textproto");

    TopologyMappingConfig config;
    config.disable_rank_bindings = true;
    const auto mapping = map_multi_mesh_to_physical(psd, pgd, mgd, config);
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 4u);
    EXPECT_THAT(
        mapped_asic_footprints(mapping),
        ::testing::UnorderedElementsAre(std::set<uint64_t>{100, 101}, std::set<uint64_t>{102, 103}));
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_2x2Mesh_HostsByColumn) {
    using namespace ::tt::tt_fabric;

    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2x2_hosts_by_column.textproto");
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
    config.disable_rank_bindings = true;
    const auto mapping = map_multi_mesh_to_physical(psd, pgd, mgd, config);
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 4u);
    verify_bidirectional_consistency(mapping);
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_NoHostRankAssigned_2x2) {
    using namespace ::tt::tt_fabric;

    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2x2_hosts_by_column.textproto");
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

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_PartialRankBinding_OneHostExplicitOthersUnset_Succeeds) {
    using namespace ::tt::tt_fabric;

    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2x2_hosts_by_column.textproto");
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
    auto asic_ranks = unset_asic_ranks(psd);
    asic_ranks[MeshId{0}][tt::tt_metal::AsicID{100}] = MeshHostRankId{0};
    asic_ranks[MeshId{0}][tt::tt_metal::AsicID{102}] = MeshHostRankId{0};
    const auto fabric_ranks = fabric_ranks_for_host_grid(MeshId{0}, 2, 2, 1, 2);
    const auto mapping = map_multi_mesh_to_physical(psd, pgd, mgd, config, /*pinnings=*/{}, asic_ranks, fabric_ranks);
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 4u);
    verify_each_rank_on_one_host(mapping, fabric_ranks.at(MeshId{0}), config.hostname_to_asics);
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_TwoHostsSplitAcrossFourRanks_EachRankWithinOneHost) {
    using namespace ::tt::tt_fabric;

    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2x2_hosts_by_column.textproto");
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
  host_topology   { dims: [ 2, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    TopologyMappingConfig config;
    config.disable_rank_bindings = false;
    fill_hosts_from_psd(config, psd);
    const auto fabric_ranks = fabric_ranks_for_host_grid(MeshId{0}, 2, 2, 2, 2);
    const auto mapping =
        map_multi_mesh_to_physical(psd, pgd, mgd, config, /*pinnings=*/{}, unset_asic_ranks(psd), fabric_ranks);
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 4u);
    verify_each_rank_on_one_host(mapping, fabric_ranks.at(MeshId{0}), config.hostname_to_asics);
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_FourNodesFourHosts_NoHostRankAssigned) {
    using namespace ::tt::tt_fabric;

    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_four_hosts_by_column.textproto");
    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh { dims: [2, 4] }
}

groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } } ]
  row_major_mesh { dims: [2, 1] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } } ]
  row_major_mesh { dims: [2, 1] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } } ]
  row_major_mesh { dims: [2, 1] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [ { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
               { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } } ]
  row_major_mesh { dims: [2, 1] } }
)")};
    MeshGraphDescriptor mgd{std::string(R"(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 4 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    TopologyMappingConfig config;
    config.disable_rank_bindings = false;
    fill_hosts_from_psd(config, psd);
    const auto fabric_ranks = fabric_ranks_for_host_grid(MeshId{0}, 2, 4, 1, 4);
    const auto mapping =
        map_multi_mesh_to_physical(psd, pgd, mgd, config, /*pinnings=*/{}, unset_asic_ranks(psd), fabric_ranks);
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 8u);
    verify_each_rank_on_one_host(mapping, fabric_ranks.at(MeshId{0}), config.hostname_to_asics);
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_TwoHostsTwoAsicsEach_SameHostSameRank) {
    using namespace ::tt::tt_fabric;

    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_8asic_2x4_hosts_by_column.textproto");
    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh { dims: [2, 4] }
}

groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } }
  ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "2x4_hosts" preset_type: HOSTS
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh { dims: [2, 2] } }
)")};
    MeshGraphDescriptor mgd{std::string(R"(
        mesh_descriptors {
          name: "M0"
          arch: WORMHOLE_B0
  device_topology { dims: [ 2, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 1, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    TopologyMappingConfig config;
    config.disable_rank_bindings = false;
    fill_hosts_from_psd(config, psd);
    const auto fabric_ranks = fabric_ranks_for_host_grid(MeshId{0}, 2, 4, 1, 2);
    const auto mapping =
        map_multi_mesh_to_physical(psd, pgd, mgd, config, /*pinnings=*/{}, unset_asic_ranks(psd), fabric_ranks);
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 8u);
    verify_each_rank_on_one_host(mapping, fabric_ranks.at(MeshId{0}), config.hostname_to_asics);
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_FourNodesFourHosts_PartialAsicRankBinding_Host0Rank1Only) {
    using namespace ::tt::tt_fabric;

    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_4asic_2x2_hosts_by_column.textproto");
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

    for (tt::tt_metal::AsicID bound : {tt::tt_metal::AsicID{101}, tt::tt_metal::AsicID{103}}) {
        auto asic_ranks = unset_asic_ranks(psd);
        asic_ranks[MeshId{0}][bound] = MeshHostRankId{1};
        const auto mapping =
            map_multi_mesh_to_physical(psd, pgd, mgd, config, /*pinnings=*/{}, asic_ranks, fabric_ranks);
        ASSERT_TRUE(mapping.success) << mapping.error_message << " bound asic " << bound.get();
        EXPECT_EQ(mapping.fabric_node_to_asic.size(), 4u);
        verify_each_rank_on_one_host(mapping, fabric_ranks.at(MeshId{0}), config.hostname_to_asics);
        EXPECT_EQ(mapping.asic_to_fabric_node.at(bound).mesh_id, MeshId{0});
        const auto bound_rank = fabric_ranks.at(MeshId{0}).at(mapping.asic_to_fabric_node.at(bound));
        EXPECT_EQ(bound_rank, MeshHostRankId{1});
    }
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_NoHostRankAssigned_4x4FourHosts) {
    using namespace ::tt::tt_fabric;

    auto psd =
        load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_16asic_4x4_four_hosts_by_quadrant.textproto");
    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "4x4_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 6 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 7 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } },
    { id: 8 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_1 } },
    { id: 9 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_2 } },
    { id: 10 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_3 } },
    { id: 11 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_4 } },
    { id: 12 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_1 } },
    { id: 13 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_2 } },
    { id: 14 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_3 } },
    { id: 15 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh { dims: [4, 4] }
}

groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } }
  ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_4 } },
    { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [
    { id: 0 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_1 } },
    { id: 3 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_2 } }
  ]
  row_major_mesh { dims: [2, 2] } }
groupings { name: "4x4_hosts" preset_type: HOSTS
  instances: [
    { id: 0 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_3 } },
    { id: 1 location { tray_id: TRAY_3 asic_location: ASIC_LOCATION_4 } },
    { id: 2 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_4 asic_location: ASIC_LOCATION_4 } }
  ]
  row_major_mesh { dims: [2, 2] } }
)")};
    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 4, 4 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 2, 2 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    TopologyMappingConfig config;
    config.disable_rank_bindings = false;
    fill_hosts_from_psd(config, psd);
    const auto fabric_ranks = fabric_ranks_for_host_grid(MeshId{0}, 4, 4, 2, 2);
    const auto mapping =
        map_multi_mesh_to_physical(psd, pgd, mgd, config, /*pinnings=*/{}, unset_asic_ranks(psd), fabric_ranks);
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 16u);
    verify_each_rank_on_one_host(mapping, fabric_ranks.at(MeshId{0}), config.hostname_to_asics);
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_NoHostRankAssigned_2x3TwoHosts) {
    using namespace ::tt::tt_fabric;

    auto psd = load_mock_psd("tests/tt_metal/tt_fabric/custom_mock_PSDs/test_6asic_2x3_hosts_by_row.textproto");
    PhysicalGroupingDescriptor pgd{std::string(R"(
groupings {
  name: "2x3_Mesh"
  preset_type: MESH
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } },
    { id: 3 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 4 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 5 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } }
  ]
  row_major_mesh { dims: [2, 3] }
}

groupings { name: "2x3_hosts" preset_type: HOSTS
  instances: [
    { id: 0 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_1 asic_location: ASIC_LOCATION_3 } }
  ]
  row_major_mesh { dims: [1, 3] } }
groupings { name: "2x3_hosts" preset_type: HOSTS
  instances: [
    { id: 0 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_1 } },
    { id: 1 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_2 } },
    { id: 2 location { tray_id: TRAY_2 asic_location: ASIC_LOCATION_3 } }
  ]
  row_major_mesh { dims: [1, 3] } }
)")};
    MeshGraphDescriptor mgd{std::string(R"(
mesh_descriptors {
  name: "M0"
  arch: WORMHOLE_B0
  device_topology { dims: [ 2, 3 ] dim_types: [ LINE, LINE ] }
  host_topology   { dims: [ 2, 1 ] }
  channels { count: 2 policy: STRICT }
}

top_level_instance { mesh { mesh_descriptor: "M0" mesh_id: 0 } }
)")};

    TopologyMappingConfig config;
    config.disable_rank_bindings = false;
    fill_hosts_from_psd(config, psd);
    const auto fabric_ranks = fabric_ranks_for_host_grid(MeshId{0}, 2, 3, 2, 1);
    const auto mapping =
        map_multi_mesh_to_physical(psd, pgd, mgd, config, /*pinnings=*/{}, unset_asic_ranks(psd), fabric_ranks);
    ASSERT_TRUE(mapping.success) << mapping.error_message;
    EXPECT_EQ(mapping.fabric_node_to_asic.size(), 6u);
    verify_each_rank_on_one_host(mapping, fabric_ranks.at(MeshId{0}), config.hostname_to_asics);
}

// Helper function to create PSD from mock cluster (similar to test_physical_grouping_descriptor.cpp)
static tt::tt_metal::PhysicalSystemDescriptor create_psd_from_mock_cluster() {
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

static std::size_t chip_count(const ::tt::tt_fabric::MeshGraph& mesh_graph) {
    std::size_t n = 0;
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        n += mesh_graph.get_chip_ids(mesh_id).size();
    }
    return n;
}

static std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> fabric_ranks_from_mesh_graph(
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

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_Sp4Glx_Blitz2x4) {
    // map_multi_mesh_to_physical using PGD, PSD, and a 10-stage Blitz MGD
    // Blitz 4x2 pipeline MGD (10 logical stages; adjacency-guided placement seats one region per stage)
    using namespace ::tt::tt_fabric;

    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";

    // Check if mock cluster descriptor is available (set by tt-run)
    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }

    // Create PSD from mock cluster
    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();

    // Load PGD - using triple_16x8_quad_bh_galaxy_physical_groupings
    const std::filesystem::path pgd_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;
    PhysicalGroupingDescriptor pgd{pgd_path};

    // Custom 10-stage 4×2 pipeline (8 ASICs/stage) — see bh_glx_10stage_4x2_pipeline.textproto
    const std::filesystem::path mgd_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_glx_10stage_4x2_pipeline.textproto";
    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;
    MeshGraphDescriptor mgd{mgd_path};

    MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, mgd_path.string());
    const std::size_t expected_fabric_nodes = chip_count(mesh_graph);
    ASSERT_GT(expected_fabric_nodes, 0u);

    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = false;

    for (const auto& [asic_id, desc] : psd.get_asic_descriptors()) {
        config.hostname_to_asics[desc.host_name].insert(asic_id);
    }

    const auto& pinnings = mgd.get_pinnings();
    for (const auto& [_, groups] : pinnings) {
        for (const auto& group : groups) {
            config.pinnings.push_back({group.fabric_nodes, group.asic_positions});
        }
    }

    if (!config.pinnings.empty()) {
        const auto& asic_descriptors = psd.get_asic_descriptors();
        for (const auto& [asic_id, _] : asic_descriptors) {
            auto tray_id = psd.get_tray_id(asic_id);
            auto asic_location = psd.get_asic_location(asic_id);
            config.asic_positions[asic_id] = std::make_pair(tray_id, asic_location);
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

    const auto mapping_result = map_multi_mesh_to_physical(
        psd, pgd, mgd, config, /*pinnings=*/{}, /*asic_id_to_mesh_rank=*/{}, fabric_ranks_from_mesh_graph(mesh_graph));
    ASSERT_TRUE(mapping_result.success) << mapping_result.error_message;

    EXPECT_EQ(mapping_result.fabric_node_to_asic.size(), expected_fabric_nodes);

    std::set<std::string> hosts_spanning_blitz_mapped;
    for (const auto& [fabric_node, asic_id] : mapping_result.fabric_node_to_asic) {
        (void)fabric_node;
        hosts_spanning_blitz_mapped.insert(psd.get_host_name_for_asic(asic_id));
    }
    EXPECT_GE(hosts_spanning_blitz_mapped.size(), 1u);
    EXPECT_LE(hosts_spanning_blitz_mapped.size(), 4u)
        << "Mapped Blitz pipeline: at most one host per logical 4×2 mesh (10 stages)";
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_Sp4Glx_Blitz2x4_11Stage) {
    // Same as MapMultiMeshToPhysical_Sp4Glx_Blitz2x4 but 11 pipeline stages
    // (bh_glx_11stage_4x2_pipeline.textproto).
    using namespace ::tt::tt_fabric;

    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();

    const std::filesystem::path pgd_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;
    PhysicalGroupingDescriptor pgd{pgd_path};

    constexpr std::size_t kPipelineStages = 11;
    constexpr std::size_t kAsicsPerStage = 8;

    const std::filesystem::path mgd_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_glx_11stage_4x2_pipeline.textproto";
    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;
    MeshGraphDescriptor mgd{mgd_path};

    MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, mgd_path.string());
    const std::size_t expected_fabric_nodes = chip_count(mesh_graph);
    ASSERT_EQ(expected_fabric_nodes, kPipelineStages * kAsicsPerStage);

    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = false;

    for (const auto& [asic_id, desc] : psd.get_asic_descriptors()) {
        config.hostname_to_asics[desc.host_name].insert(asic_id);
    }

    const auto& pinnings = mgd.get_pinnings();
    for (const auto& [_, groups] : pinnings) {
        for (const auto& group : groups) {
            config.pinnings.push_back({group.fabric_nodes, group.asic_positions});
        }
    }

    if (!config.pinnings.empty()) {
        const auto& asic_descriptors = psd.get_asic_descriptors();
        for (const auto& [asic_id, _] : asic_descriptors) {
            auto tray_id = psd.get_tray_id(asic_id);
            auto asic_location = psd.get_asic_location(asic_id);
            config.asic_positions[asic_id] = std::make_pair(tray_id, asic_location);
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

    const auto mapping_result = map_multi_mesh_to_physical(
        psd, pgd, mgd, config, /*pinnings=*/{}, /*asic_id_to_mesh_rank=*/{}, fabric_ranks_from_mesh_graph(mesh_graph));
    ASSERT_TRUE(mapping_result.success) << mapping_result.error_message;

    EXPECT_EQ(mapping_result.fabric_node_to_asic.size(), expected_fabric_nodes);

    std::set<std::string> hosts_spanning_blitz_mapped;
    for (const auto& [fabric_node, asic_id] : mapping_result.fabric_node_to_asic) {
        (void)fabric_node;
        hosts_spanning_blitz_mapped.insert(psd.get_host_name_for_asic(asic_id));
    }
    EXPECT_GE(hosts_spanning_blitz_mapped.size(), 1u);
    EXPECT_LE(hosts_spanning_blitz_mapped.size(), 5u)
        << "Mapped Blitz pipeline: at most one host per logical 4×2 mesh (11 stages)";
}

TEST_F(TopologyMapperUtilsTest, MapMultiMeshToPhysical_Sp4Glx_Blitz2x4_32Stage) {
    // Same as MapMultiMeshToPhysical_Sp4Glx_Blitz2x4 but 32 pipeline stages
    // (bh_glx_32stage_4x2_pipeline.textproto).
    using namespace ::tt::tt_fabric;

    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();

    const std::filesystem::path pgd_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/physical_groupings/bh_galaxy_rev_ab_physical_grouping_descriptor.textproto";
    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;
    PhysicalGroupingDescriptor pgd{pgd_path};

    constexpr std::size_t kPipelineStages = 32;
    constexpr std::size_t kAsicsPerStage = 8;

    const std::filesystem::path mgd_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_glx_32stage_4x2_pipeline.textproto";
    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;
    MeshGraphDescriptor mgd{mgd_path};

    MeshGraph mesh_graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, mgd_path.string());
    const std::size_t expected_fabric_nodes = chip_count(mesh_graph);
    ASSERT_EQ(expected_fabric_nodes, kPipelineStages * kAsicsPerStage);

    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = false;

    for (const auto& [asic_id, desc] : psd.get_asic_descriptors()) {
        config.hostname_to_asics[desc.host_name].insert(asic_id);
    }

    const auto& pinnings = mgd.get_pinnings();
    for (const auto& [_, groups] : pinnings) {
        for (const auto& group : groups) {
            config.pinnings.push_back({group.fabric_nodes, group.asic_positions});
        }
    }

    if (!config.pinnings.empty()) {
        const auto& asic_descriptors = psd.get_asic_descriptors();
        for (const auto& [asic_id, _] : asic_descriptors) {
            auto tray_id = psd.get_tray_id(asic_id);
            auto asic_location = psd.get_asic_location(asic_id);
            config.asic_positions[asic_id] = std::make_pair(tray_id, asic_location);
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

    const auto mapping_result = map_multi_mesh_to_physical(
        psd, pgd, mgd, config, /*pinnings=*/{}, /*asic_id_to_mesh_rank=*/{}, fabric_ranks_from_mesh_graph(mesh_graph));
    ASSERT_TRUE(mapping_result.success) << mapping_result.error_message;

    EXPECT_EQ(mapping_result.fabric_node_to_asic.size(), expected_fabric_nodes);

    std::set<std::string> hosts_spanning_blitz_mapped;
    for (const auto& [fabric_node, asic_id] : mapping_result.fabric_node_to_asic) {
        hosts_spanning_blitz_mapped.insert(psd.get_host_name_for_asic(asic_id));
    }
    EXPECT_EQ(hosts_spanning_blitz_mapped.size(), 8u) << "Mapped Blitz pipeline: should span exactly 8 hosts";
}

TEST_F(TopologyMapperUtilsTest, SweepConsumer_SolutionSpansExpectedHosts) {
    // Sweep consumer / workload: launched once per generate_rank_bindings --all-solutions solution by
    // sweep_rank_binding_solutions.py (via tt-run --rank-binding <that solution>), so the ambient
    // PhysicalSystemDescriptor reflects the hosts THAT solution occupies. Assert the solution spans exactly the
    // expected number of distinct hosts (default 8 for the 32-stage 2x4 ring pipeline on the SC36 subtorus:
    // 256 chips / 32 chips-per-galaxy = 8 galaxies). This checks that the multi-solution host-cap enforcement holds
    // on every enumerated solution end to end. Override the expectation via SWEEP_EXPECTED_HOSTS.
    const char* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP()
            << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run via the sweep / tt-run --mock-cluster-rank-binding";
    }

    std::size_t expected_hosts = 8;
    if (const char* env = getenv("SWEEP_EXPECTED_HOSTS"); env != nullptr && env[0] != '\0') {
        expected_hosts = static_cast<std::size_t>(std::stoul(env));
    }

    // In the mock each rank is a separate "board", so PhysicalSystemDescriptor reports a per-rank host_name
    // (e.g. "..._bh-glx-120-d05u02_rank_4.yaml"). Reduce each to its physical galaxy tag ("bh-glx-<aisle>-<node>")
    // so we count distinct galaxies (hosts) -- matching generate_rank_bindings' num_hosts, not the rank count.
    auto galaxy_tag = [](const std::string& h) -> std::string {
        auto pos = h.find("bh-glx-");
        if (pos == std::string::npos) {
            return h;  // non-bh-glx mock: fall back to the full host name
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

    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    ASSERT_NE(tt_metal_home, nullptr) << "TT_METAL_HOME environment variable must be set";

    auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (mock_desc == nullptr) {
        GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - run with tt-run --mock-cluster-rank-binding";
    }

    tt::tt_metal::PhysicalSystemDescriptor psd = create_psd_from_mock_cluster();

    // Rev C PGD half-pod 4x2_Mesh_horizontal ({1,3}/{2,4} tray pairs) + bh_galaxy_xyz torus links.
    const std::filesystem::path pgd_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/physical_groupings/wh_bh_rev_c_galaxy_physical_grouping_descriptor.textproto";
    ASSERT_TRUE(std::filesystem::exists(pgd_path)) << "PGD file not found: " << pgd_path;
    PhysicalGroupingDescriptor pgd{pgd_path};

    const std::filesystem::path mgd_path =
        std::filesystem::path(tt_metal_home) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_galaxy_2x4_pipeline.textproto";
    ASSERT_TRUE(std::filesystem::exists(mgd_path)) << "MGD file not found: " << mgd_path;
    MeshGraphDescriptor mgd{mgd_path};

    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = true;

    const auto mapping_result = map_multi_mesh_to_physical(psd, pgd, mgd, config);
    ASSERT_TRUE(mapping_result.success) << mapping_result.error_message;
    EXPECT_EQ(mapping_result.fabric_node_to_asic.size(), 32u);
}
}  // namespace tt::tt_metal::experimental::tt_fabric
