// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <filesystem>
#include <algorithm>
#include <unordered_set>
#include <yaml-cpp/yaml.h>

#include "fabric_fixture.hpp"
#include "t3k_mesh_descriptor_chip_mappings.hpp"
#include "utils.hpp"
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include "tt_metal/fabric/physical_system_discovery.hpp"
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include <tt-metalium/distributed_context.hpp>
#include <tt-logger/tt-logger.hpp>
#include <fmt/format.h>

namespace {

constexpr auto kFabricConfig = tt::tt_fabric::FabricConfig::FABRIC_2D;
constexpr auto kReliabilityMode = tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane(const std::filesystem::path& graph_desc) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(
        cluster, rtoptions, hal, distributed_context, graph_desc.string(), kFabricConfig, kReliabilityMode);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();

    return control_plane;
}

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane(
    const std::filesystem::path& graph_desc,
    const std::map<tt::tt_fabric::FabricNodeId, tt::ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(
        cluster,
        rtoptions,
        hal,
        distributed_context,
        graph_desc.string(),
        logical_mesh_chip_id_to_physical_chip_id_mapping,
        kFabricConfig,
        kReliabilityMode);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();

    return control_plane;
}

tt::tt_fabric::MeshGraph make_mesh_graph(const std::filesystem::path& mesh_graph_desc_file) {
    const tt::Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    return tt::tt_fabric::MeshGraph(cluster, mesh_graph_desc_file.string());
}

tt::tt_fabric::MeshGraph make_mesh_graph(
    const std::filesystem::path& mesh_graph_desc_file, tt::tt_fabric::FabricConfig fabric_config) {
    const tt::Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    return tt::tt_fabric::MeshGraph(cluster, mesh_graph_desc_file.string(), fabric_config);
}

// Helper struct to keep dependencies alive for RoutingTableGenerator tests
struct RoutingTableGeneratorTestHelper {
    std::unique_ptr<tt::tt_fabric::MeshGraph> mesh_graph;
    std::unique_ptr<tt::tt_metal::PhysicalSystemDescriptor> physical_system_descriptor;
    std::unique_ptr<tt::tt_fabric::TopologyMapper> topology_mapper;
    std::unique_ptr<tt::tt_fabric::RoutingTableGenerator> routing_table_generator;

    RoutingTableGeneratorTestHelper(const std::string& mesh_graph_desc_file) {
        const tt::Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        mesh_graph = std::make_unique<tt::tt_fabric::MeshGraph>(cluster, mesh_graph_desc_file);
        const auto& driver = cluster.get_driver();
        const auto& distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
        const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        auto& driver_ref = const_cast<tt::umd::Cluster&>(*driver);
        auto psd =
            tt::tt_metal::run_physical_system_discovery(driver_ref, distributed_context, rtoptions.get_target_device());
        physical_system_descriptor = std::make_unique<tt::tt_metal::PhysicalSystemDescriptor>(std::move(psd));

        tt::tt_fabric::LocalMeshBinding local_mesh_binding;
        local_mesh_binding.mesh_ids = {tt::tt_fabric::MeshId{0}};
        local_mesh_binding.host_rank = tt::tt_fabric::MeshHostRankId{0};

        topology_mapper = std::make_unique<tt::tt_fabric::TopologyMapper>(
            cluster, *distributed_context, *mesh_graph, *physical_system_descriptor, local_mesh_binding);

        routing_table_generator = std::make_unique<tt::tt_fabric::RoutingTableGenerator>(*topology_mapper);
    }
};

// Helper function to create RoutingTableGenerator for tests
std::unique_ptr<RoutingTableGeneratorTestHelper> make_routing_table_generator(const std::string& mesh_graph_desc_file) {
    return std::make_unique<RoutingTableGeneratorTestHelper>(mesh_graph_desc_file);
}

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane_1d_ring(const std::filesystem::path& graph_desc) {
    constexpr auto kFabricConfig1D = tt::tt_fabric::FabricConfig::FABRIC_1D_RING;
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(
        cluster, rtoptions, hal, distributed_context, graph_desc.string(), kFabricConfig1D, kReliabilityMode);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();

    return control_plane;
}

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane_1d(const std::filesystem::path& graph_desc) {
    constexpr auto kFabricConfig1D = tt::tt_fabric::FabricConfig::FABRIC_1D;
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(
        cluster, rtoptions, hal, distributed_context, graph_desc.string(), kFabricConfig1D, kReliabilityMode);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();

    return control_plane;
}

}  // namespace

namespace tt::tt_fabric::fabric_router_tests {

using ::testing::ElementsAre;

TEST(MeshGraphValidation, TestMGDConnections) {
    // TODO: This test is currently not implemented completely connection types currently cannot be mixed
    // Skip for now
    log_warning(tt::LogTest, "Skipping TestMGDConnections because connection types currently cannot be mixed");
    GTEST_SKIP();

    const std::filesystem::path test_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/mgd_test_connections.textproto";
    auto mesh_graph = make_mesh_graph(test_desc_path);

    auto connections = mesh_graph.get_requested_intermesh_connections();
    EXPECT_EQ(connections[0][1], 5); // 2 (relaxed) + 3 (mixed, treated as relaxed)

    auto ports = mesh_graph.get_requested_intermesh_ports();
    EXPECT_EQ(ports[0][1].size(), 1); // 1 strict connection
    auto [src_dev, dst_dev, count] = ports[0][1][0];
    EXPECT_EQ(src_dev, 0);
    EXPECT_EQ(dst_dev, 1);
    EXPECT_EQ(count, 1);

    // Bidirectional check
    EXPECT_EQ(connections[1][0], 5);
    EXPECT_EQ(ports[1][0].size(), 1);
    auto [rev_src_dev, rev_dst_dev, rev_count] = ports[1][0][0];
    EXPECT_EQ(rev_src_dev, 1);
    EXPECT_EQ(rev_dst_dev, 0);
    EXPECT_EQ(rev_count, 1);
}

TEST_F(ControlPlaneFixture, TestControlPlaneInitNoMGD) {
    // Reset MetalContext's control plane to ensure a clean state
    tt::tt_metal::MetalContext::instance().set_default_fabric_topology();

    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    // initialize_fabric_config() calls get_control_plane() which creates the control plane and writes the mapping file
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    EXPECT_NE(control_plane.get_mesh_graph().get_mesh_ids().size(), 0u);
}

TEST(MeshGraphValidation, TestT3kMeshGraphInit) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    auto mesh_graph_desc = make_mesh_graph(t3k_mesh_graph_desc_path);
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{0}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 3)));
}

TEST_F(ControlPlaneFixture, TestT3kControlPlaneInit) {
    // Reset MetalContext's control plane to ensure it doesn't interfere with the test's custom control plane
    tt::tt_metal::MetalContext::instance().set_default_fabric_topology();

    // Delete any existing mapping file to ensure we check the one written by the test's ControlPlane
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto& distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    int world_size = *distributed_context->size();
    int rank = *distributed_context->rank();
    std::filesystem::path root_dir = rtoptions.get_root_dir();
    std::filesystem::path generated_dir = root_dir / "generated" / "fabric";
    std::string generated_filename =
        "asic_to_fabric_node_mapping_rank_" + std::to_string(rank + 1) + "_of_" + std::to_string(world_size) + ".yaml";
    std::filesystem::path generated_file = generated_dir / generated_filename;
    if (std::filesystem::exists(generated_file)) {
        std::filesystem::remove(generated_file);
    }

    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane(t3k_mesh_graph_desc_path);

    check_asic_mapping_against_golden("TestT3kControlPlaneInit", "ControlPlaneFixture_T3k");
}

TEST_F(ControlPlaneFixture, TestT3kFabricRoutes) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane(t3k_mesh_graph_desc_path);

    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 7), chan);
        EXPECT_EQ(!path.empty(), true);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 1);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 7), chan);
        EXPECT_EQ(!path.empty(), true);
    }
}

TEST_F(ControlPlaneFixture, TestT3k1x8FabricRoutes) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane_1d_ring(t3k_mesh_graph_desc_path);

    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 7), chan);
        EXPECT_EQ(!path.empty(), true);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 1);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 7), chan);
        EXPECT_EQ(!path.empty(), true);
    }

    // Test that all forwarding directions are valid
    auto src_fabric_node_id = FabricNodeId(MeshId{0}, 0);
    for (unsigned int x = 1; x < 8; ++x) {
        auto dst_fabric_node_id = FabricNodeId(MeshId{0}, x);
        auto forwarding_direction = control_plane->get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
        EXPECT_EQ(forwarding_direction.has_value(), true);
    }
}

TEST_F(ControlPlaneFixture, TestSingleGalaxy1x32ControlPlaneInit) {
    const std::filesystem::path galaxy_6u_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_1x32_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane_1d(galaxy_6u_mesh_graph_desc_path);

    check_asic_mapping_against_golden("TestSingleGalaxy1x32ControlPlaneInit", "ControlPlaneFixture_SingleGalaxy_1x32");
}

TEST_F(ControlPlaneFixture, TestSingleGalaxy1x32FabricRoutes) {
    const std::filesystem::path galaxy_6u_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_1x32_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane_1d(galaxy_6u_mesh_graph_desc_path);

    // Test routing from first chip (0) to last chip (31) in the 1x32 topology
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 31), chan);
        EXPECT_EQ(!path.empty(), true);
    }

    // Test routing on second routing plane
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 1);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 31), chan);
        EXPECT_EQ(!path.empty(), true);
    }

    // Test that all forwarding directions are valid
    auto src_fabric_node_id = FabricNodeId(MeshId{0}, 0);
    for (unsigned int x = 1; x < 32; ++x) {
        auto dst_fabric_node_id = FabricNodeId(MeshId{0}, x);
        auto forwarding_direction = control_plane->get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
        EXPECT_EQ(forwarding_direction.has_value(), true);
    }
}

TEST_F(ControlPlaneFixture, TestSingleGalaxy1x16ControlPlaneInit) {
    GTEST_SKIP();
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_1D_RING,
        tt::tt_fabric::FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    const std::filesystem::path galaxy_6u_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x16_torus_graph_descriptor.textproto";
    auto control_plane = make_control_plane_1d(galaxy_6u_mesh_graph_desc_path);
}

TEST_F(ControlPlaneFixture, TestSingleGalaxy1x16FabricRoutes) {
    GTEST_SKIP();
    const std::filesystem::path galaxy_6u_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x16_torus_graph_descriptor.textproto";
    auto control_plane = make_control_plane_1d(galaxy_6u_mesh_graph_desc_path);

    // Test routing from first chip (0) to last chip (15) in the 1x16 topology
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 15), chan);
        EXPECT_EQ(!path.empty(), true);
    }

    // Test routing on second routing plane
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 1);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 15), chan);
        EXPECT_EQ(!path.empty(), true);
    }

    // Test that all forwarding directions are valid
    auto src_fabric_node_id = FabricNodeId(MeshId{0}, 0);
    for (unsigned int x = 1; x < 16; ++x) {
        auto dst_fabric_node_id = FabricNodeId(MeshId{0}, x);
        auto forwarding_direction = control_plane->get_forwarding_direction(src_fabric_node_id, dst_fabric_node_id);
        EXPECT_EQ(forwarding_direction.has_value(), true);
    }
}

class T3kCustomMeshGraphControlPlaneFixture
    : public ControlPlaneFixture,
      public testing::WithParamInterface<std::tuple<std::string, std::vector<std::vector<EthCoord>>>> {};

TEST_P(T3kCustomMeshGraphControlPlaneFixture, TestT3kMeshGraphInit) {
    auto [mesh_graph_desc_path, _] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto mesh_graph_desc = make_mesh_graph(t3k_mesh_graph_desc_path);
}

TEST_P(T3kCustomMeshGraphControlPlaneFixture, TestT3kControlPlaneInit) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();

    // Reset MetalContext's control plane to ensure it doesn't interfere with the test's custom control plane
    // This prevents MetalContext from creating an auto-discovery control plane that writes a mapping file first
    tt::tt_metal::MetalContext::instance().set_default_fabric_topology();

    // Delete any existing mapping file to ensure we check the one written by the test's ControlPlane
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto& distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    int world_size = *distributed_context->size();
    int rank = *distributed_context->rank();
    std::filesystem::path root_dir = rtoptions.get_root_dir();
    std::filesystem::path generated_dir = root_dir / "generated" / "fabric";
    std::string generated_filename =
        "asic_to_fabric_node_mapping_rank_" + std::to_string(rank + 1) + "_of_" + std::to_string(world_size) + ".yaml";
    std::filesystem::path generated_file = generated_dir / generated_filename;
    if (std::filesystem::exists(generated_file)) {
        std::filesystem::remove(generated_file);
    }

    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto control_plane = make_control_plane(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));

    // Extract MGD filename (without extension) for golden file naming
    std::filesystem::path mgd_path(mesh_graph_desc_path);
    std::string mgd_filename = mgd_path.stem().string();
    // Replace any special characters that might cause issues in filenames
    std::replace(mgd_filename.begin(), mgd_filename.end(), '/', '_');
    std::replace(mgd_filename.begin(), mgd_filename.end(), '-', '_');

    // Extract parameter index from test name (format: TestName/Index)
    const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string full_test_name = std::string(test_info->test_suite_name()) + "/" + test_info->name();
    int param_index = 0;
    // Test name format: TestSuiteName/TestName/Index
    size_t last_slash = full_test_name.find_last_of('/');
    if (last_slash != std::string::npos && last_slash + 1 < full_test_name.length()) {
        try {
            param_index = std::stoi(full_test_name.substr(last_slash + 1));
        } catch (...) {
            // If parsing fails, try to find index by searching through parameters
            for (size_t i = 0; i < t3k_mesh_descriptor_chip_mappings.size(); ++i) {
                if (std::get<0>(t3k_mesh_descriptor_chip_mappings[i]) == mesh_graph_desc_path &&
                    std::get<1>(t3k_mesh_descriptor_chip_mappings[i]) == mesh_graph_eth_coords) {
                    param_index = i;
                    break;
                }
            }
        }
    }

    std::string golden_name =
        "T3kCustomMeshGraph_TestT3kControlPlaneInit_" + mgd_filename + "_" + std::to_string(param_index);

    check_asic_mapping_against_golden("TestT3kControlPlaneInit", golden_name);
}

TEST_P(T3kCustomMeshGraphControlPlaneFixture, TestT3kFabricRoutes) {
    std::srand(std::time(nullptr));  // Seed the RNG
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto control_plane = make_control_plane(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));

    for (const auto& src_mesh : control_plane->get_user_physical_mesh_ids()) {
        for (const auto& dst_mesh : control_plane->get_user_physical_mesh_ids()) {
            auto src_mesh_shape = control_plane->get_physical_mesh_shape(src_mesh);
            auto src_mesh_size = src_mesh_shape.mesh_size();
            auto dst_mesh_shape = control_plane->get_physical_mesh_shape(dst_mesh);
            auto dst_mesh_size = dst_mesh_shape.mesh_size();
            auto src_fabric_node_id = FabricNodeId(src_mesh, std::rand() % src_mesh_size);
            auto active_fabric_eth_channels = control_plane->get_active_fabric_eth_channels(src_fabric_node_id);
            EXPECT_GT(active_fabric_eth_channels.size(), 0);
            for (auto [chan, direction] : active_fabric_eth_channels) {
                auto dst_fabric_node_id = FabricNodeId(dst_mesh, std::rand() % dst_mesh_size);
                auto path = control_plane->get_fabric_route(src_fabric_node_id, dst_fabric_node_id, chan);
                EXPECT_EQ(src_fabric_node_id == dst_fabric_node_id ? path.empty() : !path.empty(), true);
            }
        }
    }
}

TEST_F(ControlPlaneFixture, TestT3kDisjointFabricRoutes) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = t3k_disjoint_mesh_descriptor_chip_mappings[0];
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    auto control_plane = make_control_plane(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));

    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 3), chan);
        EXPECT_EQ(!path.empty(), true);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{1}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 3), chan);
        EXPECT_EQ(!path.empty(), true);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{1}, 3), chan);
        EXPECT_EQ(path.empty(), true);
        auto direction = control_plane->get_forwarding_direction(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{1}, 3));
        EXPECT_EQ(direction.has_value(), false);
    }
}

INSTANTIATE_TEST_SUITE_P(
    T3kCustomMeshGraphControlPlaneTests,
    T3kCustomMeshGraphControlPlaneFixture,
    ::testing::ValuesIn(t3k_mesh_descriptor_chip_mappings));

TEST_F(ControlPlaneFixture, TestSingleGalaxyControlPlaneInit) {
    const std::filesystem::path single_galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane(single_galaxy_mesh_graph_desc_path.string());

    expect_galaxy_corner_folding_check(*control_plane);

    // Create physical system descriptor to access ASIC information
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& driver = cluster.get_driver();
    const auto& distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    auto& driver_ref = const_cast<tt::umd::Cluster&>(*driver);
    auto psd =
        tt::tt_metal::run_physical_system_discovery(driver_ref, distributed_context, rtoptions.get_target_device());
    auto physical_system_descriptor = std::make_unique<tt::tt_metal::PhysicalSystemDescriptor>(std::move(psd));

    // Test that fabric node id 0 maps to a valid ASIC location and tray id
    FabricNodeId fabric_node_id_0(MeshId{0}, 0);
    auto physical_chip_id_0 = control_plane->get_physical_chip_id_from_fabric_node_id(fabric_node_id_0);
    const auto& chip_unique_ids = cluster.get_unique_chip_ids();
    uint64_t asic_id_0 = 0;
    for (const auto& [chip_id, unique_id] : chip_unique_ids) {
        if (chip_id == physical_chip_id_0) {
            asic_id_0 = unique_id;
            break;
        }
    }
    EXPECT_GT(asic_id_0, 0) << "ASIC ID should be greater than 0 for fabric node id 0";
    auto tray_id_0 = physical_system_descriptor->get_tray_id(tt::tt_metal::AsicID{asic_id_0});
    auto asic_location_0 = physical_system_descriptor->get_asic_location(tt::tt_metal::AsicID{asic_id_0});
    EXPECT_GT(*tray_id_0, 0) << "Tray ID should be greater than 0 for fabric node id 0";
    EXPECT_GE(*asic_location_0, 0) << "ASIC location should be non-negative for fabric node id 0";

    // Test that fabric node id 1 maps to tray 1, ASIC location 5 (per pinnings)
    FabricNodeId fabric_node_id_1(MeshId{0}, 1);
    auto physical_chip_id_1 = control_plane->get_physical_chip_id_from_fabric_node_id(fabric_node_id_1);
    uint64_t asic_id_1 = 0;
    for (const auto& [chip_id, unique_id] : chip_unique_ids) {
        if (chip_id == physical_chip_id_1) {
            asic_id_1 = unique_id;
            break;
        }
    }
    EXPECT_GT(asic_id_1, 0) << "ASIC ID should be greater than 0 for fabric node id 1";
    auto tray_id_1 = physical_system_descriptor->get_tray_id(tt::tt_metal::AsicID{asic_id_1});
    auto asic_location_1 = physical_system_descriptor->get_asic_location(tt::tt_metal::AsicID{asic_id_1});
    EXPECT_EQ(*tray_id_1, 1) << "Fabric node id 1 should map to tray ID 1";
    EXPECT_EQ(*asic_location_1, 5) << "Fabric node id 1 should map to ASIC location 5";

    // Test that fabric node id y_size (4) maps to tray 1, ASIC location 2 (per pinnings)
    int y_size = control_plane->get_physical_mesh_shape(MeshId{0})[1];
    FabricNodeId fabric_node_id_y_size(MeshId{0}, y_size);
    auto physical_chip_id_y_size = control_plane->get_physical_chip_id_from_fabric_node_id(fabric_node_id_y_size);
    uint64_t asic_id_y_size = 0;
    for (const auto& [chip_id, unique_id] : chip_unique_ids) {
        if (chip_id == physical_chip_id_y_size) {
            asic_id_y_size = unique_id;
            break;
        }
    }
    EXPECT_GT(asic_id_y_size, 0) << "ASIC ID should be greater than 0 for fabric node id " << y_size;
    auto tray_id_y_size = physical_system_descriptor->get_tray_id(tt::tt_metal::AsicID{asic_id_y_size});
    auto asic_location_y_size = physical_system_descriptor->get_asic_location(tt::tt_metal::AsicID{asic_id_y_size});
    EXPECT_EQ(*tray_id_y_size, 1) << "Fabric node id " << y_size << " should map to tray ID 1";
    EXPECT_EQ(*asic_location_y_size, 2) << "Fabric node id " << y_size << " should map to ASIC location 2";

    check_asic_mapping_against_golden("TestSingleGalaxyControlPlaneInit", "ControlPlaneFixture_SingleGalaxy");
}

TEST_F(ControlPlaneFixture, TestSingleGalaxyMeshAPIs) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    EXPECT_EQ(user_meshes.size(), 1);
    EXPECT_EQ(user_meshes[0], MeshId{0});
    auto mesh_shape = control_plane.get_physical_mesh_shape(MeshId{0});
    EXPECT_TRUE(
        mesh_shape == tt::tt_metal::distributed::MeshShape(8, 4) ||
        mesh_shape == tt::tt_metal::distributed::MeshShape(4, 8))
        << "Expected mesh shape to be either 8x4 or 4x8, got: (" << mesh_shape[0] << "x" << mesh_shape[1] << ")";
}

TEST(MeshGraphValidation, TestT3kDualHostMeshGraph) {
    const std::filesystem::path t3k_dual_host_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.textproto";
    auto mesh_graph = make_mesh_graph(t3k_dual_host_mesh_graph_desc_path);

    EXPECT_THAT(mesh_graph.get_mesh_ids(), ElementsAre(MeshId{0}));

    // Check host ranks by accessing the values vector
    const auto& host_ranks = mesh_graph.get_host_ranks(MeshId{0});
    EXPECT_EQ(host_ranks, MeshContainer<MeshHostRankId>(MeshShape(1, 2), {MeshHostRankId(0), MeshHostRankId(1)}));

    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}), MeshShape(2, 4));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}, MeshHostRankId(0)), MeshShape(2, 2));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}, MeshHostRankId(1)), MeshShape(2, 2));

    EXPECT_EQ(mesh_graph.get_coord_range(MeshId{0}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 3)));
    EXPECT_EQ(
        mesh_graph.get_coord_range(MeshId{0}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    EXPECT_EQ(
        mesh_graph.get_coord_range(MeshId{0}, MeshHostRankId(1)),
        MeshCoordinateRange(MeshCoordinate(0, 2), MeshCoordinate(1, 3)));

    EXPECT_THAT(mesh_graph.get_mesh_ids(), ElementsAre(MeshId{0}));

    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{0}),
        MeshContainer<ChipId>(MeshShape(2, 4), std::vector<ChipId>{0, 1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{0}, MeshHostRankId(0)),
        MeshContainer<ChipId>(MeshShape(2, 2), std::vector<ChipId>{0, 1, 4, 5}));
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{0}, MeshHostRankId(1)),
        MeshContainer<ChipId>(MeshShape(2, 2), std::vector<ChipId>{2, 3, 6, 7}));
}

TEST(MeshGraphValidation, TestT3k2x2MeshGraph) {
    const std::filesystem::path t3k_2x2_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto";
    auto mesh_graph = make_mesh_graph(t3k_2x2_mesh_graph_desc_path);

    // This configuration has two meshes (id 0 and id 1)
    EXPECT_THAT(mesh_graph.get_mesh_ids(), ElementsAre(MeshId{0}, MeshId{1}));

    // Check host ranks for mesh 0 - single host rank 0
    const auto& host_ranks_mesh0 = mesh_graph.get_host_ranks(MeshId{0});
    EXPECT_EQ(host_ranks_mesh0, MeshContainer<MeshHostRankId>(MeshShape(1, 1), {MeshHostRankId(0)}));

    // Check host ranks for mesh 1 - single host rank 0
    const auto& host_ranks_mesh1 = mesh_graph.get_host_ranks(MeshId{1});
    EXPECT_EQ(host_ranks_mesh1, MeshContainer<MeshHostRankId>(MeshShape(1, 1), {MeshHostRankId(0)}));

    // Each mesh has a 2x2 board topology
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}), MeshShape(2, 2));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{1}), MeshShape(2, 2));

    // Since there's only one host rank per mesh, mesh shape should be same
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}, MeshHostRankId(0)), MeshShape(2, 2));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{1}, MeshHostRankId(0)), MeshShape(2, 2));

    // Check coordinate ranges
    EXPECT_EQ(mesh_graph.get_coord_range(MeshId{0}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    EXPECT_EQ(mesh_graph.get_coord_range(MeshId{1}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));

    // Since each mesh has only one host rank, the coord range should be the same
    EXPECT_EQ(
        mesh_graph.get_coord_range(MeshId{0}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));
    EXPECT_EQ(
        mesh_graph.get_coord_range(MeshId{1}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 1)));

    // Check chip IDs - each mesh has 4 chips (2x2)
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{0}), MeshContainer<ChipId>(MeshShape(2, 2), std::vector<ChipId>{0, 1, 2, 3}));
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{1}), MeshContainer<ChipId>(MeshShape(2, 2), std::vector<ChipId>{0, 1, 2, 3}));

    // Check chip IDs per host rank
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{0}, MeshHostRankId(0)),
        MeshContainer<ChipId>(MeshShape(2, 2), std::vector<ChipId>{0, 1, 2, 3}));
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{1}, MeshHostRankId(0)),
        MeshContainer<ChipId>(MeshShape(2, 2), std::vector<ChipId>{0, 1, 2, 3}));

    // Check that the number of intra-mesh connections match the number of connections in the graph
    EXPECT_EQ(mesh_graph.get_intra_mesh_connectivity()[0][0].begin()->second.connected_chip_ids.size(), 2);
    EXPECT_EQ(mesh_graph.get_intra_mesh_connectivity()[0][0].size(), 2);
}

TEST(MeshGraphValidation, TestGetHostRankForChip) {
    // Test with dual host T3K configuration
    const std::filesystem::path t3k_dual_host_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.textproto";
    auto mesh_graph = make_mesh_graph(t3k_dual_host_mesh_graph_desc_path);

    // Test valid chips for mesh 0
    // Based on the dual host configuration:
    // Host rank 0 controls chips 0, 1, 4, 5 (left board)
    // Host rank 1 controls chips 2, 3, 6, 7 (right board)
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 0), MeshHostRankId(0));
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 1), MeshHostRankId(0));
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 4), MeshHostRankId(0));
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 5), MeshHostRankId(0));

    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 2), MeshHostRankId(1));
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 3), MeshHostRankId(1));
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 6), MeshHostRankId(1));
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 7), MeshHostRankId(1));

    // Test invalid chip IDs (out of range)
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 8), std::nullopt);
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{0}, 100), std::nullopt);

    // Test invalid mesh ID
    EXPECT_EQ(mesh_graph.get_host_rank_for_chip(MeshId{1}, 0), std::nullopt);

    // Test with single host T3K configuration
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    auto mesh_graph_single_host = make_mesh_graph(t3k_mesh_graph_desc_path);

    // In single host configuration, all chips should belong to host rank 0
    for (ChipId chip_id = 0; chip_id < 8; chip_id++) {
        EXPECT_EQ(mesh_graph_single_host.get_host_rank_for_chip(MeshId{0}, chip_id), MeshHostRankId(0));
    }

    // Test with 2x2 configuration (two separate meshes)
    const std::filesystem::path t3k_2x2_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto";
    auto mesh_graph_2x2 = make_mesh_graph(t3k_2x2_mesh_graph_desc_path);

    // Each mesh has only one host rank (0)
    for (ChipId chip_id = 0; chip_id < 4; chip_id++) {
        EXPECT_EQ(mesh_graph_2x2.get_host_rank_for_chip(MeshId{0}, chip_id), MeshHostRankId(0));
        EXPECT_EQ(mesh_graph_2x2.get_host_rank_for_chip(MeshId{1}, chip_id), MeshHostRankId(0));
    }

    // Test invalid chip IDs for 2x2 configuration
    EXPECT_EQ(mesh_graph_2x2.get_host_rank_for_chip(MeshId{0}, 4), std::nullopt);
    EXPECT_EQ(mesh_graph_2x2.get_host_rank_for_chip(MeshId{1}, 4), std::nullopt);
}

TEST(MeshGraphValidation, TestExplicitShapeValidationNegative) {
    // Test that invalid shapes are properly rejected
    const std::filesystem::path invalid_shape_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_invalid_shape_mesh_graph_descriptor.textproto";

    // This should throw an exception due to incompatible shape
    EXPECT_THROW(make_mesh_graph(invalid_shape_mesh_graph_desc_path), std::exception);
}

namespace single_galaxy_constants {
constexpr std::uint32_t mesh_size = 32;  // 8x4 mesh
constexpr std::uint32_t mesh_row_size = 4;
constexpr std::uint32_t num_ports_per_side = 4;
constexpr std::uint32_t nw_fabric_id = 0;
constexpr std::uint32_t ne_fabric_id = 3;
constexpr std::uint32_t sw_fabric_id = 28;
constexpr std::uint32_t se_fabric_id = 31;
}  // namespace single_galaxy_constants

TEST(MeshGraphValidation, TestSingleGalaxyMesh) {
    using namespace single_galaxy_constants;
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto";
    auto mesh_graph = make_mesh_graph(mesh_graph_desc_path);
    const auto& intra_mesh_connectivity = mesh_graph.get_intra_mesh_connectivity();

    EXPECT_EQ(intra_mesh_connectivity.size(), 1);

    EXPECT_EQ(intra_mesh_connectivity[0].size(), mesh_size);
    for (std::uint32_t i = 0; i < mesh_size; ++i) {
        int N = i - mesh_row_size;  // North neighbor
        int E = i + 1;              // East neighbor
        int S = i + mesh_row_size;  // South neighbor
        int W = i - 1;              // West neighbor

        auto row = i / mesh_row_size;
        auto col = i % mesh_row_size;
        int N_wrap = (i - mesh_row_size + mesh_size) % mesh_size;
        int E_wrap = (row * mesh_row_size) + ((col + 1) % mesh_row_size);
        int S_wrap = (i + mesh_row_size) % mesh_size;
        int W_wrap = (row * mesh_row_size) + ((col - 1 + mesh_row_size) % mesh_row_size);

        // _wrap represents the wrapped neighbor indices
        // if X == X_wrap, it means that the neighbor is within the mesh and should be connected
        // if not, the neighbour represents a wrap-around connection and should not be present in a MESH
        if (N == N_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(N_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(N_wrap).port_direction, RoutingDirection::N);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(N_wrap).connected_chip_ids,
                std::vector<ChipId>(num_ports_per_side, N_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(N_wrap), 0);
        }
        if (E == E_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(E_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(E_wrap).port_direction, RoutingDirection::E);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(E_wrap).connected_chip_ids,
                std::vector<ChipId>(num_ports_per_side, E_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(E_wrap), 0);
        }
        if (S == S_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(S_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(S_wrap).port_direction, RoutingDirection::S);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(S_wrap).connected_chip_ids,
                std::vector<ChipId>(num_ports_per_side, S_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(S_wrap), 0);
        }
        if (W == W_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(W_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(W_wrap).port_direction, RoutingDirection::W);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(W_wrap).connected_chip_ids,
                std::vector<ChipId>(num_ports_per_side, W_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(W_wrap), 0);
        }
    }
}

TEST(RoutingTableValidation, TestSingleGalaxyMesh) {
    using namespace single_galaxy_constants;
    // Testing XY dimension order routing, if algorithm changes we can remove this test
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto";
    auto helper = make_routing_table_generator(mesh_graph_desc_path.string());
    const auto& intra_mesh_routing_table = helper->routing_table_generator->get_intra_mesh_table();

    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][nw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][ne_fabric_id], RoutingDirection::E);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][sw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][se_fabric_id], RoutingDirection::S);

    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][nw_fabric_id], RoutingDirection::W);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][ne_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][sw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][se_fabric_id], RoutingDirection::S);

    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][nw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][ne_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][sw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][se_fabric_id], RoutingDirection::E);

    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][nw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][ne_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][sw_fabric_id], RoutingDirection::W);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][se_fabric_id], RoutingDirection::C);
}

TEST(MeshGraphValidation, TestSingleGalaxyTorusXY) {
    using namespace single_galaxy_constants;
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_xy_graph_descriptor.textproto";
    auto mesh_graph = make_mesh_graph(mesh_graph_desc_path);
    const auto& intra_mesh_connectivity = mesh_graph.get_intra_mesh_connectivity();

    EXPECT_EQ(intra_mesh_connectivity.size(), 1);

    EXPECT_EQ(intra_mesh_connectivity[0].size(), mesh_size);
    for (std::uint32_t i = 0; i < mesh_size; ++i) {
        auto row = i / mesh_row_size;
        auto col = i % mesh_row_size;
        int N_wrap = (i - mesh_row_size + mesh_size) % mesh_size;
        int E_wrap = (row * mesh_row_size) + ((col + 1) % mesh_row_size);
        int S_wrap = (i + mesh_row_size) % mesh_size;
        int W_wrap = (row * mesh_row_size) + ((col - 1 + mesh_row_size) % mesh_row_size);

        // _wrap represents the wrapped neighbor indices
        // check all neighbors including wrap-around connections are present in TORUS_XY
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(N_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(N_wrap).port_direction, RoutingDirection::N);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(N_wrap).connected_chip_ids,
            std::vector<ChipId>(num_ports_per_side, N_wrap));
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(E_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(E_wrap).port_direction, RoutingDirection::E);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(E_wrap).connected_chip_ids,
            std::vector<ChipId>(num_ports_per_side, E_wrap));
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(S_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(S_wrap).port_direction, RoutingDirection::S);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(S_wrap).connected_chip_ids,
            std::vector<ChipId>(num_ports_per_side, S_wrap));
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(W_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(W_wrap).port_direction, RoutingDirection::W);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(W_wrap).connected_chip_ids,
            std::vector<ChipId>(num_ports_per_side, W_wrap));
    }
}

TEST(RoutingTableValidation, TestSingleGalaxyTorusXY) {
    using namespace single_galaxy_constants;
    // Testing XY dimension order routing, if algorithm changes we can remove this test
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_xy_graph_descriptor.textproto";
    auto helper = make_routing_table_generator(mesh_graph_desc_path.string());
    const auto& intra_mesh_routing_table = helper->routing_table_generator->get_intra_mesh_table();

    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][nw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][ne_fabric_id], RoutingDirection::W);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][sw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][se_fabric_id], RoutingDirection::N);

    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][nw_fabric_id], RoutingDirection::E);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][ne_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][sw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][se_fabric_id], RoutingDirection::N);

    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][nw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][ne_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][sw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][se_fabric_id], RoutingDirection::W);

    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][nw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][ne_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][sw_fabric_id], RoutingDirection::E);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][se_fabric_id], RoutingDirection::C);
}

TEST(MeshGraphValidation, TestSingleGalaxyTorusX) {
    using namespace single_galaxy_constants;
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_x_graph_descriptor.textproto";
    auto mesh_graph = make_mesh_graph(mesh_graph_desc_path);
    const auto& intra_mesh_connectivity = mesh_graph.get_intra_mesh_connectivity();

    EXPECT_EQ(intra_mesh_connectivity.size(), 1);

    EXPECT_EQ(intra_mesh_connectivity[0].size(), mesh_size);
    for (std::uint32_t i = 0; i < mesh_size; ++i) {
        int N = i - mesh_row_size;  // North neighbor
        int S = i + mesh_row_size;  // South neighbor
        auto row = i / mesh_row_size;
        auto col = i % mesh_row_size;
        int N_wrap = (i - mesh_row_size + mesh_size) % mesh_size;
        int E_wrap = (row * mesh_row_size) + ((col + 1) % mesh_row_size);
        int S_wrap = (i + mesh_row_size) % mesh_size;
        int W_wrap = (row * mesh_row_size) + ((col - 1 + mesh_row_size) % mesh_row_size);

        // _wrap represents the wrapped neighbor indices
        // if X == X_wrap, it means that the neighbor is within the mesh and should be connected
        // if not, the neighbour represents a wrap-around connection and should not be present
        // in a TORUS_X configuration, we expect wrap around for E/W directions
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(E_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(E_wrap).port_direction, RoutingDirection::E);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(E_wrap).connected_chip_ids,
            std::vector<ChipId>(num_ports_per_side, E_wrap));
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(W_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(W_wrap).port_direction, RoutingDirection::W);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(W_wrap).connected_chip_ids,
            std::vector<ChipId>(num_ports_per_side, W_wrap));
        if (N == N_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(N_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(N_wrap).port_direction, RoutingDirection::N);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(N_wrap).connected_chip_ids,
                std::vector<ChipId>(num_ports_per_side, N_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(N_wrap), 0);
        }
        if (S == S_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(S_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(S_wrap).port_direction, RoutingDirection::S);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(S_wrap).connected_chip_ids,
                std::vector<ChipId>(num_ports_per_side, S_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(S_wrap), 0);
        }
    }
}

TEST(RoutingTableValidation, TestSingleGalaxyTorusX) {
    using namespace single_galaxy_constants;
    // Testing XY dimension order routing, if algorithm changes we can remove this test
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_x_graph_descriptor.textproto";
    auto helper = make_routing_table_generator(mesh_graph_desc_path.string());
    const auto& intra_mesh_routing_table = helper->routing_table_generator->get_intra_mesh_table();

    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][nw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][ne_fabric_id], RoutingDirection::W);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][sw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][se_fabric_id], RoutingDirection::S);

    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][nw_fabric_id], RoutingDirection::E);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][ne_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][sw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][se_fabric_id], RoutingDirection::S);

    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][nw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][ne_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][sw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][se_fabric_id], RoutingDirection::W);

    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][nw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][ne_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][sw_fabric_id], RoutingDirection::E);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][se_fabric_id], RoutingDirection::C);
}

TEST(MeshGraphValidation, TestSingleGalaxyTorusY) {
    using namespace single_galaxy_constants;
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_y_graph_descriptor.textproto";
    auto mesh_graph = make_mesh_graph(mesh_graph_desc_path);
    const auto& intra_mesh_connectivity = mesh_graph.get_intra_mesh_connectivity();

    EXPECT_EQ(intra_mesh_connectivity.size(), 1);

    EXPECT_EQ(intra_mesh_connectivity[0].size(), mesh_size);
    for (std::uint32_t i = 0; i < mesh_size; ++i) {
        int E = i + 1;  // East neighbor
        int W = i - 1;  // West neighbor
        auto row = i / mesh_row_size;
        auto col = i % mesh_row_size;
        int N_wrap = (i - mesh_row_size + mesh_size) % mesh_size;
        int E_wrap = (row * mesh_row_size) + ((col + 1) % mesh_row_size);
        int S_wrap = (i + mesh_row_size) % mesh_size;
        int W_wrap = (row * mesh_row_size) + ((col - 1 + mesh_row_size) % mesh_row_size);

        // _wrap represents the wrapped neighbor indices
        // if X == X_wrap, it means that the neighbor is within the mesh and should be connected
        // if not, the neighbour represents a wrap-around connection and should not be present
        // in a TORUS_Y configuration, we expect wrap around for N/S directions
        EXPECT_EQ(intra_mesh_connectivity[0][i].count(N_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(N_wrap).port_direction, RoutingDirection::N);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(N_wrap).connected_chip_ids,
            std::vector<ChipId>(num_ports_per_side, N_wrap));

        EXPECT_EQ(intra_mesh_connectivity[0][i].count(S_wrap), 1);
        EXPECT_EQ(intra_mesh_connectivity[0][i].at(S_wrap).port_direction, RoutingDirection::S);
        EXPECT_EQ(
            intra_mesh_connectivity[0][i].at(S_wrap).connected_chip_ids,
            std::vector<ChipId>(num_ports_per_side, S_wrap));
        if (E == E_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(E_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(E_wrap).port_direction, RoutingDirection::E);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(E_wrap).connected_chip_ids,
                std::vector<ChipId>(num_ports_per_side, E_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(E_wrap), 0);
        }
        if (W == W_wrap) {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(W_wrap), 1);
            EXPECT_EQ(intra_mesh_connectivity[0][i].at(W_wrap).port_direction, RoutingDirection::W);
            EXPECT_EQ(
                intra_mesh_connectivity[0][i].at(W_wrap).connected_chip_ids,
                std::vector<ChipId>(num_ports_per_side, W_wrap));
        } else {
            EXPECT_EQ(intra_mesh_connectivity[0][i].count(W_wrap), 0);
        }
    }
}

TEST(RoutingTableValidation, TestSingleGalaxyTorusY) {
    using namespace single_galaxy_constants;
    // Testing XY dimension order routing, if algorithm changes we can remove this test
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_y_graph_descriptor.textproto";
    auto helper = make_routing_table_generator(mesh_graph_desc_path.string());
    const auto& intra_mesh_routing_table = helper->routing_table_generator->get_intra_mesh_table();

    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][nw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][ne_fabric_id], RoutingDirection::E);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][sw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][nw_fabric_id][se_fabric_id], RoutingDirection::N);

    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][nw_fabric_id], RoutingDirection::W);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][ne_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][sw_fabric_id], RoutingDirection::N);
    EXPECT_EQ(intra_mesh_routing_table[0][ne_fabric_id][se_fabric_id], RoutingDirection::N);

    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][nw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][ne_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][sw_fabric_id], RoutingDirection::C);
    EXPECT_EQ(intra_mesh_routing_table[0][sw_fabric_id][se_fabric_id], RoutingDirection::E);

    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][nw_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][ne_fabric_id], RoutingDirection::S);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][sw_fabric_id], RoutingDirection::W);
    EXPECT_EQ(intra_mesh_routing_table[0][se_fabric_id][se_fabric_id], RoutingDirection::C);
}

TEST(MeshGraphValidation, TestDualGalaxyMeshGraph) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto";
    auto mesh_graph_desc = make_mesh_graph(mesh_graph_desc_path);
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{0}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(7, 3)));
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{0}, MeshHostRankId(1)),
        MeshCoordinateRange(MeshCoordinate(0, 4), MeshCoordinate(7, 7)));
}

TEST(MeshGraphValidation, TestSingleGalaxy1x32MeshGraph) {
    const std::filesystem::path galaxy_6u_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/galaxy_1x32_mesh_graph_descriptor.textproto";
    auto mesh_graph_desc = make_mesh_graph(galaxy_6u_mesh_graph_desc_path);

    // Verify the mesh has correct topology for 1x32 Galaxy configuration
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{0}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 31)));

    // Verify mesh shape is 1x32
    EXPECT_EQ(mesh_graph_desc.get_mesh_shape(MeshId{0}), MeshShape(1, 32));

    // Verify there's only one mesh
    EXPECT_EQ(mesh_graph_desc.get_mesh_ids().size(), 1);
    EXPECT_EQ(mesh_graph_desc.get_mesh_ids()[0], MeshId{0});
}

// Black hole tests for p150, p100, p150 x8
TEST(MeshGraphValidation, TestP150BlackHoleMeshGraph) {
    const std::filesystem::path p150_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto";
    auto mesh_graph = make_mesh_graph(p150_mesh_graph_desc_path);

    EXPECT_THAT(mesh_graph.get_mesh_ids(), ElementsAre(MeshId{0}));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}), MeshShape(1, 1));
    EXPECT_EQ(mesh_graph.get_coord_range(MeshId{0}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));

    // Check chip IDs - single chip
    EXPECT_EQ(mesh_graph.get_chip_ids(MeshId{0}), MeshContainer<ChipId>(MeshShape(1, 1), std::vector<ChipId>{0}));
}

TEST_F(ControlPlaneFixture, TestP150BlackHoleControlPlaneInit) {
    const std::filesystem::path p150_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto";
    [[maybe_unused]] auto control_plane = make_control_plane(p150_mesh_graph_desc_path);
}

TEST(MeshGraphValidation, TestP100BlackHoleMeshGraph) {
    const std::filesystem::path p100_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto";
    auto mesh_graph = make_mesh_graph(p100_mesh_graph_desc_path);

    EXPECT_THAT(mesh_graph.get_mesh_ids(), ElementsAre(MeshId{0}));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}), MeshShape(1, 1));
    EXPECT_EQ(mesh_graph.get_coord_range(MeshId{0}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));

    // Check chip IDs - single chip
    EXPECT_EQ(mesh_graph.get_chip_ids(MeshId{0}), MeshContainer<ChipId>(MeshShape(1, 1), std::vector<ChipId>{0}));
}

TEST_F(ControlPlaneFixture, TestP100BlackHoleControlPlaneInit) {
    const std::filesystem::path p100_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto";
    [[maybe_unused]] auto control_plane = make_control_plane(p100_mesh_graph_desc_path);
}

TEST(MeshGraphValidation, TestP150X8BlackHoleMeshGraph) {
    const std::filesystem::path p150_x8_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto";
    auto mesh_graph = make_mesh_graph(p150_x8_mesh_graph_desc_path);

    EXPECT_THAT(mesh_graph.get_mesh_ids(), ElementsAre(MeshId{0}));
    EXPECT_EQ(mesh_graph.get_mesh_shape(MeshId{0}), MeshShape(2, 4));
    EXPECT_EQ(mesh_graph.get_coord_range(MeshId{0}), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 3)));

    // Check chip IDs - 8 chips in 2x4 configuration
    EXPECT_EQ(
        mesh_graph.get_chip_ids(MeshId{0}),
        MeshContainer<ChipId>(MeshShape(2, 4), std::vector<ChipId>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST_F(ControlPlaneFixture, TestP150X8BlackHoleControlPlaneInit) {
    const std::filesystem::path p150_x8_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto";
    [[maybe_unused]] auto control_plane = make_control_plane(p150_x8_mesh_graph_desc_path);
}

TEST_F(ControlPlaneFixture, TestP150X8BlackHoleFabricRoutes) {
    const std::filesystem::path p150_x8_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane(p150_x8_mesh_graph_desc_path);

    // Test routing between different chips in the 2x4 mesh
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(FabricNodeId(MeshId{0}, 0), 0);
    EXPECT_GT(valid_chans.size(), 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 7), chan);
        EXPECT_EQ(!path.empty(), true);
    }
}

// Test that FabricConfig can restrict torus topology to mesh (ignore wrap-around links)
TEST(MeshGraphValidation, TestFabricConfigOverrideTorusToMesh) {
    using namespace single_galaxy_constants;
    // Use existing torus XY MGD - this MGD has physical wrap-around connections
    const std::filesystem::path torus_mgd_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_torus_xy_graph_descriptor.textproto";

    // Test 1: Without FabricConfig - should respect dim_types (RING = torus with wrap-around)
    {
        auto mesh_graph_no_override = make_mesh_graph(torus_mgd_path);
        const auto& connectivity = mesh_graph_no_override.get_intra_mesh_connectivity();

        // In torus, NW corner (chip 0) should have wrap-around connections
        const auto& nw_connections = connectivity[0][nw_fabric_id];
        // Should have all 4 directions due to wrap-around
        EXPECT_GT(nw_connections.size(), 2);  // More than just E and S
        // Check wrap-around exists (W and N)
        EXPECT_GT(nw_connections.count(3), 0);  // Has W connection (wrap-around)
    }

    // Test 2: With FabricConfig=FABRIC_2D - should restrict to mesh (ignore wrap-around links)
    {
        auto mesh_graph_override = make_mesh_graph(torus_mgd_path, tt::tt_fabric::FabricConfig::FABRIC_2D);
        const auto& connectivity = mesh_graph_override.get_intra_mesh_connectivity();

        // In mesh mode, NW corner should only have E and S connections (ignore wrap-around)
        const auto& nw_connections = connectivity[0][nw_fabric_id];
        EXPECT_EQ(nw_connections.size(), 2);    // Only E and S
        EXPECT_EQ(nw_connections.count(3), 0);  // No W connection (wrap-around ignored)
    }
}

// Test that FabricConfig cannot create topology that doesn't physically exist (mesh→torus)
TEST(MeshGraphValidation, TestFabricConfigInvalidMeshToTorus) {
    using namespace single_galaxy_constants;
    // Use existing mesh MGD - this MGD does NOT have physical wrap-around connections
    const std::filesystem::path mesh_mgd_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/single_galaxy_mesh_graph_descriptor.textproto";

    // Attempting to override mesh to torus should throw - cannot create connections that don't exist
    EXPECT_THROW(
        { auto mesh_graph_invalid = make_mesh_graph(mesh_mgd_path, tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY); },
        std::runtime_error);
}

TEST_F(ControlPlaneFixture, TestSerializeEthCoordinatesToFile) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane(t3k_mesh_graph_desc_path);

    // Get mesh shape to compute expected coordinates
    // t3k_mesh_graph_descriptor has device_topology { dims: [ 2, 4 ] } = 2 rows, 4 columns
    const auto& mesh_graph = control_plane->get_mesh_graph();
    MeshId mesh_id{0};
    MeshShape mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
    EXPECT_EQ(mesh_shape[0], 2) << "Mesh should have 2 rows";
    EXPECT_EQ(mesh_shape[1], 4) << "Mesh should have 4 columns";
    EXPECT_EQ(mesh_shape.mesh_size(), 8) << "Mesh should have 8 chips total";

    // Create a TopologyMapper for testing (similar to RoutingTableGeneratorTestHelper)
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& driver = cluster.get_driver();
    const auto& distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    auto& driver_ref = const_cast<tt::umd::Cluster&>(*driver);
    auto psd =
        tt::tt_metal::run_physical_system_discovery(driver_ref, distributed_context, rtoptions.get_target_device());
    auto physical_system_descriptor = std::make_unique<tt::tt_metal::PhysicalSystemDescriptor>(std::move(psd));

    tt::tt_fabric::LocalMeshBinding local_mesh_binding;
    local_mesh_binding.mesh_ids = {mesh_id};
    local_mesh_binding.host_rank = tt::tt_fabric::MeshHostRankId{0};

    auto topology_mapper = std::make_unique<tt::tt_fabric::TopologyMapper>(
        cluster, *distributed_context, mesh_graph, *physical_system_descriptor, local_mesh_binding);

    // Create a temporary directory for the output file
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / "test_eth_coords";
    std::filesystem::create_directories(temp_dir);

    // Serialize coordinates to file using the fabric_host_utils function
    int rank = *distributed_context->rank();
    std::filesystem::path output_file =
        temp_dir / ("physical_chip_mesh_coordinate_mapping_" + std::to_string(rank) + ".yaml");
    tt::tt_fabric::serialize_mesh_coordinates_to_file(*topology_mapper, output_file);

    // Verify the file was created
    EXPECT_TRUE(std::filesystem::exists(output_file)) << "Output file should exist: " << output_file;

    // Read and verify the file contents
    YAML::Node yaml_file = YAML::LoadFile(output_file.string());
    EXPECT_TRUE(yaml_file["chips"]) << "File should contain 'chips' key";

    const auto& chips_node = yaml_file["chips"];
    EXPECT_TRUE(chips_node.IsMap()) << "'chips' should be a map";

    // Get the mapping from topology mapper to verify physical chip IDs
    const auto& mapping = topology_mapper->get_local_logical_mesh_chip_id_to_physical_chip_id_mapping();

    // Verify that we have the correct number of chips
    EXPECT_EQ(chips_node.size(), mapping.size()) << "Should have " << mapping.size() << " chips in the file";

    // Verify coordinates for each physical chip ID match expected mesh coordinates
    for (const auto& [fabric_node_id, physical_chip_id] : mapping) {
        EXPECT_TRUE(chips_node[physical_chip_id])
            << "Physical chip " << physical_chip_id << " should exist in the file";

        const auto& coord_array = chips_node[physical_chip_id];
        ChipId logical_chip_id = fabric_node_id.chip_id;
        MeshCoordinate expected_coord = mesh_graph.chip_to_coordinate(fabric_node_id.mesh_id, logical_chip_id);

        // Verify coordinate values match expected mesh coordinates
        for (size_t dim = 0; dim < expected_coord.dims(); ++dim) {
            uint32_t actual_coord = coord_array[dim].as<uint32_t>();
            EXPECT_EQ(actual_coord, expected_coord[dim])
                << "Physical chip " << physical_chip_id << " (logical chip " << logical_chip_id << ") coordinate["
                << dim << "] should be " << expected_coord[dim] << ", got " << actual_coord;
        }
    }

    // Clean up
    std::filesystem::remove_all(temp_dir);
}

namespace {

struct Sp5BlitzPipelineStage {
    std::size_t stage_index;
    MeshCoordinate entry_node_coord;
    MeshCoordinate exit_node_coord;
};

// Split out of TEST_F to satisfy clang-tidy readability-function-cognitive-complexity for TestBody.
// NOLINTNEXTLINE(readability-function-cognitive-complexity) -- consolidated pipeline validation checks
void validate_sp5_blitz_decode_pipeline_stages(
    const tt::tt_fabric::ControlPlane& control_plane,
    const tt::tt_fabric::MeshGraph& mesh_graph,
    const std::vector<MeshId>& mesh_ids,
    const std::vector<Sp5BlitzPipelineStage>& stages) {
    const auto num_meshes = mesh_ids.size();

    auto coord_str = [](const MeshCoordinate& c) { return fmt::format("({}, {})", c[0], c[1]); };

    ASSERT_EQ(stages.size(), num_meshes + 1) << "Expected " << (num_meshes + 1) << " stages (num_meshes=" << num_meshes
                                             << " + 1 loopback), got " << stages.size();

    // 1. No stage has identical entry and exit coords
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& s = stages[i];
        EXPECT_NE(s.entry_node_coord, s.exit_node_coord)
            << "Stage [" << i << "] (stage_index=" << s.stage_index << ") has identical entry and exit coords "
            << coord_str(s.entry_node_coord);
    }

    // 2. No coord is reused across stages
    std::set<std::pair<std::size_t, std::pair<uint32_t, uint32_t>>> used_coords;
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& s = stages[i];
        auto entry_key = std::make_pair(s.stage_index, std::make_pair(s.entry_node_coord[0], s.entry_node_coord[1]));
        auto exit_key = std::make_pair(s.stage_index, std::make_pair(s.exit_node_coord[0], s.exit_node_coord[1]));
        EXPECT_TRUE(used_coords.insert(entry_key).second)
            << "Stage [" << i << "] entry coord " << coord_str(s.entry_node_coord) << " (stage_index=" << s.stage_index
            << ") overlaps with a previous stage";
        EXPECT_TRUE(used_coords.insert(exit_key).second)
            << "Stage [" << i << "] exit coord " << coord_str(s.exit_node_coord) << " (stage_index=" << s.stage_index
            << ") overlaps with a previous stage";
    }

    // 2b. Entry/exit fabric nodes chosen for the pipeline are not reused across stages.
    std::unordered_set<FabricNodeId> used_fabric_nodes;
    used_fabric_nodes.reserve(stages.size() * 2);
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& s = stages[i];
        MeshId mesh_id{static_cast<uint32_t>(s.stage_index)};
        FabricNodeId entry_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, s.entry_node_coord));
        FabricNodeId exit_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, s.exit_node_coord));
        EXPECT_TRUE(used_fabric_nodes.insert(entry_fn).second)
            << "Stage [" << i << "] entry fabric node " << entry_fn << " is reused across stages";
        EXPECT_TRUE(used_fabric_nodes.insert(exit_fn).second)
            << "Stage [" << i << "] exit fabric node " << exit_fn << " is reused across stages";
    }

    // 3a. Each stage entry and exit must have at least one active fabric ethernet channel (none empty).
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& s = stages[i];
        MeshId mesh_id{static_cast<uint32_t>(s.stage_index)};
        FabricNodeId entry_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, s.entry_node_coord));
        FabricNodeId exit_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, s.exit_node_coord));
        EXPECT_FALSE(control_plane.get_active_fabric_eth_channels(entry_fn).empty())
            << "Stage [" << i << "] entry fabric node " << entry_fn << " has no active fabric ethernet channels";
        EXPECT_FALSE(control_plane.get_active_fabric_eth_channels(exit_fn).empty())
            << "Stage [" << i << "] exit fabric node " << exit_fn << " has no active fabric ethernet channels";
    }

    // 3. Consecutive inter-mesh stages are physically connected
    for (std::size_t i = 0; i < stages.size() - 1; i++) {
        const auto& curr = stages[i];
        const auto& next = stages[i + 1];

        auto curr_mesh_id = MeshId{static_cast<uint32_t>(curr.stage_index)};
        auto next_mesh_id = MeshId{static_cast<uint32_t>(next.stage_index)};

        if (curr_mesh_id == next_mesh_id) {
            continue;
        }

        auto exit_chip_id = mesh_graph.coordinate_to_chip(curr_mesh_id, curr.exit_node_coord);
        auto entry_chip_id = mesh_graph.coordinate_to_chip(next_mesh_id, next.entry_node_coord);
        FabricNodeId exit_fn(curr_mesh_id, exit_chip_id);
        FabricNodeId entry_fn(next_mesh_id, entry_chip_id);

        auto pairs =
            control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(curr_mesh_id, next_mesh_id);

        bool found = false;
        for (const auto& [exit_node, peer_node] : pairs) {
            if (exit_node == exit_fn && peer_node == entry_fn) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Stages [" << i << "]->[" << (i + 1) << "]: exit (M" << *curr_mesh_id << "D"
                           << exit_chip_id << ") coord " << coord_str(curr.exit_node_coord)
                           << " is not physically connected to entry (M" << *next_mesh_id << "D" << entry_chip_id
                           << ") coord " << coord_str(next.entry_node_coord);
    }

    using EthDir = tt::tt_fabric::eth_chan_directions;
    auto is_z_eth_dir = [](EthDir d) { return d == EthDir::Z; };
    auto is_nesw_eth_dir = [](EthDir d) {
        return d == EthDir::NORTH || d == EthDir::SOUTH || d == EthDir::EAST || d == EthDir::WEST;
    };
    auto eth_dirs_match_kind = [&](EthDir a, EthDir b) {
        return (is_z_eth_dir(a) && is_z_eth_dir(b)) || (is_nesw_eth_dir(a) && is_nesw_eth_dir(b));
    };
    auto eth_chan_dir_cstr = [](EthDir d) -> const char* {
        switch (d) {
            case EthDir::EAST: return "EAST";
            case EthDir::WEST: return "WEST";
            case EthDir::NORTH: return "NORTH";
            case EthDir::SOUTH: return "SOUTH";
            case EthDir::Z: return "Z";
            default: return "UNKNOWN";
        }
    };

    // 3b. Stage exit -> next stage entry (full ring): Z-Z or NESW-NESW on each fabric hop.
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& stage = stages[i];
        const std::size_t next_i = (i + 1) % stages.size();
        const auto& next_stage = stages[next_i];

        MeshId mesh_id{static_cast<uint32_t>(stage.stage_index)};
        MeshId next_mesh_id{static_cast<uint32_t>(next_stage.stage_index)};
        FabricNodeId exit_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, stage.exit_node_coord));
        FabricNodeId next_entry_fn(
            next_mesh_id, mesh_graph.coordinate_to_chip(next_mesh_id, next_stage.entry_node_coord));

        bool saw_hop = false;
        for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(exit_fn)) {
            auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(exit_fn, src_chan);
            if (peer_fn != next_entry_fn) {
                continue;
            }
            saw_hop = true;
            EthDir dst_dir = control_plane.get_eth_chan_direction(peer_fn, static_cast<int>(peer_chan));
            EXPECT_TRUE(eth_dirs_match_kind(src_dir, dst_dir))
                << "Stages [" << i << "] exit -> [" << next_i << "] entry: ethernet direction mismatch " << exit_fn
                << " -> " << next_entry_fn << " (src_chan=" << static_cast<int>(src_chan)
                << " src_dir=" << static_cast<int>(src_dir) << ", peer_chan=" << static_cast<int>(peer_chan)
                << " dst_dir=" << static_cast<int>(dst_dir) << ")";
        }
        EXPECT_TRUE(saw_hop) << "Stages [" << i << "] exit " << exit_fn << " has no fabric ethernet hop to stage ["
                             << next_i << "] entry " << next_entry_fn;
    }

    // 4. Loopback stage must differ from stage 0
    const auto& stage_0 = stages[0];
    const auto& loopback = stages.back();
    EXPECT_TRUE(
        loopback.entry_node_coord != stage_0.entry_node_coord || loopback.exit_node_coord != stage_0.exit_node_coord)
        << "Loopback stage has identical entry/exit as stage 0: entry=" << coord_str(loopback.entry_node_coord)
        << ", exit=" << coord_str(loopback.exit_node_coord);

    const auto& psd = control_plane.get_physical_system_descriptor();
    const auto& topology_mapper = control_plane.get_topology_mapper();

    auto psd_has_direct_eth_link = [&](const FabricNodeId& a, const FabricNodeId& b) {
        auto asic_a = topology_mapper.get_asic_id_from_fabric_node_id(a);
        auto asic_b = topology_mapper.get_asic_id_from_fabric_node_id(b);
        if (!psd.get_eth_connections(asic_a, asic_b).empty()) {
            return true;
        }
        return !psd.get_eth_connections(asic_b, asic_a).empty();
    };

    // Every control-plane inter-mesh (exit, entry) pair: PSD shows a direct eth edge, and port directions match (Z-Z or
    // NESW-NESW) on each mapped hop.
    for (std::size_t i = 0; i < num_meshes; i++) {
        for (std::size_t j = 0; j < num_meshes; j++) {
            if (i == j) {
                continue;
            }
            MeshId src_mesh = mesh_ids[i];
            MeshId dst_mesh = mesh_ids[j];
            auto pairs = control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(src_mesh, dst_mesh);
            for (const auto& [exit_fn, entry_fn] : pairs) {
                EXPECT_TRUE(psd_has_direct_eth_link(exit_fn, entry_fn))
                    << "PhysicalSystemDescriptor: no direct ethernet edge between ASICs for exit " << exit_fn
                    << " and entry " << entry_fn;

                bool saw_exit_to_entry_hop = false;
                for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(exit_fn)) {
                    auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(exit_fn, src_chan);
                    if (peer_fn != entry_fn) {
                        continue;
                    }
                    saw_exit_to_entry_hop = true;
                    EthDir dst_dir = control_plane.get_eth_chan_direction(peer_fn, static_cast<int>(peer_chan));
                    EXPECT_TRUE(eth_dirs_match_kind(src_dir, dst_dir))
                        << "Inter-mesh direction mismatch for exit " << exit_fn << " -> entry " << entry_fn
                        << " (src_chan=" << static_cast<int>(src_chan) << " src_dir=" << static_cast<int>(src_dir)
                        << ", peer_chan=" << static_cast<int>(peer_chan) << " dst_dir=" << static_cast<int>(dst_dir)
                        << ")";
                }
                EXPECT_TRUE(saw_exit_to_entry_hop)
                    << "No active fabric ethernet channel maps exit " << exit_fn << " to entry " << entry_fn;
            }
        }
    }

    // Pipeline ring: each stage exit must be PSD-adjacent to the next stage entry (includes intra-mesh loopback).
    for (std::size_t i = 0; i < stages.size(); i++) {
        const auto& stage = stages[i];
        MeshId mesh_id{static_cast<uint32_t>(stage.stage_index)};
        FabricNodeId exit_fn(mesh_id, mesh_graph.coordinate_to_chip(mesh_id, stage.exit_node_coord));

        const std::size_t next_i = (i + 1) % stages.size();
        const auto& next_stage = stages[next_i];
        MeshId next_mesh_id{static_cast<uint32_t>(next_stage.stage_index)};
        FabricNodeId next_entry_fn(
            next_mesh_id, mesh_graph.coordinate_to_chip(next_mesh_id, next_stage.entry_node_coord));

        EXPECT_TRUE(psd_has_direct_eth_link(exit_fn, next_entry_fn))
            << "Stage [" << i << "] exit " << exit_fn << " -> stage [" << next_i << "] entry " << next_entry_fn
            << " has no direct PSD ethernet edge";
    }

    // Inter-mesh link counts: symmetric between mesh directions; every MGD-declared pair must have at least one
    // channel.
    for (std::size_t i = 0; i < num_meshes; i++) {
        for (std::size_t j = i + 1; j < num_meshes; j++) {
            std::size_t n_ij =
                control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(mesh_ids[i], mesh_ids[j])
                    .size();
            std::size_t n_ji =
                control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(mesh_ids[j], mesh_ids[i])
                    .size();
            EXPECT_EQ(n_ij, n_ji) << "Asymmetric inter-mesh pair count between M" << *mesh_ids[i] << " and M"
                                  << *mesh_ids[j] << " (" << n_ij << " vs " << n_ji << ")";
        }
    }

    const auto& requested_intermesh_connections = mesh_graph.get_requested_intermesh_connections();
    const auto& requested_intermesh_ports = mesh_graph.get_requested_intermesh_ports();

    if (!requested_intermesh_ports.empty()) {
        for (const auto& [src_u, dst_map] : requested_intermesh_ports) {
            for (const auto& kv : dst_map) {
                std::uint32_t dst_u = kv.first;
                std::size_t actual =
                    control_plane
                        .get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(MeshId{src_u}, MeshId{dst_u})
                        .size();
                EXPECT_GT(actual, 0u) << "No inter-mesh channels from M" << src_u << " to M" << dst_u
                                      << " (strict MGD declares this link)";
            }
        }
    } else {
        for (const auto& [src_u, dst_map] : requested_intermesh_connections) {
            for (const auto& kv : dst_map) {
                std::uint32_t dst_u = kv.first;
                std::size_t actual =
                    control_plane
                        .get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(MeshId{src_u}, MeshId{dst_u})
                        .size();
                EXPECT_GT(actual, 0u) << "No inter-mesh channels from M" << src_u << " to M" << dst_u
                                      << " (relaxed MGD declares this link)";
            }
        }
    }

    // ===== Comprehensive inter-mesh router configuration validation =====
    //
    // The following checks catch the class of bugs where the controller's Z-port
    // promotion/demotion and the host-side reconciliation disagree on which physical
    // ethernet channel should carry each inter-mesh connection.

    // (A) Per-chip Z-direction uniqueness: a chip may only use the Z direction toward
    //     ONE neighbor mesh. If find_available_z_port steals a Z port from mesh pair A
    //     for mesh pair B, mesh pair A gets dropped by the can_chip_use_z_for_mesh
    //     safety check. But if reconciliation doesn't fix the mapping, the router ends
    //     up on mesh pair A's cable while the controller thinks it's on mesh pair B's.
    //     Verify that per chip, all Z-direction inter-mesh channels go to the same mesh.
    {
        // map: (mesh_id, chip_id) -> set of neighbor mesh_ids reached via Z
        std::map<std::pair<uint32_t, ChipId>, std::set<uint32_t>> chip_z_neighbors;

        for (std::size_t i = 0; i < num_meshes; i++) {
            for (std::size_t j = 0; j < num_meshes; j++) {
                if (i == j) {
                    continue;
                }
                MeshId src_mesh = mesh_ids[i];
                MeshId dst_mesh = mesh_ids[j];
                auto pairs =
                    control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(src_mesh, dst_mesh);
                for (const auto& [exit_fn, entry_fn] : pairs) {
                    for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(exit_fn)) {
                        auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(exit_fn, src_chan);
                        if (peer_fn != entry_fn) {
                            continue;
                        }
                        if (src_dir == EthDir::Z) {
                            chip_z_neighbors[{*src_mesh, exit_fn.chip_id}].insert(*dst_mesh);
                        }
                    }
                }
            }
        }

        for (const auto& [chip_key, neighbor_meshes] : chip_z_neighbors) {
            EXPECT_LE(neighbor_meshes.size(), 1u)
                << "Chip " << chip_key.second << " in mesh " << chip_key.first
                << " has Z-direction connections to multiple neighbor meshes: " <<
                [&]() {
                    std::string s;
                    for (auto m : neighbor_meshes) {
                        if (!s.empty()) {
                            s += ", ";
                        }
                        s += "M" + std::to_string(m);
                    }
                    return s;
                }()
                << ". This violates the one-Z-neighbor-per-chip invariant and indicates "
                   "a bug in Z-port promotion/reconciliation.";
        }
    }

    // (B) Per-mesh-pair channel count: every MGD-declared connection should have at
    //     least the requested number of channels. Dropped connections (from unresolved
    //     Z/non-Z mismatches or stolen-port fallout) reduce this count.
    if (!requested_intermesh_ports.empty()) {
        for (const auto& [src_u, dst_map] : requested_intermesh_ports) {
            for (const auto& kv : dst_map) {
                std::uint32_t dst_u = kv.first;
                std::size_t actual =
                    control_plane
                        .get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(MeshId{src_u}, MeshId{dst_u})
                        .size();
                std::size_t requested = 0;
                for (const auto& port : kv.second) {
                    requested += std::get<2>(port);
                }
                EXPECT_GE(actual, requested)
                    << "Mesh pair M" << src_u << " -> M" << dst_u << " has " << actual
                    << " inter-mesh channels but MGD requests " << requested
                    << ". Connections may have been dropped during Z/non-Z mismatch resolution.";
            }
        }
    } else {
        // RELAXED mode: the MGD channel count is a best-effort target, not a hard
        // requirement.  The physical topology may have fewer cables than requested.
        // Only verify that at least one channel exists per declared pair.
        for (const auto& [src_u, dst_map] : requested_intermesh_connections) {
            for (const auto& kv : dst_map) {
                std::uint32_t dst_u = kv.first;
                std::size_t actual =
                    control_plane
                        .get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(MeshId{src_u}, MeshId{dst_u})
                        .size();
                EXPECT_GT(actual, 0u) << "Mesh pair M" << src_u << " -> M" << dst_u
                                      << " has 0 inter-mesh channels (relaxed MGD declares this link)";
            }
        }
    }

    // (C) Router symmetry: for every inter-mesh (exit, entry) pair, both sides must
    //     have active router channels on the matching physical ethernet channels.
    //     This is the core check for the stolen-Z-port bug: if find_available_z_port
    //     steals a Z port_id from mesh pair A for mesh pair B, and reconciliation
    //     doesn't fix the mapping, the exit side has a router on mesh pair A's cable
    //     (whose peer was dropped) while mesh pair B's cable has no router (but the
    //     peer expects one). Both sides hang at STARTED.
    for (std::size_t i = 0; i < num_meshes; i++) {
        for (std::size_t j = 0; j < num_meshes; j++) {
            if (i == j) {
                continue;
            }
            MeshId src_mesh = mesh_ids[i];
            MeshId dst_mesh = mesh_ids[j];
            auto pairs = control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(src_mesh, dst_mesh);
            for (const auto& [exit_fn, entry_fn] : pairs) {
                bool exit_has_channel_to_entry = false;
                for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(exit_fn)) {
                    auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(exit_fn, src_chan);
                    if (peer_fn != entry_fn) {
                        continue;
                    }
                    exit_has_channel_to_entry = true;

                    // (C1) Peer must have a matching active router on the same channel
                    auto entry_channels = control_plane.get_active_fabric_eth_channels(entry_fn);
                    bool entry_has_matching_channel = false;
                    for (const auto& [entry_chan, entry_dir] : entry_channels) {
                        if (entry_chan == peer_chan) {
                            entry_has_matching_channel = true;
                            break;
                        }
                    }
                    EXPECT_TRUE(entry_has_matching_channel)
                        << "Router symmetry violation: exit " << exit_fn << " chan=" << static_cast<int>(src_chan)
                        << " connects to entry " << entry_fn << " chan=" << static_cast<int>(peer_chan)
                        << ", but entry has no active router on that channel. "
                        << "This would cause the ERISC handshake to hang at STARTED.";

                    // (C2) Direction kinds must match (both Z or both NESW)
                    EthDir entry_dir_val = control_plane.get_eth_chan_direction(entry_fn, static_cast<int>(peer_chan));
                    EXPECT_TRUE(eth_dirs_match_kind(src_dir, entry_dir_val))
                        << "Router direction mismatch: exit " << exit_fn << " chan=" << static_cast<int>(src_chan)
                        << " dir=" << eth_chan_dir_cstr(src_dir) << " -> entry " << entry_fn
                        << " chan=" << static_cast<int>(peer_chan) << " dir=" << eth_chan_dir_cstr(entry_dir_val);
                }
                EXPECT_TRUE(exit_has_channel_to_entry)
                    << "Exit " << exit_fn << " has no active channel connecting to entry " << entry_fn
                    << " (inter-mesh pair M" << *src_mesh << " -> M" << *dst_mesh << ")";
            }
        }
    }

    // (D) Physical cable verification: for every inter-mesh (exit, entry) pair with
    //     an active router channel, verify that the PSD confirms a direct ethernet
    //     connection from that specific exit ASIC channel to the entry ASIC. After the
    //     intermesh_chan_to_peer_ redesign, get_connected_mesh_chip_chan_ids returns the
    //     PHYSICAL peer channel for inter-mesh links (sourced from PSD), so we now also
    //     check the (src_chan, peer_chan) pair exactly matches a PSD cable. This is the
    //     core check that catches the multi-peer cabling bug (e.g. M19D2 -> {M18D3,
    //     M18D5}) that motivated the redesign: a chip with multiple distinct peer ASICs
    //     in the same dest mesh used to silently collapse to connected_chip_ids[0].
    for (std::size_t i = 0; i < num_meshes; i++) {
        for (std::size_t j = 0; j < num_meshes; j++) {
            if (i == j) {
                continue;
            }
            MeshId src_mesh = mesh_ids[i];
            MeshId dst_mesh = mesh_ids[j];
            auto pairs = control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(src_mesh, dst_mesh);
            for (const auto& [exit_fn, entry_fn] : pairs) {
                auto exit_asic = topology_mapper.get_asic_id_from_fabric_node_id(exit_fn);
                auto entry_asic = topology_mapper.get_asic_id_from_fabric_node_id(entry_fn);
                auto eth_conns = psd.get_eth_connections(exit_asic, entry_asic);

                for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(exit_fn)) {
                    auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(exit_fn, src_chan);
                    if (peer_fn != entry_fn) {
                        continue;
                    }

                    // src_chan on exit ASIC must have a physical cable to entry ASIC
                    bool psd_has_cable_from_src_chan = false;
                    bool psd_has_exact_pair = false;
                    for (const auto& conn : eth_conns) {
                        if (conn.src_chan == src_chan) {
                            psd_has_cable_from_src_chan = true;
                            if (conn.dst_chan == peer_chan) {
                                psd_has_exact_pair = true;
                                break;
                            }
                        }
                    }
                    EXPECT_TRUE(psd_has_cable_from_src_chan)
                        << "Physical cable mismatch: exit " << exit_fn << " chan=" << static_cast<int>(src_chan)
                        << " has a router that routes toward entry " << entry_fn
                        << ", but PSD has no ethernet cable from ASIC " << exit_asic
                        << " chan=" << static_cast<int>(src_chan) << " to ASIC " << entry_asic << " (PSD has "
                        << eth_conns.size() << " cables between these ASICs)"
                        << ". The reconciliation likely mapped this port to a channel that physically "
                        << "connects to a different ASIC (stolen Z port_id bug).";
                    EXPECT_TRUE(psd_has_exact_pair)
                        << "Per-cable channel mismatch: exit " << exit_fn << " chan=" << static_cast<int>(src_chan)
                        << " -> entry " << entry_fn << " chan=" << static_cast<int>(peer_chan)
                        << ", but PSD has no cable matching this exact (src_chan, dst_chan) pair "
                        << "between ASICs " << exit_asic << " and " << entry_asic
                        << ". intermesh_chan_to_peer_ disagrees with PSD on the physical cable identity.";
                }
            }
        }
    }

    // (F) Reverse-lookup symmetry for inter-mesh hops: get_connected_mesh_chip_chan_ids
    //     must round-trip. If exit_fn:src_chan -> entry_fn:peer_chan, then querying
    //     entry_fn:peer_chan must return exit_fn:src_chan. The redesign places this
    //     invariant in intermesh_chan_to_peer_ which is populated symmetrically for
    //     both endpoints of each PSD cable; an asymmetric entry indicates that the
    //     clear-and-rebuild in convert_port_descriptors_to_intermesh_connections lost
    //     one side of a cable.
    for (std::size_t i = 0; i < num_meshes; i++) {
        for (std::size_t j = 0; j < num_meshes; j++) {
            if (i == j) {
                continue;
            }
            MeshId src_mesh = mesh_ids[i];
            MeshId dst_mesh = mesh_ids[j];
            auto pairs = control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(src_mesh, dst_mesh);
            for (const auto& [exit_fn, entry_fn] : pairs) {
                for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(exit_fn)) {
                    auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(exit_fn, src_chan);
                    if (peer_fn != entry_fn) {
                        continue;
                    }
                    auto [reverse_fn, reverse_chan] =
                        control_plane.get_connected_mesh_chip_chan_ids(peer_fn, peer_chan);
                    EXPECT_EQ(reverse_fn, exit_fn) << "Inter-mesh reverse-lookup peer mismatch: " << exit_fn
                                                   << " chan=" << static_cast<int>(src_chan) << " -> " << peer_fn
                                                   << " chan=" << static_cast<int>(peer_chan) << " but reverse returns "
                                                   << reverse_fn << " chan=" << static_cast<int>(reverse_chan)
                                                   << ". intermesh_chan_to_peer_ entries are not symmetric.";
                    EXPECT_EQ(reverse_chan, src_chan)
                        << "Inter-mesh reverse-lookup channel mismatch: " << exit_fn
                        << " chan=" << static_cast<int>(src_chan) << " -> " << peer_fn
                        << " chan=" << static_cast<int>(peer_chan) << " but reverse returns " << reverse_fn
                        << " chan=" << static_cast<int>(reverse_chan)
                        << ". A PSD cable should round-trip through intermesh_chan_to_peer_.";
                }
            }
        }
    }

    // (G) No double-claim of physical (ASIC, chan) endpoints across inter-mesh routes:
    //     Each side of a PSD cable is a single physical resource. If two distinct
    //     logical (exit_fn, peer_fn) attributions both resolve the SAME (asic, chan)
    //     to different peers, that physical channel has been double-booked and one
    //     of the routes will silently misroute. We iterate every exit chip exactly
    //     once and walk its active channels; for each inter-mesh channel we record
    //     the resolved peer keyed by physical (asic, chan). The first sighting
    //     defines the owner; later sightings must agree.
    {
        std::map<std::pair<tt::tt_metal::AsicID, chan_id_t>, std::pair<FabricNodeId, chan_id_t>> exit_chan_owner;
        std::map<std::pair<tt::tt_metal::AsicID, chan_id_t>, std::pair<FabricNodeId, chan_id_t>> entry_chan_owner;
        for (std::size_t i = 0; i < num_meshes; i++) {
            MeshId src_mesh = mesh_ids[i];
            auto coord_range = mesh_graph.get_coord_range(src_mesh);
            for (const auto& src_coord : coord_range) {
                FabricNodeId exit_fn(src_mesh, mesh_graph.coordinate_to_chip(src_mesh, src_coord));
                auto exit_asic = topology_mapper.get_asic_id_from_fabric_node_id(exit_fn);
                for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(exit_fn)) {
                    auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(exit_fn, src_chan);
                    if (peer_fn.mesh_id == src_mesh) {
                        continue;  // intra-mesh, not relevant here
                    }
                    auto peer_asic = topology_mapper.get_asic_id_from_fabric_node_id(peer_fn);
                    auto exit_key = std::make_pair(exit_asic, src_chan);
                    auto entry_key = std::make_pair(peer_asic, peer_chan);
                    auto exit_value = std::make_pair(peer_fn, peer_chan);
                    auto entry_value = std::make_pair(exit_fn, src_chan);

                    auto [exit_it, exit_inserted] = exit_chan_owner.try_emplace(exit_key, exit_value);
                    if (!exit_inserted) {
                        EXPECT_EQ(exit_it->second, exit_value)
                            << "Double-claimed exit channel: ASIC " << exit_asic
                            << " chan=" << static_cast<int>(src_chan) << " on " << exit_fn << " resolves to peer "
                            << peer_fn << " chan=" << static_cast<int>(peer_chan)
                            << " but was previously bound to peer " << exit_it->second.first
                            << " chan=" << static_cast<int>(exit_it->second.second)
                            << ". intermesh_chan_to_peer_ has inconsistent entries for this physical channel.";
                    }
                    auto [entry_it, entry_inserted] = entry_chan_owner.try_emplace(entry_key, entry_value);
                    if (!entry_inserted) {
                        EXPECT_EQ(entry_it->second, entry_value)
                            << "Double-claimed entry channel: ASIC " << peer_asic
                            << " chan=" << static_cast<int>(peer_chan) << " on " << peer_fn << " is the peer of "
                            << exit_fn << " chan=" << static_cast<int>(src_chan) << " but was previously claimed by "
                            << entry_it->second.first << " chan=" << static_cast<int>(entry_it->second.second)
                            << ". A physical channel can only terminate one inter-mesh route.";
                    }
                }
            }
        }
    }

    // (H) Multi-peer cabling explicit coverage: this is the exact scenario that
    //     motivated the connection_hash + intermesh_chan_to_peer_ redesign. Find
    //     every (exit_fn, dst_mesh) where PSD shows cables to >= 2 distinct peer
    //     ASICs in dst_mesh, and verify that:
    //       - the control plane reports >= 2 distinct (exit_fn, entry_fn) pairs
    //       - per-channel routing distributes to all peer ASICs (no silent
    //         collapse to a single connected_chip_ids[0])
    //       - each per-channel hop's (src_chan, peer_chan) matches a PSD cable
    //         to the SAME peer ASIC the routing reports.
    {
        for (std::size_t i = 0; i < num_meshes; i++) {
            MeshId src_mesh = mesh_ids[i];
            auto coord_range_src = mesh_graph.get_coord_range(src_mesh);
            for (const auto& src_coord : coord_range_src) {
                FabricNodeId exit_fn(src_mesh, mesh_graph.coordinate_to_chip(src_mesh, src_coord));
                auto exit_asic = topology_mapper.get_asic_id_from_fabric_node_id(exit_fn);
                for (std::size_t j = 0; j < num_meshes; j++) {
                    if (i == j) {
                        continue;
                    }
                    MeshId dst_mesh = mesh_ids[j];
                    // Enumerate distinct peer ASICs in dst_mesh that are physically
                    // cabled to exit_fn per PSD.
                    std::set<tt::tt_metal::AsicID> psd_peer_asics;
                    auto coord_range_dst = mesh_graph.get_coord_range(dst_mesh);
                    for (const auto& dst_coord : coord_range_dst) {
                        FabricNodeId candidate(dst_mesh, mesh_graph.coordinate_to_chip(dst_mesh, dst_coord));
                        auto candidate_asic = topology_mapper.get_asic_id_from_fabric_node_id(candidate);
                        if (!psd.get_eth_connections(exit_asic, candidate_asic).empty()) {
                            psd_peer_asics.insert(candidate_asic);
                        }
                    }
                    if (psd_peer_asics.size() < 2) {
                        continue;  // not a multi-peer chip for this dst_mesh
                    }

                    auto pairs =
                        control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(src_mesh, dst_mesh);
                    std::set<tt::tt_metal::AsicID> control_plane_peer_asics_for_exit;
                    for (const auto& [p_exit, p_entry] : pairs) {
                        if (p_exit != exit_fn) {
                            continue;
                        }
                        control_plane_peer_asics_for_exit.insert(
                            topology_mapper.get_asic_id_from_fabric_node_id(p_entry));
                    }
                    EXPECT_EQ(control_plane_peer_asics_for_exit, psd_peer_asics)
                        << "Multi-peer cabling not fully represented in control plane: exit " << exit_fn
                        << " has PSD cables to " << psd_peer_asics.size() << " distinct ASICs in M" << *dst_mesh
                        << ", but control plane only knows about " << control_plane_peer_asics_for_exit.size()
                        << " of them. The lossy connected_chip_ids[0] fallback may have collapsed peers.";

                    // Per-channel routing must distribute across all PSD peers and
                    // each (src_chan, peer_chan) must match a PSD cable to the
                    // reported peer ASIC.
                    std::set<tt::tt_metal::AsicID> peer_asics_via_routing;
                    for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(exit_fn)) {
                        auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(exit_fn, src_chan);
                        if (peer_fn.mesh_id != dst_mesh) {
                            continue;
                        }
                        auto routed_peer_asic = topology_mapper.get_asic_id_from_fabric_node_id(peer_fn);
                        peer_asics_via_routing.insert(routed_peer_asic);
                        bool psd_confirms = false;
                        for (const auto& conn : psd.get_eth_connections(exit_asic, routed_peer_asic)) {
                            if (conn.src_chan == src_chan && conn.dst_chan == peer_chan) {
                                psd_confirms = true;
                                break;
                            }
                        }
                        EXPECT_TRUE(psd_confirms)
                            << "Multi-peer per-channel routing mismatch: exit " << exit_fn
                            << " chan=" << static_cast<int>(src_chan) << " -> " << peer_fn
                            << " chan=" << static_cast<int>(peer_chan)
                            << " but PSD has no cable matching that exact (src_chan, dst_chan) pair "
                            << "between ASICs " << exit_asic << " and " << routed_peer_asic
                            << ". A different peer ASIC may have stolen this channel.";
                    }
                    EXPECT_EQ(peer_asics_via_routing, psd_peer_asics)
                        << "Per-channel routing does not cover all PSD peer ASICs for exit " << exit_fn << " in M"
                        << *dst_mesh << " (routing reaches " << peer_asics_via_routing.size() << " of "
                        << psd_peer_asics.size() << " peers). intermesh_chan_to_peer_ may be incomplete.";
                }
            }
        }
    }

    // (I) Routing-table forwarding is ready for every inter-mesh hop. The decode
    //     pipeline ultimately relies on get_forwarding_direction / get_fabric_route
    //     (driven by inter_mesh_routing_tables_ / intra_mesh_routing_tables_) to
    //     push packets across a hop; if the pair is in the control plane but the
    //     routing table is missing the entry, sockets will handshake but packets
    //     will never arrive. Runs on all ranks: forwarding-direction is derived
    //     from mesh_graph and is globally valid; get_fabric_route is local-only
    //     (uses per-chip routing tables on this host) so we gate it on locality.
    {
        auto local_meshes = control_plane.get_local_mesh_id_bindings();
        std::set<MeshId> local_mesh_set(local_meshes.begin(), local_meshes.end());
        for (std::size_t i = 0; i < num_meshes; i++) {
            for (std::size_t j = 0; j < num_meshes; j++) {
                if (i == j) {
                    continue;
                }
                MeshId src_mesh = mesh_ids[i];
                MeshId dst_mesh = mesh_ids[j];
                auto pairs =
                    control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(src_mesh, dst_mesh);
                for (const auto& [exit_fn, entry_fn] : pairs) {
                    auto fwd_dir = control_plane.get_forwarding_direction(exit_fn, entry_fn);
                    EXPECT_TRUE(fwd_dir.has_value())
                        << "Routing table has no forwarding direction from " << exit_fn << " (M" << *src_mesh << ") to "
                        << entry_fn << " (M" << *dst_mesh
                        << "). A decode-pipeline socket between these nodes would handshake but never deliver.";

                    // Local src only: validate channels carry an end-to-end route whose
                    // final hop lands on entry_fn.
                    if (!local_mesh_set.contains(exit_fn.mesh_id)) {
                        continue;
                    }
                    auto fwd_chans = control_plane.get_forwarding_eth_chans_to_chip(exit_fn, entry_fn);
                    EXPECT_FALSE(fwd_chans.empty())
                        << "No forwarding eth channels from " << exit_fn << " to " << entry_fn
                        << " despite the control plane listing them as an inter-mesh pair. "
                        << "get_forwarding_eth_chans_to_chip would return empty, socket send would fail.";
                    for (chan_id_t fwd_chan : fwd_chans) {
                        auto route = control_plane.get_fabric_route(exit_fn, entry_fn, fwd_chan);
                        EXPECT_FALSE(route.empty())
                            << "get_fabric_route returned no route: " << exit_fn
                            << " chan=" << static_cast<int>(fwd_chan) << " -> " << entry_fn
                            << ". The forwarding channel is advertised but routing tables cannot resolve an "
                            << "end-to-end path.";
                        if (!route.empty()) {
                            EXPECT_EQ(route.back().first, entry_fn)
                                << "Route from " << exit_fn << " chan=" << static_cast<int>(fwd_chan)
                                << " claims destination " << entry_fn << " but terminates at " << route.back().first
                                << ". Routing table hop sequence is inconsistent.";
                        }
                    }
                }
            }
        }
    }

    // (E) No orphaned inter-mesh routers: every active router channel on an exit node
    //     that connects to a peer in a different mesh must appear in the
    //     intermesh_exit_peer_fabric_node_id_pairs. If an exit node has a router
    //     channel to a remote mesh but the pair isn't tracked, the peer side won't
    //     know about it and may not have a matching router.
    for (std::size_t i = 0; i < num_meshes; i++) {
        MeshId src_mesh = mesh_ids[i];
        auto exit_node_ids = control_plane.get_exit_fabric_node_ids_between_meshes(src_mesh, src_mesh);
        // Get all exit nodes for all neighbor meshes
        std::set<FabricNodeId> all_exit_nodes;
        for (std::size_t j = 0; j < num_meshes; j++) {
            if (i == j) {
                continue;
            }
            MeshId dst_mesh = mesh_ids[j];
            auto pairs = control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(src_mesh, dst_mesh);
            for (const auto& [exit_fn, entry_fn] : pairs) {
                all_exit_nodes.insert(exit_fn);
            }
        }

        for (const auto& exit_fn : all_exit_nodes) {
            for (const auto& [src_chan, src_dir] : control_plane.get_active_fabric_eth_channels(exit_fn)) {
                auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(exit_fn, src_chan);
                if (peer_fn.mesh_id == src_mesh) {
                    continue;  // Intra-mesh, not relevant
                }
                // This inter-mesh channel must be tracked in the pairs
                MeshId peer_mesh = peer_fn.mesh_id;
                auto pairs =
                    control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(src_mesh, peer_mesh);
                bool found_in_pairs = false;
                for (const auto& [p_exit, p_entry] : pairs) {
                    if (p_exit == exit_fn && p_entry == peer_fn) {
                        found_in_pairs = true;
                        break;
                    }
                }
                EXPECT_TRUE(found_in_pairs)
                    << "Orphaned inter-mesh router: exit " << exit_fn << " chan=" << static_cast<int>(src_chan)
                    << " connects to " << peer_fn << " (M" << *peer_mesh
                    << "), but this pair is not in intermesh_exit_peer_fabric_node_id_pairs. "
                    << "The peer may not have a matching router.";
            }
        }
    }
}

}  // namespace

TEST_F(ControlPlaneFixture, TestBlitzDecodePipelineBuilder) {
    tt::tt_metal::MetalContext::instance().set_default_fabric_topology();

    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    auto mesh_ids = mesh_graph.get_mesh_ids();
    std::sort(mesh_ids.begin(), mesh_ids.end());
    const auto num_meshes = mesh_ids.size();

    ASSERT_GE(num_meshes, 2u) << "Pipeline builder requires at least 2 meshes";

    auto fn_to_coord = [&](const FabricNodeId& fn) { return mesh_graph.chip_to_coordinate(fn.mesh_id, fn.chip_id); };

    // --- build_pipeline_from_topology ---
    std::set<FabricNodeId> used_nodes;

    // Select one inter-mesh pair per hop, avoiding collisions.
    // hop[i] connects mesh_ids[i] -> mesh_ids[(i+1) % N].
    std::vector<std::pair<FabricNodeId, FabricNodeId>> hops;
    hops.reserve(num_meshes);
    for (std::size_t i = 0; i < num_meshes; i++) {
        auto pairs = control_plane.get_intermesh_exit_peer_fabric_node_id_pairs_between_meshes(
            mesh_ids[i], mesh_ids[(i + 1) % num_meshes]);
        ASSERT_FALSE(pairs.empty()) << "No inter-mesh connection from mesh " << *mesh_ids[i] << " to mesh "
                                    << *mesh_ids[(i + 1) % num_meshes];

        bool found = false;
        for (const auto& pair : pairs) {
            if (used_nodes.contains(pair.first) || used_nodes.contains(pair.second)) {
                continue;
            }
            hops.push_back(pair);
            used_nodes.insert(pair.first);
            used_nodes.insert(pair.second);
            found = true;
            break;
        }
        ASSERT_TRUE(found) << "No non-colliding inter-mesh pair from mesh " << *mesh_ids[i] << " to mesh "
                           << *mesh_ids[(i + 1) % num_meshes] << " (all " << pairs.size()
                           << " candidate pairs overlap with already-claimed nodes)";
    }

    // Find two unclaimed nodes on mesh_0 for stage 0 entry and loopback exit.
    // We need a pair that has a direct intra-mesh ethernet link between them
    // (loopback_exit -> stage_0_entry). Prefer non-Z direction links.
    auto mesh_0_coord_range = mesh_graph.get_coord_range(mesh_ids[0]);
    std::vector<FabricNodeId> unclaimed_mesh_0_nodes;
    for (const auto& coord : mesh_0_coord_range) {
        auto chip_id = mesh_graph.coordinate_to_chip(mesh_ids[0], coord);
        FabricNodeId fn(mesh_ids[0], chip_id);
        if (!used_nodes.contains(fn)) {
            unclaimed_mesh_0_nodes.push_back(fn);
        }
    }
    ASSERT_GE(unclaimed_mesh_0_nodes.size(), 2u)
        << "Need at least 2 unclaimed nodes on mesh " << *mesh_ids[0] << " for stage 0 entry and loopback exit, found "
        << unclaimed_mesh_0_nodes.size();

    std::optional<FabricNodeId> stage_0_entry_fn;
    std::optional<FabricNodeId> loopback_exit_fn;
    bool found_non_z_pair = false;

    for (std::size_t a = 0; a < unclaimed_mesh_0_nodes.size() && !found_non_z_pair; a++) {
        auto fn_a = unclaimed_mesh_0_nodes[a];
        auto channels_a = control_plane.get_active_fabric_eth_channels(fn_a);
        for (const auto& [chan_id, direction] : channels_a) {
            auto [peer_fn, peer_chan] = control_plane.get_connected_mesh_chip_chan_ids(fn_a, chan_id);
            if (peer_fn.mesh_id != mesh_ids[0]) {
                continue;
            }
            bool peer_unclaimed = !used_nodes.contains(peer_fn) &&
                                  std::find(unclaimed_mesh_0_nodes.begin(), unclaimed_mesh_0_nodes.end(), peer_fn) !=
                                      unclaimed_mesh_0_nodes.end();
            if (!peer_unclaimed) {
                continue;
            }
            bool is_non_z = (direction != tt::tt_fabric::eth_chan_directions::Z);
            if (!loopback_exit_fn.has_value() || (is_non_z && !found_non_z_pair)) {
                loopback_exit_fn = fn_a;
                stage_0_entry_fn = peer_fn;
                found_non_z_pair = is_non_z;
            }
        }
    }

    ASSERT_TRUE(loopback_exit_fn.has_value()) << "Could not find a directly-connected unclaimed pair on mesh "
                                              << *mesh_ids[0] << " for loopback exit -> stage 0 entry";

    std::vector<Sp5BlitzPipelineStage> stages;
    stages.reserve(num_meshes + 1);

    stages.push_back(
        {static_cast<std::size_t>(*mesh_ids[0]), fn_to_coord(*stage_0_entry_fn), fn_to_coord(hops[0].first)});

    for (std::size_t i = 1; i < num_meshes; i++) {
        stages.push_back(
            {static_cast<std::size_t>(*mesh_ids[i]), fn_to_coord(hops[i - 1].second), fn_to_coord(hops[i].first)});
    }

    stages.push_back(
        {static_cast<std::size_t>(*mesh_ids[0]),
         fn_to_coord(hops[num_meshes - 1].second),
         fn_to_coord(*loopback_exit_fn)});

    validate_sp5_blitz_decode_pipeline_stages(control_plane, mesh_graph, mesh_ids, stages);
}
}  // namespace tt::tt_fabric::fabric_router_tests
