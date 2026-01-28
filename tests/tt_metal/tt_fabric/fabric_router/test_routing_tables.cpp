// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <filesystem>
#include <yaml-cpp/yaml.h>

#include "fabric_fixture.hpp"
#include "t3k_mesh_descriptor_chip_mappings.hpp"
#include "utils.hpp"
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"

namespace {

constexpr auto kFabricConfig = tt::tt_fabric::FabricConfig::FABRIC_2D;
constexpr auto kReliabilityMode = tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane(const std::filesystem::path& graph_desc) {
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(graph_desc.string());
    control_plane->initialize_fabric_context(kFabricConfig, tt::tt_fabric::FabricRouterConfig{});
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(kFabricConfig, kReliabilityMode);

    return control_plane;
}

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane(
    const std::filesystem::path& graph_desc,
    const std::map<tt::tt_fabric::FabricNodeId, tt::ChipId>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(
        graph_desc.string(), logical_mesh_chip_id_to_physical_chip_id_mapping);
    control_plane->initialize_fabric_context(kFabricConfig, tt::tt_fabric::FabricRouterConfig{});
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(kFabricConfig, kReliabilityMode);

    return control_plane;
}

constexpr auto kFabricConfig1D = tt::tt_fabric::FabricConfig::FABRIC_1D_RING;

// Helper struct to keep dependencies alive for RoutingTableGenerator tests
struct RoutingTableGeneratorTestHelper {
    std::unique_ptr<tt::tt_fabric::MeshGraph> mesh_graph;
    std::unique_ptr<tt::tt_metal::PhysicalSystemDescriptor> physical_system_descriptor;
    std::unique_ptr<tt::tt_fabric::TopologyMapper> topology_mapper;
    std::unique_ptr<tt::tt_fabric::RoutingTableGenerator> routing_table_generator;

    RoutingTableGeneratorTestHelper(const std::string& mesh_graph_desc_file) {
        mesh_graph = std::make_unique<tt::tt_fabric::MeshGraph>(mesh_graph_desc_file);
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        const auto& driver = cluster.get_driver();
        const auto& distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
        const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        physical_system_descriptor = std::make_unique<tt::tt_metal::PhysicalSystemDescriptor>(
            driver, distributed_context, &tt::tt_metal::MetalContext::instance().hal(), rtoptions);

        tt::tt_fabric::LocalMeshBinding local_mesh_binding;
        local_mesh_binding.mesh_ids = {tt::tt_fabric::MeshId{0}};
        local_mesh_binding.host_rank = tt::tt_fabric::MeshHostRankId{0};

        topology_mapper = std::make_unique<tt::tt_fabric::TopologyMapper>(
            *mesh_graph, *physical_system_descriptor, local_mesh_binding);

        routing_table_generator = std::make_unique<tt::tt_fabric::RoutingTableGenerator>(*topology_mapper);
    }
};

// Helper function to create RoutingTableGenerator for tests
std::unique_ptr<RoutingTableGeneratorTestHelper> make_routing_table_generator(const std::string& mesh_graph_desc_file) {
    return std::make_unique<RoutingTableGeneratorTestHelper>(mesh_graph_desc_file);
}

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane_1d(const std::filesystem::path& graph_desc) {
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(graph_desc.string());
    control_plane->initialize_fabric_context(kFabricConfig1D, tt::tt_fabric::FabricRouterConfig{});
    control_plane->configure_routing_tables_for_fabric_ethernet_channels(kFabricConfig1D, kReliabilityMode);

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
    MeshGraph mesh_graph(test_desc_path.string());

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
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    tt::tt_metal::MetalContext::instance().get_control_plane();
}

TEST(MeshGraphValidation, TestT3kMeshGraphInit) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    MeshGraph mesh_graph_desc(t3k_mesh_graph_desc_path.string());
    EXPECT_EQ(
        mesh_graph_desc.get_coord_range(MeshId{0}, MeshHostRankId(0)),
        MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(1, 3)));
}

TEST_F(ControlPlaneFixture, TestT3kControlPlaneInit) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.textproto";
    auto control_plane = make_control_plane(t3k_mesh_graph_desc_path);
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
    auto control_plane = make_control_plane_1d(t3k_mesh_graph_desc_path);

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

class T3kCustomMeshGraphControlPlaneFixture
    : public ControlPlaneFixture,
      public testing::WithParamInterface<std::tuple<std::string, std::vector<std::vector<EthCoord>>>> {};

TEST_P(T3kCustomMeshGraphControlPlaneFixture, TestT3kMeshGraphInit) {
    auto [mesh_graph_desc_path, _] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    MeshGraph mesh_graph_desc(t3k_mesh_graph_desc_path.string());
}

TEST_P(T3kCustomMeshGraphControlPlaneFixture, TestT3kControlPlaneInit) {
    auto [mesh_graph_desc_path, mesh_graph_eth_coords] = GetParam();
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / mesh_graph_desc_path;
    [[maybe_unused]] auto control_plane = make_control_plane(
        t3k_mesh_graph_desc_path.string(), get_physical_chip_mapping_from_eth_coords_mapping(mesh_graph_eth_coords));
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

    // Create physical system descriptor to access ASIC information
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& driver = cluster.get_driver();
    const auto& distributed_context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    auto physical_system_descriptor = std::make_unique<tt::tt_metal::PhysicalSystemDescriptor>(
        driver, distributed_context, &tt::tt_metal::MetalContext::instance().hal(), rtoptions);

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
    tt_fabric::MeshGraph mesh_graph(t3k_dual_host_mesh_graph_desc_path.string());

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
    tt_fabric::MeshGraph mesh_graph(t3k_2x2_mesh_graph_desc_path.string());

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
    tt_fabric::MeshGraph mesh_graph(t3k_dual_host_mesh_graph_desc_path.string());

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
    tt_fabric::MeshGraph mesh_graph_single_host(t3k_mesh_graph_desc_path.string());

    // In single host configuration, all chips should belong to host rank 0
    for (ChipId chip_id = 0; chip_id < 8; chip_id++) {
        EXPECT_EQ(mesh_graph_single_host.get_host_rank_for_chip(MeshId{0}, chip_id), MeshHostRankId(0));
    }

    // Test with 2x2 configuration (two separate meshes)
    const std::filesystem::path t3k_2x2_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.textproto";
    tt_fabric::MeshGraph mesh_graph_2x2(t3k_2x2_mesh_graph_desc_path.string());

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
    EXPECT_THROW(tt_fabric::MeshGraph(invalid_shape_mesh_graph_desc_path.string()), std::exception);
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
    MeshGraph mesh_graph(mesh_graph_desc_path.string());
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
    MeshGraph mesh_graph(mesh_graph_desc_path.string());
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
    MeshGraph mesh_graph(mesh_graph_desc_path.string());
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
    MeshGraph mesh_graph(mesh_graph_desc_path.string());
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
    MeshGraph mesh_graph_desc(mesh_graph_desc_path.string());
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
    MeshGraph mesh_graph_desc(galaxy_6u_mesh_graph_desc_path.string());

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
    MeshGraph mesh_graph(p150_mesh_graph_desc_path.string());

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
    MeshGraph mesh_graph(p100_mesh_graph_desc_path.string());

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
    MeshGraph mesh_graph(p150_x8_mesh_graph_desc_path.string());

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
        MeshGraph mesh_graph_no_override(torus_mgd_path.string());
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
        MeshGraph mesh_graph_override(torus_mgd_path.string(), tt::tt_fabric::FabricConfig::FABRIC_2D);
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
        { MeshGraph mesh_graph_invalid(mesh_mgd_path.string(), tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY); },
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
    auto physical_system_descriptor = std::make_unique<tt::tt_metal::PhysicalSystemDescriptor>(
        driver, distributed_context, &tt::tt_metal::MetalContext::instance().hal(), rtoptions);

    tt::tt_fabric::LocalMeshBinding local_mesh_binding;
    local_mesh_binding.mesh_ids = {mesh_id};
    local_mesh_binding.host_rank = tt::tt_fabric::MeshHostRankId{0};

    auto topology_mapper =
        std::make_unique<tt::tt_fabric::TopologyMapper>(mesh_graph, *physical_system_descriptor, local_mesh_binding);

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
}  // namespace tt::tt_fabric::fabric_router_tests
