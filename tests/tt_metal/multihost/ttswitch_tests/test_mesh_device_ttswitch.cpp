// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <memory>
#include <optional>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/fabric/fabric_switch_manager.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/experimental/fabric/fabric.hpp>

namespace tt::tt_metal {

class MeshDeviceTTSwitchFixture : public ::testing::Test {
protected:
    int trace_region_size_ = DEFAULT_TRACE_REGION_SIZE;
    int l1_small_size_ = DEFAULT_L1_SMALL_SIZE;

    void SetUp() override {
        // Check if we're running in a multi-process environment
        const char* mesh_id_str = std::getenv("TT_MESH_ID");
        if (mesh_id_str == nullptr) {
            GTEST_SKIP() << "This test requires TT_MESH_ID environment variable (run with tt-run)";
        }

        // Check if this is T3k
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() !=
            tt::tt_metal::ClusterType::N300_2x2) {
            GTEST_SKIP() << "This test is only for N300 2x2";
        }

        auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        if (control_plane.is_local_host_on_switch_mesh()) {
            // setup tt-switch manager
            tt::tt_fabric::FabricSwitchManager::instance().setup(tt::tt_fabric::FabricConfig::FABRIC_2D);
            GTEST_SKIP() << "This test is only for compute mesh switch mesh just needs to setup tt-switch manager";
        }
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
    }

    void TearDown() override {
        auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        if (control_plane.is_local_host_on_switch_mesh()) {
            tt::tt_fabric::FabricSwitchManager::instance().teardown();
        } else {
            tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
        }
    }
};

TEST_F(MeshDeviceTTSwitchFixture, TestOpenCloseComputeMeshDevice) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    auto mesh_id_val = 0;

    // Verify this is a compute mesh, not a switch
    tt::tt_fabric::MeshId mesh_id(mesh_id_val);
    ASSERT_FALSE(mesh_graph.is_switch_mesh(mesh_id))
        << "Mesh ID " << mesh_id_val << " should be a compute mesh, not a switch";

    // Get mesh shape
    const auto mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
    EXPECT_EQ(mesh_shape, tt::tt_metal::distributed::MeshShape(2, 2)) << "Compute mesh should have 2x2 shape";

    // Open mesh device - by default, this uses SystemMesh which is initialized with
    // the mesh_id from TT_MESH_ID environment variable. The SystemMesh singleton
    // maps logical coordinates to physical device IDs based on the mesh_id.
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
    {
        mesh_device = tt::tt_metal::distributed::MeshDevice::create(
            tt::tt_metal::distributed::MeshDeviceConfig(mesh_shape),
            l1_small_size_,
            trace_region_size_,
            1,  // num_command_queues
            tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::WORKER});

        ASSERT_NE(mesh_device, nullptr) << "Failed to open compute mesh device";
        EXPECT_EQ(mesh_device->shape(), mesh_shape) << "Mesh device shape mismatch";

        // Verify that the mesh device uses the correct mesh_id from TT_MESH_ID
        // by checking fabric node IDs
        auto fabric_node_id = mesh_device->get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(0, 0));
        EXPECT_EQ(*fabric_node_id.mesh_id, mesh_id_val)
            << "Mesh device should use mesh_id from TT_MESH_ID environment variable";
    }
}

TEST_F(MeshDeviceTTSwitchFixture, TestOpenMeshDeviceWithExplicitPhysicalDeviceIds) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // Get chip IDs for mesh_id 0
    tt::tt_fabric::MeshId mesh_id_0(0);
    const auto& chip_ids_container_0 = mesh_graph.get_chip_ids(mesh_id_0);
    std::vector<int> device_ids_0;
    // Convert logical chip IDs to physical chip IDs
    for (const auto& logical_chip_id : chip_ids_container_0.values()) {
        tt::tt_fabric::FabricNodeId fabric_node_id(mesh_id_0, logical_chip_id);
        int physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
        device_ids_0.push_back(physical_chip_id);
    }

    EXPECT_EQ(device_ids_0.size(), 4) << "Mesh 0 should have 4 devices";

    // When you provide physical_device_ids explicitly, MeshDevice::create will use those
    // specific devices. The fabric node IDs will be determined from the physical device IDs,
    // which will map to the mesh_id that those devices belong to.
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
    {
        // Create mesh device with explicit physical device IDs
        // This bypasses SystemMesh and directly uses the provided device IDs
        mesh_device = tt::tt_metal::distributed::MeshDevice::create(
            tt::tt_metal::distributed::MeshDeviceConfig(
                tt::tt_metal::distributed::MeshShape(2, 2),
                std::nullopt,   // offset
                device_ids_0),  // explicit physical_device_ids
            l1_small_size_,
            trace_region_size_,
            1,  // num_command_queues
            tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::WORKER});

        ASSERT_NE(mesh_device, nullptr) << "Failed to open mesh device with explicit device IDs";
        EXPECT_EQ(mesh_device->shape(), tt::tt_metal::distributed::MeshShape(2, 2)) << "Mesh device shape mismatch";

        // Verify fabric node IDs match the mesh_id of the physical devices
        // (which should be mesh_id 0 in this case)
        auto fabric_node_id = mesh_device->get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(0, 0));
        EXPECT_EQ(*fabric_node_id.mesh_id, 0) << "Fabric node ID should match the mesh_id of the physical devices";
    }

    // Close mesh device
    mesh_device->close();
    mesh_device.reset();
}

TEST_F(MeshDeviceTTSwitchFixture, TestOpenUnitMeshesOnComputeMeshFabricNodes) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    auto mesh_id_val = 0;

    // Verify this is a compute mesh, not a switch
    tt::tt_fabric::MeshId mesh_id(mesh_id_val);
    const auto& switch_ids = mesh_graph.get_switch_ids();
    bool is_switch = false;
    for (const auto& switch_id : switch_ids) {
        if (*switch_id == mesh_id_val) {
            is_switch = true;
            break;
        }
    }
    ASSERT_FALSE(is_switch) << "Mesh ID " << mesh_id_val << " should be a compute mesh, not a switch";

    // Get chip IDs for this mesh
    const auto& chip_ids_container = mesh_graph.get_chip_ids(mesh_id);
    std::vector<int> device_ids;
    // Convert logical chip IDs to physical chip IDs
    for (const auto& logical_chip_id : chip_ids_container.values()) {
        tt::tt_fabric::FabricNodeId fabric_node_id(mesh_id, logical_chip_id);
        int physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
        device_ids.push_back(physical_chip_id);
    }

    EXPECT_EQ(device_ids.size(), 4) << "Compute mesh should have 4 devices (2x2)";

    // Create unit meshes for each device (this tests fabric node mapping)
    std::map<int, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> unit_meshes;
    {
        unit_meshes = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
            device_ids,
            l1_small_size_,
            trace_region_size_,
            1,  // num_command_queues
            tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::WORKER});

        EXPECT_EQ(unit_meshes.size(), device_ids.size()) << "Should create one unit mesh per device";

        // Verify each unit mesh has correct fabric node ID mapping
        for (const auto& [device_id, unit_mesh] : unit_meshes) {
            ASSERT_NE(unit_mesh, nullptr) << "Unit mesh for device " << device_id << " should not be null";
            EXPECT_EQ(unit_mesh->shape(), tt::tt_metal::distributed::MeshShape(1, 1)) << "Unit mesh should be 1x1";

            // Verify fabric node ID is correctly mapped
            auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
            auto expected_fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device_id);
            auto actual_fabric_node_id = unit_mesh->get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(0, 0));
            EXPECT_EQ(actual_fabric_node_id.mesh_id, expected_fabric_node_id.mesh_id)
                << "Fabric node mesh ID mismatch for device " << device_id;
            EXPECT_EQ(actual_fabric_node_id.chip_id, expected_fabric_node_id.chip_id)
                << "Fabric node chip ID mismatch for device " << device_id;
        }
    }

    // Close all unit meshes
    for (auto& [device_id, unit_mesh] : unit_meshes) {
        unit_mesh->close();
    }
    unit_meshes.clear();
}

}  // namespace tt::tt_metal
