// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <memory>
#include <optional>

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/impl/device/device_pool.hpp"
#include <llrt/tt_cluster.hpp>
#include "ttnn/device.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn {
namespace test {

class MeshDeviceTTSwitchFixture : public TTNNFixtureBase {
protected:
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

        // Set fabric config (only if not already set)
        // Note: initialize_fabric_config() is called automatically by the system
        // when opening mesh devices, so we don't need to call it here
        tt::tt_metal::MetalContext::instance().set_fabric_config(
            tt::tt_fabric::FabricConfig::FABRIC_2D,
            tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    }

    void initialize_device_pool() {
        // For switch meshes, DevicePool will be automatically initialized when Control Plane
        // is accessed (via get_control_plane()). No explicit initialization needed.

        // initialize device pool still needs to be called to add switch devices to the pool
        // Init the device pool - pass empty list and let DevicePool::initialize() add switch devices automatically
        tt::DevicePool::initialize(
            {},  // Empty device_ids - DevicePool will automatically add switch devices if on switch mesh
            1,   // num_command_queues
            l1_small_size_,
            trace_region_size_,
            tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::WORKER},
            {},     // l1_bank_remap
            false,  // init_profiler
            true,   // use_max_eth_core_count_on_all_devices
            true);  // initialize_fabric_and_dispatch_fw
    }

    void TearDown() override {
        // Close all active devices to ensure proper fabric handshake between tests.
        // This is critical because fabric routers wait for peer handshake, and if
        // devices remain open from a previous test, the handshake won't be re-initiated,
        // causing subsequent tests to hang.
        if (tt::DevicePool::is_initialized()) {
            auto active_devices = tt::DevicePool::instance().get_all_active_devices();
            if (!active_devices.empty()) {
                tt::DevicePool::instance().close_devices(active_devices);
            }
        }
    }
};

TEST_F(MeshDeviceTTSwitchFixture, TestOpenCloseComputeMeshDevice) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // Only test compute mesh (mesh_id 0), not the switch
    if (control_plane.is_local_host_on_switch_mesh()) {
        initialize_device_pool();
        GTEST_SKIP() << "This test is for compute mesh only (mesh_id 0)";
    }

    auto mesh_id_val = *control_plane.get_local_mesh_id_bindings()[0];

    // Verify this is a compute mesh, not a switch
    tt::tt_fabric::MeshId mesh_id(mesh_id_val);
    ASSERT_FALSE(mesh_graph.is_switch(mesh_id))
        << "Mesh ID " << mesh_id_val << " should be a compute mesh, not a switch";

    // Get mesh shape
    const auto mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
    EXPECT_EQ(mesh_shape, tt::tt_metal::distributed::MeshShape(2, 2)) << "Compute mesh should have 2x2 shape";

    // Open mesh device - by default, this uses SystemMesh which is initialized with
    // the mesh_id from TT_MESH_ID environment variable. The SystemMesh singleton
    // maps logical coordinates to physical device IDs based on the mesh_id.
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
    {
        mesh_device = ttnn::distributed::open_mesh_device(
            mesh_shape,
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

    // Close mesh device
    ttnn::distributed::close_mesh_device(mesh_device);
    mesh_device.reset();

    // Verify device is closed (should not crash on reset)
    EXPECT_EQ(mesh_device, nullptr);
}

TEST_F(MeshDeviceTTSwitchFixture, TestOpenMeshDeviceWithExplicitPhysicalDeviceIds) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // Only test compute mesh (mesh_id 0), not the switch
    if (control_plane.is_local_host_on_switch_mesh()) {
        initialize_device_pool();
        GTEST_SKIP() << "This test is for compute mesh only (mesh_id 0)";
    }

    // Get chip IDs for mesh_id 0
    tt::tt_fabric::MeshId mesh_id_0(0);
    const auto& chip_ids_container_0 = mesh_graph.get_chip_ids(mesh_id_0);
    std::vector<int> device_ids_0;
    for (const auto& chip_id : chip_ids_container_0.values()) {
        device_ids_0.push_back(chip_id);
    }

    EXPECT_EQ(device_ids_0.size(), 4) << "Mesh 0 should have 4 devices";

    // When you provide physical_device_ids explicitly, open_mesh_device will use those
    // specific devices. The fabric node IDs will be determined from the physical device IDs,
    // which will map to the mesh_id that those devices belong to.
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
    {
        // Open mesh device with explicit physical device IDs
        // This bypasses SystemMesh and directly uses the provided device IDs
        mesh_device = ttnn::distributed::open_mesh_device(
            tt::tt_metal::distributed::MeshShape(2, 2),
            l1_small_size_,
            trace_region_size_,
            1,  // num_command_queues
            tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::WORKER},
            std::nullopt,   // offset
            device_ids_0);  // explicit physical_device_ids

        ASSERT_NE(mesh_device, nullptr) << "Failed to open mesh device with explicit device IDs";
        EXPECT_EQ(mesh_device->shape(), tt::tt_metal::distributed::MeshShape(2, 2)) << "Mesh device shape mismatch";

        // Verify fabric node IDs match the mesh_id of the physical devices
        // (which should be mesh_id 0 in this case)
        auto fabric_node_id = mesh_device->get_fabric_node_id(tt::tt_metal::distributed::MeshCoordinate(0, 0));
        EXPECT_EQ(*fabric_node_id.mesh_id, 0) << "Fabric node ID should match the mesh_id of the physical devices";
    }

    // Close mesh device
    ttnn::distributed::close_mesh_device(mesh_device);
    mesh_device.reset();

    // Note: You CANNOT open devices from mesh_id 1 on a host configured for mesh_id 0
    // because:
    // 1. TT_VISIBLE_DEVICES is set per-rank and only exposes devices for that mesh
    // 2. Device locks prevent accessing devices assigned to other ranks
    // 3. The fabric node IDs are determined by the physical device IDs, which must
    //    belong to the mesh_id specified in the mesh graph descriptor
}

TEST_F(MeshDeviceTTSwitchFixture, TestOpenCloseSwitchMeshDevice) {
    GTEST_SKIP() << "This test is disabled because it hangs when running in multi-process environment";
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // Only test switch mesh - skip if not on switch mesh
    if (!control_plane.is_local_host_on_switch_mesh()) {
        initialize_device_pool();
        GTEST_SKIP() << "This test is for switch mesh only";
    }

    auto mesh_id_val = *control_plane.get_local_mesh_id_bindings()[0];

    // Verify this is a switch
    tt::tt_fabric::MeshId mesh_id(mesh_id_val);
    const auto& switch_ids = mesh_graph.get_switch_ids();
    bool is_switch = false;
    for (const auto& switch_id : switch_ids) {
        if (*switch_id == mesh_id_val) {
            is_switch = true;
            break;
        }
    }
    ASSERT_TRUE(is_switch) << "Mesh ID " << mesh_id_val << " should be a switch";

    // Get mesh shape for switch
    const auto mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
    EXPECT_EQ(mesh_shape, tt::tt_metal::distributed::MeshShape(2, 2)) << "Switch mesh should have 2x2 shape";

    // Verify switch connectivity
    tt::tt_fabric::SwitchId switch_id(mesh_id_val);
    const auto& connected_meshes = mesh_graph.get_meshes_connected_to_switch(switch_id);
    EXPECT_EQ(connected_meshes.size(), 1) << "Switch should be connected to 1 compute mesh";

    // Attempting to open mesh device on switch should fail
    // Devices cannot be created on tt-switch meshes
    EXPECT_THROW(
        {
            ttnn::distributed::open_mesh_device(
                mesh_shape,
                l1_small_size_,
                trace_region_size_,
                1,  // num_command_queues
                tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::WORKER});
        },
        std::exception)
        << "Opening mesh device on switch should fail";
}

TEST_F(MeshDeviceTTSwitchFixture, TestOpenUnitMeshesOnComputeMeshFabricNodes) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // Only test compute mesh (mesh_id 0), not the switch
    if (control_plane.is_local_host_on_switch_mesh()) {
        initialize_device_pool();
        GTEST_SKIP() << "This test is for compute mesh only (mesh_id 0)";
    }

    auto mesh_id_val = *control_plane.get_local_mesh_id_bindings()[0];

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
    for (const auto& chip_id : chip_ids_container.values()) {
        device_ids.push_back(chip_id);
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

TEST_F(MeshDeviceTTSwitchFixture, TestOpenUnitMeshesOnSwitchFabricNodes) {
    GTEST_SKIP() << "This test is disabled because it hangs when running in multi-process environment";
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();

    // Only test switch mesh - skip if not on switch mesh
    if (!control_plane.is_local_host_on_switch_mesh()) {
        initialize_device_pool();
        GTEST_SKIP() << "This test is for switch mesh only";
    }

    auto mesh_id_val = *control_plane.get_local_mesh_id_bindings()[0];

    // Verify this is a switch
    tt::tt_fabric::MeshId mesh_id(mesh_id_val);
    const auto& switch_ids = mesh_graph.get_switch_ids();
    bool is_switch = false;
    for (const auto& switch_id : switch_ids) {
        if (*switch_id == mesh_id_val) {
            is_switch = true;
            break;
        }
    }
    ASSERT_TRUE(is_switch) << "Mesh ID " << mesh_id_val << " should be a switch";

    // Get chip IDs for the switch mesh
    const auto& chip_ids_container = mesh_graph.get_chip_ids(mesh_id);
    std::vector<int> device_ids;
    for (const auto& chip_id : chip_ids_container.values()) {
        device_ids.push_back(chip_id);
    }

    EXPECT_EQ(device_ids.size(), 4) << "Switch mesh should have 4 devices (2x2)";

    // Attempting to create unit meshes on switch devices should fail
    // Devices cannot be created on tt-switch meshes
    EXPECT_THROW(
        {
            tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
                device_ids,
                l1_small_size_,
                trace_region_size_,
                1,  // num_command_queues
                tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::WORKER});
        },
        std::exception)
        << "Creating unit meshes on switch devices should fail";
}

}  // namespace test
}  // namespace ttnn
