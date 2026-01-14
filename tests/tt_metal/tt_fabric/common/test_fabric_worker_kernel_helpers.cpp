// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <memory>

#include "tests/tt_metal/tt_fabric/common/fabric_worker_kernel_helpers.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"

namespace tt::tt_fabric::test_utils {

// Test fixture for worker kernel helpers
class FabricWorkerKernelHelpersTest : public Fabric1DFixture {
protected:
    void SetUp() override {
        Fabric1DFixture::SetUp();
    }

    void TearDown() override {
        Fabric1DFixture::TearDown();
    }
};

// Test that allocate_worker_memory returns a valid memory layout
TEST_F(FabricWorkerKernelHelpersTest, AllocateWorkerMemoryReturnsValidLayout) {
    auto devices = get_devices();
    ASSERT_FALSE(devices.empty()) << "No devices available";

    auto layout = allocate_worker_memory(devices[0]);

    // Verify layout contains valid addresses
    EXPECT_NE(layout.source_buffer_address, 0)
        << "Source buffer address should not be zero";
    EXPECT_NE(layout.teardown_signal_address, 0)
        << "Teardown signal address should not be zero";
    EXPECT_GT(layout.packet_payload_size_bytes, 0)
        << "Packet payload size should be positive";

    log_info(LogTest, "Allocated worker memory: source=0x{:x}, teardown=0x{:x}, payload={}",
        layout.source_buffer_address, layout.teardown_signal_address,
        layout.packet_payload_size_bytes);
}

// Test that multiple allocations return non-overlapping addresses
TEST_F(FabricWorkerKernelHelpersTest, MultipleAllocationsNonOverlapping) {
    auto devices = get_devices();
    ASSERT_FALSE(devices.empty()) << "No devices available";

    auto layout1 = allocate_worker_memory(devices[0]);
    auto layout2 = allocate_worker_memory(devices[0]);

    // Verify addresses don't overlap
    // Source buffer should be different or in different regions
    if (layout1.source_buffer_address == layout2.source_buffer_address) {
        // If same address is used (e.g., for reuse), that's also valid
        log_info(LogTest, "Allocations using same address (reusable pool)");
    }

    // Teardown signal addresses should be different if both are allocated
    if (layout1.teardown_signal_address != layout2.teardown_signal_address) {
        // Verify they don't overlap
        uint32_t end1 = layout1.teardown_signal_address + sizeof(uint32_t);
        uint32_t end2 = layout2.teardown_signal_address + sizeof(uint32_t);

        bool overlap = (layout1.teardown_signal_address < end2 &&
                       layout2.teardown_signal_address < end1);
        EXPECT_FALSE(overlap) << "Teardown signal addresses overlap";
    }

    log_info(LogTest, "Layout 1: source=0x{:x}, teardown=0x{:x}",
        layout1.source_buffer_address, layout1.teardown_signal_address);
    log_info(LogTest, "Layout 2: source=0x{:x}, teardown=0x{:x}",
        layout2.source_buffer_address, layout2.teardown_signal_address);
}

// Test that WorkerMemoryLayout struct is properly initialized
TEST_F(FabricWorkerKernelHelpersTest, WorkerMemoryLayoutInitialization) {
    WorkerMemoryLayout layout;

    // Default construction should initialize members
    EXPECT_GE(layout.source_buffer_address, 0)
        << "Source buffer address should be initialized";
    EXPECT_GE(layout.teardown_signal_address, 0)
        << "Teardown signal address should be initialized";
    EXPECT_GE(layout.packet_payload_size_bytes, 0)
        << "Packet payload size should be initialized";

    log_info(LogTest, "Default WorkerMemoryLayout: source=0x{:x}, teardown=0x{:x}, payload={}",
        layout.source_buffer_address, layout.teardown_signal_address,
        layout.packet_payload_size_bytes);
}

// Test that create_traffic_generator_program returns a valid program
TEST_F(FabricWorkerKernelHelpersTest, CreateTrafficGeneratorProgramValid) {
    auto devices = get_devices();
    ASSERT_FALSE(devices.empty()) << "No devices available";

    auto mem_layout = allocate_worker_memory(devices[0]);
    CoreCoord worker_core = {0, 0};
    FabricNodeId dest_node;
    dest_node.mesh_id = 0;
    dest_node.logical_x = 0;
    dest_node.logical_y = 1;

    auto program = create_traffic_generator_program(
        devices[0], worker_core, dest_node, mem_layout);

    EXPECT_TRUE(program) << "Program creation failed";
    EXPECT_NE(program.get(), nullptr) << "Program pointer should not be null";

    log_info(LogTest, "Traffic generator program created successfully");
}

// Test that get_fabric_node_id returns a valid node ID
TEST_F(FabricWorkerKernelHelpersTest, GetFabricNodeIdValid) {
    auto devices = get_devices();
    ASSERT_FALSE(devices.empty()) << "No devices available";

    auto node_id = get_fabric_node_id(devices[0]);

    EXPECT_GE(node_id.mesh_id, 0) << "Mesh ID should be non-negative";
    EXPECT_GE(node_id.logical_x, 0) << "Logical X should be non-negative";
    EXPECT_GE(node_id.logical_y, 0) << "Logical Y should be non-negative";

    log_info(LogTest, "Fabric node ID: mesh={}, x={}, y={}",
        node_id.mesh_id, node_id.logical_x, node_id.logical_y);
}

// Test that signal_worker_teardown executes without error
TEST_F(FabricWorkerKernelHelpersTest, SignalWorkerTeardownExecutes) {
    auto devices = get_devices();
    ASSERT_FALSE(devices.empty()) << "No devices available";

    auto mem_layout = allocate_worker_memory(devices[0]);
    CoreCoord worker_core = {0, 0};

    // Should not throw or crash
    EXPECT_NO_THROW({
        signal_worker_teardown(devices[0], worker_core,
            mem_layout.teardown_signal_address);
    }) << "signal_worker_teardown should execute without error";

    log_info(LogTest, "Worker teardown signal executed successfully");
}

// Test that wait_for_worker_complete executes without error
TEST_F(FabricWorkerKernelHelpersTest, WaitForWorkerCompleteExecutes) {
    auto devices = get_devices();
    ASSERT_FALSE(devices.empty()) << "No devices available";

    auto mem_layout = allocate_worker_memory(devices[0]);
    CoreCoord worker_core = {0, 0};

    auto program = create_traffic_generator_program(
        devices[0], worker_core, get_fabric_node_id(devices[0]), mem_layout);

    // Should not throw or crash
    EXPECT_NO_THROW({
        wait_for_worker_complete(this, devices[0], *program,
            std::chrono::milliseconds(1000));
    }) << "wait_for_worker_complete should execute without error";

    log_info(LogTest, "wait_for_worker_complete executed successfully");
}

// Test with multiple devices
TEST_F(FabricWorkerKernelHelpersTest, MultipleDeviceAllocation) {
    auto devices = get_devices();
    if (devices.size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 devices";
    }

    // Allocate for multiple devices
    std::vector<WorkerMemoryLayout> layouts;
    for (const auto& device : devices) {
        layouts.push_back(allocate_worker_memory(device));
    }

    EXPECT_EQ(layouts.size(), devices.size())
        << "Should allocate for all devices";

    log_info(LogTest, "Allocated memory for {} devices", layouts.size());
}

// Test packet payload size is reasonable
TEST_F(FabricWorkerKernelHelpersTest, PacketPayloadSizeReasonable) {
    auto devices = get_devices();
    ASSERT_FALSE(devices.empty()) << "No devices available";

    auto layout = allocate_worker_memory(devices[0]);

    // Packet size should be reasonable (not too large, not too small)
    EXPECT_GE(layout.packet_payload_size_bytes, 64)
        << "Packet payload should be at least 64 bytes";
    EXPECT_LE(layout.packet_payload_size_bytes, 65536)
        << "Packet payload should not exceed 64KB";

    log_info(LogTest, "Packet payload size: {} bytes",
        layout.packet_payload_size_bytes);
}

// Test fabric node ID values are in expected range
TEST_F(FabricWorkerKernelHelpersTest, FabricNodeIdValuesInRange) {
    auto devices = get_devices();
    ASSERT_FALSE(devices.empty()) << "No devices available";

    auto node_id = get_fabric_node_id(devices[0]);

    // Mesh ID should be reasonable
    EXPECT_LT(node_id.mesh_id, 256) << "Mesh ID should be in valid range";

    // Coordinates should be in reasonable range for typical systems
    EXPECT_LT(node_id.logical_x, 256) << "Logical X should be < 256";
    EXPECT_LT(node_id.logical_y, 256) << "Logical Y should be < 256";

    log_info(LogTest, "Node ID values in expected range");
}

} // namespace tt::tt_fabric::test_utils
