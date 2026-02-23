// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device-side functional tests for wrap-around multicast on NoC torus architectures (WH/BH).
//
// Verifies that hardware multicasts work correctly with wrap-around coordinates (end < start).
// Tests both noc0 and noc1 behavior. Some tests are disabled due to hardware limitations.

#include <gtest/gtest.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "debug_tools_fixture.hpp"
#include <tt-logger/tt-logger.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

// Helper to run a wrap-around multicast test on device
bool RunDeviceMcastWrapAroundTest(
    IDevice* device,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    NOC noc_id,
    CoreCoord sender_logical,
    CoreCoord mcast_start_logical,
    CoreCoord mcast_end_logical,
    const std::string& test_name) {
    bool pass = true;

    // Convert logical to NOC coordinates
    CoreCoord sender_noc = device->worker_core_from_logical_core(sender_logical);
    CoreCoord mcast_start_noc = device->worker_core_from_logical_core(mcast_start_logical);
    CoreCoord mcast_end_noc = device->worker_core_from_logical_core(mcast_end_logical);

    log_info(
        LogTest,
        "{}: sender_noc={}, mcast_start_noc={}, mcast_end_noc={}, noc={}",
        test_name,
        sender_noc.str(),
        mcast_start_noc.str(),
        mcast_end_noc.str(),
        noc_id == NOC::RISCV_0_default ? "noc0" : "noc1");

    // Create program
    Program program = CreateProgram();

    // Test data
    constexpr uint32_t data_size_bytes = 64;  // Small test payload
    constexpr uint32_t sender_l1_addr = 400 * 1024;
    constexpr uint32_t receiver_l1_addr = 500 * 1024;

    std::vector<uint32_t> test_data(data_size_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < test_data.size(); i++) {
        test_data[i] = 0xABCD0000 + i;  // Unique pattern
    }

    // Calculate number of destination cores
    // For wrap-around, need to count cores in the wrapped region
    uint32_t num_dests = 0;

    // Determine which cores are in the multicast region (accounting for wrap-around)
    // Only include valid worker cores (exclude dispatch cores, etc.)
    std::vector<CoreCoord> receiver_cores;
    CoreCoord grid = device->compute_with_storage_grid_size();

    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            CoreCoord logical_core(x, y);

            // Skip sender core
            if (logical_core == sender_logical) {
                continue;
            }

            // Try to convert to NOC coordinates - if it fails, this isn't a valid worker core
            CoreCoord noc_core;
            try {
                noc_core = device->worker_core_from_logical_core(logical_core);
            } catch (...) {
                // Not a valid worker core (e.g., dispatch core, harvested core)
                continue;
            }

            // Check if this core is in the multicast rectangle (with wrap-around)
            bool in_x_range, in_y_range;

            if (mcast_start_noc.x <= mcast_end_noc.x) {
                // Normal X range
                in_x_range = (noc_core.x >= mcast_start_noc.x && noc_core.x <= mcast_end_noc.x);
            } else {
                // Wrap-around X range
                in_x_range = (noc_core.x >= mcast_start_noc.x || noc_core.x <= mcast_end_noc.x);
            }

            if (mcast_start_noc.y <= mcast_end_noc.y) {
                // Normal Y range
                in_y_range = (noc_core.y >= mcast_start_noc.y && noc_core.y <= mcast_end_noc.y);
            } else {
                // Wrap-around Y range
                in_y_range = (noc_core.y >= mcast_start_noc.y || noc_core.y <= mcast_end_noc.y);
            }

            if (in_x_range && in_y_range) {
                receiver_cores.push_back(logical_core);
                num_dests++;
            }
        }
    }

    log_info(LogTest, "Number of receiver cores: {}", num_dests);

    if (num_dests == 0) {
        log_warning(LogTest, "No receiver cores found, skipping test");
        return true;
    }

    // Create sender kernel
    // Use RISCV_0 (BRISC) for noc0, RISCV_1 (NCRISC) for noc1 (conventional pairing)
    DataMovementProcessor sender_processor =
        (noc_id == NOC::RISCV_0_default) ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1;
    DataMovementProcessor receiver_processor =
        (noc_id == NOC::RISCV_0_default) ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0;

    auto sender_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/mcast_wrap_around_sender.cpp",
        sender_logical,
        DataMovementConfig{.processor = sender_processor, .noc = noc_id});

    // Create receiver kernels on all destination cores
    for (const auto& receiver_core : receiver_cores) {
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/mcast_wrap_around_receiver.cpp",
            receiver_core,
            DataMovementConfig{.processor = receiver_processor, .noc = noc_id});
    }

    // Set runtime args for sender
    SetRuntimeArgs(
        program,
        sender_kernel,
        sender_logical,
        {sender_l1_addr,
         receiver_l1_addr,
         mcast_start_noc.x,
         mcast_start_noc.y,
         mcast_end_noc.x,
         mcast_end_noc.y,
         num_dests,
         data_size_bytes});

    // Write test data to sender's L1
    detail::WriteToDeviceL1(device, sender_logical, sender_l1_addr, test_data);

    // Execute program using fast dispatch (compatible with MeshWatcherFixture)
    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord{0, 0};
    distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());

    // Verify data on all receiver cores
    for (const auto& receiver_core : receiver_cores) {
        std::vector<uint32_t> readback_data;
        detail::ReadFromDeviceL1(device, receiver_core, receiver_l1_addr, data_size_bytes, readback_data);

        if (readback_data != test_data) {
            log_error(LogTest, "Data mismatch on receiver core {}", receiver_core.str());
            pass = false;
        }
    }

    return pass;
}

}  // namespace

// Test Y-dimension wrap-around with noc0 (single column)
TEST_F(MeshWatcherFixture, DeviceMcastWrapAroundY_Down_Noc0_SingleCol) {
    auto* device = this->devices_[0]->get_devices()[0];
    CoreCoord grid = device->logical_grid_size();

    if (grid.y < 2) {
        GTEST_SKIP() << "Need grid.y >= 2 for Y-wrap test";
    }

    CoreCoord sender(1, 0);
    CoreCoord mcast_start(0, grid.y - 1);
    CoreCoord mcast_end(0, 0);

    bool pass = RunDeviceMcastWrapAroundTest(
        device, this->devices_[0], NOC::RISCV_0_default, sender, mcast_start, mcast_end, "Y-wrap single column (noc0)");

    EXPECT_TRUE(pass);
}

// Test S4 pattern from DeepSeek: multi-column Y-wrap
TEST_F(MeshWatcherFixture, DeviceMcastWrapAroundY_Down_Noc0_MultiCol_S4) {
    auto* device = this->devices_[0]->get_devices()[0];
    CoreCoord grid = device->logical_grid_size();

    if (grid.y < 2 || grid.x < 5) {
        GTEST_SKIP() << "Need grid.y >= 2 and grid.x >= 5 for S4 pattern test";
    }

    // S4 pattern: multi-column wrap from bottom to top
    // Covers approximately 4 columns (x=1 to x=4)
    CoreCoord sender(5, 5);                // Safe sender position
    CoreCoord mcast_start(1, grid.y - 1);  // Bottom row, x=1
    CoreCoord mcast_end(4, 0);             // Top row, x=4 (wraps in Y, spans 4 columns)

    bool pass = RunDeviceMcastWrapAroundTest(
        device,
        this->devices_[0],
        NOC::RISCV_0_default,
        sender,
        mcast_start,
        mcast_end,
        "S4 pattern: multi-column Y-wrap (noc0)");

    EXPECT_TRUE(pass);
}

// Test S8 pattern from DeepSeek: multi-column Y-wrap with noc1
TEST_F(MeshWatcherFixture, DeviceMcastWrapAroundY_Down_Noc1_MultiCol_S8) {
    auto* device = this->devices_[0]->get_devices()[0];
    CoreCoord grid = device->logical_grid_size();

    if (grid.y < 2 || grid.x < 12) {
        GTEST_SKIP() << "Need grid.y >= 2 and grid.x >= 12 for S8 pattern test";
    }

    // S8 pattern: multi-column wrap from bottom to top
    // Adjusted to fit Blackhole grid: covers 4 columns (x=7 to x=10)
    CoreCoord sender(5, 5);                // Safe sender position
    CoreCoord mcast_start(7, grid.y - 1);  // Bottom row, x=7
    CoreCoord mcast_end(10, 0);            // Top row, x=10 (wraps in Y, spans 4 columns)

    bool pass = RunDeviceMcastWrapAroundTest(
        device,
        this->devices_[0],
        NOC::RISCV_1_default,
        sender,
        mcast_start,
        mcast_end,
        "S8 pattern: multi-column Y-wrap (noc1)");

    EXPECT_TRUE(pass);
}

// Test Y-dimension wrap-around with noc1 (single column)
TEST_F(MeshWatcherFixture, DeviceMcastWrapAroundY_Down_Noc1_SingleCol) {
    auto* device = this->devices_[0]->get_devices()[0];
    CoreCoord grid = device->logical_grid_size();

    if (grid.y < 2) {
        GTEST_SKIP() << "Need grid.y >= 2 for Y-wrap test";
    }

    CoreCoord sender(1, 0);
    CoreCoord mcast_start(0, grid.y - 1);
    CoreCoord mcast_end(0, 0);

    bool pass = RunDeviceMcastWrapAroundTest(
        device, this->devices_[0], NOC::RISCV_1_default, sender, mcast_start, mcast_end, "Y-wrap single column (noc1)");

    EXPECT_TRUE(pass);
}

// Test X-dimension wrap-around with noc0
TEST_F(MeshWatcherFixture, DeviceMcastWrapAroundX_Right_Noc0) {
    auto* device = this->devices_[0]->get_devices()[0];
    CoreCoord grid = device->logical_grid_size();

    if (grid.x < 2) {
        GTEST_SKIP() << "Need grid.x >= 2 for X-wrap test";
    }

    CoreCoord sender(0, 1);
    CoreCoord mcast_start(grid.x - 1, 0);
    CoreCoord mcast_end(0, 0);

    bool pass = RunDeviceMcastWrapAroundTest(
        device, this->devices_[0], NOC::RISCV_0_default, sender, mcast_start, mcast_end, "X-wrap right (noc0)");

    EXPECT_TRUE(pass);
}

// Test X-dimension wrap-around with noc1
TEST_F(MeshWatcherFixture, DeviceMcastWrapAroundX_Right_Noc1) {
    auto* device = this->devices_[0]->get_devices()[0];
    CoreCoord grid = device->logical_grid_size();

    if (grid.x < 2) {
        GTEST_SKIP() << "Need grid.x >= 2 for X-wrap test";
    }

    CoreCoord sender(0, 1);
    CoreCoord mcast_start(grid.x - 1, 0);
    CoreCoord mcast_end(0, 0);

    bool pass = RunDeviceMcastWrapAroundTest(
        device, this->devices_[0], NOC::RISCV_1_default, sender, mcast_start, mcast_end, "X-wrap right (noc1)");

    EXPECT_TRUE(pass);
}

// X-wrap spanning multiple Y rows (should work like S4/S8)
TEST_F(MeshWatcherFixture, DeviceMcastMixedWrap_XWrapYNormal_Noc0) {
    auto* device = this->devices_[0]->get_devices()[0];
    CoreCoord grid = device->logical_grid_size();

    if (grid.x < 2 || grid.y < 3) {
        GTEST_SKIP() << "Need grid.x >= 2 and grid.y >= 3 for mixed wrap test";
    }

    // Try to mirror S4 pattern but in X dimension
    // S4: Y-wrap from y=11 to y=0, spanning x=1 to x=4 (4 columns)
    // This: X-wrap from x=11 to x=0, spanning y=1 to y=4 (4 rows)
    CoreCoord sender(5, 5);
    CoreCoord mcast_start(11, 1);  // Right side, starting at row 1
    CoreCoord mcast_end(0, 4);     // Left side, ending at row 4 (X wraps, Y spans 4 rows)

    bool pass = RunDeviceMcastWrapAroundTest(
        device,
        this->devices_[0],
        NOC::RISCV_0_default,
        sender,
        mcast_start,
        mcast_end,
        "Mixed: X-wrap + Y-normal (noc0)");

    EXPECT_TRUE(pass);
}

// Test dual-dimension wrap: down and right (both X and Y wrap)
// DISABLED: Dual-dimension wraps hang on hardware
TEST_F(MeshWatcherFixture, DISABLED_DeviceMcastWrapAroundXY_DownRight_Noc0) {
    auto* device = this->devices_[0]->get_devices()[0];
    CoreCoord grid = device->logical_grid_size();

    if (grid.x < 3 || grid.y < 3) {
        GTEST_SKIP() << "Need grid.x >= 3 and grid.y >= 3 for XY-wrap test";
    }

    // Use safe coordinates that won't go out of bounds
    // X-wrap: right to left, Y-wrap: bottom to top
    CoreCoord sender(5, 5);
    CoreCoord mcast_start(10, grid.y - 2);  // Near bottom-right
    CoreCoord mcast_end(1, 1);              // Near top-left (both dimensions wrap)

    bool pass = RunDeviceMcastWrapAroundTest(
        device, this->devices_[0], NOC::RISCV_0_default, sender, mcast_start, mcast_end, "XY-wrap down-right (noc0)");

    EXPECT_TRUE(pass);
}

// Test dual-dimension wrap: left and up (both X and Y wrap with noc1)
// DISABLED: Dual-dimension wraps hang on hardware
TEST_F(MeshWatcherFixture, DISABLED_DeviceMcastWrapAroundXY_LeftUp_Noc1) {
    auto* device = this->devices_[0]->get_devices()[0];
    CoreCoord grid = device->logical_grid_size();

    if (grid.x < 3 || grid.y < 3) {
        GTEST_SKIP() << "Need grid.x >= 3 and grid.y >= 3 for XY-wrap test";
    }

    // Use safe coordinates for noc1
    // X-wrap: right to left, Y-wrap: bottom to top
    CoreCoord sender(5, 5);
    CoreCoord mcast_start(9, grid.y - 2);  // Near bottom-right
    CoreCoord mcast_end(1, 1);             // Near top-left (both dimensions wrap)

    bool pass = RunDeviceMcastWrapAroundTest(
        device, this->devices_[0], NOC::RISCV_1_default, sender, mcast_start, mcast_end, "XY-wrap left-up (noc1)");

    EXPECT_TRUE(pass);
}
