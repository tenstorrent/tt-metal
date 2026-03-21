// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device-side functional tests for wrap-around multicast on NoC torus architectures (WH/BH).
//
// Verifies that hardware multicasts work correctly with wrap-around coordinates (end < start).
// Tests both noc0 and noc1 for single-dimension wraps (Y-only or X-only).

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

    Program program = CreateProgram();

    constexpr uint32_t data_size_bytes = 64;

    const uint32_t sender_l1_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(
                                        HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE) +
                                    (400 * 1024);
    const uint32_t receiver_l1_addr = sender_l1_addr + data_size_bytes;

    std::vector<uint32_t> test_data(data_size_bytes / sizeof(uint32_t));
    for (size_t i = 0; i < test_data.size(); i++) {
        test_data[i] = 0xABCD0000 + i;
    }

    uint32_t num_dests = 0;
    std::vector<CoreCoord> receiver_cores;
    CoreCoord grid = device->compute_with_storage_grid_size();

    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            CoreCoord logical_core(x, y);

            if (logical_core == sender_logical) {
                continue;
            }

            CoreCoord noc_core = device->worker_core_from_logical_core(logical_core);
            bool in_x_range, in_y_range;

            if (mcast_start_noc.x <= mcast_end_noc.x) {
                in_x_range = (noc_core.x >= mcast_start_noc.x && noc_core.x <= mcast_end_noc.x);
            } else {
                in_x_range = (noc_core.x >= mcast_start_noc.x || noc_core.x <= mcast_end_noc.x);
            }

            if (mcast_start_noc.y <= mcast_end_noc.y) {
                in_y_range = (noc_core.y >= mcast_start_noc.y && noc_core.y <= mcast_end_noc.y);
            } else {
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

    DataMovementProcessor sender_processor =
        (noc_id == NOC::RISCV_0_default) ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1;

    auto sender_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/mcast_wrap_around_sender.cpp",
        sender_logical,
        DataMovementConfig{.processor = sender_processor, .noc = noc_id});

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

    detail::WriteToDeviceL1(device, sender_logical, sender_l1_addr, test_data);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinate zero_coord{0, 0};
    distributed::MeshCoordinateRange device_range{zero_coord, zero_coord};
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
    distributed::Finish(mesh_device->mesh_command_queue());

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
    if (!this->IsSlowDispatch()) {
        GTEST_SKIP() << "Wrap-around multicast device tests require Slow Dispatch mode";
    }
    auto* device = this->devices_[0]->impl().get_devices()[0];
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.y < 2) {
        GTEST_SKIP() << "Need grid.y >= 2 for Y-wrap test";
    }

    CoreCoord sender(1, 0);
    CoreCoord mcast_start(0, grid.y - 1);
    CoreCoord mcast_end(0, 0);

    bool pass = false;
    EXPECT_NO_THROW({
        pass = RunDeviceMcastWrapAroundTest(
            device,
            this->devices_[0],
            NOC::RISCV_0_default,
            sender,
            mcast_start,
            mcast_end,
            "Y-wrap single column (noc0)");
    });

    EXPECT_TRUE(pass);
}

TEST_F(MeshWatcherFixture, DeviceMcastWrapAroundY_Down_Noc0_MultiCol_S4) {
    if (!this->IsSlowDispatch()) {
        GTEST_SKIP() << "Wrap-around multicast device tests require Slow Dispatch mode";
    }
    auto* device = this->devices_[0]->impl().get_devices()[0];
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.y < 2 || grid.x < 5) {
        GTEST_SKIP() << "Need grid.y >= 2 and grid.x >= 5 for S4 pattern test";
    }

    CoreCoord sender(5, 5);
    CoreCoord mcast_start(1, grid.y - 1);
    CoreCoord mcast_end(4, 0);

    bool pass = false;
    EXPECT_NO_THROW({
        pass = RunDeviceMcastWrapAroundTest(
            device,
            this->devices_[0],
            NOC::RISCV_0_default,
            sender,
            mcast_start,
            mcast_end,
            "S4 pattern: multi-column Y-wrap (noc0)");
    });

    EXPECT_TRUE(pass);
}

TEST_F(MeshWatcherFixture, DeviceMcastWrapAroundY_Down_Noc1_MultiCol_S8) {
    if (!this->IsSlowDispatch()) {
        GTEST_SKIP() << "Wrap-around multicast device tests require Slow Dispatch mode";
    }
    auto* device = this->devices_[0]->impl().get_devices()[0];
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.y < 2 || grid.x < 12) {
        GTEST_SKIP() << "Need grid.y >= 2 and grid.x >= 12 for S8 pattern test";
    }

    CoreCoord sender(5, 5);
    CoreCoord mcast_start(7, grid.y - 1);
    CoreCoord mcast_end(10, 0);

    bool pass = false;
    EXPECT_NO_THROW({
        pass = RunDeviceMcastWrapAroundTest(
            device,
            this->devices_[0],
            NOC::RISCV_1_default,
            sender,
            mcast_start,
            mcast_end,
            "S8 pattern: multi-column Y-wrap (noc1)");
    });

    EXPECT_TRUE(pass);
}

TEST_F(MeshWatcherFixture, DeviceMcastWrapAroundY_Down_Noc1_SingleCol) {
    if (!this->IsSlowDispatch()) {
        GTEST_SKIP() << "Wrap-around multicast device tests require Slow Dispatch mode";
    }
    auto* device = this->devices_[0]->impl().get_devices()[0];
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.y < 2) {
        GTEST_SKIP() << "Need grid.y >= 2 for Y-wrap test";
    }

    CoreCoord sender(1, 0);
    CoreCoord mcast_start(0, grid.y - 1);
    CoreCoord mcast_end(0, 0);

    bool pass = false;
    EXPECT_NO_THROW({
        pass = RunDeviceMcastWrapAroundTest(
            device,
            this->devices_[0],
            NOC::RISCV_1_default,
            sender,
            mcast_start,
            mcast_end,
            "Y-wrap single column (noc1)");
    });

    EXPECT_TRUE(pass);
}

TEST_F(MeshWatcherFixture, DeviceMcastWrapAroundX_Right_Noc0) {
    if (!this->IsSlowDispatch()) {
        GTEST_SKIP() << "Wrap-around multicast device tests require Slow Dispatch mode";
    }
    auto* device = this->devices_[0]->impl().get_devices()[0];
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.x < 2) {
        GTEST_SKIP() << "Need grid.x >= 2 for X-wrap test";
    }

    CoreCoord sender(0, 1);
    CoreCoord mcast_start(grid.x - 1, 0);
    CoreCoord mcast_end(0, 0);

    bool pass = false;
    EXPECT_NO_THROW({
        pass = RunDeviceMcastWrapAroundTest(
            device, this->devices_[0], NOC::RISCV_0_default, sender, mcast_start, mcast_end, "X-wrap right (noc0)");
    });

    EXPECT_TRUE(pass);
}

TEST_F(MeshWatcherFixture, DeviceMcastWrapAroundX_Right_Noc1) {
    if (!this->IsSlowDispatch()) {
        GTEST_SKIP() << "Wrap-around multicast device tests require Slow Dispatch mode";
    }
    auto* device = this->devices_[0]->impl().get_devices()[0];
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.x < 2) {
        GTEST_SKIP() << "Need grid.x >= 2 for X-wrap test";
    }

    CoreCoord sender(0, 1);
    CoreCoord mcast_start(grid.x - 1, 0);
    CoreCoord mcast_end(0, 0);

    bool pass = false;
    EXPECT_NO_THROW({
        pass = RunDeviceMcastWrapAroundTest(
            device, this->devices_[0], NOC::RISCV_1_default, sender, mcast_start, mcast_end, "X-wrap right (noc1)");
    });

    EXPECT_TRUE(pass);
}

TEST_F(MeshWatcherFixture, DeviceMcastMixedWrap_XWrapYNormal_Noc0) {
    if (!this->IsSlowDispatch()) {
        GTEST_SKIP() << "Wrap-around multicast device tests require Slow Dispatch mode";
    }
    auto* device = this->devices_[0]->impl().get_devices()[0];
    CoreCoord grid = device->compute_with_storage_grid_size();

    if (grid.x < 2 || grid.y < 3) {
        GTEST_SKIP() << "Need grid.x >= 2 and grid.y >= 3 for mixed wrap test";
    }

    CoreCoord sender(5, 5);
    CoreCoord mcast_start(grid.x - 2, 1);
    CoreCoord mcast_end(0, 4);

    bool pass = false;
    EXPECT_NO_THROW({
        pass = RunDeviceMcastWrapAroundTest(
            device,
            this->devices_[0],
            NOC::RISCV_0_default,
            sender,
            mcast_start,
            mcast_end,
            "Mixed: X-wrap + Y-normal (noc0)");
    });

    EXPECT_TRUE(pass);
}
