// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "umd/device/tt_core_coordinates.h"
#include "umd/device/types/arch.h"
#include "umd/device/types/xy_pair.h"
#include <tt-metalium/utils.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::basic::test_noc {

const uint32_t init_value = 0x1234B33F;

uint32_t read_reg(tt::tt_metal::IDevice* device, CoreCoord logical_node, uint32_t reg_addr) {
    // Read and return reg value form reading
    uint32_t reg_data = unit_tests::basic::test_noc::init_value;
    tt_metal::detail::ReadRegFromDevice(device, logical_node, reg_addr, reg_data);
    return reg_data;
}

void read_translation_table(
    tt::tt_metal::IDevice* device,
    CoreCoord logical_node,
    std::vector<unsigned int>& x_remap,
    std::vector<unsigned int>& y_remap) {
    auto x_reg_addrs_full = tt::tt_metal::MetalContext::instance().hal().get_noc_x_id_translate_table();
    auto y_reg_addrs_full = tt::tt_metal::MetalContext::instance().hal().get_noc_y_id_translate_table();
    std::vector<uint32_t> x_reg_addrs = {
        x_reg_addrs_full[0], x_reg_addrs_full[1], x_reg_addrs_full[2], x_reg_addrs_full[3]};
    x_remap.clear();
    std::vector<uint32_t> y_reg_addrs = {
        y_reg_addrs_full[0], y_reg_addrs_full[1], y_reg_addrs_full[2], y_reg_addrs_full[3]};
    y_remap.clear();
    for (const auto& reg_addr : x_reg_addrs) {
        auto regval = read_reg(device, logical_node, reg_addr);
        for (int i = 0; i < 8; i++) {
            x_remap.push_back(regval & 0xF);
            regval = regval >> 4;
        }
    }
    for (const auto& reg_addr : y_reg_addrs) {
        auto regval = read_reg(device, logical_node, reg_addr);
        ASSERT_NE(regval, init_value);  // Need to make sure we read in valid reg
        for (int i = 0; i < 8; i++) {
            y_remap.push_back(regval & 0xF);
            regval = regval >> 4;
        }
    }
}

}  // namespace unit_tests::basic::test_noc

TEST(NOC, TensixSingleDeviceHarvestingPrints) {
    auto arch = tt::get_arch_from_string(get_umd_arch_name());
    tt::tt_metal::IDevice* device;
    const unsigned int device_id = 0;
    device = tt::tt_metal::CreateDevice(device_id);
    CoreCoord unharvested_logical_grid_size;
    switch (arch) {
        case tt::ARCH::GRAYSKULL: unharvested_logical_grid_size = CoreCoord(12, 10); break;
        case tt::ARCH::WORMHOLE_B0: unharvested_logical_grid_size = CoreCoord(8, 10); break;
        case tt::ARCH::BLACKHOLE: unharvested_logical_grid_size = CoreCoord(14, 10); break;
        default: TT_THROW("Unsupported arch {}", get_umd_arch_name());
    }
    auto logical_grid_size = device->logical_grid_size();
    if (logical_grid_size == unharvested_logical_grid_size) {
        log_info(tt::LogTest, "Harvesting Disabled in SW");
    } else {
        log_info(tt::LogTest, "Harvesting Enabled in SW");
        log_info(tt::LogTest, "Number of Harvested Rows={}", unharvested_logical_grid_size.y - logical_grid_size.y);
    }

    log_info(tt::LogTest, "Logical -- Virtual Mapping");
    log_info(tt::LogTest, "[Logical <-> Virtual] Coordinates");
    for (int r = 0; r < logical_grid_size.y; r++) {
        string output_row = "";
        for (int c = 0; c < logical_grid_size.x; c++) {
            const CoreCoord logical_coord(c, r);
            const auto noc_coord = device->worker_core_from_logical_core(logical_coord);
            output_row += "{L[x" + std::to_string(c);
            output_row += "-y" + std::to_string(r);
            output_row += "]:V[x" + std::to_string(noc_coord.x);
            output_row += "-y" + std::to_string(noc_coord.y);
            output_row += "]}, ";
        }
        log_info(tt::LogTest, "{}", output_row);
    }
    ASSERT_TRUE(tt::tt_metal::CloseDevice(device));
}

TEST(NOC, TensixVerifyNocNodeIDs) {
    auto arch = tt::get_arch_from_string(get_umd_arch_name());
    tt::tt_metal::IDevice* device;
    const unsigned int device_id = 0;
    device = tt::tt_metal::CreateDevice(device_id);

    uint32_t MY_NOC_ENCODING_REG = tt::tt_metal::MetalContext::instance().hal().get_noc_encoding_reg();

    // Ping all the Noc Nodes
    auto logical_grid_size = device->logical_grid_size();
    for (size_t y = 0; y < logical_grid_size.y; y++) {
        for (size_t x = 0; x < logical_grid_size.x; x++) {
            auto worker_core = device->worker_core_from_logical_core(CoreCoord(x, y));
            // Read register from specific node
            uint32_t node_id_regval;
            node_id_regval = unit_tests::basic::test_noc::read_reg(device, CoreCoord(x, y), MY_NOC_ENCODING_REG);
            ASSERT_NE(
                node_id_regval, unit_tests::basic::test_noc::init_value);  // Need to make sure we read in valid reg
            // Check it matches software translated xy
            uint32_t node_id_mask = tt::tt_metal::MetalContext::instance().hal().get_noc_node_id_mask();
            uint32_t node_id_bits = tt::tt_metal::MetalContext::instance().hal().get_noc_addr_node_id_bits();
            uint32_t my_x = node_id_regval & node_id_mask;
            uint32_t my_y = (node_id_regval >> node_id_bits) & node_id_mask;
            EXPECT_EQ(my_x, worker_core.x);
            EXPECT_EQ(my_y, worker_core.y);
        }
    }
    ASSERT_TRUE(tt::tt_metal::CloseDevice(device));
}
TEST(NOC, TensixVerifyNocIdentityTranslationTable) {
    auto arch = tt::get_arch_from_string(get_umd_arch_name());
    if (arch == tt::ARCH::BLACKHOLE) {
        GTEST_SKIP();
    }
    tt::tt_metal::IDevice* device;
    const unsigned int device_id = 0;
    device = tt::tt_metal::CreateDevice(device_id);
    // Ping all the registers for NOC
    auto logical_grid_size = device->logical_grid_size();
    for (size_t y = 0; y < logical_grid_size.y; y++) {
        for (size_t x = 0; x < logical_grid_size.x; x++) {
            std::vector<unsigned int> x_remap = {};
            std::vector<unsigned int> y_remap = {};
            unit_tests::basic::test_noc::read_translation_table(device, CoreCoord(x, y), x_remap, y_remap);
            bool core_has_translation_error = false;
            // bottom 16 values are not remapped --> identity
            for (int x = 0; x < 16; x++) {
                EXPECT_EQ(x, x_remap[x]);
                core_has_translation_error |= (x != x_remap[x]);
            }
            // bottom 16 values are not remapped --> identity
            for (int y = 0; y < 16; y++) {
                EXPECT_EQ(y, y_remap[y]);
                core_has_translation_error |= (y != y_remap[y]);
            }
            ASSERT_FALSE(core_has_translation_error);
        }
    }
    ASSERT_TRUE(tt::tt_metal::CloseDevice(device));
}

namespace tt::tt_metal {

// Tests that kernel can write to and read from a stream register address
// This is meant to exercise noc_inline_dw_write API
TEST_F(DeviceFixture, TensixDirectedStreamRegWriteRead) {
    CoreCoord start_core{0, 0};
    const uint32_t stream_id = 0;
    const uint32_t stream_reg = 4;

    for (tt_metal::IDevice* device : this->devices_) {
        tt_metal::Program program = tt_metal::CreateProgram();
        CoreCoord logical_grid_size = device->compute_with_storage_grid_size();
        CoreCoord end_core{logical_grid_size.x - 1, logical_grid_size.y - 1};
        CoreRange all_cores(start_core, end_core);
        tt_metal::KernelHandle kernel_id = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_reg_read_write.cpp",
            all_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_0});

        uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
        uint32_t value_to_write = 0x1234;
        for (uint32_t x = 0; x < logical_grid_size.x; x++) {
            for (uint32_t y = 0; y < logical_grid_size.y; y++) {
                CoreCoord logical_core(x, y);
                uint32_t logical_target_x = (x + 1) % logical_grid_size.x;
                uint32_t logical_target_y = (y + 1) % logical_grid_size.y;
                CoreCoord logical_target_core(logical_target_x, logical_target_y);
                CoreCoord worker_target_core = device->worker_core_from_logical_core(logical_target_core);

                tt_metal::SetRuntimeArgs(
                    program,
                    kernel_id,
                    logical_core,
                    {worker_target_core.x,
                     worker_target_core.y,
                     stream_id,
                     stream_reg,
                     value_to_write,
                     l1_unreserved_base});

                value_to_write++;
            }
        }

        tt_metal::detail::LaunchProgram(device, program);

        uint32_t expected_value_to_read = 0x1234;
        for (uint32_t x = 0; x < logical_grid_size.x; x++) {
            for (uint32_t y = 0; y < logical_grid_size.y; y++) {
                CoreCoord logical_core(x, y);
                std::vector<uint32_t> readback = {0xDEADBEEF};
                tt_metal::detail::ReadFromDeviceL1(
                    device, logical_core, l1_unreserved_base, sizeof(uint32_t), readback);
                EXPECT_EQ(readback[0], expected_value_to_read);
                expected_value_to_read++;
            }
        }
    }
}

// Test inline writes from many cores to an auto-incrementing register on one core.
TEST_F(DeviceFixture, TensixIncrementStreamRegWrite) {
    CoreCoord start_core{0, 0};
    const uint32_t stream_id = 1;

    for (tt_metal::IDevice* device : this->devices_) {
        tt_metal::Program program = tt_metal::CreateProgram();
        CoreCoord logical_grid_size = device->compute_with_storage_grid_size();
        CoreCoord end_core{logical_grid_size.x - 1, logical_grid_size.y - 1};
        CoreRange all_cores(start_core, end_core);
        tt_metal::KernelHandle kernel_id = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_increment_reg_write.cpp",
            all_cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_0});
        const uint32_t logical_target_x = 0;
        const uint32_t logical_target_y = 0;
        CoreCoord logical_target_core(logical_target_x, logical_target_y);
        CoreCoord worker_target_core = device->worker_core_from_logical_core(logical_target_core);

        uint32_t semaphore = tt_metal::CreateSemaphore(program, all_cores, 0);
        auto top_left = device->virtual_core_from_logical_core({0, 0}, CoreType::WORKER);
        auto bottom_right = device->virtual_core_from_logical_core(end_core, CoreType::WORKER);

        for (uint32_t x = 0; x < logical_grid_size.x; x++) {
            for (uint32_t y = 0; y < logical_grid_size.y; y++) {
                CoreCoord logical_core(x, y);

                tt_metal::SetRuntimeArgs(
                    program,
                    kernel_id,
                    logical_core,
                    {worker_target_core.x,
                     worker_target_core.y,
                     stream_id,
                     logical_core == logical_target_core ? logical_grid_size.x * logical_grid_size.y : 0,
                     semaphore,
                     top_left.x,
                     bottom_right.x,
                     top_left.y,
                     bottom_right.y,
                     all_cores.size() - 1});
            }
        }

        tt_metal::detail::LaunchProgram(device, program);
    }
}

TEST_F(DeviceFixture, TensixInlineWrite4BAlignment) {
    CoreCoord writer_core{0, 0};
    CoreCoord receiver_core(0, 1);
    uint32_t value_to_write = 39;
    for (tt_metal::IDevice* device : this->devices_) {
        uint32_t receiver_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1) + 4;
        EXPECT_EQ(receiver_addr % 4, 0)
            << "Expected dest address to be 4B aligned to test noc_inline_dw_write alignment rule";
        std::vector<uint32_t> readback(sizeof(uint32_t), 0);
        tt_metal::detail::WriteToDeviceL1(device, receiver_core, receiver_addr, readback);

        CoreCoord virtual_receiver_core = device->worker_core_from_logical_core(receiver_core);

        tt_metal::Program program = tt_metal::CreateProgram();
        tt_metal::KernelHandle kernel0 = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/inline_writer.cpp",
            writer_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_0});

        tt_metal::SetRuntimeArgs(
            program,
            kernel0,
            writer_core,
            {virtual_receiver_core.x, virtual_receiver_core.y, receiver_addr, value_to_write, 1, 0});

        tt_metal::detail::LaunchProgram(device, program);

        tt_metal::detail::ReadFromDeviceL1(device, receiver_core, receiver_addr, sizeof(uint32_t), readback);
        EXPECT_EQ(readback[0], value_to_write);
    }
}

// Both data movement riscs issue inline writes
TEST_F(DeviceFixture, TensixInlineWriteDedicatedNoc) {
    CoreCoord writer_core{0, 0};
    CoreCoord receiver_core(0, 1);
    uint32_t value_to_write = 39;

    for (tt_metal::IDevice* device : this->devices_) {
        uint32_t first_receiver_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
        uint32_t second_receiver_addr =
            first_receiver_addr + MetalContext::instance().hal().get_alignment(HalMemType::L1);
        std::vector<uint32_t> readback(32 / sizeof(uint32_t), 0);
        tt_metal::detail::WriteToDeviceL1(device, receiver_core, first_receiver_addr, readback);

        CoreCoord virtual_receiver_core = device->worker_core_from_logical_core(receiver_core);

        tt_metal::Program program = tt_metal::CreateProgram();
        tt_metal::KernelHandle kernel0 = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/inline_writer.cpp",
            writer_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_0});

        tt_metal::SetRuntimeArgs(
            program,
            kernel0,
            writer_core,
            {virtual_receiver_core.x, virtual_receiver_core.y, first_receiver_addr, value_to_write, 1, 0});

        tt_metal::KernelHandle kernel1 = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/inline_writer.cpp",
            writer_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::NOC_1});

        tt_metal::SetRuntimeArgs(
            program,
            kernel1,
            writer_core,
            {virtual_receiver_core.x, virtual_receiver_core.y, second_receiver_addr, value_to_write + 1, 1, 0});

        tt_metal::detail::LaunchProgram(device, program);

        tt_metal::detail::ReadFromDeviceL1(device, receiver_core, first_receiver_addr, 32, readback);
        EXPECT_EQ(readback[0], value_to_write);
        EXPECT_EQ(readback[4], value_to_write + 1);
    }
}

TEST_F(DeviceFixture, TensixInlineWriteDedicatedNocMisaligned) {
    CoreCoord writer_core{0, 0};
    CoreCoord receiver_core(0, 1);
    uint32_t value_to_write = 39;
    uint32_t num_writes = 8;

    for (tt_metal::IDevice* device : this->devices_) {
        uint32_t base_receiver_addr = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1) + 4;
        std::vector<uint32_t> readback(num_writes * sizeof(uint32_t), 0);
        tt_metal::detail::WriteToDeviceL1(device, receiver_core, base_receiver_addr, readback);

        CoreCoord virtual_receiver_core = device->worker_core_from_logical_core(receiver_core);

        tt_metal::Program program = tt_metal::CreateProgram();
        tt_metal::KernelHandle kernel0 = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/inline_writer.cpp",
            writer_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::NOC_0});

        tt_metal::SetRuntimeArgs(
            program,
            kernel0,
            writer_core,
            {virtual_receiver_core.x,
             virtual_receiver_core.y,
             base_receiver_addr,
             value_to_write,
             num_writes,
             sizeof(uint32_t)});

        tt_metal::detail::LaunchProgram(device, program);

        tt_metal::detail::ReadFromDeviceL1(
            device, receiver_core, base_receiver_addr, num_writes * sizeof(uint32_t), readback);
        uint32_t expected_value = value_to_write;
        for (int i = 0; i < num_writes; i++) {
            EXPECT_EQ(readback[i], expected_value);
            expected_value++;
        }
    }
}

// Both data movement riscs issue inline writes using the same noc
TEST_F(DeviceFixture, TensixInlineWriteDynamicNoc) {
    // #21082
    auto arch = tt::get_arch_from_string(get_umd_arch_name());
    if (arch == tt::ARCH::BLACKHOLE) {
        GTEST_SKIP();
    }
    CoreCoord writer_core{0, 0};
    CoreCoord receiver_core(0, 1);
    uint32_t value_to_write = 39;

    for (tt_metal::IDevice* device : this->devices_) {
        uint32_t receiver_addr0 = device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);
        uint32_t receiver_addr2 = receiver_addr0 + (2 * MetalContext::instance().hal().get_alignment(HalMemType::L1));
        std::vector<uint32_t> readback(80 / sizeof(uint32_t), 0);
        tt_metal::detail::WriteToDeviceL1(device, receiver_core, receiver_addr0, readback);

        CoreCoord virtual_receiver_core = device->worker_core_from_logical_core(receiver_core);

        tt_metal::Program program = tt_metal::CreateProgram();
        tt_metal::KernelHandle kernel0 = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/inline_writer.cpp",
            writer_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::NOC_0,
                .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC});

        tt_metal::SetRuntimeArgs(
            program,
            kernel0,
            writer_core,
            {virtual_receiver_core.x,
             virtual_receiver_core.y,
             receiver_addr0,
             value_to_write,
             2,
             MetalContext::instance().hal().get_alignment(HalMemType::L1)});

        tt_metal::KernelHandle kernel1 = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/inline_writer.cpp",
            writer_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::NOC_1,
                .noc_mode = tt_metal::NOC_MODE::DM_DYNAMIC_NOC});

        tt_metal::SetRuntimeArgs(
            program,
            kernel1,
            writer_core,
            {virtual_receiver_core.x,
             virtual_receiver_core.y,
             receiver_addr2,
             value_to_write + 2,
             2,
             MetalContext::instance().hal().get_alignment(HalMemType::L1)});

        tt_metal::detail::LaunchProgram(device, program);

        tt_metal::detail::ReadFromDeviceL1(device, receiver_core, receiver_addr0, 64, readback);
        uint32_t expected_value = value_to_write;
        for (int i = 0; i < 4; i++) {
            EXPECT_EQ(readback[i * 4], expected_value);
            expected_value++;
        }
    }
}

}  // namespace tt::tt_metal
