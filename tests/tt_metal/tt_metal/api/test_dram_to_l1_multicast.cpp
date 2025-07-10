// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <fmt/base.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "dispatch_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "umd/device/types/arch.h"

using namespace tt;

namespace unit_tests_common::dram::test_dram_to_l1_multicast {

struct DRAMtoL1MulticastConfig {
    std::uint32_t dest_buffer_addr;
    std::uint32_t target_grid_offset;
    std::string kernel_file;
    CoreCoord exclude_start;
    CoreCoord exclude_direction;
};

bool dram_to_l1_multicast(
    tt::tt_metal::DispatchFixture* fixture, tt_metal::IDevice* device, const DRAMtoL1MulticastConfig& cfg) {
    bool pass = true;
    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size =
        single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    uint32_t local_buffer_addr = 200 * 1024;

    // same address as local_buffer
    // Note: src will NOT write into its dst buffer address
    // since we are not setting NOC_CMD_BRCST_SRC_INCLUDE
    uint32_t dest_buffer_addr = 200 * 1024;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};
    auto dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_addr = dram_buffer->address();


    CoreCoord core_start = {0, 0};
    CoreCoord grid_size = device->logical_grid_size();
    CoreCoord core_end = {core_start.x + (grid_size.x - 1), core_start.y + (grid_size.y - 1)};
    auto core_start_physical = device->worker_core_from_logical_core(core_start);
    auto core_end_physical = device->worker_core_from_logical_core(core_end);
    auto core_exclude_physical = device->worker_core_from_logical_core(cfg.exclude_start);
    auto num_dests = (grid_size.x * grid_size.y) - cfg.target_grid_offset;
    // calculate number of destination cores, taking exluded ones into account
    if (cfg.exclude_start.x != 0 || cfg.exclude_start.y != 0) {
        auto num_x = cfg.exclude_direction.x == 1 ? grid_size.x - cfg.exclude_start.x : cfg.exclude_start.x + 1;
        auto num_y = cfg.exclude_direction.y == 1 ? grid_size.y - cfg.exclude_start.y : cfg.exclude_start.y + 1;
        num_dests = (grid_size.x * grid_size.y) - num_x * num_y - cfg.target_grid_offset;
    }
    std::vector<uint32_t> mcast_reader_args = {
        (std::uint32_t)dram_buffer_addr,
        0,
        (std::uint32_t)dram_buffer_size,
        (std::uint32_t)local_buffer_addr,
        (std::uint32_t)dest_buffer_addr,
        (std::uint32_t)core_end_physical.x,
        (std::uint32_t)core_end_physical.y,
        (std::uint32_t)core_start_physical.x,
        (std::uint32_t)core_start_physical.y,
        (std::uint32_t)num_dests,
        (std::uint32_t)core_exclude_physical.x,
        (std::uint32_t)core_exclude_physical.y,
        (std::uint32_t)cfg.exclude_direction.x,
        (std::uint32_t)cfg.exclude_direction.y,
    };  // Note: exclude src from acks, since we are not setting NOC_CMD_BRCST_SRC_INCLUDE

    log_debug(LogTest, "Start = {}, {}", core_start_physical.x, core_start_physical.y);
    log_debug(LogTest, "End = {}, {}", core_end_physical.x, core_end_physical.y);
    log_debug(LogTest, "Exclude = {}, {}", core_exclude_physical.x, core_exclude_physical.y);
    auto mcast_reader_kernel = tt_metal::CreateKernel(
        program,
        cfg.kernel_file,
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    SHAPE shape = {1, 1, 32, 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(
        shape, tt::deprecated::Initialize::RANDOM, 0, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto activations = pack_bfloat16_vec_into_uint32_vec(tensor.get_values());
    fixture->WriteBuffer(device, dram_buffer, activations);

    tt_metal::SetRuntimeArgs(program, mcast_reader_kernel, core, mcast_reader_args);

    log_debug(LogTest, "Launching kernels");
    fixture->RunProgram(device, program);
    log_debug(LogTest, "Kernels done");

    for (int i = 0; i < grid_size.y; i++) {
        for (int j = 0; j < grid_size.x; j++) {
            // don't compare on skipped cores
            if (((cfg.exclude_direction.x == 0 && j <= cfg.exclude_start.x) ||
                 (cfg.exclude_direction.x == 1 && j >= cfg.exclude_start.x)) &&
                ((cfg.exclude_direction.y == 0 && i <= cfg.exclude_start.y) ||
                 (cfg.exclude_direction.y == 1 && i >= cfg.exclude_start.y))) {
                log_debug(
                    tt::LogTest, "Skipping core {},{}", j, i);  // debug print to verify we don't skip unnecessary cores
                continue;
            }
            CoreCoord dest_core = {(std::size_t)core_start.x + j, (std::size_t)core_start.y + i};
            std::vector<uint32_t> dest_core_data;
            tt_metal::detail::ReadFromDeviceL1(device, dest_core, dest_buffer_addr, dram_buffer_size, dest_core_data);
            auto dest_core_data_unpacked = unpack_uint32_vec_into_bfloat16_vec(dest_core_data);
            pass &= (dest_core_data_unpacked == tensor.get_values());
            if (not(dest_core_data_unpacked == tensor.get_values())) {
                log_info(LogTest, "Mismatch on core {}, {}", dest_core.x, dest_core.y);
                print_vec_of_bfloat16(dest_core_data_unpacked, 1, "Result");
            }
        }
    }
    return pass;
}
}  // namespace unit_tests_common::dram::test_dram_to_l1_multicast

namespace tt::tt_metal {

TEST_F(DispatchFixture, TensixDRAMtoL1Multicast) {
    unit_tests_common::dram::test_dram_to_l1_multicast::DRAMtoL1MulticastConfig test_config = {
        .dest_buffer_addr = 200 * 1024,
        .target_grid_offset = 1,
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_multicast.cpp",
    };
    for (unsigned int id = 0; id < devices_.size(); id++) {
        ASSERT_TRUE(unit_tests_common::dram::test_dram_to_l1_multicast::dram_to_l1_multicast(
            this, devices_.at(id), test_config));
    }
}
TEST_F(DispatchFixture, TensixDRAMtoL1MulticastLoopbackSrc) {
    unit_tests_common::dram::test_dram_to_l1_multicast::DRAMtoL1MulticastConfig test_config = {
        .dest_buffer_addr = 500 * 1024,
        .target_grid_offset = 0,
        .kernel_file = "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_multicast_include_src.cpp",
    };
    for (unsigned int id = 0; id < devices_.size(); id++) {
        ASSERT_TRUE(unit_tests_common::dram::test_dram_to_l1_multicast::dram_to_l1_multicast(
            this, devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
