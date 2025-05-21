// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <exception>
#include <map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/data_types.hpp>
#include "dispatch_fixture.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal {

TEST_F(DispatchFixture, TensixCreateGlobalCircularBuffers) {
    CoreRangeSet cores(CoreRange({1, 1}, {1, 1}));
    CoreRangeSet cores2(CoreRange({1, 1}, {2, 2}));
    CoreRangeSet cores3(CoreRange({3, 3}, {3, 3}));

    auto device = devices_[0];
    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {{CoreCoord(0, 0), cores}};
        auto global_cb = tt::tt_metal::experimental::CreateGlobalCircularBuffer(
            device, sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1);
        auto buffer_address = global_cb.buffer_address();
        auto config_address = global_cb.config_address();
    }
    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {
            {CoreCoord(0, 0), cores}, {CoreCoord(1, 1), cores3}};
        // sender receiver cores overlap
        EXPECT_THROW(
            tt::tt_metal::experimental::CreateGlobalCircularBuffer(
                device, sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1),
            std::exception);
    }
    {
        std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {
            {CoreCoord(0, 0), cores}, {CoreCoord(0, 1), cores2}};
        // receiver cores overlap
        EXPECT_THROW(
            tt::tt_metal::experimental::CreateGlobalCircularBuffer(
                device, sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1),
            std::exception);
    }
}

TEST_F(DispatchFixture, TensixProgramGlobalCircularBuffers) {
    CoreCoord sender_core = CoreCoord(0, 0);
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    CoreRangeSet receiver_cores(CoreRange({1, 1}, {2, 2}));
    CoreRangeSet dummy_receiver_cores(CoreRange({3, 3}, {3, 3}));
    uint32_t global_cb_size = 3200;
    uint32_t cb_page_size = 32;
    tt::DataFormat tile_format = tt::DataFormat::Float16_b;
    auto all_cores = sender_cores.merge(receiver_cores).merge(dummy_receiver_cores);
    auto device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {{sender_core, receiver_cores}};
    auto global_cb = tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1);
    std::vector<std::pair<CoreCoord, CoreRangeSet>> dummy_sender_receiver_core_mapping = {
        {CoreCoord(0, 0), dummy_receiver_cores}};
    auto dummy_global_cb = tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        device, dummy_sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1);
    {
        tt::tt_metal::Program program = CreateProgram();
        tt::tt_metal::KernelHandle blank_kernel = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
        uint32_t remote_cb_index = 31;
        uint32_t local_cb_index = 0;
        tt::tt_metal::CircularBufferConfig global_cb_config = tt::tt_metal::CircularBufferConfig(cb_page_size);
        global_cb_config.remote_index(remote_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
        global_cb_config.index(local_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
        EXPECT_THROW(global_cb_config.remote_index(2), std::exception);
        EXPECT_THROW(
            tt::tt_metal::experimental::CreateCircularBuffer(
                program, CoreRangeSet(CoreRange({3, 3})), global_cb_config, global_cb),
            std::exception);
        auto remote_cb =
            tt::tt_metal::experimental::CreateCircularBuffer(program, receiver_cores, global_cb_config, global_cb);
        tt::tt_metal::detail::CompileProgram(device, program);
        program.finalize_offsets(device);
        tt::tt_metal::experimental::UpdateDynamicCircularBufferAddress(program, remote_cb, global_cb);
        EXPECT_THROW(UpdateDynamicCircularBufferAddress(program, remote_cb, dummy_global_cb), std::exception);
    }
    {
        tt::tt_metal::Program program = CreateProgram();
        tt::tt_metal::KernelHandle blank_kernel = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/blank.cpp",
            all_cores,
            tt::tt_metal::DataMovementConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});
        uint32_t remote_cb_index = 16;
        uint32_t local_cb_index = 17;
        tt::tt_metal::CircularBufferConfig global_cb_config = tt::tt_metal::CircularBufferConfig(cb_page_size);
        global_cb_config.remote_index(remote_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
        global_cb_config.index(local_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
        auto remote_cb =
            tt::tt_metal::experimental::CreateCircularBuffer(program, receiver_cores, global_cb_config, global_cb);
        tt::tt_metal::detail::CompileProgram(device, program);
        EXPECT_THROW(program.finalize_offsets(device), std::exception);
    }
}

}  // namespace tt::tt_metal
