// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdint.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/device.hpp>
#include "dispatch_fixture.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "umd/device/types/xy_pair.h"

namespace tt::tt_metal {

TEST_F(DispatchFixture, TensixProgramGlobalCircularBuffers) {
    CoreCoord sender_core = CoreCoord(0, 0);
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    CoreRangeSet receiver_cores(CoreRange({1, 1}, {2, 2}));
    uint32_t global_cb_size = 3200;
    uint32_t cb_page_size = 32;
    tt::DataFormat tile_format = tt::DataFormat::Float16_b;
    auto all_cores = sender_cores.merge(receiver_cores);
    auto device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {{sender_core, receiver_cores}};
    auto global_cb = tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        device, sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1);

    tt::tt_metal::Program program = CreateProgram();
    uint32_t remote_cb_index = 31;
    uint32_t local_cb_index = 0;
    tt::tt_metal::CircularBufferConfig global_cb_config = tt::tt_metal::CircularBufferConfig(cb_page_size);
    global_cb_config.remote_index(remote_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
    global_cb_config.index(local_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
    auto remote_cb = tt::tt_metal::experimental::CreateCircularBuffer(program, all_cores, global_cb_config, global_cb);

    std::vector<uint32_t> compile_args = {remote_cb_index};
    tt::tt_metal::KernelHandle dm0_sender_kernel = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/global_circular_buffer/validate_sender_config.cpp",
        sender_cores,
        tt::tt_metal::ReaderDataMovementConfig(compile_args));
    tt::tt_metal::KernelHandle dm1_sender_kernel = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/global_circular_buffer/validate_sender_config.cpp",
        sender_cores,
        tt::tt_metal::WriterDataMovementConfig(compile_args));
    tt::tt_metal::KernelHandle compute_sender_kernel = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/global_circular_buffer/validate_sender_config.cpp",
        sender_cores,
        tt::tt_metal::ComputeConfig{.compile_args = compile_args});
    tt::tt_metal::KernelHandle dm0_receiver_kernel = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/global_circular_buffer/validate_receiver_config.cpp",
        receiver_cores,
        tt::tt_metal::ReaderDataMovementConfig(compile_args));
    tt::tt_metal::KernelHandle dm1_receiver_kernel = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/global_circular_buffer/validate_receiver_config.cpp",
        receiver_cores,
        tt::tt_metal::WriterDataMovementConfig(compile_args));
    tt::tt_metal::KernelHandle compute_receiver_kernel = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/global_circular_buffer/validate_receiver_config.cpp",
        receiver_cores,
        tt::tt_metal::ComputeConfig{.compile_args = compile_args});

    for (const auto& [sender_core, receiver_cores] : sender_receiver_core_mapping) {
        auto sender_noc_coords = device->worker_core_from_logical_core(sender_core);
        std::vector<CoreCoord> receiver_noc_coords;
        for (const auto& receiver_core_range : receiver_cores.ranges()) {
            const auto& receiver_cores_vec = corerange_to_cores(receiver_core_range);
            for (const auto& receiver_core : receiver_cores_vec) {
                receiver_noc_coords.push_back(device->worker_core_from_logical_core(receiver_core));
            }
        }
        std::vector<uint32_t> sender_runtime_args(11 + receiver_noc_coords.size() * 2);
        uint32_t sender_args_idx = 0;
        sender_runtime_args[sender_args_idx++] = global_cb.config_address();  // config_addr
        sender_runtime_args[sender_args_idx++] = 1;                           // is_sender
        sender_runtime_args[sender_args_idx++] = receiver_noc_coords.size();  // num_receivers
        sender_runtime_args[sender_args_idx++] = global_cb.buffer_address();  // fifo_start_addr
        sender_runtime_args[sender_args_idx++] = global_cb.size();            // fifo_size
        sender_runtime_args[sender_args_idx++] = global_cb.buffer_address();  // fifo_ptr
        for (const auto& receiver_noc_coord : receiver_noc_coords) {
            sender_runtime_args[sender_args_idx++] = receiver_noc_coord.x;  // remote_noc_x
            sender_runtime_args[sender_args_idx++] = receiver_noc_coord.y;  // remote_noc_y
        }
        sender_runtime_args[sender_args_idx++] = 0;                           // aligned_pages_sent
        sender_runtime_args[sender_args_idx++] = 0;                           // aligned_pages_acked
        sender_runtime_args[sender_args_idx++] = global_cb.buffer_address();  // fifo_wr_ptr
        sender_runtime_args[sender_args_idx++] =
            global_cb.buffer_address() + global_cb.size();      // fifo_limit_page_aligned
        sender_runtime_args[sender_args_idx++] = cb_page_size;  // fifo_page_size

        std::vector<uint32_t> receiver_runtime_args = {
            global_cb.config_address(),                     // config_addr
            0,                                              // is_sender
            global_cb.buffer_address(),                     // fifo_start_addr
            global_cb.size(),                               // fifo_size
            global_cb.buffer_address(),                     // fifo_ptr
            sender_noc_coords.x,                            // sender_noc_x
            sender_noc_coords.y,                            // sender_noc_y
            0,                                              // aligned_pages_sent
            0,                                              // aligned_pages_acked
            global_cb.buffer_address(),                     // fifo_rd_ptr
            global_cb.buffer_address() + global_cb.size(),  // fifo_limit_page_aligned
            cb_page_size,                                   // fifo_page_size
        };
        tt::tt_metal::SetRuntimeArgs(program, dm0_sender_kernel, sender_cores, sender_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, dm1_sender_kernel, sender_cores, sender_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_sender_kernel, sender_cores, sender_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, dm0_receiver_kernel, receiver_cores, receiver_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, dm1_receiver_kernel, receiver_cores, receiver_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_receiver_kernel, receiver_cores, receiver_runtime_args);
    }
    this->RunProgram(device, program);
}

}  // namespace tt::tt_metal
