// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/allocator.hpp>
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
#include "mesh_dispatch_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/context/metal_context.hpp"

namespace tt::tt_metal {

TEST_F(MeshDispatchFixture, TensixProgramGlobalCircularBuffers) {
    CoreCoord sender_core = CoreCoord(0, 0);
    CoreRangeSet sender_cores = CoreRangeSet(CoreRange(sender_core));
    CoreRangeSet receiver_cores(CoreRange({1, 1}, {2, 2}));
    uint32_t cb_page_size = 32;
    tt::DataFormat tile_format = tt::DataFormat::Float16_b;
    auto all_cores = sender_cores.merge(receiver_cores);
    auto mesh_device = devices_[0];
    std::vector<std::pair<CoreCoord, CoreRangeSet>> sender_receiver_core_mapping = {{sender_core, receiver_cores}};
    auto global_cb = tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        mesh_device.get(), sender_receiver_core_mapping, 3200, tt::tt_metal::BufferType::L1);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    tt::tt_metal::Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    uint32_t remote_cb_index = 31;
    uint32_t local_cb_index = 0;
    tt::tt_metal::CircularBufferConfig global_cb_config = tt::tt_metal::CircularBufferConfig(cb_page_size);
    global_cb_config.remote_index(remote_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
    global_cb_config.index(local_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
    tt::tt_metal::experimental::CreateCircularBuffer(program_, all_cores, global_cb_config, global_cb);

    std::vector<uint32_t> compile_args = {remote_cb_index};
    tt::tt_metal::KernelHandle dm0_sender_kernel = tt::tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/global_circular_buffer/validate_sender_config.cpp",
        sender_cores,
        tt::tt_metal::ReaderDataMovementConfig(compile_args));
    tt::tt_metal::KernelHandle dm1_sender_kernel = tt::tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/global_circular_buffer/validate_sender_config.cpp",
        sender_cores,
        tt::tt_metal::WriterDataMovementConfig(compile_args));
    tt::tt_metal::KernelHandle compute_sender_kernel = tt::tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/global_circular_buffer/validate_sender_config.cpp",
        sender_cores,
        tt::tt_metal::ComputeConfig{.compile_args = compile_args});
    tt::tt_metal::KernelHandle dm0_receiver_kernel = tt::tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/global_circular_buffer/validate_receiver_config.cpp",
        receiver_cores,
        tt::tt_metal::ReaderDataMovementConfig(compile_args));
    tt::tt_metal::KernelHandle dm1_receiver_kernel = tt::tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/global_circular_buffer/validate_receiver_config.cpp",
        receiver_cores,
        tt::tt_metal::WriterDataMovementConfig(compile_args));
    tt::tt_metal::KernelHandle compute_receiver_kernel = tt::tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/global_circular_buffer/validate_receiver_config.cpp",
        receiver_cores,
        tt::tt_metal::ComputeConfig{.compile_args = compile_args});

    for (const auto& [sender_core, receiver_cores] : sender_receiver_core_mapping) {
        auto sender_noc_coords = mesh_device->worker_core_from_logical_core(sender_core);
        std::vector<CoreCoord> receiver_noc_coords;
        for (const auto& receiver_core_range : receiver_cores.ranges()) {
            const auto& receiver_cores_vec = corerange_to_cores(receiver_core_range);
            for (const auto& receiver_core : receiver_cores_vec) {
                receiver_noc_coords.push_back(mesh_device->worker_core_from_logical_core(receiver_core));
            }
        }
        std::vector<uint32_t> sender_runtime_args(11 + (receiver_noc_coords.size() * 2));
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
        tt::tt_metal::SetRuntimeArgs(program_, dm0_sender_kernel, sender_cores, sender_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program_, dm1_sender_kernel, sender_cores, sender_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program_, compute_sender_kernel, sender_cores, sender_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program_, dm0_receiver_kernel, receiver_cores, receiver_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program_, dm1_receiver_kernel, receiver_cores, receiver_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program_, compute_receiver_kernel, receiver_cores, receiver_runtime_args);
    }
    this->RunProgram(mesh_device, workload);
}

TEST_F(MeshDispatchFixture, TensixProgramClearsStaleRemoteCircularBufferConfig) {
    const CoreCoord sender_core(0, 0);
    const CoreCoord receiver_core(1, 0);
    const CoreCoord idle_core(1, 1);
    const CoreRangeSet receiver_cores{CoreRange(receiver_core)};
    const CoreRangeSet all_cores(CoreRange({0, 0}, {1, 1}));
    constexpr uint32_t cb_page_size = 32;
    constexpr uint32_t remote_cb_index = 31;
    constexpr uint32_t local_cb_index = 0;
    constexpr tt::DataFormat tile_format = tt::DataFormat::Float16_b;

    auto mesh_device = devices_[0];
    auto* device = mesh_device->get_devices()[0];
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    auto make_cb_config = [&]() {
        CircularBufferConfig config(cb_page_size);
        config.remote_index(remote_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
        config.index(local_cb_index).set_page_size(cb_page_size).set_data_format(tile_format);
        return config;
    };
    // Run a rectangular kernel grid with a remote CB on only one receiver. Dispatch must overwrite stale config on
    // idle_core with the zero sentinel that setup_remote_cb_interfaces() skips.
    std::vector<std::pair<CoreCoord, CoreRangeSet>> sparse_mapping = {{sender_core, receiver_cores}};
    auto sparse_global_cb =
        experimental::CreateGlobalCircularBuffer(mesh_device.get(), sparse_mapping, 3200, BufferType::L1);
    distributed::MeshWorkload sparse_workload;
    Program sparse_program = CreateProgram();
    experimental::CreateCircularBuffer(sparse_program, receiver_cores, make_cb_config(), sparse_global_cb);
    CreateKernel(
        sparse_program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        all_cores,
        ReaderDataMovementConfig{});
    sparse_workload.add_program(device_range, std::move(sparse_program));
    auto& sparse_program_in_workload = sparse_workload.get_programs().at(device_range);

    const auto& hal = MetalContext::instance().hal();
    const uint32_t programmable_core_index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    const uint32_t poison_base = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::KERNEL_CONFIG);
    const uint32_t poison_size = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1) - poison_base;
    ASSERT_GT(poison_size, 0);
    ASSERT_EQ(poison_size % sizeof(uint32_t), 0);
    std::vector<uint32_t> poison(poison_size / sizeof(uint32_t), 0xffffffff);
    detail::WriteToDeviceL1(device, idle_core, poison_base, poison);

    this->RunProgram(mesh_device, sparse_workload);

    const uint32_t remote_config_address =
        sparse_workload.get_cb_base_addr(mesh_device, idle_core, CoreType::WORKER) +
        sparse_program_in_workload.impl().get_program_config(programmable_core_index).local_cb_size;
    std::vector<uint32_t> remote_config;
    detail::ReadFromDeviceL1(
        device,
        idle_core,
        remote_config_address,
        UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t),
        remote_config);
    EXPECT_EQ(remote_config, std::vector<uint32_t>(UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG, 0));
}

}  // namespace tt::tt_metal
