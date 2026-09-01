// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "impl/program/program_impl.hpp"
#include <umd/device/types/core_coordinates.hpp>

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

TEST_F(UnitMeshFixture, DatacopyOutputInL1) {
    Program program = CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 32;
    uint32_t buffer_size = single_tile_size * num_tiles;

    distributed::ReplicatedBufferConfig buffer_config{.size = buffer_size};

    auto src_dram_buffer = distributed::MeshBuffer::create(
        buffer_config, {.page_size = buffer_size, .buffer_type = BufferType::DRAM}, &this->device());
    auto dst_l1_buffer = distributed::MeshBuffer::create(
        buffer_config, {.page_size = buffer_size, .buffer_type = BufferType::L1}, &this->device());

    auto l1_dst_noc_xy = this->device().virtual_core_from_logical_core(
        this->device().allocator()->get_logical_core_from_bank_id(0), CoreType::WORKER);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 8;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    auto unary_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_1.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {uint(num_tiles)};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});

    // Execute
    std::vector<uint32_t> src_vec =
        create_random_vector_of_bfloat16(buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    slow_dispatch::WriteToBuffer(*src_dram_buffer, src_vec);

    SetRuntimeArgs(program, unary_reader_kernel, core, {(std::uint32_t)src_dram_buffer->address(), 0, num_tiles});
    SetRuntimeArgs(
        program,
        unary_writer_kernel,
        core,
        {(std::uint32_t)dst_l1_buffer->address(),
         (std::uint32_t)l1_dst_noc_xy.x,
         (std::uint32_t)l1_dst_noc_xy.y,
         num_tiles});

    LaunchProgram(this->device(), std::move(program), /*wait_until_cores_done=*/true);

    std::vector<uint32_t> result_vec;
    slow_dispatch::ReadFromBuffer(*dst_l1_buffer, result_vec);

    // Validation
    EXPECT_EQ(src_vec, result_vec);
}
