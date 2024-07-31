// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "common/test_tiles.hpp"  // FIXME: Remove dependency on this or move to test_utils like tilize/untilize
#include "device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/tilization.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::matmul {

void create_CBs_for_fused_matmul(
    tt_metal::Program& program,
    tt_metal::Device* device,
    CoreCoord core,
    bool activations_rm,
    bool output_rm,
    uint32_t M,
    uint32_t N,
    uint32_t in0_block_w,
    uint32_t out_subblock_h) {
    uint32_t num_bytes_for_df = 2;
    uint32_t in0_cb = 0;
    uint32_t in1_cb = 1;
    uint32_t tilize_mode_tilized_in0_cb = 24;
    uint32_t matmul_partials_cb = 25;
    uint32_t untilize_mode_final_matmul_partials_cb = 26;
    uint32_t untilize_mode_reblock_cb = 27;
    uint32_t out0_cb = 16;

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = M * N;

    // Invariants
    uint32_t cb0_tiles = M * in0_block_w * 2;
    tt_metal::CircularBufferConfig l1_input0_cb_config = tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{in0_cb, tt::DataFormat::Float16_b}})
        .set_page_size(in0_cb, single_tile_size);
    auto l1_input0_cb = tt_metal::CreateCircularBuffer(program, core, l1_input0_cb_config);

    uint32_t cb1_tiles = N * in0_block_w * 2;
    tt_metal::CircularBufferConfig cb_in1_config = tt_metal::CircularBufferConfig(cb1_tiles * single_tile_size, {{in1_cb, tt::DataFormat::Float16_b}})
        .set_page_size(in1_cb, single_tile_size);
    auto cb_in1 = tt_metal::CreateCircularBuffer(program, core, cb_in1_config);

    if (not activations_rm and not output_rm) {  // no tilize, no untilize
        tt_metal::CircularBufferConfig cb_matmul_partials_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
            .set_page_size(matmul_partials_cb, single_tile_size);
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Partials share same L1 address space as output
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(out0_cb, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else if (not activations_rm and output_rm) {  // no tilize, just untilize

        tt_metal::CircularBufferConfig cb_matmul_partials_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
            .set_page_size(matmul_partials_cb, single_tile_size);
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Need a new CB to push output block to since other
        // intermediate read pointer changes in enable reload
        // block
        tt_metal::CircularBufferConfig cb_final_matmul_partials_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
            .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = N;  // Only space for one row
        tt_metal::CircularBufferConfig cb_reblock_config = tt_metal::CircularBufferConfig(reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
            .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        auto cb_reblock = tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(out0_cb, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else if (activations_rm and not output_rm) {  // just tilize, no untilize

        tt_metal::CircularBufferConfig cb_src0_tilized_config = tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{tilize_mode_tilized_in0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(tilize_mode_tilized_in0_cb, single_tile_size);
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

        tt_metal::CircularBufferConfig cb_matmul_partials_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
            .set_page_size(matmul_partials_cb, single_tile_size);
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(out0_cb, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else {  // tilize activations and untilize output

        // Used for placing tilized activations
        tt_metal::CircularBufferConfig cb_src0_tilized_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{tilize_mode_tilized_in0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(tilize_mode_tilized_in0_cb, single_tile_size);
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

        tt_metal::CircularBufferConfig cb_matmul_partials_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
            .set_page_size(matmul_partials_cb, single_tile_size);
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Shares same address space as matmul partials
        tt_metal::CircularBufferConfig cb_final_matmul_partials_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{untilize_mode_final_matmul_partials_cb, tt::DataFormat::Float16_b}})
            .set_page_size(untilize_mode_final_matmul_partials_cb, single_tile_size);
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = N;  // Only space for one row
        tt_metal::CircularBufferConfig cb_reblock_config = tt_metal::CircularBufferConfig(reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
            .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        auto cb_reblock = tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(out0_cb, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
    }
}

bool single_tile_matmul(tt_metal::Device* device) {

    bool pass = true;
    // FIXME: Convert to config
    CoreCoord core(0, 0);
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const size_t byte_size = 1 * 2 * 32 * 32;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = byte_size,
                    .page_size = byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

    tt_metal::Program program = tt_metal::CreateProgram();
    auto input0_dram_buffer = CreateBuffer(dram_config);
    const uint32_t in0_dram_addr = input0_dram_buffer->address();
    auto input0_dram_noc_xy = input0_dram_buffer->noc_coordinates();
    auto input1_dram_buffer = CreateBuffer(dram_config);
    const uint32_t in1_dram_addr = input1_dram_buffer->address();
    auto input1_dram_noc_xy = input1_dram_buffer->noc_coordinates();
    auto output_dram_buffer = CreateBuffer(dram_config);
    const uint32_t out_dram_addr = output_dram_buffer->address();
    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

    tt_metal::CircularBufferConfig l1_input0_cb_config = tt_metal::CircularBufferConfig(byte_size, {{in0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(in0_cb_index, byte_size);
    auto l1_input0_cb = tt_metal::CreateCircularBuffer(program, core, l1_input0_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config = tt_metal::CircularBufferConfig(byte_size, {{in1_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(in1_cb_index, byte_size);
    auto l1_input1_cb = tt_metal::CreateCircularBuffer(program, core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config = tt_metal::CircularBufferConfig(byte_size, {{out_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(out_cb_index, byte_size);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(program, core, l1_output_cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    auto simple_matmul_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/single_tile_compute.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = {in0_cb_index, in1_cb_index, out_cb_index}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        1.0f, 1.0f, byte_size / tt::test_utils::df::bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        1.0f / 32.0f,
        1.0f / 32.0f,
        byte_size / tt::test_utils::df::bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());
    // Setup the weights such that final result is the original input.

    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto packed_golden = packed_input0;

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);


    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)input0_dram_noc_xy.x,
            (uint32_t)input0_dram_noc_xy.y,
            (uint32_t)in1_dram_addr,
            (uint32_t)input1_dram_noc_xy.x,
            (uint32_t)input1_dram_noc_xy.y,
            (uint32_t)1,  // num_tiles
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)1,
        });


    tt_metal::detail::LaunchProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
        dest_buffer_data, packed_golden, [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) { return is_close(a, b, 0.015f); });
    return pass;
}
// blocked matmul has blocking, but still fits within dst, so no spill/reloads or intermediates
bool single_block_matmul(tt_metal::Device* device, uint32_t M, uint32_t K, uint32_t N) {

    bool pass = true;
    // FIXME: Convert to config
    CoreCoord core(0, 0);
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const size_t cb_page_size = 2 * 32 * 32;
    const size_t in0_byte_size = M * K * cb_page_size;
    const size_t in1_byte_size = K * N * cb_page_size;
    const size_t out_byte_size = M * N * cb_page_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    tt::tt_metal::InterleavedBufferConfig dram_config_0{
                    .device=device,
                    .size = in0_byte_size,
                    .page_size = in0_byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

    tt::tt_metal::InterleavedBufferConfig dram_config_1{
                    .device=device,
                    .size = in1_byte_size,
                    .page_size = in1_byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

    tt::tt_metal::InterleavedBufferConfig dram_config_out{
                    .device=device,
                    .size = out_byte_size,
                    .page_size = out_byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };


    tt_metal::Program program = tt_metal::CreateProgram();
    auto input0_dram_buffer = CreateBuffer(dram_config_0);
    const uint32_t in0_dram_addr = input0_dram_buffer->address();
    auto input0_dram_noc_xy = input0_dram_buffer->noc_coordinates();
    auto input1_dram_buffer = CreateBuffer(dram_config_1);
    const uint32_t in1_dram_addr = input1_dram_buffer->address();
    auto input1_dram_noc_xy = input1_dram_buffer->noc_coordinates();
    auto output_dram_buffer = CreateBuffer(dram_config_out);
    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();
    const uint32_t out_dram_addr = output_dram_buffer->address();

    tt_metal::CircularBufferConfig l1_input0_cb_config = tt_metal::CircularBufferConfig(in0_byte_size, {{in0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(in0_cb_index, cb_page_size);
    auto l1_input0_cb = tt_metal::CreateCircularBuffer(program, core, l1_input0_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config = tt_metal::CircularBufferConfig(in1_byte_size, {{in1_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(in1_cb_index, cb_page_size);
    auto l1_input1_cb = tt_metal::CreateCircularBuffer(program, core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config = tt_metal::CircularBufferConfig(out_byte_size, {{out_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(out_cb_index, cb_page_size);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(program, core, l1_output_cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary_blocked.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    auto simple_matmul_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_tile_compute.cpp",
        core,
        tt_metal::ComputeConfig{
            .compile_args = {in0_cb_index, in1_cb_index, out_cb_index, M * K, K * N, M * N, M, N, K}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        1.0f, 1.0f, in0_byte_size / tt::test_utils::df::bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        0.03125f,
        0.03125f,
        in1_byte_size / tt::test_utils::df::bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto packed_golden = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        1.0f * K,
        1.0f * K,
        (out_byte_size) / tt::test_utils::df::bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);


    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)input0_dram_noc_xy.x,
            (uint32_t)input0_dram_noc_xy.y,
            (uint32_t)in1_dram_addr,
            (uint32_t)input1_dram_noc_xy.x,
            (uint32_t)input1_dram_noc_xy.y,
            (uint32_t)1,              // num_blocks
            (uint32_t)M * K,          // in0_block_tile_cnt
            (uint32_t)K * N,          // in1_block_tile_cnt
            (uint32_t)in0_byte_size,  // in0_block_size_bytes
            (uint32_t)in1_byte_size,  // in1_block_size_bytes
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)M * N,
        });

    tt_metal::detail::LaunchProgram(device, program);
    sleep(1);
    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    int failed_index;
    pass &= is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
        dest_buffer_data,
        packed_golden,
        [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) { return is_close(a, b, 0.015f); },
        &failed_index);
    if (not pass) {
        log_info("Failed Index={}", failed_index);
        print_vector_fixed_numel_per_row(unpack_vector<tt::test_utils::df::bfloat16, uint32_t>(dest_buffer_data), 32);
    }
    return pass;
}
// blocked matmul has blocking on output, spill/reloads using intermediate
bool blocked_matmul(tt_metal::Device* device, uint32_t M, uint32_t K, uint32_t N) {

    bool pass = true;
    // FIXME: Convert to config
    CoreCoord core(0, 0);
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const uint32_t partials_cb_index = 24;
    const size_t cb_page_size = 2 * 32 * 32;
    const size_t in0_byte_size = M * K * cb_page_size;
    const size_t in1_byte_size = K * N * cb_page_size;
    const size_t out_byte_size = M * N * cb_page_size;
    const size_t num_blocks = 1;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    tt::tt_metal::InterleavedBufferConfig dram_config_0{
                    .device=device,
                    .size = in0_byte_size,
                    .page_size = in0_byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

    tt::tt_metal::InterleavedBufferConfig dram_config_1{
                    .device=device,
                    .size = in1_byte_size,
                    .page_size = in1_byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

    tt::tt_metal::InterleavedBufferConfig dram_config_out{
                    .device=device,
                    .size = out_byte_size,
                    .page_size = out_byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

    tt_metal::Program program = tt_metal::CreateProgram();
    auto input0_dram_buffer = CreateBuffer(dram_config_0);
    const uint32_t in0_dram_addr = input0_dram_buffer->address();
    auto input0_dram_noc_xy = input0_dram_buffer->noc_coordinates();
    auto input1_dram_buffer = CreateBuffer(dram_config_1);
    const uint32_t in1_dram_addr = input1_dram_buffer->address();
    auto input1_dram_noc_xy = input1_dram_buffer->noc_coordinates();
    auto output_dram_buffer = CreateBuffer(dram_config_out);
    const uint32_t out_dram_addr = output_dram_buffer->address();

    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

    tt_metal::CircularBufferConfig l1_input0_cb_config = tt_metal::CircularBufferConfig(in0_byte_size, {{in0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(in0_cb_index, cb_page_size);
    auto l1_input0_cb = tt_metal::CreateCircularBuffer(program, core, l1_input0_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config = tt_metal::CircularBufferConfig(in1_byte_size, {{in1_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(in1_cb_index, cb_page_size);
    auto l1_input1_cb = tt_metal::CreateCircularBuffer(program, core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config = tt_metal::CircularBufferConfig(out_byte_size, {{out_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(out_cb_index, cb_page_size);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(program, core, l1_output_cb_config);

    tt_metal::CircularBufferConfig l1_partials_cb_config = tt_metal::CircularBufferConfig(out_byte_size, {{partials_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(partials_cb_index, cb_page_size);
    auto l1_partials_cb = tt_metal::CreateCircularBuffer(program, core, l1_partials_cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary_blocked.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    auto simple_matmul_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_block_compute.cpp",
        core,
        tt_metal::ComputeConfig{
            .compile_args = {
                in0_cb_index,
                in1_cb_index,
                out_cb_index,
                partials_cb_index,
                M * K,
                K * N,
                M * N,
                M,
                N,
                K,
                num_blocks}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        1.0f, 1.0f, in0_byte_size / tt::test_utils::df::bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        0.03125f,
        0.03125f,
        in1_byte_size / tt::test_utils::df::bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto packed_golden = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(
        1.0f * K,
        1.0f * K,
        (out_byte_size) / tt::test_utils::df::bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);


    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)input0_dram_noc_xy.x,
            (uint32_t)input0_dram_noc_xy.y,
            (uint32_t)in1_dram_addr,
            (uint32_t)input1_dram_noc_xy.x,
            (uint32_t)input1_dram_noc_xy.y,
            (uint32_t)1,              // num_blocks
            (uint32_t)M * K,          // in0_block_tile_cnt
            (uint32_t)K * N,          // in1_block_tile_cnt
            (uint32_t)in0_byte_size,  // in0_block_size_bytes
            (uint32_t)in1_byte_size,  // in1_block_size_bytes
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)M * N,
        });

    tt_metal::detail::LaunchProgram(device, program);
    sleep(1);
    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    int failed_index;
    pass &= is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
        dest_buffer_data,
        packed_golden,
        [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) { return is_close(a, b, 0.015f); },
        &failed_index);
    if (not pass) {
        log_info("Failed Index={}", failed_index);
        print_vector_fixed_numel_per_row(unpack_vector<tt::test_utils::df::bfloat16, uint32_t>(dest_buffer_data), 32);
    }
    return pass;
}
}  // namespace unit_tests::compute::matmul

TEST_F(DeviceFixture, TestSingleCoreSingleTileComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_tile_matmul(this->devices_.at(id)));
    }
}
TEST_F(DeviceFixture, TestSingleCoreSingleBlockSingleTileComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(this->devices_.at(id), 1, 1, 1));
    }
}
TEST_F(DeviceFixture, TestSingleCoreSingleBlockSingleTileAccumulationComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(this->devices_.at(id), 1, 2, 1));
    }
}
TEST_F(DeviceFixture, TestSingleCoreSingleBlockSingleTileNoAccumulationComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(this->devices_.at(id), 2, 1, 2));
    }
}
