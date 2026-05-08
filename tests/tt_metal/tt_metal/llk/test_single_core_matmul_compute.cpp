// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <unistd.h>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "llk_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::matmul {

// Per-block matmul: out[M x N] = in0[M x K] * in1[K x N]
// Repeated num_blocks times along K with partials accumulation.
// If K > 1 -> dest accumulation within each block
// If num_blocks > 1 -> partials accumulation (either l1 accumulation or spill and reload)
struct BlockedMatmulConfig {
    uint32_t M = 1;           // per-block rows (tiles)
    uint32_t K = 1;           // per-block inner dim (tiles)
    uint32_t N = 1;           // per-block cols (tiles)
    uint32_t num_blocks = 1;  // number of K-blocks
    bool packer_l1_acc = false;
};

void create_CBs_for_fused_matmul(
    distributed::MeshWorkload& workload,
    const std::shared_ptr<distributed::MeshDevice>& /*mesh_device*/,
    CoreCoord core,
    bool activations_rm,
    bool output_rm,
    uint32_t M,
    uint32_t N,
    uint32_t in0_block_w,
    uint32_t /*out_subblock_h*/) {
    uint32_t num_bytes_for_df = 2;
    uint32_t in0_cb = 0;
    uint32_t in1_cb = 1;
    uint32_t tilize_mode_tilized_in0_cb = 24;
    uint32_t matmul_partials_cb = 25;
    uint32_t untilize_mode_final_matmul_partials_cb = 26;
    uint32_t untilize_mode_reblock_cb = 27;
    uint32_t out0_cb = 16;

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto& program = workload.get_programs().at(device_range);

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = M * N;

    // Invariants
    uint32_t cb0_tiles = M * in0_block_w * 2;
    tt_metal::CircularBufferConfig l1_input0_cb_config =
        tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{in0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(in0_cb, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, l1_input0_cb_config);

    uint32_t cb1_tiles = N * in0_block_w * 2;
    tt_metal::CircularBufferConfig cb_in1_config =
        tt_metal::CircularBufferConfig(cb1_tiles * single_tile_size, {{in1_cb, tt::DataFormat::Float16_b}})
            .set_page_size(in1_cb, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_in1_config);

    if (not activations_rm and not output_rm) {  // no tilize, no untilize
        tt_metal::CircularBufferConfig cb_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Partials share same L1 address space as output
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else if (not activations_rm and output_rm) {  // no tilize, just untilize

        tt_metal::CircularBufferConfig cb_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Need a new CB to push output block to since other
        // intermediate read pointer changes in enable reload
        // block
        tt_metal::CircularBufferConfig cb_final_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = N;  // Only space for one row
        tt_metal::CircularBufferConfig cb_reblock_config =
            tt_metal::CircularBufferConfig(
                reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else if (activations_rm and not output_rm) {  // just tilize, no untilize

        tt_metal::CircularBufferConfig cb_src0_tilized_config =
            tt_metal::CircularBufferConfig(
                cb0_tiles * single_tile_size, {{tilize_mode_tilized_in0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(tilize_mode_tilized_in0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

        tt_metal::CircularBufferConfig cb_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else {  // tilize activations and untilize output

        // Used for placing tilized activations
        tt_metal::CircularBufferConfig cb_src0_tilized_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{tilize_mode_tilized_in0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(tilize_mode_tilized_in0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

        tt_metal::CircularBufferConfig cb_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Shares same address space as matmul partials
        tt_metal::CircularBufferConfig cb_final_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size,
                {{untilize_mode_final_matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_final_matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = N;  // Only space for one row
        tt_metal::CircularBufferConfig cb_reblock_config =
            tt_metal::CircularBufferConfig(
                reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);
    }
}

bool single_tile_matmul(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    bool pass = true;
    // FIXME: Convert to config
    CoreCoord core(0, 0);
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const size_t byte_size = 1 * 2 * 32 * 32;

    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto input0_dram_buffer = CreateBuffer(dram_config);
    const uint32_t in0_dram_addr = input0_dram_buffer->address();
    auto input1_dram_buffer = CreateBuffer(dram_config);
    const uint32_t in1_dram_addr = input1_dram_buffer->address();
    auto output_dram_buffer = CreateBuffer(dram_config);
    const uint32_t out_dram_addr = output_dram_buffer->address();

    tt_metal::CircularBufferConfig l1_input0_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{in0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in0_cb_index, byte_size);
    tt_metal::CreateCircularBuffer(program_, core, l1_input0_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{in1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in1_cb_index, byte_size);
    tt_metal::CreateCircularBuffer(program_, core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config =
        tt_metal::CircularBufferConfig(byte_size, {{out_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(out_cb_index, byte_size);
    tt_metal::CreateCircularBuffer(program_, core, l1_output_cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/single_tile_compute.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = {in0_cb_index, in1_cb_index, out_cb_index}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f, 1.0f, byte_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f / 32.0f,
        1.0f / 32.0f,
        byte_size / sizeof(bfloat16),
        std::chrono::system_clock::now().time_since_epoch().count());
    // Setup the weights such that final result is the original input.

    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    auto packed_golden = packed_input0;

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)0,  // in_0 dram bank id
            (uint32_t)in1_dram_addr,
            (uint32_t)0,
            (uint32_t)1,  // num_tiles
        });
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)0,
            (uint32_t)1,  // num_tiles
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.015f); });
    return pass;
}
// blocked matmul has blocking, but still fits within dst, so no spill/reloads or intermediates
bool single_block_matmul(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t M, uint32_t K, uint32_t N) {
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
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config_0{
        .device = device,
        .size = in0_byte_size,
        .page_size = in0_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::InterleavedBufferConfig dram_config_1{
        .device = device,
        .size = in1_byte_size,
        .page_size = in1_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::InterleavedBufferConfig dram_config_out{
        .device = device,
        .size = out_byte_size,
        .page_size = out_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto input0_dram_buffer = CreateBuffer(dram_config_0);
    const uint32_t in0_dram_addr = input0_dram_buffer->address();
    auto input1_dram_buffer = CreateBuffer(dram_config_1);
    const uint32_t in1_dram_addr = input1_dram_buffer->address();
    auto output_dram_buffer = CreateBuffer(dram_config_out);
    const uint32_t out_dram_addr = output_dram_buffer->address();

    tt_metal::CircularBufferConfig l1_input0_cb_config =
        tt_metal::CircularBufferConfig(in0_byte_size, {{in0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in0_cb_index, cb_page_size);
    tt_metal::CreateCircularBuffer(program_, core, l1_input0_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config =
        tt_metal::CircularBufferConfig(in1_byte_size, {{in1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in1_cb_index, cb_page_size);
    tt_metal::CreateCircularBuffer(program_, core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config =
        tt_metal::CircularBufferConfig(out_byte_size, {{out_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(out_cb_index, cb_page_size);
    tt_metal::CreateCircularBuffer(program_, core, l1_output_cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary_blocked.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_tile_compute.cpp",
        core,
        tt_metal::ComputeConfig{
            .compile_args = {in0_cb_index, in1_cb_index, out_cb_index, M * K, K * N, M * N, M, N, K}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f, 1.0f, in0_byte_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        0.03125f,
        0.03125f,
        in1_byte_size / sizeof(bfloat16),
        std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto packed_golden = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f * K,
        1.0f * K,
        (out_byte_size) / sizeof(bfloat16),
        std::chrono::system_clock::now().time_since_epoch().count());

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)0,
            (uint32_t)in1_dram_addr,
            (uint32_t)0,
            (uint32_t)1,              // num_blocks
            (uint32_t)M * K,          // in0_block_tile_cnt
            (uint32_t)K * N,          // in1_block_tile_cnt
            (uint32_t)in0_byte_size,  // in0_block_size_bytes
            (uint32_t)in1_byte_size,  // in1_block_size_bytes
        });
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)0,
            (uint32_t)M * N,
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    int failed_index;
    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data,
        packed_golden,
        [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.015f); },
        &failed_index);
    if (not pass) {
        log_info(tt::LogTest, "Failed Index={}", failed_index);
        print_vector_fixed_numel_per_row(unpack_vector<bfloat16, uint32_t>(dest_buffer_data), 32);
    }
    return pass;
}
// blocked matmul has blocking on output, spill/reloads or does l1 accumulation using intermediate
bool blocked_matmul(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const BlockedMatmulConfig& cfg) {
    const uint32_t M = cfg.M;
    const uint32_t K = cfg.K;
    const uint32_t N = cfg.N;
    const uint32_t num_blocks = cfg.num_blocks;

    const bool is_quasar = MetalContext::instance().get_cluster().arch() == ARCH::QUASAR;

    bool pass = true;
    CoreCoord core(0, 0);
    const size_t cb_page_size = 2 * 32 * 32;
    const size_t in0_block_byte_size = M * K * cb_page_size;
    const size_t in1_block_byte_size = K * N * cb_page_size;
    const size_t in0_total_byte_size = num_blocks * in0_block_byte_size;
    const size_t in1_total_byte_size = num_blocks * in1_block_byte_size;
    const size_t out_byte_size = M * N * cb_page_size;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config_0{
        .device = device,
        .size = in0_total_byte_size,
        .page_size = in0_total_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::InterleavedBufferConfig dram_config_1{
        .device = device,
        .size = in1_total_byte_size,
        .page_size = in1_total_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::InterleavedBufferConfig dram_config_out{
        .device = device,
        .size = out_byte_size,
        .page_size = out_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto input0_dram_buffer = CreateBuffer(dram_config_0);
    const uint32_t in0_dram_addr = input0_dram_buffer->address();
    auto input1_dram_buffer = CreateBuffer(dram_config_1);
    const uint32_t in1_dram_addr = input1_dram_buffer->address();
    auto output_dram_buffer = CreateBuffer(dram_config_out);
    const uint32_t out_dram_addr = output_dram_buffer->address();

    uint32_t in0_id = 0;
    uint32_t in1_id = 0;
    uint32_t out_id = 0;
    uint32_t partials_id = 0;

    if (is_quasar) {
        auto make_dfb = [&](uint32_t entry_size, uint32_t num_entries, uint16_t producer_mask, uint16_t consumer_mask) {
            return tt_metal::experimental::dfb::CreateDataflowBuffer(
                program_,
                core,
                tt_metal::experimental::dfb::DataflowBufferConfig{
                    .entry_size = entry_size,
                    .num_entries = num_entries,
                    .producer_risc_mask = producer_mask,
                    .num_producers = 1,
                    .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
                    .consumer_risc_mask = consumer_mask,
                    .num_consumers = 1,
                    .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
                    .enable_implicit_sync = false,
                    .data_format = tt::DataFormat::Float16_b});
        };
        in0_id = make_dfb(cb_page_size, M * K, 0x1, 0x100);
        in1_id = make_dfb(cb_page_size, K * N, 0x1, 0x100);
        out_id = make_dfb(cb_page_size, M * N, 0x100, 0x2);
        partials_id = tt_metal::experimental::dfb::CreateDataflowBuffer(
            program_,
            core,
            tt_metal::experimental::dfb::DataflowBufferConfig{
                .entry_size = cb_page_size,
                .num_entries = M * N,
                .num_producers = 1,
                .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
                .num_consumers = 1,
                .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
                .enable_implicit_sync = false,
                .data_format = tt::DataFormat::Float16_b,
                .tensix_scope = tt_metal::experimental::dfb::TensixScope::INTRA});
    } else {
        const uint32_t in0_cb_index = 0;
        const uint32_t in1_cb_index = 1;
        const uint32_t out_cb_index = 16;
        const uint32_t partials_cb_index = 24;

        tt_metal::CircularBufferConfig l1_input0_cb_config =
            tt_metal::CircularBufferConfig(in0_block_byte_size, {{in0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(in0_cb_index, cb_page_size);
        tt_metal::CreateCircularBuffer(program_, core, l1_input0_cb_config);

        tt_metal::CircularBufferConfig l1_input1_cb_config =
            tt_metal::CircularBufferConfig(in1_block_byte_size, {{in1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(in1_cb_index, cb_page_size);
        tt_metal::CreateCircularBuffer(program_, core, l1_input1_cb_config);

        tt_metal::CircularBufferConfig l1_output_cb_config =
            tt_metal::CircularBufferConfig(out_byte_size, {{out_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(out_cb_index, cb_page_size);
        tt_metal::CreateCircularBuffer(program_, core, l1_output_cb_config);

        tt_metal::CircularBufferConfig l1_partials_cb_config =
            tt_metal::CircularBufferConfig(out_byte_size, {{partials_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(partials_cb_index, cb_page_size);
        tt_metal::CreateCircularBuffer(program_, core, l1_partials_cb_config);

        in0_id = in0_cb_index;
        in1_id = in1_cb_index;
        out_id = out_cb_index;
        partials_id = partials_cb_index;
    }

    std::map<std::string, std::string> compute_defines;
    if (cfg.packer_l1_acc) {
        compute_defines["PACKER_L1_ACC"] = "1";
    }

    std::vector<uint32_t> compute_compile_args = {
        in0_id, in1_id, out_id, partials_id, M * K, K * N, M * N, M, N, K, num_blocks};

    KernelHandle reader_kernel;
    KernelHandle writer_kernel;
    KernelHandle compute_kernel;

    if (is_quasar) {
        reader_kernel = tt_metal::experimental::quasar::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary_blocked.cpp",
            core,
            tt_metal::experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1, .compile_args = {in0_id, in1_id}});

        writer_kernel = tt_metal::experimental::quasar::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
            core,
            tt_metal::experimental::quasar::QuasarDataMovementConfig{
                .num_threads_per_cluster = 1, .compile_args = {out_id}});

        compute_kernel = tt_metal::experimental::quasar::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_block_compute.cpp",
            core,
            tt_metal::experimental::quasar::QuasarComputeConfig{
                .num_threads_per_cluster = 1, .compile_args = compute_compile_args, .defines = compute_defines});

        tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program_, in0_id, reader_kernel, compute_kernel);
        tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program_, in1_id, reader_kernel, compute_kernel);
        tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program_, out_id, compute_kernel, writer_kernel);
        tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
            program_, partials_id, compute_kernel, compute_kernel);
    } else {
        reader_kernel = tt_metal::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary_blocked.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .compile_args = {in0_id, in1_id}});

        writer_kernel = tt_metal::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = {out_id}});

        compute_kernel = tt_metal::CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_block_compute.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_compile_args, .defines = compute_defines});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f,
        1.0f,
        in0_total_byte_size / sizeof(bfloat16),
        std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        0.03125f,
        0.03125f,
        in1_total_byte_size / sizeof(bfloat16),
        std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto packed_golden = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f * K * num_blocks,
        1.0f * K * num_blocks,
        (out_byte_size) / sizeof(bfloat16),
        std::chrono::system_clock::now().time_since_epoch().count());

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)0,
            (uint32_t)in1_dram_addr,
            (uint32_t)0,
            (uint32_t)num_blocks,
            (uint32_t)(M * K),
            (uint32_t)(K * N),
            (uint32_t)in0_block_byte_size,
            (uint32_t)in1_block_byte_size,
        });
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)0,
            (uint32_t)(M * N),
        });

    auto blocking = is_quasar;
    distributed::EnqueueMeshWorkload(cq, workload, blocking);
    if (not blocking) {
        distributed::Finish(cq);
    }
    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    int failed_index;
    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data,
        packed_golden,
        [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.015f); },
        &failed_index);
    if (not pass) {
        log_info(tt::LogTest, "Failed Index={}", failed_index);
        print_vector_fixed_numel_per_row(unpack_vector<bfloat16, uint32_t>(dest_buffer_data), 32);
    }
    return pass;
}
}  // namespace unit_tests::compute::matmul

TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleTileComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_tile_matmul(this->devices_.at(id)));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleBlockSingleTileComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(this->devices_.at(id), 1, 1, 1));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleBlockSingleTileAccumulationComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(this->devices_.at(id), 1, 2, 1));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreSingleBlockSingleTileNoAccumulationComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(this->devices_.at(id), 2, 1, 2));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreBlockedComputeMatmul) {
    unit_tests::compute::matmul::BlockedMatmulConfig config{.M = 2, .K = 2, .N = 2, .num_blocks = 1};
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::blocked_matmul(this->devices_.at(id), config));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreMultiBlockSpillReloadComputeMatmul) {
    unit_tests::compute::matmul::BlockedMatmulConfig config{
        .M = 2, .K = 2, .N = 2, .num_blocks = 2, .packer_l1_acc = false};
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::blocked_matmul(this->devices_.at(id), config));
    }
}
TEST_F(LLKMeshDeviceFixture, TensixTestSingleCoreMultiBlockL1AccComputeMatmul) {
    unit_tests::compute::matmul::BlockedMatmulConfig config{
        .M = 2, .K = 2, .N = 2, .num_blocks = 2, .packer_l1_acc = true};
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::blocked_matmul(this->devices_.at(id), config));
    }
}

}  // namespace tt::tt_metal
