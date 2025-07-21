// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include "dispatch_fixture.hpp"
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"

namespace tt {
namespace tt_metal {
class CommandQueue;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

using namespace tt;

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

std::shared_ptr<Buffer> MakeBuffer(IDevice* device, uint32_t size, uint32_t page_size, bool sram) {
    InterleavedBufferConfig config{
        .device = device,
        .size = size,
        .page_size = page_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)};
    return CreateBuffer(config);
}

std::shared_ptr<Buffer> MakeBufferBFP16(IDevice* device, uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    // For simplicity, all DRAM buffers have page size = tile size.
    const uint32_t page_tiles = sram ? n_tiles : 1;
    return MakeBuffer(device, tile_size * n_tiles, page_tiles * tile_size, sram);
}

CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format) {
    CircularBufferConfig cb_src0_config = CircularBufferConfig(size, {{cb, format}}).set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_src0_config);
}

CBHandle MakeCircularBufferBFP16(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float16_b);
}

namespace unit_tests_common::vecadd::test_vecadd_multi_core {

bool vecadd_multi_core(DispatchFixture* fixture, IDevice* device, uint32_t n_tiles) {
    const uint32_t num_core = 4;
    TT_FATAL(n_tiles >= num_core, "Parameter mismatch {} {}", n_tiles, num_core);

    bool pass = true;

    int seed = 0x1234567;
    Program program = CreateProgram();

    // designate 4 cores for utilization - cores (0,0), (0,1), (0,2), (0,3)
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {0, 3};
    CoreRange cores(start_core, end_core);

    CommandQueue& cq = device->command_queue();
    const uint32_t tile_size = tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;
    const uint32_t tiles_per_core = n_tiles / num_core;

    // Create 3 buffers on DRAM. These will hold the input and output data. A
    // and B are the input buffers, C is the output buffer.
    auto a = MakeBufferBFP16(device, n_tiles, false);
    auto b = MakeBufferBFP16(device, n_tiles, false);
    auto c = MakeBufferBFP16(device, n_tiles, false);

    std::mt19937 rng(seed);
    std::vector<bfloat16> a_data = create_random_vector_of_bfloat16_native(tile_size * n_tiles * 2, 10, rng());
    std::vector<bfloat16> b_data = create_random_vector_of_bfloat16_native(tile_size * n_tiles * 2, 10, rng());

    const uint32_t cir_buffer_title = 4;
    CBHandle cb_a = MakeCircularBufferBFP16(program, cores, tt::CBIndex::c_0, cir_buffer_title);
    CBHandle cb_b = MakeCircularBufferBFP16(program, cores, tt::CBIndex::c_1, cir_buffer_title);
    CBHandle cb_c = MakeCircularBufferBFP16(program, cores, tt::CBIndex::c_2, cir_buffer_title);

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)tt::CBIndex::c_0, (std::uint32_t)tt::CBIndex::c_1};
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)tt::CBIndex::c_2};
    std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t)tt::CBIndex::c_0, (std::uint32_t)tt::CBIndex::c_1, (std::uint32_t)tt::CBIndex::c_2};
    auto reader = CreateKernel(
        program,
        "tt_metal/programming_examples/vecadd_multi_core/kernels/"
        "interleaved_tile_read_multi_core.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});
    auto writer = CreateKernel(
        program,
        "tt_metal/programming_examples/vecadd_multi_core/kernels/"
        "tile_write_multi_core.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});
    auto compute = CreateKernel(
        program,
        "tt_metal/programming_examples/vecadd_multi_core/"
        "kernels/add_multi_core.cpp",
        cores,
        ComputeConfig{.math_approx_mode = false, .compile_args = compute_compile_time_args, .defines = {}});

    for (int i = 0; i < num_core; ++i) {
        // Set runtime arguments for each core.
        CoreCoord core = {0, i};
        SetRuntimeArgs(program, reader, core, {a->address(), b->address(), tiles_per_core, i});
        SetRuntimeArgs(program, writer, core, {c->address(), tiles_per_core, i});
        SetRuntimeArgs(program, compute, core, {tiles_per_core, i});
    }

    EnqueueWriteBuffer(cq, a, a_data, false);
    EnqueueWriteBuffer(cq, b, b_data, false);
    // Enqueue the program
    EnqueueProgram(cq, program, true);

    log_debug(LogTest, "Kernel execution finished");

    // Read the output buffer.
    std::vector<bfloat16> c_data;
    EnqueueReadBuffer(cq, c, c_data, true);

    size_t data_per_core = tile_size * tiles_per_core;

    for (int core = 0; core < num_core; ++core) {
        const auto core_offset = core * (tile_size + tiles_per_core);
        for (int index = 0; index < data_per_core; index++) {
            const auto i = core_offset + index;
            float golden = a_data[i].to_float() + b_data[i].to_float();
            pass &= tt::test_utils::is_close<float>(golden, c_data[i].to_float(), 0.015f);
        }
    }

    return pass;
}
}  // namespace unit_tests_common::vecadd::test_vecadd_multi_core

TEST_F(DispatchFixture, VecaddMultiCore) {
    uint32_t num_tiles = 64;
    ASSERT_TRUE(unit_tests_common::vecadd::test_vecadd_multi_core::vecadd_multi_core(this, devices_.at(0), num_tiles));
}

}  // namespace tt::tt_metal
