// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <vector>

#include <tt-metalium/data_formats/bfloat16.hpp>
#include <tt-metalium/data_formats/tile_constants.hpp>
#include <tt-metalium/memory/buffer/buffer.hpp>
#include <tt-metalium/memory/buffer/buffer_types.hpp>
#include <tt-metalium/workload/legacy/circular_buffer_config.hpp>
#include <tt-metalium/vocabulary/core_coord.hpp>
#include <tt-metalium/execution/distributed.hpp>
#include <tt-metalium/workload/legacy/kernel_types.hpp>
#include <tt-metalium/mesh/legacy/device.hpp>
#include <tt-metalium/workload/legacy/host_api.hpp>
#include <tt-metalium/workload/program.hpp>
#include <tt-metalium/execution/slow_dispatch.hpp>
#include "impl/program/program_impl.hpp"
#include <tt-metalium/workload/legacy/tensor_accessor_args.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/types/arch.hpp>

#include "llk_device_fixture.hpp"
#include "single_node_dfb_program.hpp"
#include "test_golden_impls.hpp"
#include "tt_metal/impl/data_format/bfloat16_utils.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal::unit_tests::compute::mul_reduce_scalar {

struct MulReduceScalarConfig {
    uint32_t num_tiles = 1;
    // Tile row height. 32 -> standard 32x32 tiles (4 faces); 16 -> 16x32 "tiny
    // tiles" (2 faces, one face-row). Column dimension is always 32.
    uint32_t tile_height = 32;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t seed = 12345;
    bool accumulate_in_one_tile = false;
    float rel_tol = 0.01f;
};

using InputPair = std::pair<std::vector<uint32_t>, std::vector<uint32_t>>;

// in0 is random in [0, 1) and in1 is all ones, so the result is the sum of in0.
InputPair make_inputs(const MulReduceScalarConfig& config, uint32_t byte_size) {
    auto packed_input0 = test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        0, 1.0f, byte_size / sizeof(bfloat16), config.seed);

    bfloat16 one_bf16 = bfloat16(1.0f);
    uint16_t one_u16 = std::bit_cast<uint16_t>(one_bf16);
    std::vector<uint32_t> packed_input1(byte_size / sizeof(uint32_t));
    for (uint32_t& val : packed_input1) {
        val = (static_cast<uint32_t>(one_u16) << 16) | one_u16;
    }
    return {std::move(packed_input0), std::move(packed_input1)};
}

bool check_result(
    const MulReduceScalarConfig& config,
    const std::vector<uint32_t>& packed_input0,
    const std::vector<uint32_t>& packed_input1,
    const std::vector<uint32_t>& result_vec) {
    auto u16_src0_vec = u16_from_u32_vector(packed_input0);
    auto u16_src1_vec = u16_from_u32_vector(packed_input1);

    float golden_scalar = 0.0f;
    for (size_t i = 0; i < u16_src0_vec.size(); ++i) {
        float val0 = static_cast<float>(std::bit_cast<bfloat16>(u16_src0_vec[i]));
        float val1 = static_cast<float>(std::bit_cast<bfloat16>(u16_src1_vec[i]));
        golden_scalar += val0 * val1;
    }

    auto u16_result_vec = u16_from_u32_vector(result_vec);
    float device_scalar = static_cast<float>(std::bit_cast<bfloat16>(u16_result_vec[0]));

    log_info(
        LogTest,
        "num_tiles={}: Golden={}, Device={}, Diff={}",
        config.num_tiles,
        golden_scalar,
        device_scalar,
        std::abs(device_scalar - golden_scalar));

    float abs_tol = 0.01f;
    float tolerance = std::max(config.rel_tol * std::abs(golden_scalar), abs_tol);
    bool pass = std::abs(device_scalar - golden_scalar) < tolerance;

    return pass;
}

// Quasar only accepts the 2.0 host API.
bool run_mul_reduce_scalar_test_quasar(distributed::MeshDevice& mesh_device, const MulReduceScalarConfig& config) {
    namespace sn = tt::tt_metal::unit_tests::llk::single_node_dfb;
    // The stimulus and golden are bf16, so every DFB and the tile size use it.
    constexpr tt::DataFormat format = tt::DataFormat::Float16_b;
    const uint32_t tile_byte_size = tt::tile_size(format);
    auto [packed_input0, packed_input1] = make_inputs(config, config.num_tiles * tile_byte_size);
    const std::vector<uint32_t> result_vec = sn::run(
        mesh_device,
        {.name = "mul_reduce_scalar_quasar",
         .inputs =
             {{.data = packed_input0,
               .format = format,
               .num_tiles = config.num_tiles,
               .accessor = "in0",
               .num_entries = config.num_tiles},
              {.data = packed_input1,
               .format = format,
               .num_tiles = config.num_tiles,
               .accessor = "in1",
               .num_entries = config.num_tiles}},
         .output = {.format = format, .num_tiles = 1, .accessor = "out", .num_entries = 1},
         .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/mul_reduce_scalar_2_0.cpp",
         .compile_time_args = {{"num_tiles", config.num_tiles}},
         .hw_config = ComputeHardwareConfig{.fpu_math_fidelity = config.math_fidelity}});
    return check_result(config, packed_input0, packed_input1, result_vec);
}

bool run_mul_reduce_scalar_test(distributed::MeshDevice& mesh_device, const MulReduceScalarConfig& config) {
    if (mesh_device.arch() == tt::ARCH::QUASAR) {
        return run_mul_reduce_scalar_test_quasar(mesh_device, config);
    }
    tt_metal::Program program = tt_metal::CreateProgram(mesh_device);
    CoreCoord core = {0, 0};

    // bfloat16: 2 bytes per element; a 16x32 tiny tile is half a full tile.
    const uint32_t tile_byte_size = 2 * config.tile_height * tt::constants::TILE_WIDTH;
    const tt::tt_metal::Tile cb_tile({config.tile_height, tt::constants::TILE_WIDTH});
    const bool tiny_tile = (config.tile_height != tt::constants::TILE_HEIGHT);

    uint32_t input_buffer_size = config.num_tiles * tile_byte_size;
    distributed::ReplicatedBufferConfig input_global_config{.size = input_buffer_size};
    distributed::DeviceLocalBufferConfig input_local_config{
        .page_size = tile_byte_size, .buffer_type = tt_metal::BufferType::DRAM};
    auto src0_dram_buffer = distributed::MeshBuffer::create(input_global_config, input_local_config, &mesh_device);
    auto src1_dram_buffer = distributed::MeshBuffer::create(input_global_config, input_local_config, &mesh_device);

    auto dst_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = tile_byte_size},
        {.page_size = tile_byte_size, .buffer_type = tt_metal::BufferType::DRAM},
        &mesh_device);

    uint32_t cb_tiles = std::max(8u, config.num_tiles);
    uint32_t cb_size = cb_tiles * tile_byte_size;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(cb_size, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, tile_byte_size);
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(cb_size, {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_1, tile_byte_size);
    tt_metal::CircularBufferConfig cb_out_config =
        tt_metal::CircularBufferConfig(cb_size, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_16, tile_byte_size);
    if (tiny_tile) {
        // Advertise the 16x32 tile geometry so the compute kernel derives
        // num_faces=2 from the operand CBs via get_operand_num_faces().
        cb_src0_config.set_tile_dims(tt::CBIndex::c_0, cb_tile);
        cb_src1_config.set_tile_dims(tt::CBIndex::c_1, cb_tile);
        cb_out_config.set_tile_dims(tt::CBIndex::c_16, cb_tile);
    }
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Set up compile-time arguments for the reader kernel using TensorAccessor
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    auto dual_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_multiple_tiles.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    const std::map<std::string, std::string> compute_defines = {
        {"REDUCE_OP", "PoolType::SUM"}, {"ACCUMULATE_IN_ONE_TILE", config.accumulate_in_one_tile ? "true" : "false"}};
    auto mul_reduce_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/mul_reduce_scalar.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = config.math_fidelity, .compile_args = {}, .defines = compute_defines});

    SetRuntimeArgs(
        program,
        dual_reader_kernel,
        core,
        {src0_dram_buffer->address(), 0, src1_dram_buffer->address(), 0, config.num_tiles});

    SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0, 1});

    SetRuntimeArgs(program, mul_reduce_kernel, core, {config.num_tiles});

    auto [packed_input0, packed_input1] = make_inputs(config, config.num_tiles * tile_byte_size);

    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, packed_input0, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, packed_input1, /*blocking=*/true);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    return check_result(config, packed_input0, packed_input1, result_vec);
}

// A bf16 DEST rounds each element's running sum at every one-tile accumulate, so the bound widens with the tile
// count. It stays under the 1/num_tiles a dropped tile would cost for every count the suites use.
float one_tile_rel_tol(int num_tiles) { return 0.01f + 0.001f * static_cast<float>(num_tiles); }

}  // namespace tt::tt_metal::unit_tests::compute::mul_reduce_scalar

using namespace tt::tt_metal::unit_tests::compute::mul_reduce_scalar;

// Runs on all architectures.
class MulReduceScalarTest : public LLKMeshDeviceSingleCardFixture, public testing::WithParamInterface<int> {};

// Standard 32x32-tile suite parametrized by tile count.
TEST_P(MulReduceScalarTest, MulReduceScalar) {
    int num_tiles = GetParam();
    ASSERT_TRUE(run_mul_reduce_scalar_test(this->device(), {.num_tiles = num_tiles, .tile_height = 32}));
}

// Instantiate the test suite with different tile counts
INSTANTIATE_TEST_SUITE_P(
    MulReduceScalarTests,
    MulReduceScalarTest,
    testing::Values(1, 2, 3, 7, 8),
    [](const testing::TestParamInfo<int>& info) { return "MulReduceScalar_" + std::to_string(info.param) + "_Tiles"; });

// 16x32 "tiny tile" (num_faces=2) suite parametrized by tile count.
class MulReduceScalarTinyTileTest : public LLKMeshDeviceSingleCardFixture, public testing::WithParamInterface<int> {};

TEST_P(MulReduceScalarTinyTileTest, MulReduceScalarTinyTile) {
    if (this->arch_ == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "the Quasar binary unpack supports full 32x32 tiles only (tenstorrent/tt-metal#47597)";
    }
    int num_tiles = GetParam();
    ASSERT_TRUE(run_mul_reduce_scalar_test(this->device(), {.num_tiles = num_tiles, .tile_height = 16}));
}

INSTANTIATE_TEST_SUITE_P(
    MulReduceScalarTinyTileTests,
    MulReduceScalarTinyTileTest,
    testing::Values(1, 2, 3, 7, 8),
    [](const testing::TestParamInfo<int>& info) {
        return "MulReduceScalar_16x32_" + std::to_string(info.param) + "_Tiles";
    });

// One-tile accumulation, parametrized past the 8-tile DEST capacity.
class MulReduceScalarOneTileTest : public LLKBlackholeSingleCardFixture, public testing::WithParamInterface<int> {};

TEST_P(MulReduceScalarOneTileTest, MulReduceScalarOneTile) {
    int num_tiles = GetParam();
    ASSERT_TRUE(run_mul_reduce_scalar_test(
        this->device(),
        {.num_tiles = num_tiles,
         .tile_height = 32,
         .accumulate_in_one_tile = true,
         .rel_tol = one_tile_rel_tol(num_tiles)}));
}

INSTANTIATE_TEST_SUITE_P(
    MulReduceScalarOneTileTests,
    MulReduceScalarOneTileTest,
    testing::Values(1, 2, 8, 16, 24),
    [](const testing::TestParamInfo<int>& info) {
        return "MulReduceScalarOneTile_" + std::to_string(info.param) + "_Tiles";
    });

class MulReduceScalarOneTileTinyTileTest : public LLKBlackholeSingleCardFixture,
                                           public testing::WithParamInterface<int> {};

TEST_P(MulReduceScalarOneTileTinyTileTest, MulReduceScalarOneTileTinyTile) {
    int num_tiles = GetParam();
    ASSERT_TRUE(run_mul_reduce_scalar_test(
        this->device(),
        {.num_tiles = num_tiles,
         .tile_height = 16,
         .accumulate_in_one_tile = true,
         .rel_tol = one_tile_rel_tol(num_tiles)}));
}

INSTANTIATE_TEST_SUITE_P(
    MulReduceScalarOneTileTinyTileTests,
    MulReduceScalarOneTileTinyTileTest,
    testing::Values(1, 8, 16),
    [](const testing::TestParamInfo<int>& info) {
        return "MulReduceScalarOneTile_16x32_" + std::to_string(info.param) + "_Tiles";
    });
