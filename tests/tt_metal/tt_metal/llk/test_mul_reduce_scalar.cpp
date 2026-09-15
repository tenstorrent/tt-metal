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

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/program/program_impl.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/arch.hpp>

#include "llk_device_fixture.hpp"
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
    // 0 selects the non-chunked mul_reduce_scalar_tile. Non-zero selects
    // mul_reduce_scalar_chunked_tile with this dst_capacity, which the compute
    // API only permits when num_tiles > dst_capacity.
    uint32_t chunked_dst_capacity = 0;
    // Tile row height. 32 -> standard 32x32 tiles (4 faces); 16 -> 16x32 "tiny
    // tiles" (2 faces, one face-row). Column dimension is always 32.
    uint32_t tile_height = 32;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t seed = 12345;
};

bool run_mul_reduce_scalar_test(distributed::MeshDevice& mesh_device, const MulReduceScalarConfig& config) {
    tt_metal::Program program = tt_metal::CreateProgram();
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

    const bool chunked = config.chunked_dst_capacity != 0;

    // num_tiles and dst_capacity are template parameters of
    // mul_reduce_scalar_chunked_tile, so the chunked kernel takes them as
    // defines; the non-chunked one reads num_tiles as a runtime arg.
    std::map<std::string, std::string> compute_defines = {{"REDUCE_OP", "PoolType::SUM"}};
    if (chunked) {
        compute_defines["CHUNKED_NUM_TILES"] = std::to_string(config.num_tiles);
        compute_defines["CHUNKED_DST_CAPACITY"] = std::to_string(config.chunked_dst_capacity);
    }
    auto mul_reduce_kernel = tt_metal::CreateKernel(
        program,
        chunked ? "tests/tt_metal/tt_metal/test_kernels/compute/mul_reduce_scalar_chunked.cpp"
                : "tests/tt_metal/tt_metal/test_kernels/compute/mul_reduce_scalar.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = config.math_fidelity, .compile_args = {}, .defines = compute_defines});

    SetRuntimeArgs(
        program,
        dual_reader_kernel,
        core,
        {src0_dram_buffer->address(), 0, src1_dram_buffer->address(), 0, config.num_tiles});

    SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0, 1});

    SetRuntimeArgs(program, mul_reduce_kernel, core, {config.num_tiles});

    uint32_t byte_size = config.num_tiles * tile_byte_size;
    auto packed_input0 = test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        0, 1.0f, byte_size / sizeof(bfloat16), config.seed);

    bfloat16 one_bf16 = bfloat16(1.0f);
    uint16_t one_u16 = std::bit_cast<uint16_t>(one_bf16);
    std::vector<uint32_t> packed_input1(byte_size / sizeof(uint32_t));
    for (uint32_t& val : packed_input1) {
        val = (static_cast<uint32_t>(one_u16) << 16) | one_u16;
    }

    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, packed_input0, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, packed_input1, /*blocking=*/true);

    LaunchProgram(mesh_device, std::move(program));

    std::vector<uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

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
        "num_tiles={} dst_capacity={}: Golden={}, Device={}, Diff={}, Device/Golden={}",
        config.num_tiles,
        config.chunked_dst_capacity,
        golden_scalar,
        device_scalar,
        std::abs(device_scalar - golden_scalar),
        golden_scalar != 0.0f ? device_scalar / golden_scalar : 0.0f);

    float rel_tol = 0.01f;
    float abs_tol = 0.01f;
    float tolerance = std::max(rel_tol * std::abs(golden_scalar), abs_tol);
    bool pass = std::abs(device_scalar - golden_scalar) < tolerance;

    return pass;
}

}  // namespace tt::tt_metal::unit_tests::compute::mul_reduce_scalar

using namespace tt::tt_metal::unit_tests::compute::mul_reduce_scalar;

// Runs on any single card (Wormhole or Blackhole); mul_reduce_scalar is
// supported on both architectures.
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

// Chunked suites: mul_reduce_scalar_chunked_tile, the path taken once a row no
// longer fits DST. Chunking is a pure reordering of the same sum, so the golden
// and the tolerance are unchanged from the non-chunked suites -- the only thing
// that differs is which compute API computes it.
//
// DISABLED because the primitive is broken, not because the tests are: every
// case below fails on Blackhole today. Run them with
// --gtest_also_run_disabled_tests. Measured (Blackhole p150b, bfloat16 DEST,
// HiFi4, A and B ~ U[0,1] so the golden is sum(A*B), default scaler 1.0):
//
//   dst_capacity=8 (batch_size 7), varying tile count:
//     num_tiles  batches   Golden      Device    Device/Golden
//             9        2  4603.68     30848            6.70
//            12        2  6143.16     33536            5.46
//            16        3  8204.25    262144           31.95
//            21        3 10760.21    270336           25.12
//            28        4 14310.35   1949696          136.24
//
//   num_tiles=8 fixed (identical data and golden), varying dst_capacity:
//     dst_capacity  batch_size  batches    Device    Device/Golden
//                7           6        2     26752             6.52
//                4           3        3    119296            29.06
//                2           1        8   7.55e+08        183885.61
//
// For reference the non-chunked mul_reduce_scalar_tile at 8 tiles returns 4080
// against the same golden of 4105.68, i.e. 0.994 -- so the inputs and the
// golden are fine and only the chunked driver is not.
//
// Two things follow that a single case cannot show. First, the second table
// holds the data, the tile count and the golden completely fixed and varies
// only dst_capacity, so the error cannot be attributed to the amount of data
// or to the input values -- it tracks the number of chunks. Second, the error
// is not a consistent over-count: at dst_capacity=2 the result is enormous,
// and passing a small scaler instead of the default 1.0 makes the result
// collapse to O(scaler) and become nearly independent of the input, i.e. the
// summed data stops reaching the output at all. The dst_capacity=2 case is
// also not run-order stable (observed as both 7.55 and 7.55e+08 -- same
// mantissa, different exponent), which is consistent with the accumulator
// picking up residual DEST state.
class MulReduceScalarChunkedTest : public LLKMeshDeviceSingleCardFixture, public testing::WithParamInterface<int> {};

TEST_P(MulReduceScalarChunkedTest, DISABLED_MulReduceScalarChunked) {
    int num_tiles = GetParam();
    ASSERT_TRUE(run_mul_reduce_scalar_test(
        this->device(), {.num_tiles = num_tiles, .chunked_dst_capacity = 8, .tile_height = 32}));
}

INSTANTIATE_TEST_SUITE_P(
    MulReduceScalarChunkedTests,
    MulReduceScalarChunkedTest,
    testing::Values(9, 12, 16, 21, 28),
    [](const testing::TestParamInfo<int>& info) {
        return "MulReduceScalarChunked_" + std::to_string(info.param) + "_Tiles";
    });

// The controlled version of the suite above: the tile count is FIXED at 8, so
// the input data, the golden and every buffer are identical across the whole
// suite, and the only thing that varies is dst_capacity -- which is to say, the
// number of chunks (batch_size = dst_capacity - 1). Also DISABLED; see above.
class MulReduceScalarChunkedCapacityTest : public LLKMeshDeviceSingleCardFixture,
                                           public testing::WithParamInterface<int> {};

TEST_P(MulReduceScalarChunkedCapacityTest, DISABLED_MulReduceScalarChunkedCapacity) {
    int dst_capacity = GetParam();
    ASSERT_TRUE(run_mul_reduce_scalar_test(
        this->device(),
        {.num_tiles = 8, .chunked_dst_capacity = static_cast<uint32_t>(dst_capacity), .tile_height = 32}));
}

INSTANTIATE_TEST_SUITE_P(
    MulReduceScalarChunkedCapacityTests,
    MulReduceScalarChunkedCapacityTest,
    testing::Values(2, 4, 7),
    [](const testing::TestParamInfo<int>& info) {
        return "MulReduceScalarChunked_8_Tiles_DstCapacity_" + std::to_string(info.param);
    });
