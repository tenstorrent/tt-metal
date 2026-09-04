// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <string>
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
#include <tt-logger/tt-logger.hpp>

#include "llk_device_fixture.hpp"
#include "tt_metal/impl/data_format/bfloat16_utils.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal::unit_tests::compute::sum_reduce_scalar {

struct SumReduceScalarConfig {
    uint32_t num_tiles = 1;
    // Tile row height. 32 -> standard 32x32 tiles (4 faces); 16 -> 16x32 "tiny
    // tiles" (2 faces, one face-row). Column dimension is always 32.
    uint32_t tile_height = 32;
    float scaler = 1.0f;
    // Native fp32 DEST. The reduce tail is templated on DST_ACCUM_MODE, so this
    // selects a different instantiation of the whole op. Inputs stay bfloat16; only
    // DEST and the output CB widen, matching mul_reduce_scalar's own fp32 coverage.
    bool fp32_dest_acc = false;
    // Full-sync DEST doubles the tile budget (8 fp32 / 16 bf16 instead of 4 / 8) and
    // changes DST_SYNC_MODE, which the op threads into its SFPU fills. blaze couples
    // this to fp32 (compiler.py: dst_full_sync_en=fp32_dest_acc_en), so the fp32
    // suites below set both together to reproduce its real configuration.
    bool dst_full_sync = false;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t seed = 12345;
    bool use_constant_input = false;
    uint16_t constant_input_bits = 0;
    bool expect_exact_zero = false;
};

bool run_sum_reduce_scalar_test(distributed::MeshDevice& mesh_device, const SumReduceScalarConfig& config) {
    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord core = {0, 0};

    // bfloat16: 2 bytes per element; a 16x32 tiny tile is half a full tile.
    const uint32_t tile_byte_size = 2 * config.tile_height * tt::constants::TILE_WIDTH;
    const tt::DataFormat out_format = config.fp32_dest_acc ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const uint32_t out_tile_byte_size = config.fp32_dest_acc ? 2 * tile_byte_size : tile_byte_size;
    const tt::tt_metal::Tile cb_tile({config.tile_height, tt::constants::TILE_WIDTH});
    const bool tiny_tile = (config.tile_height != tt::constants::TILE_HEIGHT);

    // One page for the whole input, so it lands contiguously in a single DRAM bank:
    // reader_unary_push_n walks a bank linearly, which would otherwise stride across
    // interleaved banks and hand the compute kernel the wrong tiles.
    const uint32_t input_byte_size = config.num_tiles * tile_byte_size;
    auto src_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = input_byte_size},
        {.page_size = input_byte_size, .buffer_type = tt_metal::BufferType::DRAM},
        &mesh_device);

    auto dst_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = out_tile_byte_size},
        {.page_size = out_tile_byte_size, .buffer_type = tt_metal::BufferType::DRAM},
        &mesh_device);

    uint32_t cb_tiles = std::max(8u, config.num_tiles);
    tt_metal::CircularBufferConfig cb_src_config =
        tt_metal::CircularBufferConfig(cb_tiles * tile_byte_size, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, tile_byte_size);
    tt_metal::CircularBufferConfig cb_out_config =
        tt_metal::CircularBufferConfig(cb_tiles * out_tile_byte_size, {{tt::CBIndex::c_16, out_format}})
            .set_page_size(tt::CBIndex::c_16, out_tile_byte_size);
    if (tiny_tile) {
        // Advertise the 16x32 tile geometry so the compute kernel derives
        // num_faces=2 from the operand CBs via get_operand_num_faces().
        cb_src_config.set_tile_dims(tt::CBIndex::c_0, cb_tile);
        cb_out_config.set_tile_dims(tt::CBIndex::c_16, cb_tile);
    }
    tt_metal::CreateCircularBuffer(program, core, cb_src_config);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto sum_reduce_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/sum_reduce_scalar.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = config.math_fidelity,
            .fp32_dest_acc_en = config.fp32_dest_acc,
            .dst_full_sync_en = config.dst_full_sync});

    SetRuntimeArgs(
        program,
        unary_reader_kernel,
        core,
        {src_dram_buffer->address(),
         0,
         config.num_tiles,
         tt::CBIndex::c_0,
         1 /*ublock_size_tiles*/,
         0 /*reader_only*/});

    SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0, 1});

    SetRuntimeArgs(program, sum_reduce_kernel, core, {config.num_tiles, std::bit_cast<uint32_t>(config.scaler)});

    uint32_t byte_size = config.num_tiles * tile_byte_size;
    std::vector<uint32_t> packed_input;
    if (config.use_constant_input) {
        const uint32_t packed_value =
            (static_cast<uint32_t>(config.constant_input_bits) << 16) | config.constant_input_bits;
        packed_input.assign(byte_size / sizeof(uint32_t), packed_value);
    } else {
        packed_input = test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
            0, 1.0f, byte_size / sizeof(bfloat16), config.seed);
    }

    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, packed_input, /*blocking=*/true);

    // Wrap the program into a MeshWorkload and dispatch via the mesh command queue.
    // This path works under both fast dispatch and slow dispatch, unlike detail::LaunchProgram.
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, /*blocking=*/true);

    auto u16_src_vec = u16_from_u32_vector(packed_input);

    float golden_scalar = 0.0f;
    if (!config.expect_exact_zero) {
        for (uint16_t packed : u16_src_vec) {
            golden_scalar += static_cast<float>(std::bit_cast<bfloat16>(packed));
        }
        // The scaler sits in SrcB for both GAPOOL passes (column accumulate and final
        // collapse), so it multiplies the result twice.
        golden_scalar *= config.scaler * config.scaler;
    }

    // The reduced scalar lives in element [0] of the output tile, in the output CB's
    // format: raw fp32 with native fp32 DEST, otherwise the low bfloat16 of word 0.
    float device_scalar = config.fp32_dest_acc
                              ? std::bit_cast<float>(result_vec[0])
                              : static_cast<float>(std::bit_cast<bfloat16>(u16_from_u32_vector(result_vec)[0]));

    log_info(
        LogTest,
        "num_tiles={}, tile_height={}, fp32_dest_acc={}, full_sync={}, fidelity={}, scaler={}: "
        "Golden={}, Device={}, Diff={}",
        config.num_tiles,
        config.tile_height,
        config.fp32_dest_acc,
        config.dst_full_sync,
        static_cast<int>(config.math_fidelity),
        config.scaler,
        golden_scalar,
        device_scalar,
        std::abs(device_scalar - golden_scalar));

    // LoFi truncates the SrcA/SrcB mantissas feeding both GAPOOL passes. With
    // all-positive stimuli that truncation is one-directional, so the reduced scalar
    // comes out ~3% low no matter how many tiles are summed -- a per-element precision
    // effect, not accumulation drift. Budget for it rather than asserting HiFi
    // precision at LoFi; the tile-count-independence is what rules out a real bug, and
    // blaze absorbs the bias because layernorm subtracts the mean and divides by the
    // variance, so a common scale error largely cancels.
    const float rel_tol = (config.math_fidelity == MathFidelity::LoFi) ? 0.05f : 0.01f;
    const float abs_tol = 0.01f;
    const float tolerance = std::max(rel_tol * std::abs(golden_scalar), abs_tol);
    if (config.expect_exact_zero) {
        return device_scalar == 0.0f;
    }
    return std::abs(device_scalar - golden_scalar) < tolerance;
}

}  // namespace tt::tt_metal::unit_tests::compute::sum_reduce_scalar

using namespace tt::tt_metal::unit_tests::compute::sum_reduce_scalar;

// Runs on any single card (Wormhole or Blackhole): sum_reduce_scalar builds on
// mul_reduce_scalar's reduce tail, which is supported on both architectures.
//
// Match Blaze LayerNorm's interpreted-tile selection exactly. It picks the tallest
// legal height in {32, 16, 8, 4, 2, 1} that covers the width with at most eight
// whole tiles, so only the height/count pairs below are generated today.
class SumReduceScalarBlazeShapeTest : public LLKMeshDeviceSingleCardFixture,
                                      public testing::WithParamInterface<SumReduceScalarConfig> {};

TEST_P(SumReduceScalarBlazeShapeTest, SumReduceScalarBlazeShape) {
    auto& mesh_device = *devices_[0];
    ASSERT_TRUE(run_sum_reduce_scalar_test(mesh_device, GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    SumReduceScalarBlazeShapes,
    SumReduceScalarBlazeShapeTest,
    testing::Values(
        SumReduceScalarConfig{.num_tiles = 1, .tile_height = 1},
        SumReduceScalarConfig{.num_tiles = 1, .tile_height = 2},
        SumReduceScalarConfig{.num_tiles = 3, .tile_height = 2},
        SumReduceScalarConfig{.num_tiles = 5, .tile_height = 2},
        SumReduceScalarConfig{.num_tiles = 7, .tile_height = 2},
        SumReduceScalarConfig{.num_tiles = 1, .tile_height = 4},
        SumReduceScalarConfig{.num_tiles = 3, .tile_height = 4},
        SumReduceScalarConfig{.num_tiles = 5, .tile_height = 4},
        SumReduceScalarConfig{.num_tiles = 7, .tile_height = 4},
        SumReduceScalarConfig{.num_tiles = 1, .tile_height = 8},
        SumReduceScalarConfig{.num_tiles = 3, .tile_height = 8},
        SumReduceScalarConfig{.num_tiles = 5, .tile_height = 8},
        SumReduceScalarConfig{.num_tiles = 7, .tile_height = 8},
        SumReduceScalarConfig{.num_tiles = 1, .tile_height = 16},
        SumReduceScalarConfig{.num_tiles = 3, .tile_height = 16},
        SumReduceScalarConfig{.num_tiles = 1, .tile_height = 32},
        SumReduceScalarConfig{.num_tiles = 2, .tile_height = 32},
        SumReduceScalarConfig{.num_tiles = 4, .tile_height = 32},
        SumReduceScalarConfig{.num_tiles = 8, .tile_height = 32}),
    [](const testing::TestParamInfo<SumReduceScalarConfig>& info) {
        return "SumReduceScalar_" + std::to_string(info.param.tile_height) + "x32_" +
               std::to_string(info.param.num_tiles) + "_Tiles";
    });

// Native fp32 DEST with full sync, parametrized by tile count. A separate instantiation
// of the op: DST_ACCUM_MODE and DST_SYNC_MODE are both threaded into the reduce init and
// the SFPU fills. blaze couples the two flags (compiler.py:
// dst_full_sync_en=fp32_dest_acc_en) and never uses the other diagonal. Full-sync 32-bit
// DEST holds 8 tiles, which is what blaze's widest layernorm asks for (width 8192, one
// 32x32 tile per 1024 elements).
class SumReduceScalarFp32DestTest : public LLKMeshDeviceSingleCardFixture, public testing::WithParamInterface<int> {};

TEST_P(SumReduceScalarFp32DestTest, SumReduceScalarFp32Dest) {
    auto& mesh_device = *devices_[0];
    int num_tiles = GetParam();
    ASSERT_TRUE(run_sum_reduce_scalar_test(
        mesh_device, {.num_tiles = num_tiles, .tile_height = 32, .fp32_dest_acc = true, .dst_full_sync = true}));
}

INSTANTIATE_TEST_SUITE_P(
    SumReduceScalarFp32DestTests,
    SumReduceScalarFp32DestTest,
    testing::Values(1, 8),
    [](const testing::TestParamInfo<int>& info) {
        return "SumReduceScalar_fp32dest_" + std::to_string(info.param) + "_Tiles";
    });

// LoFi is what blaze runs this op at (LayerNorm.math_fidelity = "LoFi"), and
// MATH_FIDELITY parametrizes both GAPOOL passes, so it is its own instantiation. Run it
// on both DEST widths; the fp32 case is blaze's exact layernorm configuration.
class SumReduceScalarLoFiTest : public LLKMeshDeviceSingleCardFixture, public testing::WithParamInterface<bool> {};

TEST_P(SumReduceScalarLoFiTest, SumReduceScalarLoFi) {
    auto& mesh_device = *devices_[0];
    bool fp32_dest_acc = GetParam();
    ASSERT_TRUE(run_sum_reduce_scalar_test(
        mesh_device,
        {.num_tiles = 8,
         .tile_height = 32,
         .fp32_dest_acc = fp32_dest_acc,
         .dst_full_sync = fp32_dest_acc,
         .math_fidelity = MathFidelity::LoFi}));
}

INSTANTIATE_TEST_SUITE_P(
    SumReduceScalarLoFiTests, SumReduceScalarLoFiTest, testing::Bool(), [](const testing::TestParamInfo<bool>& info) {
        return std::string("SumReduceScalar_LoFi_8_Tiles_") + (info.param ? "fp32dest" : "bf16dest");
    });

// The scaler is what separates this from a plain sum, and it lands on the result twice.
// One value either side of 1.0 pins the square. The doubling case also runs on fp32
// DEST, since the scaler arrives through an SFPU fill whose access width is selected by
// DST_ACCUM_MODE.
class SumReduceScalarScalerTest : public LLKMeshDeviceSingleCardFixture, public testing::WithParamInterface<bool> {};

TEST_P(SumReduceScalarScalerTest, SumReduceScalarScaler) {
    auto& mesh_device = *devices_[0];
    bool doubling = GetParam();
    ASSERT_TRUE(run_sum_reduce_scalar_test(
        mesh_device,
        {.num_tiles = 4,
         .tile_height = 32,
         .scaler = doubling ? 2.0f : 0.5f,
         .fp32_dest_acc = doubling,
         .dst_full_sync = doubling}));
}

INSTANTIATE_TEST_SUITE_P(
    SumReduceScalarScalerTests,
    SumReduceScalarScalerTest,
    testing::Bool(),
    [](const testing::TestParamInfo<bool>& info) {
        return info.param ? "SumReduceScalar_Scaler_2_fp32dest" : "SumReduceScalar_Scaler_0p5";
    });

class SumReduceScalarZeroFlagTest : public LLKMeshDeviceSingleCardFixture {};

TEST_F(SumReduceScalarZeroFlagTest, RestoresDefaultAfterCopyBeforeDenormalScaler) {
    auto& mesh_device = *devices_[0];
    constexpr uint16_t largest_finite_bfloat16 = 0x7f7f;
    const float denormal_scaler = static_cast<float>(std::bit_cast<bfloat16>(uint16_t{1}));

    ASSERT_TRUE(run_sum_reduce_scalar_test(
        mesh_device,
        {.num_tiles = 1,
         .tile_height = 32,
         .scaler = denormal_scaler,
         .use_constant_input = true,
         .constant_input_bits = largest_finite_bfloat16,
         .expect_exact_zero = true}));
}
