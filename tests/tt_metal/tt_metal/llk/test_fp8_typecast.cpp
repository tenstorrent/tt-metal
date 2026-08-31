// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/float8.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/span.hpp>
#include <tt-logger/tt-logger.hpp>
#include "llk_device_fixture.hpp"
#include "tt_metal/test_utils/bfloat_utils.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/float8_utils.hpp"

namespace tt::tt_metal {

using std::vector;

namespace unit_tests::llk::fp8_typecast {

// Run a datacopy kernel with different input/output CB formats.
// The hardware unpacker reads input_fmt and the packer writes output_fmt,
// performing the format conversion implicitly. fp32_dest_acc_en controls
// whether the Dest register operates in 32-bit mode.
static vector<uint32_t> run_fp8_typecast(
    distributed::MeshDevice& mesh_device,
    tt::DataFormat input_fmt,
    tt::DataFormat output_fmt,
    const vector<uint32_t>& src_vec,
    uint32_t num_tiles,
    bool fp32_dest_acc_en) {
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t input_tile_size = tt::tile_size(input_fmt);
    uint32_t output_tile_size = tt::tile_size(output_fmt);

    auto src_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = num_tiles * input_tile_size},
        {.page_size = num_tiles * input_tile_size, .buffer_type = BufferType::DRAM},
        &mesh_device);

    auto dst_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = num_tiles * output_tile_size},
        {.page_size = num_tiles * output_tile_size, .buffer_type = BufferType::DRAM},
        &mesh_device);

    CircularBufferConfig cb_src_config = CircularBufferConfig(input_tile_size, {{tt::CBIndex::c_0, input_fmt}})
                                             .set_page_size(tt::CBIndex::c_0, input_tile_size);
    CreateCircularBuffer(program, core, cb_src_config);

    CircularBufferConfig cb_dst_config = CircularBufferConfig(output_tile_size, {{tt::CBIndex::c_16, output_fmt}})
                                             .set_page_size(tt::CBIndex::c_16, output_tile_size);
    CreateCircularBuffer(program, core, cb_dst_config);

    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_fp8.cpp",
        core,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {num_tiles}});

    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, src_buffer, src_vec, /*blocking=*/true);
    SetRuntimeArgs(program, reader, core, {src_buffer->address(), 0, num_tiles});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, num_tiles});

    // Wrap the program into a MeshWorkload so we can dispatch via the mesh command queue.
    // This path works under both fast dispatch and slow dispatch, unlike detail::LaunchProgram.
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    vector<uint32_t> result_vec;
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_buffer, /*blocking=*/true);
    return result_vec;
}

// --- Format-to-float unpackers ---
// fp8_to_floats / bf16_to_floats live in tt_metal/test_utils/float8_utils.hpp;
// expose them in this namespace so existing unqualified call sites resolve.

using tt::test_utils::bf16_to_floats;
using tt::test_utils::fp8_to_floats;

static vector<float> bfp8_to_floats(const vector<uint32_t>& packed) {
    return unpack_bfp8_tiles_into_float_vec(
        ttsl::make_const_span(packed), /*row_major_output=*/false, /*is_exp_a=*/false);
}

// --- Validation ---
// is_close_vectors + is_close + check_pcc all live in tt_metal/test_utils/comparison.hpp.

using tt::test_utils::check_pcc;
using tt::test_utils::is_close;
using tt::test_utils::is_close_vectors;

struct LocalFp32EpochResult {
    vector<uint32_t> fp32;
    vector<uint32_t> bf16;
};

static LocalFp32EpochResult run_local_fp32_epoch(
    distributed::MeshDevice& mesh_device,
    const vector<uint32_t>& fp8_input,
    const vector<uint32_t>& fp32_scale,
    const vector<uint32_t>& bf16_input) {
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    constexpr auto fp8_cb = tt::CBIndex::c_0;
    constexpr auto fp32_scale_cb = tt::CBIndex::c_1;
    constexpr auto bf16_cb = tt::CBIndex::c_2;
    constexpr auto fp32_out_cb = tt::CBIndex::c_16;
    constexpr auto bf16_out_cb = tt::CBIndex::c_17;

    const uint32_t fp8_tile_size = tt::tile_size(tt::DataFormat::Fp8_e4m3);
    const uint32_t fp32_tile_size = tt::tile_size(tt::DataFormat::Float32);
    const uint32_t bf16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    auto make_dram_buffer = [&](uint32_t size) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = size},
            {.page_size = size, .buffer_type = BufferType::DRAM},
            &mesh_device);
    };

    auto fp8_buffer = make_dram_buffer(fp8_tile_size);
    auto fp32_scale_buffer = make_dram_buffer(fp32_tile_size);
    auto bf16_buffer = make_dram_buffer(bf16_tile_size);
    auto fp32_out_buffer = make_dram_buffer(fp32_tile_size);
    auto bf16_out_buffer = make_dram_buffer(bf16_tile_size);

    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(fp8_tile_size, {{fp8_cb, tt::DataFormat::Fp8_e4m3}}).set_page_size(fp8_cb, fp8_tile_size));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(fp32_tile_size, {{fp32_scale_cb, tt::DataFormat::Float32}})
            .set_page_size(fp32_scale_cb, fp32_tile_size));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(bf16_tile_size, {{bf16_cb, tt::DataFormat::Float16_b}})
            .set_page_size(bf16_cb, bf16_tile_size));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(fp32_tile_size, {{fp32_out_cb, tt::DataFormat::Float32}})
            .set_page_size(fp32_out_cb, fp32_tile_size));
    CreateCircularBuffer(
        program,
        core,
        CircularBufferConfig(bf16_tile_size, {{bf16_out_cb, tt::DataFormat::Float16_b}})
            .set_page_size(bf16_out_cb, bf16_tile_size));

    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .defines = {{"LOAD_BUF2_DATA", "1"}}});
    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_dual_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[fp32_scale_cb] = UnpackToDestMode::UnpackToDestFp32;
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/fp8_local_fp32_epoch.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = false,
            .enable_local_fp32_dest_epoch = true,
            .unpack_to_dest_mode = unpack_to_dest_mode});

    SetRuntimeArgs(
        program,
        reader,
        core,
        {
            fp8_buffer->address(),
            0,
            fp32_scale_buffer->address(),
            0,
            1,
            bf16_buffer->address(),
            0,
        });
    SetRuntimeArgs(
        program,
        writer,
        core,
        {
            fp32_out_buffer->address(),
            0,
            bf16_out_buffer->address(),
            0,
            1,
        });

    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, fp8_buffer, fp8_input, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, fp32_scale_buffer, fp32_scale, /*blocking=*/true);
    distributed::EnqueueWriteMeshBuffer(cq, bf16_buffer, bf16_input, /*blocking=*/true);

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    LocalFp32EpochResult result;
    distributed::EnqueueReadMeshBuffer(cq, result.fp32, fp32_out_buffer, /*blocking=*/true);
    distributed::EnqueueReadMeshBuffer(cq, result.bf16, bf16_out_buffer, /*blocking=*/true);
    return result;
}

}  // namespace unit_tests::llk::fp8_typecast

using namespace unit_tests::llk::fp8_typecast;

// ============================================================================
// fp8_e4m3 → Float16_b
// Widening conversion: every fp8 value is exactly representable in BF16.
// Expected: no precision loss → rtol=0.0, atol=0.0.
// ============================================================================

TEST_F(LLKBlackholeSingleCardFixture, TensixFp8e4m3ToFloat16b) {
    auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_float8_e4m3(
        tt::tile_size(tt::DataFormat::Fp8_e4m3) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        mesh_device,
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Float16_b,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/true);  // BH: Fp8 requires fp32_dest_acc_en=true (JIT-enforced)
    auto src_floats = fp8_to_floats(src_vec);
    auto dst_floats = bf16_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.0f, /*atol=*/0.0f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(LLKBlackholeSingleCardFixture, TensixFp8LocalFp32EpochRestoresBf16Dest) {
    auto& mesh_device = *devices_[0];
    constexpr float scale = 1.000244140625f;  // 1 + 2^-12: retained by FP32, rounded away by TF32.

    auto fp8_input = create_random_vector_of_float8_e4m3(
        tt::tile_size(tt::DataFormat::Fp8_e4m3), /*rand_max_float=*/16, /*seed=*/42, /*offset=*/-8.0f);
    vector<uint32_t> fp32_scale(
        tt::tile_size(tt::DataFormat::Float32) / sizeof(uint32_t), std::bit_cast<uint32_t>(scale));
    auto bf16_input = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b), /*rand_max_float=*/4, /*seed=*/17, /*offset=*/-2.0f);

    const auto result = run_local_fp32_epoch(mesh_device, fp8_input, fp32_scale, bf16_input);

    const auto fp8_values = fp8_to_floats(fp8_input);
    vector<float> expected_fp32(fp8_values.size());
    std::transform(
        fp8_values.begin(), fp8_values.end(), expected_fp32.begin(), [](float value) { return value * scale; });

    vector<float> actual_fp32;
    actual_fp32.reserve(result.fp32.size());
    for (uint32_t word : result.fp32) {
        actual_fp32.push_back(std::bit_cast<float>(word));
    }
    EXPECT_TRUE(is_close_vectors<float>(
        expected_fp32, actual_fp32, [](float expected, float actual) { return expected == actual; }));

    auto expected_bf16 = bf16_to_floats(bf16_input);
    std::transform(expected_bf16.begin(), expected_bf16.end(), expected_bf16.begin(), [](float value) {
        return std::max(value, 0.0f);
    });
    const auto actual_bf16 = bf16_to_floats(result.bf16);
    EXPECT_TRUE(is_close_vectors<float>(
        expected_bf16, actual_bf16, [](float expected, float actual) { return expected == actual; }));
}

// ============================================================================
// Float16_b → fp8_e4m3
// Narrowing: BF16 has 7 mantissa bits vs fp8's 3 → precision loss expected.
// rtol=0.125 covers the max relative quantization error of fp8 (~1/8).
// ============================================================================

TEST_F(LLKBlackholeSingleCardFixture, TensixFloat16bToFp8e4m3) {
    auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::Fp8_e4m3,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/true);  // BH: Fp8 requires fp32_dest_acc_en=true (JIT-enforced)
    auto src_floats = bf16_to_floats(src_vec);
    auto dst_floats = fp8_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.125f, /*atol=*/0.015625f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.999));
}

// ============================================================================
// fp8_e4m3 → Bfp8_b
// Widening: Bfp8_b has 8 mantissa bits and a shared exponent per 16-element
// row. For test data within [-10, 10], fp8 values may lose significant
// precision due to the blocking forming process.
// ============================================================================

TEST_F(LLKBlackholeSingleCardFixture, TensixFp8e4m3ToBfp8b) {
    auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_float8_e4m3(
        tt::tile_size(tt::DataFormat::Fp8_e4m3) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        mesh_device,
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Bfp8_b,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/true);  // BH: Fp8 requires fp32_dest_acc_en=true (JIT-enforced)
    auto src_floats = fp8_to_floats(src_vec);
    auto dst_floats = bfp8_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.3f, /*atol=*/0.3f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.9999));
}

// ============================================================================
// Bfp8_b → fp8_e4m3
// Narrowing: Bfp8_b has 8 mantissa bits vs fp8's 3 → precision loss expected.
// ============================================================================

TEST_F(LLKBlackholeSingleCardFixture, TensixBfp8bToFp8e4m3) {
    auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = tt::test_utils::create_random_vector_of_bfp8(
        tt::tile_size(tt::DataFormat::Bfp8_b) * num_tiles,
        /*is_exp_a=*/false,
        /*rand_max_float=*/20,
        /*seed=*/42,
        /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        mesh_device,
        tt::DataFormat::Bfp8_b,
        tt::DataFormat::Fp8_e4m3,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/true);  // BH: Fp8 requires fp32_dest_acc_en=true (JIT-enforced)
    auto src_floats = bfp8_to_floats(src_vec);
    auto dst_floats = fp8_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.125f, /*atol=*/0.015625f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.999));
}

// ============================================================================
// Bfp8_b → Bfp8_b (identity)
// Same format on both sides. The unpack→Dest→repack round-trip through the
// shared-exponent blocking process may introduce minor rounding, but PCC
// should remain very high.
// ============================================================================

TEST_F(LLKBlackholeSingleCardFixture, TensixBfp8bToBfp8b) {
    auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = tt::test_utils::create_random_vector_of_bfp8(
        tt::tile_size(tt::DataFormat::Bfp8_b) * num_tiles,
        /*is_exp_a=*/false,
        /*rand_max_float=*/20,
        /*seed=*/42,
        /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        mesh_device, tt::DataFormat::Bfp8_b, tt::DataFormat::Bfp8_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = bfp8_to_floats(src_vec);
    auto dst_floats = bfp8_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.3f, /*atol=*/0.3f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.9999));
}

TEST_F(LLKBlackholeSingleCardFixture, TensixBfp8bToBfp8bFp32Dest) {
    auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = tt::test_utils::create_random_vector_of_bfp8(
        tt::tile_size(tt::DataFormat::Bfp8_b) * num_tiles,
        /*is_exp_a=*/false,
        /*rand_max_float=*/20,
        /*seed=*/42,
        /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        mesh_device, tt::DataFormat::Bfp8_b, tt::DataFormat::Bfp8_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = bfp8_to_floats(src_vec);
    auto dst_floats = bfp8_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.3f, /*atol=*/0.3f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.9999));
}

// ============================================================================
// fp8_e4m3 → fp8_e4m3 (identity)
// Same format on both sides. The round-trip should be lossless since every
// fp8 value survives the unpack→Dest→repack cycle exactly.
// ============================================================================

TEST_F(LLKBlackholeSingleCardFixture, TensixFp8e4m3ToFp8e4m3) {
    auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_float8_e4m3(
        tt::tile_size(tt::DataFormat::Fp8_e4m3) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        mesh_device,
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Fp8_e4m3,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/true);  // BH: Fp8 requires fp32_dest_acc_en=true (JIT-enforced)
    auto src_floats = fp8_to_floats(src_vec);
    auto dst_floats = fp8_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.0f, /*atol=*/0.0f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// ============================================================================
// Float16_b → Float16_b (identity)
// Same format on both sides. The round-trip should be lossless since every
// BF16 value survives the unpack→Dest→repack cycle exactly.
// ============================================================================

TEST_F(LLKBlackholeSingleCardFixture, TensixFloat16bToFloat16b) {
    auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/false);
    auto src_floats = bf16_to_floats(src_vec);
    auto dst_floats = bf16_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.0f, /*atol=*/0.0f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(LLKBlackholeSingleCardFixture, TensixFloat16bToFloat16bFp32Dest) {
    auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/true);
    auto src_floats = bf16_to_floats(src_vec);
    auto dst_floats = bf16_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.0f, /*atol=*/0.0f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

}  // namespace tt::tt_metal
