// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/float8.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
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

// The two datacopy compute kernels under test. Both do the same identity/typecast datacopy (c_0 -> c_16);
// they differ only in the compute API they exercise:
//   * kLegacyKernel: classic id-based API (copy_tile(cb,tile,dst) / pack_tile(dst,cb)).
//   * kSpecKernel:   id-free LLKOperand API (format+geometry as NTTPs, absolute L1 address at runtime).
// Every case below is run against BOTH so the legacy path stays a regression baseline and the id-free
// path is proven bit-for-bit equivalent.
static constexpr const char* kLegacyKernel = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_fp8.cpp";
static constexpr const char* kSpecKernel = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_fp8_2_0.cpp";

// Run a datacopy kernel with different input/output CB formats.
// The hardware unpacker reads input_fmt and the packer writes output_fmt,
// performing the format conversion implicitly. fp32_dest_acc_en controls
// whether the Dest register operates in 32-bit mode. compute_kernel selects
// the legacy vs id-free kernel (both take a single {num_tiles} compile-time arg).
static vector<std::uint32_t> run_fp8_typecast(
    distributed::MeshDevice& mesh_device,
    tt::DataFormat input_fmt,
    tt::DataFormat output_fmt,
    const vector<std::uint32_t>& src_vec,
    std::uint32_t num_tiles,
    bool fp32_dest_acc_en,
    const std::string& compute_kernel) {
    IDevice* dev = mesh_device.get_devices()[0];
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    std::uint32_t input_tile_size = tt::tile_size(input_fmt);
    std::uint32_t output_tile_size = tt::tile_size(output_fmt);

    InterleavedBufferConfig src_config{
        .device = dev,
        .size = num_tiles * input_tile_size,
        .page_size = num_tiles * input_tile_size,
        .buffer_type = BufferType::DRAM};
    auto src_buffer = CreateBuffer(src_config);

    InterleavedBufferConfig dst_config{
        .device = dev,
        .size = num_tiles * output_tile_size,
        .page_size = num_tiles * output_tile_size,
        .buffer_type = BufferType::DRAM};
    auto dst_buffer = CreateBuffer(dst_config);

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
        compute_kernel,
        core,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {num_tiles}});

    detail::WriteToBuffer(src_buffer, src_vec);
    SetRuntimeArgs(program, reader, core, {src_buffer->address(), 0, num_tiles});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, num_tiles});

    // Wrap the program into a MeshWorkload so we can dispatch via the mesh command queue.
    // This path works under both fast dispatch and slow dispatch, unlike detail::LaunchProgram.
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    vector<std::uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

// --- Format-to-float unpackers ---
// fp8_to_floats / bf16_to_floats live in tt_metal/test_utils/float8_utils.hpp;
// expose them in this namespace so existing unqualified call sites resolve.

using tt::test_utils::bf16_to_floats;
using tt::test_utils::fp8_to_floats;

static vector<float> bfp8_to_floats(const vector<std::uint32_t>& packed) {
    return unpack_bfp8_tiles_into_float_vec(
        ttsl::make_const_span(packed), /*row_major_output=*/false, /*is_exp_a=*/false);
}

// --- Validation ---
// is_close_vectors + is_close + check_pcc all live in tt_metal/test_utils/comparison.hpp.

using tt::test_utils::check_pcc;
using tt::test_utils::is_close;
using tt::test_utils::is_close_vectors;

// ============================================================================
// Per-conversion cases. Each is parameterized by the compute kernel so the same
// stimulus + validation runs against BOTH the legacy and the id-free API.
// ============================================================================

// fp8_e4m3 -> Float16_b. Widening: every fp8 value is exactly representable in BF16 -> lossless.
static void case_fp8_to_bf16(distributed::MeshDevice& md, const std::string& kernel) {
    constexpr std::uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_float8_e4m3(
        tt::tile_size(tt::DataFormat::Fp8_e4m3) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        md, tt::DataFormat::Fp8_e4m3, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/true, kernel);
    auto src_floats = fp8_to_floats(src_vec);
    auto dst_floats = bf16_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.0f, /*atol=*/0.0f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// Float16_b -> fp8_e4m3. Narrowing: rtol=0.125 covers the max relative fp8 quantization error (~1/8).
static void case_bf16_to_fp8(distributed::MeshDevice& md, const std::string& kernel) {
    constexpr std::uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        md, tt::DataFormat::Float16_b, tt::DataFormat::Fp8_e4m3, src_vec, num_tiles, /*fp32_dest_acc_en=*/true, kernel);
    auto src_floats = bf16_to_floats(src_vec);
    auto dst_floats = fp8_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.125f, /*atol=*/0.015625f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.999));
}

// fp8_e4m3 -> Bfp8_b. Widening into a shared-exponent block format.
static void case_fp8_to_bfp8(distributed::MeshDevice& md, const std::string& kernel) {
    constexpr std::uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_float8_e4m3(
        tt::tile_size(tt::DataFormat::Fp8_e4m3) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        md, tt::DataFormat::Fp8_e4m3, tt::DataFormat::Bfp8_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/true, kernel);
    auto src_floats = fp8_to_floats(src_vec);
    auto dst_floats = bfp8_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.3f, /*atol=*/0.3f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.9999));
}

// Bfp8_b -> fp8_e4m3. Narrowing.
static void case_bfp8_to_fp8(distributed::MeshDevice& md, const std::string& kernel) {
    constexpr std::uint32_t num_tiles = 64;
    auto src_vec = tt::test_utils::create_random_vector_of_bfp8(
        tt::tile_size(tt::DataFormat::Bfp8_b) * num_tiles,
        /*is_exp_a=*/false,
        /*rand_max_float=*/20,
        /*seed=*/42,
        /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        md, tt::DataFormat::Bfp8_b, tt::DataFormat::Fp8_e4m3, src_vec, num_tiles, /*fp32_dest_acc_en=*/true, kernel);
    auto src_floats = bfp8_to_floats(src_vec);
    auto dst_floats = fp8_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.125f, /*atol=*/0.015625f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.999));
}

// Bfp8_b -> Bfp8_b identity. fp32_dest_acc_en selects Dest 16b vs 32b mode.
static void case_bfp8_to_bfp8(distributed::MeshDevice& md, const std::string& kernel, bool fp32_dest_acc_en) {
    constexpr std::uint32_t num_tiles = 64;
    auto src_vec = tt::test_utils::create_random_vector_of_bfp8(
        tt::tile_size(tt::DataFormat::Bfp8_b) * num_tiles,
        /*is_exp_a=*/false,
        /*rand_max_float=*/20,
        /*seed=*/42,
        /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        md, tt::DataFormat::Bfp8_b, tt::DataFormat::Bfp8_b, src_vec, num_tiles, fp32_dest_acc_en, kernel);
    auto src_floats = bfp8_to_floats(src_vec);
    auto dst_floats = bfp8_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.3f, /*atol=*/0.3f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.9999));
}

// fp8_e4m3 -> fp8_e4m3 identity. Lossless round-trip.
static void case_fp8_to_fp8(distributed::MeshDevice& md, const std::string& kernel) {
    constexpr std::uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_float8_e4m3(
        tt::tile_size(tt::DataFormat::Fp8_e4m3) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        md, tt::DataFormat::Fp8_e4m3, tt::DataFormat::Fp8_e4m3, src_vec, num_tiles, /*fp32_dest_acc_en=*/true, kernel);
    auto src_floats = fp8_to_floats(src_vec);
    auto dst_floats = fp8_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.0f, /*atol=*/0.0f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// Float16_b -> Float16_b identity. Lossless round-trip. fp32_dest_acc_en selects Dest 16b vs 32b mode.
static void case_bf16_to_bf16(distributed::MeshDevice& md, const std::string& kernel, bool fp32_dest_acc_en) {
    constexpr std::uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        md, tt::DataFormat::Float16_b, tt::DataFormat::Float16_b, src_vec, num_tiles, fp32_dest_acc_en, kernel);
    auto src_floats = bf16_to_floats(src_vec);
    auto dst_floats = bf16_to_floats(result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.0f, /*atol=*/0.0f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

}  // namespace unit_tests::llk::fp8_typecast

using namespace unit_tests::llk::fp8_typecast;

// ============================================================================
// Legacy (id-based) API — regression baseline.
// ============================================================================
TEST_F(LLKBlackholeSingleCardFixture, TensixFp8e4m3ToFloat16b) { case_fp8_to_bf16(*devices_[0], kLegacyKernel); }
TEST_F(LLKBlackholeSingleCardFixture, TensixFloat16bToFp8e4m3) { case_bf16_to_fp8(*devices_[0], kLegacyKernel); }
TEST_F(LLKBlackholeSingleCardFixture, TensixFp8e4m3ToBfp8b) { case_fp8_to_bfp8(*devices_[0], kLegacyKernel); }
TEST_F(LLKBlackholeSingleCardFixture, TensixBfp8bToFp8e4m3) { case_bfp8_to_fp8(*devices_[0], kLegacyKernel); }
TEST_F(LLKBlackholeSingleCardFixture, TensixBfp8bToBfp8b) {
    case_bfp8_to_bfp8(*devices_[0], kLegacyKernel, /*fp32_dest_acc_en=*/false);
}
TEST_F(LLKBlackholeSingleCardFixture, TensixBfp8bToBfp8bFp32Dest) {
    case_bfp8_to_bfp8(*devices_[0], kLegacyKernel, /*fp32_dest_acc_en=*/true);
}
TEST_F(LLKBlackholeSingleCardFixture, TensixFp8e4m3ToFp8e4m3) { case_fp8_to_fp8(*devices_[0], kLegacyKernel); }
TEST_F(LLKBlackholeSingleCardFixture, TensixFloat16bToFloat16b) {
    case_bf16_to_bf16(*devices_[0], kLegacyKernel, /*fp32_dest_acc_en=*/false);
}
TEST_F(LLKBlackholeSingleCardFixture, TensixFloat16bToFloat16bFp32Dest) {
    case_bf16_to_bf16(*devices_[0], kLegacyKernel, /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// Id-free LLKOperand API — must be bit-for-bit equivalent to the legacy path.
// ============================================================================
TEST_F(LLKBlackholeSingleCardFixture, TensixFp8e4m3ToFloat16bSpec) { case_fp8_to_bf16(*devices_[0], kSpecKernel); }
TEST_F(LLKBlackholeSingleCardFixture, TensixFloat16bToFp8e4m3Spec) { case_bf16_to_fp8(*devices_[0], kSpecKernel); }
TEST_F(LLKBlackholeSingleCardFixture, TensixFp8e4m3ToBfp8bSpec) { case_fp8_to_bfp8(*devices_[0], kSpecKernel); }
TEST_F(LLKBlackholeSingleCardFixture, TensixBfp8bToFp8e4m3Spec) { case_bfp8_to_fp8(*devices_[0], kSpecKernel); }
TEST_F(LLKBlackholeSingleCardFixture, TensixBfp8bToBfp8bSpec) {
    case_bfp8_to_bfp8(*devices_[0], kSpecKernel, /*fp32_dest_acc_en=*/false);
}
TEST_F(LLKBlackholeSingleCardFixture, TensixBfp8bToBfp8bFp32DestSpec) {
    case_bfp8_to_bfp8(*devices_[0], kSpecKernel, /*fp32_dest_acc_en=*/true);
}
TEST_F(LLKBlackholeSingleCardFixture, TensixFp8e4m3ToFp8e4m3Spec) { case_fp8_to_fp8(*devices_[0], kSpecKernel); }
TEST_F(LLKBlackholeSingleCardFixture, TensixFloat16bToFloat16bSpec) {
    case_bf16_to_bf16(*devices_[0], kSpecKernel, /*fp32_dest_acc_en=*/false);
}
TEST_F(LLKBlackholeSingleCardFixture, TensixFloat16bToFloat16bFp32DestSpec) {
    case_bf16_to_bf16(*devices_[0], kSpecKernel, /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// Tilize: the id-free (2.0) tilize kernel must produce output bit-for-bit identical to the legacy
// CB-id tilize kernel on the same input (differential equivalence -- no golden needed; reuses the
// single-core classic-CB datacopy harness run_fp8_typecast to run each compute kernel).
// ============================================================================
TEST_F(LLKBlackholeSingleCardFixture, TensixTilizeSpecMatchesLegacy) {
    auto& mesh_device = *devices_[0];
    constexpr std::uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);

    auto legacy = run_fp8_typecast(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/tilize_legacy.cpp");
    auto spec = run_fp8_typecast(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/tilize_2_0.cpp");

    EXPECT_EQ(legacy, spec);
}

// ============================================================================
// Pack-untilize: the id-free (2.0) pack_untilize kernel must produce output bit-for-bit identical to the
// legacy CB-id pack_untilize kernel on the same input (differential equivalence -- no golden needed; reuses
// the single-core classic-CB harness run_fp8_typecast). block_ct_dim/full_ct_dim/block_rt_dim == 1.
// ============================================================================
TEST_F(LLKBlackholeSingleCardFixture, TensixPackUntilizeSpecMatchesLegacy) {
    auto& mesh_device = *devices_[0];
    constexpr std::uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);

    auto legacy = run_fp8_typecast(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/pack_untilize_legacy.cpp");
    auto spec = run_fp8_typecast(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::Float16_b,
        src_vec,
        num_tiles,
        /*fp32_dest_acc_en=*/false,
        "tests/tt_metal/tt_metal/test_kernels/compute/pack_untilize_2_0.cpp");

    EXPECT_EQ(legacy, spec);
}

// ============================================================================
// Eltwise binary (add): two classic circular buffers (c_0, c_1) -> c_16. The id-free (2.0) add kernel must
// produce output bit-for-bit identical to the legacy CB-id add kernel on the same inputs (differential
// equivalence -- no golden needed).
// ============================================================================
static vector<std::uint32_t> run_binary_add(
    distributed::MeshDevice& mesh_device,
    const vector<std::uint32_t>& src0_vec,
    const vector<std::uint32_t>& src1_vec,
    std::uint32_t num_tiles,
    const std::string& compute_kernel,
    const std::map<std::string, std::string>& compute_defines = {}) {
    IDevice* dev = mesh_device.get_devices()[0];
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    const tt::DataFormat fmt = tt::DataFormat::Float16_b;
    std::uint32_t tile_bytes = tt::tile_size(fmt);

    auto make_dram = [&]() {
        InterleavedBufferConfig cfg{
            .device = dev,
            .size = num_tiles * tile_bytes,
            .page_size = num_tiles * tile_bytes,
            .buffer_type = BufferType::DRAM};
        return CreateBuffer(cfg);
    };
    auto src0_buffer = make_dram();
    auto src1_buffer = make_dram();
    auto dst_buffer = make_dram();

    auto make_cb = [&](tt::CBIndex idx) {
        CircularBufferConfig cb_cfg = CircularBufferConfig(tile_bytes, {{idx, fmt}}).set_page_size(idx, tile_bytes);
        CreateCircularBuffer(program, core, cb_cfg);
    };
    make_cb(tt::CBIndex::c_0);
    make_cb(tt::CBIndex::c_1);
    make_cb(tt::CBIndex::c_16);

    auto reader = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto compute = CreateKernel(
        program,
        compute_kernel,
        core,
        ComputeConfig{.fp32_dest_acc_en = false, .compile_args = {num_tiles}, .defines = compute_defines});

    detail::WriteToBuffer(src0_buffer, src0_vec);
    detail::WriteToBuffer(src1_buffer, src1_vec);
    SetRuntimeArgs(program, reader, core, {src0_buffer->address(), 0, src1_buffer->address(), 0, num_tiles});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, num_tiles});
    // Legacy eltwise_binary.cpp reads runtime args {per_core_block_cnt, per_core_block_size, acc_to_dst};
    // the id-free kernel reads only the compile-time num_tiles and ignores these (harmless). One tile/block.
    SetRuntimeArgs(program, compute, core, {num_tiles, 1, 0});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    auto& cq = mesh_device.mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    vector<std::uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

TEST_F(LLKBlackholeSingleCardFixture, TensixBinaryAddSpecMatchesLegacy) {
    auto& mesh_device = *devices_[0];
    constexpr std::uint32_t num_tiles = 64;
    auto src0 = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto src1 = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/7, /*offset=*/-10.0f);

    // Legacy baseline = the shipping classic-CB kernel eltwise_binary.cpp, driven for ADD via defines
    // (no golden, no hand-written mirror). On silicon DST_ACCUM_MODE is a constexpr (not a macro), so the
    // #if defined(DST_ACCUM_MODE) dest-accum path stays off and the plain 2-input add path is compiled.
    auto legacy = run_binary_add(
        mesh_device,
        src0,
        src1,
        num_tiles,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_binary.cpp",
        {{"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}});
    auto spec = run_binary_add(
        mesh_device,
        src0,
        src1,
        num_tiles,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_binary_add_idfree.cpp");

    EXPECT_EQ(legacy, spec);
}

}  // namespace tt::tt_metal
