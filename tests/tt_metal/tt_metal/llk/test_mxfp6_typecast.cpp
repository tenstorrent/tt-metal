// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mxfp6.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/experimental/dataflow_buffer/dataflow_buffer.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include <tt-logger/tt-logger.hpp>

#include "device_fixture.hpp"

namespace tt::tt_metal {

using std::vector;

namespace unit_tests::llk::mxfp6_typecast {

// Run a datacopy kernel with different input/output formats. Mirrors the
// MXFP4 typecast harness — for Quasar, data is moved via DataflowBuffers
// (DFBs) and the hardware unpacker/packer performs the format conversion
// implicitly.
static vector<uint32_t> run_mxfp6_typecast(
    IDevice* dev,
    tt::DataFormat input_fmt,
    tt::DataFormat output_fmt,
    const vector<uint32_t>& src_vec,
    uint32_t num_tiles,
    bool fp32_dest_acc_en) {
    Program program = CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t input_tile_size = tt::tile_size(input_fmt);
    uint32_t output_tile_size = tt::tile_size(output_fmt);

    InterleavedBufferConfig src_config{
        .device = dev,
        .size = num_tiles * input_tile_size,
        .page_size = input_tile_size,
        .buffer_type = BufferType::DRAM};
    auto src_buffer = CreateBuffer(src_config);

    InterleavedBufferConfig dst_config{
        .device = dev,
        .size = num_tiles * output_tile_size,
        .page_size = output_tile_size,
        .buffer_type = BufferType::DRAM};
    auto dst_buffer = CreateBuffer(dst_config);

    tt_metal::experimental::dfb::DataflowBufferConfig l1_input_dfb_config = {
        .entry_size = input_tile_size,
        .num_entries = 2,
        .num_producers = 1,
        .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = true,
        .data_format = input_fmt};
    tt_metal::experimental::dfb::DataflowBufferConfig l1_output_dfb_config = {
        .entry_size = output_tile_size,
        .num_entries = 2,
        .num_producers = 1,
        .pap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .num_consumers = 1,
        .cap = tt_metal::experimental::dfb::AccessPattern::STRIDED,
        .enable_implicit_sync = true,
        .data_format = output_fmt};

    uint32_t l1_input_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program, core, l1_input_dfb_config);
    uint32_t l1_output_dfb = tt_metal::experimental::dfb::CreateDataflowBuffer(program, core, l1_output_dfb_config);

    KernelHandle reader = tt_metal::experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        core,
        tt_metal::experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1, .compile_args = {l1_input_dfb, /*use_dfbs=*/true}});

    KernelHandle writer = tt_metal::experimental::quasar::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        core,
        tt_metal::experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1, .compile_args = {l1_output_dfb, /*use_dfbs=*/true}});

    KernelHandle compute = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        core,
        tt_metal::experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 1,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = {num_tiles, /*use_dfbs=*/true}});

    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program, l1_input_dfb, reader, compute);
    tt_metal::experimental::dfb::BindDataflowBufferToProducerConsumerKernels(program, l1_output_dfb, compute, writer);

    detail::WriteToBuffer(src_buffer, src_vec);
    uint32_t src_dram_stride = static_cast<uint32_t>(src_buffer->aligned_page_size());
    uint32_t dst_dram_stride = static_cast<uint32_t>(dst_buffer->aligned_page_size());
    SetRuntimeArgs(program, reader, core, {src_buffer->address(), 0, num_tiles, src_dram_stride});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, num_tiles, dst_dram_stride});

    detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

// --- Random data generators ---

static vector<uint32_t> create_random_vector_of_mxfp6r(
    uint32_t num_bytes, int rand_max_float, int seed, float offset = 0.0f) {
    uint32_t single_tile_size = tt::tile_size(tt::DataFormat::MxFp6R);
    TT_FATAL(
        num_bytes % single_tile_size == 0,
        "num_bytes {} must be divisible by MXFP6R tile_size {}",
        num_bytes,
        single_tile_size);
    uint32_t num_tiles = num_bytes / single_tile_size;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(rand_max_float));

    constexpr uint32_t kNumFloatsPerTile = 1024;
    vector<float> fp32_vec(num_tiles * kNumFloatsPerTile);
    for (float& v : fp32_vec) {
        v = dist(rng) + offset;
    }

    vector<uint32_t> packed = pack_as_mxfp6r_tiles(tt::stl::make_const_span(fp32_vec), /*row_major_input=*/true);
    TT_FATAL(
        packed.size() * sizeof(uint32_t) == num_tiles * single_tile_size,
        "MXFP6R packed size {} bytes does not match expected {} bytes",
        packed.size() * sizeof(uint32_t),
        num_tiles * single_tile_size);
    return packed;
}

static vector<uint32_t> create_random_vector_of_mxfp6p(
    uint32_t num_bytes, int rand_max_float, int seed, float offset = 0.0f) {
    uint32_t single_tile_size = tt::tile_size(tt::DataFormat::MxFp6P);
    TT_FATAL(
        num_bytes % single_tile_size == 0,
        "num_bytes {} must be divisible by MXFP6P tile_size {}",
        num_bytes,
        single_tile_size);
    uint32_t num_tiles = num_bytes / single_tile_size;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(rand_max_float));

    constexpr uint32_t kNumFloatsPerTile = 1024;
    vector<float> fp32_vec(num_tiles * kNumFloatsPerTile);
    for (float& v : fp32_vec) {
        v = dist(rng) + offset;
    }

    vector<uint32_t> packed = pack_as_mxfp6p_tiles(tt::stl::make_const_span(fp32_vec), /*row_major_input=*/true);
    TT_FATAL(
        packed.size() * sizeof(uint32_t) == num_tiles * single_tile_size,
        "MXFP6P packed size {} bytes does not match expected {} bytes",
        packed.size() * sizeof(uint32_t),
        num_tiles * single_tile_size);
    return packed;
}

// --- Format-to-float unpackers ---

static vector<float> mxfp6r_to_floats(const vector<uint32_t>& packed) {
    return unpack_mxfp6r_tiles_into_float_vec(tt::stl::make_const_span(packed), /*row_major_output=*/false);
}

static vector<float> mxfp6p_to_floats(const vector<uint32_t>& packed) {
    return unpack_mxfp6p_tiles_into_float_vec(tt::stl::make_const_span(packed), /*row_major_output=*/false);
}

static vector<float> bf16_to_floats(const vector<uint32_t>& packed) {
    auto bf16_vec = unpack_uint32_vec_into_bfloat16_vec(packed);
    vector<float> floats;
    floats.reserve(bf16_vec.size());
    for (const auto& v : bf16_vec) {
        floats.push_back(static_cast<float>(v));
    }
    return floats;
}

// --- Validation ---

static bool check_floats_close(const vector<float>& a, const vector<float>& b, float rtol, float atol) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!is_close(a[i], b[i], rtol, atol)) {
            log_info(tt::LogTest, "check_floats_close: mismatch at index {} - a[i] = {}, b[i] = {}", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

static double compute_pcc(const vector<float>& a, const vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    const size_t n = a.size();
    double sum_a = 0.0, sum_b = 0.0;
    double sum_a2 = 0.0, sum_b2 = 0.0, sum_ab = 0.0;
    for (size_t i = 0; i < n; i++) {
        double ai = a[i], bi = b[i];
        sum_a += ai;
        sum_b += bi;
        sum_a2 += ai * ai;
        sum_b2 += bi * bi;
        sum_ab += ai * bi;
    }
    double denom_a = (n * sum_a2) - (sum_a * sum_a);
    double denom_b = (n * sum_b2) - (sum_b * sum_b);
    if (denom_a == 0.0 || denom_b == 0.0) {
        return 1.0;
    }
    return (n * sum_ab - sum_a * sum_b) / std::sqrt(denom_a * denom_b);
}

static bool check_pcc(const vector<float>& a, const vector<float>& b, double min_pcc) {
    double pcc = compute_pcc(a, b);
    if (pcc < min_pcc) {
        log_info(tt::LogTest, "check_pcc: PCC = {} < min_pcc = {}", pcc, min_pcc);
        return false;
    }
    return true;
}

// --- Special-case rule testing infrastructure ---
//
// Helpers for hand-crafting raw MXFP6 tile bytes and reading raw BF16 outputs.
// MXFP6 stores one byte per element (6 bits in [7:2], bits [1:0] are zero),
// which makes patching simpler than MXFP4's nibble-packed layout.

struct TileLayout {
    size_t total_words = 0;
    size_t exp_bytes = 0;  // byte offset where the elem region begins
};

// Pack an all-zero MXFP6R tile to discover the runtime tile layout (which
// depends on HAL L1 alignment). MXFP6R and MXFP6P share the same per-tile
// byte layout — same block size (32), same elem storage width (8) — so the
// caller can use the same layout for both.
static TileLayout get_mxfp6_tile_layout() {
    constexpr uint32_t kTileHW = 1024;
    std::vector<float> zeros(kTileHW, 0.0f);
    auto packed = pack_as_mxfp6r_tiles(tt::stl::make_const_span(zeros), /*row_major_input=*/true);
    const size_t elem_words = kTileHW / 4;  // 1024 bytes / 4 bytes per word = 256 words
    const size_t exp_words = packed.size() - elem_words;
    return TileLayout{.total_words = packed.size(), .exp_bytes = exp_words * 4};
}

struct ScalePatch {
    uint32_t block_idx;
    uint8_t scale_byte;
};
struct ElemPatch {
    uint32_t elem_idx;
    uint8_t elem_byte;  // full 8-bit storage byte; bits [1:0] should be 0
};

// Build a single-tile packed MXFP6 buffer: all 32 scale bytes set to
// scale_default, all 1024 elem bytes set to elem_byte_default, then patches
// applied. elem_idx is in face-major order; one byte per element.
static vector<uint32_t> build_mxfp6_tile_raw(
    const TileLayout& layout,
    uint8_t scale_default,
    uint8_t elem_byte_default,
    std::initializer_list<ScalePatch> scale_patches,
    std::initializer_list<ElemPatch> elem_patches) {
    vector<uint32_t> packed(layout.total_words, 0);
    auto* bytes = reinterpret_cast<uint8_t*>(packed.data());
    for (uint32_t s = 0; s < 32; ++s) {
        bytes[s] = scale_default;
    }
    for (uint32_t b = 0; b < 1024; ++b) {
        bytes[layout.exp_bytes + b] = elem_byte_default;
    }
    for (const auto& p : scale_patches) {
        TT_FATAL(p.block_idx < 32, "block_idx {} out of range", p.block_idx);
        bytes[p.block_idx] = p.scale_byte;
    }
    for (const auto& p : elem_patches) {
        TT_FATAL(p.elem_idx < 1024, "elem_idx {} out of range", p.elem_idx);
        bytes[layout.exp_bytes + p.elem_idx] = p.elem_byte;
    }
    return packed;
}

static uint16_t bf16_raw_at(const vector<uint32_t>& packed, uint32_t i) {
    return static_cast<uint16_t>((packed[i / 2] >> ((i % 2) * 16)) & 0xFFFFu);
}

enum class Bf16Class { Zero, Subnormal, Normal, PosInf, NegInf, NaN };

static Bf16Class classify_bf16(uint16_t bits) {
    uint16_t sign = (bits >> 15) & 0x1u;
    uint16_t exp = (bits >> 7) & 0xFFu;
    uint16_t mant = bits & 0x7Fu;
    if (exp == 0xFF) {
        if (mant == 0) {
            return sign ? Bf16Class::NegInf : Bf16Class::PosInf;
        }
        return Bf16Class::NaN;
    }
    if (exp == 0) {
        return mant == 0 ? Bf16Class::Zero : Bf16Class::Subnormal;
    }
    return Bf16Class::Normal;
}

}  // namespace unit_tests::llk::mxfp6_typecast

namespace mxfp6_tc = unit_tests::llk::mxfp6_typecast;

// ============================================================================
// MXFP6R → Float16_b
// Widening conversion: every MXFP6R value should be representable in BF16.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6RToFloat16b) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp6_tc::create_random_vector_of_mxfp6r(
        tt::tile_size(tt::DataFormat::MxFp6R) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp6_tc::run_mxfp6_typecast(
        dev, tt::DataFormat::MxFp6R, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp6_tc::mxfp6r_to_floats(src_vec);
    auto dst_floats = mxfp6_tc::bf16_to_floats(result_vec);
    EXPECT_TRUE(mxfp6_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp6_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6RToFloat16bFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp6_tc::create_random_vector_of_mxfp6r(
        tt::tile_size(tt::DataFormat::MxFp6R) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp6_tc::run_mxfp6_typecast(
        dev, tt::DataFormat::MxFp6R, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp6_tc::mxfp6r_to_floats(src_vec);
    auto dst_floats = mxfp6_tc::bf16_to_floats(result_vec);
    EXPECT_TRUE(mxfp6_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp6_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// ============================================================================
// Float16_b → MXFP6R
// Narrowing conversion: BF16 → MXFP6R introduces quantization. MXFP6R has
// 2 mantissa bits and a wide exponent (E3), so block scaling preserves a
// large dynamic range but per-element rounding error scales with magnitude.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp6R) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp6_tc::run_mxfp6_typecast(
        dev, tt::DataFormat::Float16_b, tt::DataFormat::MxFp6R, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp6_tc::bf16_to_floats(src_vec);
    auto dst_floats = mxfp6_tc::mxfp6r_to_floats(result_vec);
    EXPECT_TRUE(mxfp6_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.5f, /*atol=*/0.5f));
    EXPECT_TRUE(mxfp6_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/0.98));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp6RFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp6_tc::run_mxfp6_typecast(
        dev, tt::DataFormat::Float16_b, tt::DataFormat::MxFp6R, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp6_tc::bf16_to_floats(src_vec);
    auto dst_floats = mxfp6_tc::mxfp6r_to_floats(result_vec);
    EXPECT_TRUE(mxfp6_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.5f, /*atol=*/0.5f));
    EXPECT_TRUE(mxfp6_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/0.98));
}

// ============================================================================
// MXFP6R → MXFP6R (identity)
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6RToMxFp6R) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp6_tc::create_random_vector_of_mxfp6r(
        tt::tile_size(tt::DataFormat::MxFp6R) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp6_tc::run_mxfp6_typecast(
        dev, tt::DataFormat::MxFp6R, tt::DataFormat::MxFp6R, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp6_tc::mxfp6r_to_floats(src_vec);
    auto dst_floats = mxfp6_tc::mxfp6r_to_floats(result_vec);
    EXPECT_TRUE(mxfp6_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp6_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6RToMxFp6RFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp6_tc::create_random_vector_of_mxfp6r(
        tt::tile_size(tt::DataFormat::MxFp6R) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp6_tc::run_mxfp6_typecast(
        dev, tt::DataFormat::MxFp6R, tt::DataFormat::MxFp6R, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp6_tc::mxfp6r_to_floats(src_vec);
    auto dst_floats = mxfp6_tc::mxfp6r_to_floats(result_vec);
    EXPECT_TRUE(mxfp6_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp6_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// ============================================================================
// MXFP6P → Float16_b
// Widening conversion: every MXFP6P value should be representable in BF16.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6PToFloat16b) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp6_tc::create_random_vector_of_mxfp6p(
        tt::tile_size(tt::DataFormat::MxFp6P) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp6_tc::run_mxfp6_typecast(
        dev, tt::DataFormat::MxFp6P, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp6_tc::mxfp6p_to_floats(src_vec);
    auto dst_floats = mxfp6_tc::bf16_to_floats(result_vec);
    EXPECT_TRUE(mxfp6_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp6_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6PToFloat16bFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp6_tc::create_random_vector_of_mxfp6p(
        tt::tile_size(tt::DataFormat::MxFp6P) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp6_tc::run_mxfp6_typecast(
        dev, tt::DataFormat::MxFp6P, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp6_tc::mxfp6p_to_floats(src_vec);
    auto dst_floats = mxfp6_tc::bf16_to_floats(result_vec);
    EXPECT_TRUE(mxfp6_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp6_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// ============================================================================
// Float16_b → MXFP6P
// Narrowing conversion: BF16 → MXFP6P. MXFP6P has 3 mantissa bits and a
// narrow exponent (E2), so per-element precision is finer but block scaling
// has less headroom than MXFP6R.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp6P) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp6_tc::run_mxfp6_typecast(
        dev, tt::DataFormat::Float16_b, tt::DataFormat::MxFp6P, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp6_tc::bf16_to_floats(src_vec);
    auto dst_floats = mxfp6_tc::mxfp6p_to_floats(result_vec);
    EXPECT_TRUE(mxfp6_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.5f, /*atol=*/0.5f));
    EXPECT_TRUE(mxfp6_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/0.98));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp6PFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp6_tc::run_mxfp6_typecast(
        dev, tt::DataFormat::Float16_b, tt::DataFormat::MxFp6P, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp6_tc::bf16_to_floats(src_vec);
    auto dst_floats = mxfp6_tc::mxfp6p_to_floats(result_vec);
    EXPECT_TRUE(mxfp6_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.5f, /*atol=*/0.5f));
    EXPECT_TRUE(mxfp6_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/0.98));
}

// ============================================================================
// MXFP6P → MXFP6P (identity)
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6PToMxFp6P) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp6_tc::create_random_vector_of_mxfp6p(
        tt::tile_size(tt::DataFormat::MxFp6P) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp6_tc::run_mxfp6_typecast(
        dev, tt::DataFormat::MxFp6P, tt::DataFormat::MxFp6P, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp6_tc::mxfp6p_to_floats(src_vec);
    auto dst_floats = mxfp6_tc::mxfp6p_to_floats(result_vec);
    EXPECT_TRUE(mxfp6_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp6_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6PToMxFp6PFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp6_tc::create_random_vector_of_mxfp6p(
        tt::tile_size(tt::DataFormat::MxFp6P) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp6_tc::run_mxfp6_typecast(
        dev, tt::DataFormat::MxFp6P, tt::DataFormat::MxFp6P, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp6_tc::mxfp6p_to_floats(src_vec);
    auto dst_floats = mxfp6_tc::mxfp6p_to_floats(result_vec);
    EXPECT_TRUE(mxfp6_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp6_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// ============================================================================
// Device special-case tests for hardware MXFP6R → BF16 typecast.
// MXFP6R (E3M2) is finite-only — every "unit_exp = all 1" pattern is a normal
// finite value (±16, ±20, ±24, ±28), so spec rule "leave as is" applies to
// all such elements. Block-exp = 0xFF still emits NaN; over/underflow follow
// BF16 saturation.
//
// 6-bit storage layout: bits [7:2] hold the fp6 value, bits [1:0] are zero.
// 6-bit pattern (sign:1 exp:3 mant:2, bias=3):
//   0b011100 (storage 0x70) = +16.0   (unit_exp=all 1, mant=00)
//   0b011101 (storage 0x74) = +20.0
//   0b011110 (storage 0x78) = +24.0
//   0b011111 (storage 0x7C) = +28.0   (max normal)
//   0b111100 (storage 0xF0) = -16.0
//   0b111111 (storage 0xFC) = -28.0
//   0b001100 (storage 0x30) = +1.0    (exp=011=biased 3 → unbiased 0)
//   0b000001 (storage 0x04) = +0.0625 (smallest +subnormal)
//   0b100001 (storage 0x84) = -0.0625
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6RToBf16SpecialCases) {
    IDevice* dev = devices_[0]->get_devices()[0];
    auto layout = mxfp6_tc::get_mxfp6_tile_layout();

    // Block 0: scale = 0xFF → all 32 elements should be NaN.
    // Block 1: scale = 1. elem 32-35: +16, +28, -16, -28 — every "unit exp =
    //          all 1" pattern with mant=00 / mant=11, both signs. All should
    //          appear in BF16 unchanged.
    // Block 2: scale = 2^127 (block_exp_biased=0xFE). elem 64-65: ±28 → 28 *
    //          2^127 ≈ 2^131.8, overflows BF16 → ±Inf.
    // Block 3: scale = 1. elem 96: +1.0 — sanity, must remain finite.
    // Block 4: scale = 2^-127 (block_exp_biased=0x00). elem 128-129: ±0.0625
    //          subnormal → ±2^-131. Below BF16 normal range; accept Zero or
    //          Subnormal (silicon may flush or retain BF16 subnormal precision).
    auto packed = mxfp6_tc::build_mxfp6_tile_raw(
        layout,
        /*scale_default=*/0x7F,
        /*elem_byte_default=*/0x00,
        {{0, 0xFF}, {1, 0x7F}, {2, 0xFE}, {3, 0x7F}, {4, 0x00}},
        {{32, 0x70}, {33, 0x7C}, {34, 0xF0}, {35, 0xFC}, {64, 0x7C}, {65, 0xFC}, {96, 0x30}, {128, 0x04}, {129, 0x84}});

    auto result = mxfp6_tc::run_mxfp6_typecast(
        dev,
        tt::DataFormat::MxFp6R,
        tt::DataFormat::Float16_b,
        packed,
        /*num_tiles=*/1,
        /*fp32_dest_acc_en=*/false);

    using Cls = mxfp6_tc::Bf16Class;
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_EQ(mxfp6_tc::classify_bf16(mxfp6_tc::bf16_raw_at(result, i)), Cls::NaN)
            << "block 0 (NaN-scale) elem " << i;
    }
    EXPECT_EQ(mxfp6_tc::bf16_raw_at(result, 32), 0x4180u);  // +16.0
    EXPECT_EQ(mxfp6_tc::bf16_raw_at(result, 33), 0x41E0u);  // +28.0
    EXPECT_EQ(mxfp6_tc::bf16_raw_at(result, 34), 0xC180u);  // -16.0
    EXPECT_EQ(mxfp6_tc::bf16_raw_at(result, 35), 0xC1E0u);  // -28.0
    EXPECT_EQ(mxfp6_tc::classify_bf16(mxfp6_tc::bf16_raw_at(result, 64)), Cls::PosInf);
    EXPECT_EQ(mxfp6_tc::classify_bf16(mxfp6_tc::bf16_raw_at(result, 65)), Cls::NegInf);
    EXPECT_EQ(mxfp6_tc::bf16_raw_at(result, 96), 0x3F80u);  // +1.0

    const auto bits_pos = mxfp6_tc::bf16_raw_at(result, 128);
    const auto cls_pos = mxfp6_tc::classify_bf16(bits_pos);
    EXPECT_TRUE(cls_pos == Cls::Zero || cls_pos == Cls::Subnormal)
        << "elem 128 expected Zero/Subnormal, got bits=0x" << std::hex << bits_pos;
    const auto bits_neg = mxfp6_tc::bf16_raw_at(result, 129);
    const auto cls_neg = mxfp6_tc::classify_bf16(bits_neg);
    EXPECT_TRUE(cls_neg == Cls::Zero || cls_neg == Cls::Subnormal)
        << "elem 129 expected Zero/Subnormal, got bits=0x" << std::hex << bits_neg;
}

// ============================================================================
// Device special-case tests for hardware MXFP6P → BF16 typecast.
// MXFP6P (E2M3) is finite-only — every "unit_exp = all 1" pattern is a normal
// finite value (±4, ±4.5, ..., ±7.5), so spec rule "leave as is" applies to
// all such elements.
//
// 6-bit pattern (sign:1 exp:2 mant:3, bias=1):
//   0b011000 (storage 0x60) = +4.0   (unit_exp=all 1, mant=000)
//   0b011111 (storage 0x7C) = +7.5   (max normal)
//   0b111000 (storage 0xE0) = -4.0
//   0b111111 (storage 0xFC) = -7.5
//   0b001000 (storage 0x20) = +1.0   (exp=01=biased 1 → unbiased 0)
//   0b000001 (storage 0x04) = +0.125 (smallest +subnormal)
//   0b100001 (storage 0x84) = -0.125
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6PToBf16SpecialCases) {
    IDevice* dev = devices_[0]->get_devices()[0];
    auto layout = mxfp6_tc::get_mxfp6_tile_layout();

    // Block 0: scale = 0xFF → all 32 elements should be NaN.
    // Block 1: scale = 1. elem 32-35: +4, +7.5, -4, -7.5.
    // Block 2: scale = 2^127. elem 64-65: ±7.5 → 7.5 * 2^127 ≈ 2^129.9,
    //          overflows BF16 → ±Inf.
    // Block 3: scale = 1. elem 96: +1.0 sanity.
    // Block 4: scale = 2^-127. elem 128-129: ±0.125 subnormal → ±2^-130.
    //          Below BF16 normal range; accept Zero or Subnormal.
    auto packed = mxfp6_tc::build_mxfp6_tile_raw(
        layout,
        /*scale_default=*/0x7F,
        /*elem_byte_default=*/0x00,
        {{0, 0xFF}, {1, 0x7F}, {2, 0xFE}, {3, 0x7F}, {4, 0x00}},
        {{32, 0x60}, {33, 0x7C}, {34, 0xE0}, {35, 0xFC}, {64, 0x7C}, {65, 0xFC}, {96, 0x20}, {128, 0x04}, {129, 0x84}});

    auto result = mxfp6_tc::run_mxfp6_typecast(
        dev,
        tt::DataFormat::MxFp6P,
        tt::DataFormat::Float16_b,
        packed,
        /*num_tiles=*/1,
        /*fp32_dest_acc_en=*/false);

    using Cls = mxfp6_tc::Bf16Class;
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_EQ(mxfp6_tc::classify_bf16(mxfp6_tc::bf16_raw_at(result, i)), Cls::NaN)
            << "block 0 (NaN-scale) elem " << i;
    }
    EXPECT_EQ(mxfp6_tc::bf16_raw_at(result, 32), 0x4080u);  // +4.0
    EXPECT_EQ(mxfp6_tc::bf16_raw_at(result, 33), 0x40F0u);  // +7.5
    EXPECT_EQ(mxfp6_tc::bf16_raw_at(result, 34), 0xC080u);  // -4.0
    EXPECT_EQ(mxfp6_tc::bf16_raw_at(result, 35), 0xC0F0u);  // -7.5
    EXPECT_EQ(mxfp6_tc::classify_bf16(mxfp6_tc::bf16_raw_at(result, 64)), Cls::PosInf);
    EXPECT_EQ(mxfp6_tc::classify_bf16(mxfp6_tc::bf16_raw_at(result, 65)), Cls::NegInf);
    EXPECT_EQ(mxfp6_tc::bf16_raw_at(result, 96), 0x3F80u);  // +1.0

    const auto bits_pos = mxfp6_tc::bf16_raw_at(result, 128);
    const auto cls_pos = mxfp6_tc::classify_bf16(bits_pos);
    EXPECT_TRUE(cls_pos == Cls::Zero || cls_pos == Cls::Subnormal)
        << "elem 128 expected Zero/Subnormal, got bits=0x" << std::hex << bits_pos;
    const auto bits_neg = mxfp6_tc::bf16_raw_at(result, 129);
    const auto cls_neg = mxfp6_tc::classify_bf16(bits_neg);
    EXPECT_TRUE(cls_neg == Cls::Zero || cls_neg == Cls::Subnormal)
        << "elem 129 expected Zero/Subnormal, got bits=0x" << std::hex << bits_neg;
}

}  // namespace tt::tt_metal
