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
#include <tt-metalium/mxfp8.hpp>
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

namespace unit_tests::llk::mxfp8_typecast {

// Run a datacopy kernel with different input/output formats.
// For Quasar, data is moved via DataflowBuffers (DFBs) and the hardware
// unpacker/packer performs the format conversion implicitly.
static vector<uint32_t> run_mxfp8_typecast(
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

    // DFB configs for Quasar.
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
    // Pass aligned DRAM page stride so the reader/writer advance the DRAM
    // pointer by the allocator's aligned_page_size while the DFB streams the
    // native tile size (e.g. 1056 bytes for MxFp8 on Quasar; the allocator
    // rounds up to 1088 due to 64B DRAM alignment).
    uint32_t src_dram_stride = static_cast<uint32_t>(src_buffer->aligned_page_size());
    uint32_t dst_dram_stride = static_cast<uint32_t>(dst_buffer->aligned_page_size());
    SetRuntimeArgs(program, reader, core, {src_buffer->address(), 0, num_tiles, src_dram_stride});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, num_tiles, dst_dram_stride});

    detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

// Data generators follow the fp8_typecast tests' convention: generate
// row-major floats in U(0, rand_max_float) + offset, then pack into tiles.
static vector<uint32_t> create_random_vector_of_mxfp8_e5m2(
    uint32_t num_bytes, int rand_max_float, int seed, float offset = 0.0f) {
    uint32_t single_tile_size = tt::tile_size(tt::DataFormat::MxFp8R);
    TT_FATAL(
        num_bytes % single_tile_size == 0,
        "num_bytes {} must be divisible by MXFP8 (E5M2) tile_size {}",
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

    vector<uint32_t> packed = pack_as_mxfp8_e5m2_tiles(tt::stl::make_const_span(fp32_vec), /*row_major_input=*/true);
    TT_FATAL(
        packed.size() * sizeof(uint32_t) == num_tiles * single_tile_size,
        "MXFP8 (E5M2) packed size {} bytes does not match expected {} bytes",
        packed.size() * sizeof(uint32_t),
        num_tiles * single_tile_size);
    return packed;
}

static vector<uint32_t> create_random_vector_of_mxfp8_e4m3(
    uint32_t num_bytes, int rand_max_float, int seed, float offset = 0.0f) {
    uint32_t single_tile_size = tt::tile_size(tt::DataFormat::MxFp8P);
    TT_FATAL(
        num_bytes % single_tile_size == 0,
        "num_bytes {} must be divisible by MXFP8 (E4M3) tile_size {}",
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

    vector<uint32_t> packed = pack_as_mxfp8_e4m3_tiles(tt::stl::make_const_span(fp32_vec), /*row_major_input=*/true);
    TT_FATAL(
        packed.size() * sizeof(uint32_t) == num_tiles * single_tile_size,
        "MXFP8 (E4M3) packed size {} bytes does not match expected {} bytes",
        packed.size() * sizeof(uint32_t),
        num_tiles * single_tile_size);
    return packed;
}

// --- Format-to-float unpackers ---

static vector<float> mxfp8_e5m2_to_floats(const vector<uint32_t>& packed) {
    return unpack_mxfp8_e5m2_tiles_into_float_vec(tt::stl::make_const_span(packed), /*row_major_output=*/false);
}

static vector<float> mxfp8_e4m3_to_floats(const vector<uint32_t>& packed) {
    return unpack_mxfp8_e4m3_tiles_into_float_vec(tt::stl::make_const_span(packed), /*row_major_output=*/false);
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
// Helpers for hand-crafting raw MXFP8 tile bytes and reading raw BF16 outputs,
// used to verify the spec rules for unit-exp = all-1, block-exp = 0xFF, and
// over/underflow (which random-data tests never exercise).

struct TileLayout {
    size_t total_words = 0;
    size_t exp_bytes = 0;  // byte offset where the elem region begins
};

// Pack an all-zero tile to discover the runtime tile layout (which depends on
// HAL L1 alignment), then derive exp_bytes from the resulting word count.
static TileLayout get_e5m2_tile_layout() {
    constexpr uint32_t kTileHW = 1024;
    std::vector<float> zeros(kTileHW, 0.0f);
    auto packed = pack_as_mxfp8_e5m2_tiles(tt::stl::make_const_span(zeros), /*row_major_input=*/true);
    const size_t elem_words = kTileHW / 4;
    const size_t exp_words = packed.size() - elem_words;
    return TileLayout{.total_words = packed.size(), .exp_bytes = exp_words * 4};
}

static TileLayout get_e4m3_tile_layout() {
    constexpr uint32_t kTileHW = 1024;
    std::vector<float> zeros(kTileHW, 0.0f);
    auto packed = pack_as_mxfp8_e4m3_tiles(tt::stl::make_const_span(zeros), /*row_major_input=*/true);
    const size_t elem_words = kTileHW / 4;
    const size_t exp_words = packed.size() - elem_words;
    return TileLayout{.total_words = packed.size(), .exp_bytes = exp_words * 4};
}

struct ScalePatch {
    uint32_t block_idx;
    uint8_t scale_byte;
};
struct ElemPatch {
    uint32_t elem_idx;
    uint8_t elem_byte;
};

// Build a single-tile packed MXFP8 buffer: all 32 scale bytes set to
// scale_default, all 1024 elem bytes set to elem_default, then patches applied.
// elem_idx is in face-major order (the natural ordering produced by the
// pack helper), so block b covers elem indices [b*32, (b+1)*32).
static vector<uint32_t> build_mxfp8_tile_raw(
    const TileLayout& layout,
    uint8_t scale_default,
    uint8_t elem_default,
    std::initializer_list<ScalePatch> scale_patches,
    std::initializer_list<ElemPatch> elem_patches) {
    vector<uint32_t> packed(layout.total_words, 0);
    auto* bytes = reinterpret_cast<uint8_t*>(packed.data());
    for (uint32_t s = 0; s < 32; ++s) {
        bytes[s] = scale_default;
    }
    for (uint32_t e = 0; e < 1024; ++e) {
        bytes[layout.exp_bytes + e] = elem_default;
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

// Extract raw BF16 bits at face-major position `i` from a packed BF16 readback.
// BF16 readback packs two values per uint32 (LSB = lower index, MSB = higher).
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

}  // namespace unit_tests::llk::mxfp8_typecast

namespace mxfp8_tc = unit_tests::llk::mxfp8_typecast;

// ============================================================================
// MXFP8 (E5M2) → Float16_b
// Widening conversion: every MXFP8 E5M2 value (with a power-of-two block
// scale) is exactly representable in BF16.
// Expected: no precision loss → rtol=0.0, atol=0.0.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8RToFloat16b) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp8_tc::create_random_vector_of_mxfp8_e5m2(
        tt::tile_size(tt::DataFormat::MxFp8R) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp8_tc::run_mxfp8_typecast(
        dev, tt::DataFormat::MxFp8R, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp8_tc::mxfp8_e5m2_to_floats(src_vec);
    auto dst_floats = mxfp8_tc::bf16_to_floats(result_vec);
    EXPECT_TRUE(mxfp8_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp8_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8RToFloat16bFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp8_tc::create_random_vector_of_mxfp8_e5m2(
        tt::tile_size(tt::DataFormat::MxFp8R) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp8_tc::run_mxfp8_typecast(
        dev, tt::DataFormat::MxFp8R, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp8_tc::mxfp8_e5m2_to_floats(src_vec);
    auto dst_floats = mxfp8_tc::bf16_to_floats(result_vec);
    EXPECT_TRUE(mxfp8_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp8_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// ============================================================================
// MXFP8 (E4M3) → Float16_b
// Widening conversion: every MXFP8 E4M3 value (with a power-of-two block
// scale) is exactly representable in BF16.
// Expected: no precision loss → rtol=0.0, atol=0.0.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8PToFloat16b) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp8_tc::create_random_vector_of_mxfp8_e4m3(
        tt::tile_size(tt::DataFormat::MxFp8P) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp8_tc::run_mxfp8_typecast(
        dev, tt::DataFormat::MxFp8P, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp8_tc::mxfp8_e4m3_to_floats(src_vec);
    auto dst_floats = mxfp8_tc::bf16_to_floats(result_vec);
    EXPECT_TRUE(mxfp8_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp8_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8PToFloat16bFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp8_tc::create_random_vector_of_mxfp8_e4m3(
        tt::tile_size(tt::DataFormat::MxFp8P) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp8_tc::run_mxfp8_typecast(
        dev, tt::DataFormat::MxFp8P, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp8_tc::mxfp8_e4m3_to_floats(src_vec);
    auto dst_floats = mxfp8_tc::bf16_to_floats(result_vec);
    EXPECT_TRUE(mxfp8_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp8_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// ============================================================================
// Float16_b → MXFP8 (E5M2)
// Narrowing conversion: BF16 → MXFP8 E5M2 introduces quantization. E5M2
// has 2 mantissa bits, so the worst-case relative error is ~1/4. Tolerances
// are loose to account for block-scaling behavior.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp8R) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp8_tc::run_mxfp8_typecast(
        dev, tt::DataFormat::Float16_b, tt::DataFormat::MxFp8R, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp8_tc::bf16_to_floats(src_vec);
    auto dst_floats = mxfp8_tc::mxfp8_e5m2_to_floats(result_vec);
    EXPECT_TRUE(mxfp8_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.25f, /*atol=*/0.1f));
    EXPECT_TRUE(mxfp8_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/0.999));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp8RFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp8_tc::run_mxfp8_typecast(
        dev, tt::DataFormat::Float16_b, tt::DataFormat::MxFp8R, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp8_tc::bf16_to_floats(src_vec);
    auto dst_floats = mxfp8_tc::mxfp8_e5m2_to_floats(result_vec);
    EXPECT_TRUE(mxfp8_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.25f, /*atol=*/0.1f));
    EXPECT_TRUE(mxfp8_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/0.999));
}

// ============================================================================
// Float16_b → MXFP8 (E4M3)
// Narrowing conversion: BF16 → MXFP8 E4M3 introduces quantization. E4M3
// has 3 mantissa bits, so the worst-case relative error is ~1/8 — tighter
// than E5M2.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp8P) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp8_tc::run_mxfp8_typecast(
        dev, tt::DataFormat::Float16_b, tt::DataFormat::MxFp8P, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp8_tc::bf16_to_floats(src_vec);
    auto dst_floats = mxfp8_tc::mxfp8_e4m3_to_floats(result_vec);
    EXPECT_TRUE(mxfp8_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.125f, /*atol=*/0.05f));
    EXPECT_TRUE(mxfp8_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/0.999));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp8PFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp8_tc::run_mxfp8_typecast(
        dev, tt::DataFormat::Float16_b, tt::DataFormat::MxFp8P, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp8_tc::bf16_to_floats(src_vec);
    auto dst_floats = mxfp8_tc::mxfp8_e4m3_to_floats(result_vec);
    EXPECT_TRUE(mxfp8_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.125f, /*atol=*/0.05f));
    EXPECT_TRUE(mxfp8_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/0.999));
}

// ============================================================================
// MXFP8 (E5M2) → MXFP8 (E5M2) (identity)
// Same format on both sides. The round-trip should be lossless.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8RToMxFp8R) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp8_tc::create_random_vector_of_mxfp8_e5m2(
        tt::tile_size(tt::DataFormat::MxFp8R) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp8_tc::run_mxfp8_typecast(
        dev, tt::DataFormat::MxFp8R, tt::DataFormat::MxFp8R, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp8_tc::mxfp8_e5m2_to_floats(src_vec);
    auto dst_floats = mxfp8_tc::mxfp8_e5m2_to_floats(result_vec);
    EXPECT_TRUE(mxfp8_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp8_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8RToMxFp8RFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp8_tc::create_random_vector_of_mxfp8_e5m2(
        tt::tile_size(tt::DataFormat::MxFp8R) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp8_tc::run_mxfp8_typecast(
        dev, tt::DataFormat::MxFp8R, tt::DataFormat::MxFp8R, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp8_tc::mxfp8_e5m2_to_floats(src_vec);
    auto dst_floats = mxfp8_tc::mxfp8_e5m2_to_floats(result_vec);
    EXPECT_TRUE(mxfp8_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp8_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// ============================================================================
// MXFP8 (E4M3) → MXFP8 (E4M3) (identity)
// Same format on both sides. The round-trip should be lossless.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8PToMxFp8P) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp8_tc::create_random_vector_of_mxfp8_e4m3(
        tt::tile_size(tt::DataFormat::MxFp8P) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp8_tc::run_mxfp8_typecast(
        dev, tt::DataFormat::MxFp8P, tt::DataFormat::MxFp8P, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp8_tc::mxfp8_e4m3_to_floats(src_vec);
    auto dst_floats = mxfp8_tc::mxfp8_e4m3_to_floats(result_vec);
    EXPECT_TRUE(mxfp8_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp8_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8PToMxFp8PFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp8_tc::create_random_vector_of_mxfp8_e4m3(
        tt::tile_size(tt::DataFormat::MxFp8P) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp8_tc::run_mxfp8_typecast(
        dev, tt::DataFormat::MxFp8P, tt::DataFormat::MxFp8P, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp8_tc::mxfp8_e4m3_to_floats(src_vec);
    auto dst_floats = mxfp8_tc::mxfp8_e4m3_to_floats(result_vec);
    EXPECT_TRUE(mxfp8_tc::check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(mxfp8_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// ============================================================================
// Device special-case tests for hardware MXFP8 → BF16 typecast.
// Each test bundles multiple edge cases into a single tile (one rule per
// block, since each block has its own scale byte) to keep simulator runtime
// low. Output BF16 bits are inspected directly — is_close() can't be used
// for NaN/Inf comparisons.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8RToBf16SpecialCases) {
    IDevice* dev = devices_[0]->get_devices()[0];
    auto layout = mxfp8_tc::get_e5m2_tile_layout();

    // Block 0: scale = 0xFF → all 32 elements should be NaN (rule 1).
    // Block 1: scale = 1. elem 32: +Inf, 33: -Inf, 34/35/36: NaN encodings.
    // Block 2: scale = 2^127. elem 64: max normal, 65: -max normal → overflow → ±Inf.
    // Block 3: scale = 1. elem 96: +1.0 — sanity, must remain finite.
    auto packed = mxfp8_tc::build_mxfp8_tile_raw(
        layout,
        /*scale_default=*/0x7F,
        /*elem_default=*/0,
        {{0, 0xFF}, {1, 0x7F}, {2, 0xFE}, {3, 0x7F}},
        {{32, 0x7C},
         {33, 0xFC},
         {34, 0x7D},
         {35, 0x7E},
         {36, 0x7F},
         {64, 0x7B},
         {65, 0xFB},
         {96, 0x3C}});

    auto result = mxfp8_tc::run_mxfp8_typecast(
        dev,
        tt::DataFormat::MxFp8R,
        tt::DataFormat::Float16_b,
        packed,
        /*num_tiles=*/1,
        /*fp32_dest_acc_en=*/false);

    using Cls = mxfp8_tc::Bf16Class;
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_EQ(mxfp8_tc::classify_bf16(mxfp8_tc::bf16_raw_at(result, i)), Cls::NaN)
            << "block 0 (NaN-scale) elem " << i;
    }
    EXPECT_EQ(mxfp8_tc::classify_bf16(mxfp8_tc::bf16_raw_at(result, 32)), Cls::PosInf);
    EXPECT_EQ(mxfp8_tc::classify_bf16(mxfp8_tc::bf16_raw_at(result, 33)), Cls::NegInf);
    EXPECT_EQ(mxfp8_tc::classify_bf16(mxfp8_tc::bf16_raw_at(result, 34)), Cls::NaN);
    EXPECT_EQ(mxfp8_tc::classify_bf16(mxfp8_tc::bf16_raw_at(result, 35)), Cls::NaN);
    EXPECT_EQ(mxfp8_tc::classify_bf16(mxfp8_tc::bf16_raw_at(result, 36)), Cls::NaN);
    EXPECT_EQ(mxfp8_tc::classify_bf16(mxfp8_tc::bf16_raw_at(result, 64)), Cls::PosInf);
    EXPECT_EQ(mxfp8_tc::classify_bf16(mxfp8_tc::bf16_raw_at(result, 65)), Cls::NegInf);
    EXPECT_EQ(mxfp8_tc::bf16_raw_at(result, 96), 0x3F80u);  // BF16 +1.0
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8PToBf16SpecialCases) {
    IDevice* dev = devices_[0]->get_devices()[0];
    auto layout = mxfp8_tc::get_e4m3_tile_layout();

    // Block 0: scale = 0xFF → all 32 elements should be NaN (rule 1).
    // Block 1: scale = 1. elem 32: NaN encoding (mant=all 1), 33: +256 (mant=0,
    //          "leave as is"), 34: +352 (mid mant), 35: +448 (max normal).
    // Block 2: scale = 2^127. elem 64: max normal → overflow → +Inf.
    // Block 3: scale = 1. elem 96: +1.0 — sanity.
    auto packed = mxfp8_tc::build_mxfp8_tile_raw(
        layout,
        /*scale_default=*/0x7F,
        /*elem_default=*/0,
        {{0, 0xFF}, {1, 0x7F}, {2, 0xFE}, {3, 0x7F}},
        {{32, 0x7F},
         {33, 0x78},
         {34, 0x7B},
         {35, 0x7E},
         {64, 0x7E},
         {65, 0xFE},
         {96, 0x38}});

    auto result = mxfp8_tc::run_mxfp8_typecast(
        dev,
        tt::DataFormat::MxFp8P,
        tt::DataFormat::Float16_b,
        packed,
        /*num_tiles=*/1,
        /*fp32_dest_acc_en=*/false);

    using Cls = mxfp8_tc::Bf16Class;
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_EQ(mxfp8_tc::classify_bf16(mxfp8_tc::bf16_raw_at(result, i)), Cls::NaN)
            << "block 0 (NaN-scale) elem " << i;
    }
    EXPECT_EQ(mxfp8_tc::classify_bf16(mxfp8_tc::bf16_raw_at(result, 32)), Cls::NaN);
    EXPECT_EQ(mxfp8_tc::bf16_raw_at(result, 33), 0x4380u);  // BF16 +256.0
    EXPECT_EQ(mxfp8_tc::bf16_raw_at(result, 34), 0x43B0u);  // BF16 +352.0
    EXPECT_EQ(mxfp8_tc::bf16_raw_at(result, 35), 0x43E0u);  // BF16 +448.0
    EXPECT_EQ(mxfp8_tc::classify_bf16(mxfp8_tc::bf16_raw_at(result, 64)), Cls::PosInf);
    EXPECT_EQ(mxfp8_tc::classify_bf16(mxfp8_tc::bf16_raw_at(result, 65)), Cls::NegInf);
    EXPECT_EQ(mxfp8_tc::bf16_raw_at(result, 96), 0x3F80u);  // BF16 +1.0
}

}  // namespace tt::tt_metal
