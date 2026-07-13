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
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mxfp6.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include <tt-logger/tt-logger.hpp>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/float8_utils.hpp"
#include "tt_metal/test_utils/mx_utils.hpp"

namespace tt::tt_metal {

using std::vector;

namespace unit_tests::llk::mxfp6_typecast {

using tt::test_utils::bf16_to_floats;
using tt::test_utils::check_pcc;
using tt::test_utils::is_close;
using tt::test_utils::is_close_vectors;
using tt::test_utils::mx_to_floats;
using tt::test_utils::pack_as_mx_tiles;

// Run a datacopy kernel with different input/output formats. Mirrors the
// MXFP4 typecast harness — for Quasar, data is moved via DataflowBuffers
// (DFBs) and the hardware unpacker/packer performs the format conversion
// implicitly.
static vector<uint32_t> run_mxfp6_typecast(
    const distributed::MeshDevice& mesh_device,
    tt::DataFormat input_fmt,
    tt::DataFormat output_fmt,
    const vector<uint32_t>& src_vec,
    uint32_t num_tiles,
    bool fp32_dest_acc_en) {
    IDevice* dev = mesh_device.get_devices()[0];
    const experimental::NodeCoord node{0, 0};

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

    const experimental::DFBSpecName INPUT_DFB{"input_dfb"};
    const experimental::DFBSpecName OUTPUT_DFB{"output_dfb"};
    const experimental::KernelSpecName READER{"reader"};
    const experimental::KernelSpecName WRITER{"writer"};
    const experimental::KernelSpecName COMPUTE{"compute"};

    experimental::DataflowBufferSpec input_dfb_spec{
        .unique_id = INPUT_DFB,
        .entry_size = input_tile_size,
        .num_entries = 2,
        .data_format_metadata = input_fmt,
    };
    experimental::DataflowBufferSpec output_dfb_spec{
        .unique_id = OUTPUT_DFB,
        .entry_size = output_tile_size,
        .num_entries = 2,
        .data_format_metadata = output_fmt,
    };

    experimental::KernelSpec reader_spec{
        .unique_id = READER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = INPUT_DFB,
            .accessor_name = "out",
            .endpoint_type = experimental::DFBEndpointType::PRODUCER,
            .access_pattern = experimental::DFBAccessPattern::STRIDED,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {{
            .dfb_spec_name = OUTPUT_DFB,
            .accessor_name = "in",
            .endpoint_type = experimental::DFBEndpointType::CONSUMER,
            .access_pattern = experimental::DFBAccessPattern::STRIDED,
        }},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings =
            {{
                 .dfb_spec_name = INPUT_DFB,
                 .accessor_name = "in",
                 .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             },
             {
                 .dfb_spec_name = OUTPUT_DFB,
                 .accessor_name = "out",
                 .endpoint_type = experimental::DFBEndpointType::PRODUCER,
                 .access_pattern = experimental::DFBAccessPattern::STRIDED,
             }},
        .compile_time_args = {{"per_core_tile_cnt", num_tiles}},
        .hw_config =
            experimental::ComputeGen2Config{
                .fp32_dest_acc_en = fp32_dest_acc_en,
            },
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "mxfp6_typecast",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    detail::WriteToBuffer(src_buffer, src_vec);
    uint32_t src_dram_stride = static_cast<uint32_t>(src_buffer->aligned_page_size());
    uint32_t dst_dram_stride = static_cast<uint32_t>(dst_buffer->aligned_page_size());

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values =
                {{node,
                  {{"src_addr", src_buffer->address()},
                   {"src_bank_id", 0u},
                   {"num_tiles", num_tiles},
                   {"dram_page_stride", src_dram_stride}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values =
                {{node,
                  {{"dst_addr", dst_buffer->address()},
                   {"dst_bank_id", 0u},
                   {"num_tiles", num_tiles},
                   {"dram_page_stride", dst_dram_stride}}}},
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program, params);

    detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

// --- Random data generators ---

static vector<uint32_t> create_random_vector_of_mxfp6(
    tt::DataFormat fmt, uint32_t num_bytes, int rand_max_float, int seed, float offset = 0.0f) {
    TT_FATAL(
        fmt == tt::DataFormat::MxFp6R || fmt == tt::DataFormat::MxFp6P,
        "Unsupported MXFP6 DataFormat: {}",
        static_cast<int>(fmt));
    uint32_t single_tile_size = tt::tile_size(fmt);
    TT_FATAL(
        num_bytes % single_tile_size == 0,
        "num_bytes {} must be divisible by MXFP6 tile_size {}",
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

    auto span = ttsl::make_const_span(fp32_vec);
    vector<uint32_t> packed = pack_as_mx_tiles(fmt, span, /*row_major_input=*/true);
    TT_FATAL(
        packed.size() * sizeof(uint32_t) == num_tiles * single_tile_size,
        "MXFP6 packed size {} bytes does not match expected {} bytes",
        packed.size() * sizeof(uint32_t),
        num_tiles * single_tile_size);
    return packed;
}

// --- Random typecast test driver ---
//
// Shared parameters for the random-input typecast tests below. Centralized so
// changing num_tiles / seed / range happens in one place.
constexpr uint32_t kDefaultNumTiles = 64;
constexpr int kRandMaxFloat = 20;
constexpr int kSeed = 42;
constexpr float kOffset = -10.0f;  // U(0, kRandMaxFloat) + kOffset = U(-10, 10)

static vector<uint32_t> generate_random_src(tt::DataFormat fmt, uint32_t num_tiles) {
    uint32_t bytes = tt::tile_size(fmt) * num_tiles;
    switch (fmt) {
        case tt::DataFormat::MxFp6R:
        case tt::DataFormat::MxFp6P: return create_random_vector_of_mxfp6(fmt, bytes, kRandMaxFloat, kSeed, kOffset);
        case tt::DataFormat::Float16_b: return create_random_vector_of_bfloat16(bytes, kRandMaxFloat, kSeed, kOffset);
        default: TT_THROW("Unsupported source DataFormat for mxfp6 typecast test: {}", static_cast<int>(fmt));
    }
}

static vector<float> unpack_to_floats(tt::DataFormat fmt, const vector<uint32_t>& packed) {
    switch (fmt) {
        case tt::DataFormat::MxFp6R:
        case tt::DataFormat::MxFp6P: return mx_to_floats(fmt, packed);
        case tt::DataFormat::Float16_b: return bf16_to_floats(packed);
        default: TT_THROW("Unsupported DataFormat for mxfp6 unpack: {}", static_cast<int>(fmt));
    }
}

// Drive one random typecast test: generate U(-10, 10) data in input_fmt, run
// the device, unpack both sides to floats, and check element-wise tolerance
// + PCC. Used by all random-input TEST_F bodies below.
static void run_random_typecast_test(
    const distributed::MeshDevice& mesh_device,
    tt::DataFormat input_fmt,
    tt::DataFormat output_fmt,
    float rtol,
    float atol,
    double min_pcc,
    bool fp32_dest_acc_en) {
    auto src_vec = generate_random_src(input_fmt, kDefaultNumTiles);
    auto result_vec =
        run_mxfp6_typecast(mesh_device, input_fmt, output_fmt, src_vec, kDefaultNumTiles, fp32_dest_acc_en);
    auto src_floats = unpack_to_floats(input_fmt, src_vec);
    auto dst_floats = unpack_to_floats(output_fmt, result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [rtol, atol](float a, float b) { return is_close(a, b, rtol, atol); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, min_pcc));
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
    auto packed = pack_as_mx_tiles(tt::DataFormat::MxFp6R, ttsl::make_const_span(zeros), /*row_major_input=*/true);
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

// MXFP6 element-byte classifier. MXFP6R/P are finite-only formats — neither
// has an Inf or NaN encoding, so NaN can only appear via the OCP MX block-
// level rule (scale = 0xFF). Combine with the scale byte via
// effective_*_class below.
enum class MxFp6Class { Zero, Subnormal, Normal, MaxNormalPos, MaxNormalNeg, NaN };

// MXFP6R (E3M2): 1 sign / 3 exp / 2 mantissa, bias 3. Storage byte holds the
// 6-bit value in bits [7:2]; bits [1:0] are zero.
//   Max normal: 0_111_11 / 1_111_11 = 0x7C / 0xFC (= ±28.0)
static MxFp6Class classify_mxfp6r(uint8_t bits) {
    uint8_t sign = (bits >> 7) & 0x1u;
    uint8_t exp = (bits >> 4) & 0x7u;
    uint8_t mant = (bits >> 2) & 0x3u;
    if (exp == 0x7 && mant == 0x3) {
        return sign ? MxFp6Class::MaxNormalNeg : MxFp6Class::MaxNormalPos;
    }
    if (exp == 0) {
        return mant == 0 ? MxFp6Class::Zero : MxFp6Class::Subnormal;
    }
    return MxFp6Class::Normal;
}

// MXFP6P (E2M3): 1 sign / 2 exp / 3 mantissa, bias 1.
//   Max normal: 0_11_111 / 1_11_111 = 0x7C / 0xFC (= ±7.5)
static MxFp6Class classify_mxfp6p(uint8_t bits) {
    uint8_t sign = (bits >> 7) & 0x1u;
    uint8_t exp = (bits >> 5) & 0x3u;
    uint8_t mant = (bits >> 2) & 0x7u;
    if (exp == 0x3 && mant == 0x7) {
        return sign ? MxFp6Class::MaxNormalNeg : MxFp6Class::MaxNormalPos;
    }
    if (exp == 0) {
        return mant == 0 ? MxFp6Class::Zero : MxFp6Class::Subnormal;
    }
    return MxFp6Class::Normal;
}

static uint8_t mxfp6_elem_byte_at(const vector<uint32_t>& packed, const TileLayout& layout, uint32_t i) {
    return reinterpret_cast<const uint8_t*>(packed.data())[layout.exp_bytes + i];
}

static uint8_t mxfp6_scale_byte_at(const vector<uint32_t>& packed, uint32_t block_idx) {
    return reinterpret_cast<const uint8_t*>(packed.data())[block_idx];
}

// Apply the OCP MX block-level rule: a block with scale = 0xFF reads as NaN
// for every element. Otherwise the element class stands.
static MxFp6Class effective_mxfp6_class(tt::DataFormat fmt, uint8_t scale_byte, uint8_t elem_byte) {
    if (scale_byte == 0xFF) {
        return MxFp6Class::NaN;
    }
    if (fmt == tt::DataFormat::MxFp6R) {
        return classify_mxfp6r(elem_byte);
    }
    if (fmt == tt::DataFormat::MxFp6P) {
        return classify_mxfp6p(elem_byte);
    }
    TT_THROW("Unsupported MXFP6 DataFormat: {}", static_cast<int>(fmt));
}

// Build a 1024-element BF16 tile (1 tile = 32 blocks × 32 elements) where
// blocks 0..N-1 are filled with the corresponding raw BF16 word from
// `block_values`, and remaining blocks are filled with the last value (or 0
// if the list is empty). Used to inject per-block special inputs (NaN,
// ±Inf, finite sanity) for the BF16 → MXFP6 special-case tests.
static vector<uint32_t> build_bf16_tile_with_block_values(std::initializer_list<uint16_t> block_values) {
    constexpr uint32_t kNumBlocksPerTile = 32;
    constexpr uint32_t kElementsPerBlock = 32;
    constexpr uint32_t kWordsPerBlock = kElementsPerBlock / 2;  // 2 BF16 per uint32
    vector<uint32_t> packed(kNumBlocksPerTile * kWordsPerBlock, 0);

    uint16_t default_val = block_values.size() > 0 ? *(block_values.end() - 1) : 0;
    const auto* it = block_values.begin();
    for (uint32_t b = 0; b < kNumBlocksPerTile; ++b) {
        uint16_t val = (it != block_values.end()) ? *it++ : default_val;
        // Pack two identical BF16 values per uint32 (LSB = lower face-major
        // index, MSB = higher) — mirrors bf16_raw_at's read pattern.
        uint32_t paired = (static_cast<uint32_t>(val) << 16) | static_cast<uint32_t>(val);
        for (uint32_t w = 0; w < kWordsPerBlock; ++w) {
            packed[b * kWordsPerBlock + w] = paired;
        }
    }
    return packed;
}

}  // namespace unit_tests::llk::mxfp6_typecast

namespace mxfp6_tc = unit_tests::llk::mxfp6_typecast;

// ============================================================================
// MXFP6R → Float16_b
// Widening conversion: every MXFP6R value should be representable in BF16.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6RToFloat16b) {
    const auto& mesh_device = *devices_[0];
    mxfp6_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp6R,
        tt::DataFormat::Float16_b,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/false);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6RToFloat16bFp32Dest) {
    const auto& mesh_device = *devices_[0];
    mxfp6_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp6R,
        tt::DataFormat::Float16_b,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// Float16_b → MXFP6R
// Narrowing conversion: BF16 → MXFP6R introduces quantization. MXFP6R has
// 2 mantissa bits and a wide exponent (E3), so block scaling preserves a
// large dynamic range but per-element rounding error scales with magnitude.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp6R) {
    const auto& mesh_device = *devices_[0];
    mxfp6_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::MxFp6R,
        /*rtol=*/0.5f,
        /*atol=*/0.5f,
        /*min_pcc=*/0.98,
        /*fp32_dest_acc_en=*/false);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp6RFp32Dest) {
    const auto& mesh_device = *devices_[0];
    mxfp6_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::MxFp6R,
        /*rtol=*/0.5f,
        /*atol=*/0.5f,
        /*min_pcc=*/0.98,
        /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// MXFP6R → MXFP6R (identity)
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6RToMxFp6R) {
    const auto& mesh_device = *devices_[0];
    mxfp6_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp6R,
        tt::DataFormat::MxFp6R,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/false);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6RToMxFp6RFp32Dest) {
    const auto& mesh_device = *devices_[0];
    mxfp6_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp6R,
        tt::DataFormat::MxFp6R,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// MXFP6P → Float16_b
// Widening conversion: every MXFP6P value should be representable in BF16.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6PToFloat16b) {
    const auto& mesh_device = *devices_[0];
    mxfp6_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp6P,
        tt::DataFormat::Float16_b,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/false);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6PToFloat16bFp32Dest) {
    const auto& mesh_device = *devices_[0];
    mxfp6_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp6P,
        tt::DataFormat::Float16_b,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// Float16_b → MXFP6P
// Narrowing conversion: BF16 → MXFP6P. MXFP6P has 3 mantissa bits and a
// narrow exponent (E2), so per-element precision is finer but block scaling
// has less headroom than MXFP6R.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp6P) {
    const auto& mesh_device = *devices_[0];
    mxfp6_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::MxFp6P,
        /*rtol=*/0.5f,
        /*atol=*/0.5f,
        /*min_pcc=*/0.98,
        /*fp32_dest_acc_en=*/false);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp6PFp32Dest) {
    const auto& mesh_device = *devices_[0];
    mxfp6_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::MxFp6P,
        /*rtol=*/0.5f,
        /*atol=*/0.5f,
        /*min_pcc=*/0.98,
        /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// MXFP6P → MXFP6P (identity)
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6PToMxFp6P) {
    const auto& mesh_device = *devices_[0];
    mxfp6_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp6P,
        tt::DataFormat::MxFp6P,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/false);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp6PToMxFp6PFp32Dest) {
    const auto& mesh_device = *devices_[0];
    mxfp6_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp6P,
        tt::DataFormat::MxFp6P,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/true);
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
    const auto& mesh_device = *devices_[0];
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
        mesh_device,
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
    const auto& mesh_device = *devices_[0];
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
        mesh_device,
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

// ============================================================================
// Float16_b → MXFP6R special cases.
// MXFP6R is finite-only — there is no Inf encoding — so BF16 ±Inf inputs must
// produce either NaN (NaN-scale block per OCP MX rule) or saturate to the
// ±max-normal element. NaN inputs propagate via NaN-scale.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp6RSpecialCases) {
    const auto& mesh_device = *devices_[0];

    // Block layout (32 BF16 elements per block):
    //   0: all +NaN  → block must read as NaN (NaN propagation).
    //   1: all +Inf  → NaN or +max-normal (saturation; MXFP6R has no Inf).
    //   2: all -Inf  → NaN or -max-normal.
    //   3: all +1.0  → sanity, must round-trip exactly to MXFP6R 1.0.
    constexpr uint16_t kBf16PosNaN = 0x7FC0;
    constexpr uint16_t kBf16PosInf = 0x7F80;
    constexpr uint16_t kBf16NegInf = 0xFF80;
    constexpr uint16_t kBf16PosOne = 0x3F80;

    auto src = mxfp6_tc::build_bf16_tile_with_block_values({kBf16PosNaN, kBf16PosInf, kBf16NegInf, kBf16PosOne});

    auto result = mxfp6_tc::run_mxfp6_typecast(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::MxFp6R,
        src,
        /*num_tiles=*/1,
        /*fp32_dest_acc_en=*/false);

    auto layout = mxfp6_tc::get_mxfp6_tile_layout();
    using Cls = mxfp6_tc::MxFp6Class;

    // Block 0: NaN BF16 in → NaN out (NaN-scale).
    uint8_t scale0 = mxfp6_tc::mxfp6_scale_byte_at(result, 0);
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_EQ(
            mxfp6_tc::effective_mxfp6_class(
                tt::DataFormat::MxFp6R, scale0, mxfp6_tc::mxfp6_elem_byte_at(result, layout, i)),
            Cls::NaN)
            << "block 0 (BF16 NaN in) elem " << i;
    }

    // Block 1: +Inf BF16 in → NaN or +max-normal.
    uint8_t scale1 = mxfp6_tc::mxfp6_scale_byte_at(result, 1);
    for (uint32_t i = 32; i < 64; ++i) {
        auto cls = mxfp6_tc::effective_mxfp6_class(
            tt::DataFormat::MxFp6R, scale1, mxfp6_tc::mxfp6_elem_byte_at(result, layout, i));
        EXPECT_TRUE(cls == Cls::NaN || cls == Cls::MaxNormalPos)
            << "block 1 (BF16 +Inf in) elem " << i << " cls=" << static_cast<int>(cls);
    }

    // Block 2: -Inf BF16 in → NaN or -max-normal.
    uint8_t scale2 = mxfp6_tc::mxfp6_scale_byte_at(result, 2);
    for (uint32_t i = 64; i < 96; ++i) {
        auto cls = mxfp6_tc::effective_mxfp6_class(
            tt::DataFormat::MxFp6R, scale2, mxfp6_tc::mxfp6_elem_byte_at(result, layout, i));
        EXPECT_TRUE(cls == Cls::NaN || cls == Cls::MaxNormalNeg)
            << "block 2 (BF16 -Inf in) elem " << i << " cls=" << static_cast<int>(cls);
    }

    // Block 3: +1.0 sanity — verify each element round-trips exactly via the
    // float unpack (1.0 is exactly representable in MXFP6R with scale=2^0).
    auto floats = mxfp6_tc::mx_to_floats(tt::DataFormat::MxFp6R, result);
    for (uint32_t i = 96; i < 128; ++i) {
        EXPECT_EQ(floats[i], 1.0f) << "block 3 (BF16 +1.0 in) elem " << i;
    }
}

// ============================================================================
// Float16_b → MXFP6P special cases. Mirrors the MXFP6R test — same Inf/NaN
// expectations (MXFP6P is also finite-only).
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp6PSpecialCases) {
    const auto& mesh_device = *devices_[0];

    constexpr uint16_t kBf16PosNaN = 0x7FC0;
    constexpr uint16_t kBf16PosInf = 0x7F80;
    constexpr uint16_t kBf16NegInf = 0xFF80;
    constexpr uint16_t kBf16PosOne = 0x3F80;

    auto src = mxfp6_tc::build_bf16_tile_with_block_values({kBf16PosNaN, kBf16PosInf, kBf16NegInf, kBf16PosOne});

    auto result = mxfp6_tc::run_mxfp6_typecast(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::MxFp6P,
        src,
        /*num_tiles=*/1,
        /*fp32_dest_acc_en=*/false);

    auto layout = mxfp6_tc::get_mxfp6_tile_layout();
    using Cls = mxfp6_tc::MxFp6Class;

    uint8_t scale0 = mxfp6_tc::mxfp6_scale_byte_at(result, 0);
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_EQ(
            mxfp6_tc::effective_mxfp6_class(
                tt::DataFormat::MxFp6P, scale0, mxfp6_tc::mxfp6_elem_byte_at(result, layout, i)),
            Cls::NaN)
            << "block 0 (BF16 NaN in) elem " << i;
    }

    uint8_t scale1 = mxfp6_tc::mxfp6_scale_byte_at(result, 1);
    for (uint32_t i = 32; i < 64; ++i) {
        auto cls = mxfp6_tc::effective_mxfp6_class(
            tt::DataFormat::MxFp6P, scale1, mxfp6_tc::mxfp6_elem_byte_at(result, layout, i));
        EXPECT_TRUE(cls == Cls::NaN || cls == Cls::MaxNormalPos)
            << "block 1 (BF16 +Inf in) elem " << i << " cls=" << static_cast<int>(cls);
    }

    uint8_t scale2 = mxfp6_tc::mxfp6_scale_byte_at(result, 2);
    for (uint32_t i = 64; i < 96; ++i) {
        auto cls = mxfp6_tc::effective_mxfp6_class(
            tt::DataFormat::MxFp6P, scale2, mxfp6_tc::mxfp6_elem_byte_at(result, layout, i));
        EXPECT_TRUE(cls == Cls::NaN || cls == Cls::MaxNormalNeg)
            << "block 2 (BF16 -Inf in) elem " << i << " cls=" << static_cast<int>(cls);
    }

    auto floats = mxfp6_tc::mx_to_floats(tt::DataFormat::MxFp6P, result);
    for (uint32_t i = 96; i < 128; ++i) {
        EXPECT_EQ(floats[i], 1.0f) << "block 3 (BF16 +1.0 in) elem " << i;
    }
}

}  // namespace tt::tt_metal
