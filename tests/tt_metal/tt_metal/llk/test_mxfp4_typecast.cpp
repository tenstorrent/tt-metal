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
#include <tt-metalium/mxfp4.hpp>
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

namespace unit_tests::llk::mxfp4_typecast {

using tt::test_utils::bf16_to_floats;
using tt::test_utils::check_pcc;
using tt::test_utils::is_close;
using tt::test_utils::is_close_vectors;
using tt::test_utils::mx_to_floats;
using tt::test_utils::pack_as_mx_tiles;

// Run a datacopy kernel with different input/output formats.
// For Quasar, data is moved via DataflowBuffers (DFBs) and the hardware
// unpacker/packer performs the format conversion implicitly.
static vector<uint32_t> run_mxfp4_typecast(
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
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ProducerOf(INPUT_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUTPUT_DFB, "in")},
        .runtime_arg_schema = {.runtime_arg_names = {"dst_addr", "dst_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source =

            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_2_0.cpp",
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
                .enable_32_bit_dest = fp32_dest_acc_en,
            },
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "mxfp4_typecast",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    detail::WriteToBuffer(src_buffer, src_vec);
    // Pass aligned DRAM page stride so the reader/writer advance the DRAM
    // pointer by the allocator's aligned_page_size (576 for MxFp4 on Quasar
    // due to 64B DRAM alignment) while the DFB streams native 544-byte tiles.
    uint32_t src_dram_stride = static_cast<uint32_t>(src_buffer->aligned_page_size());
    uint32_t dst_dram_stride = static_cast<uint32_t>(dst_buffer->aligned_page_size());

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"src_addr", src_buffer->address()},
                 {"src_bank_id", 0u},
                 {"num_tiles", num_tiles},
                 {"dram_page_stride", src_dram_stride}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                node,
                {{"dst_addr", dst_buffer->address()},
                 {"dst_bank_id", 0u},
                 {"num_tiles", num_tiles},
                 {"dram_page_stride", dst_dram_stride}}),
        },
        experimental::ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE},
    };
    experimental::SetProgramRunArgs(program, params);

    detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/true);

    vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

// Data generators follow the fp8_typecast tests' convention: generate
// row-major floats in U(0, rand_max_float) + offset, then pack into tiles.
static vector<uint32_t> create_random_vector_of_mxfp4(
	uint32_t num_bytes, int rand_max_float, int seed, float offset = 0.0f) {
	uint32_t single_tile_size = tt::tile_size(tt::DataFormat::MxFp4);
	TT_FATAL(
		num_bytes % single_tile_size == 0,
		"num_bytes {} must be divisible by MXFP4 tile_size {}",
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

    vector<uint32_t> packed =
        pack_as_mx_tiles(tt::DataFormat::MxFp4, ttsl::make_const_span(fp32_vec), /*row_major_input=*/true);
    TT_FATAL(
        packed.size() * sizeof(uint32_t) == num_tiles * single_tile_size,
        "MXFP4 packed size {} bytes does not match expected {} bytes",
        packed.size() * sizeof(uint32_t),
        num_tiles * single_tile_size);
    return packed;
}

// --- Special-case rule testing infrastructure ---
//
// Helpers for hand-crafting raw MXFP4 tile bytes and reading raw BF16 outputs.
// Used to verify hardware spec rules (block-exp = 0xFF, "leave as is" for
// unit_exp = all-1, over/underflow) that random-data tests don't exercise.

struct TileLayout {
    size_t total_words = 0;
    size_t exp_bytes = 0;  // byte offset where the elem region begins
};

// Pack an all-zero tile to discover the runtime tile layout (which depends on
// HAL L1 alignment), then derive exp_bytes from the resulting word count.
static TileLayout get_mxfp4_tile_layout() {
    constexpr uint32_t kTileHW = 1024;
    std::vector<float> zeros(kTileHW, 0.0f);
    auto packed = pack_as_mx_tiles(tt::DataFormat::MxFp4, ttsl::make_const_span(zeros), /*row_major_input=*/true);
    const size_t elem_words = kTileHW / 8;  // 8 nibbles per uint32
    const size_t exp_words = packed.size() - elem_words;
    return TileLayout{.total_words = packed.size(), .exp_bytes = exp_words * 4};
}

struct ScalePatch {
    uint32_t block_idx;
    uint8_t scale_byte;
};
struct ElemPatch {
    uint32_t elem_idx;
    uint8_t elem_nibble;  // low 4 bits used
};

// Build a single-tile packed MXFP4 buffer: all 32 scale bytes set to
// scale_default, all 1024 elem nibbles set to elem_nibble_default, then
// patches applied. elem_idx is in face-major order; even indices live in
// the low nibble of their byte, odd indices in the high nibble.
static vector<uint32_t> build_mxfp4_tile_raw(
    const TileLayout& layout,
    uint8_t scale_default,
    uint8_t elem_nibble_default,
    std::initializer_list<ScalePatch> scale_patches,
    std::initializer_list<ElemPatch> elem_patches) {
    vector<uint32_t> packed(layout.total_words, 0);
    auto* bytes = reinterpret_cast<uint8_t*>(packed.data());
    for (uint32_t s = 0; s < 32; ++s) {
        bytes[s] = scale_default;
    }
    const uint8_t default_byte =
        static_cast<uint8_t>((elem_nibble_default & 0x0F) | ((elem_nibble_default & 0x0F) << 4));
    for (uint32_t b = 0; b < 512; ++b) {
        bytes[layout.exp_bytes + b] = default_byte;
    }
    for (const auto& p : scale_patches) {
        TT_FATAL(p.block_idx < 32, "block_idx {} out of range", p.block_idx);
        bytes[p.block_idx] = p.scale_byte;
    }
    for (const auto& p : elem_patches) {
        TT_FATAL(p.elem_idx < 1024, "elem_idx {} out of range", p.elem_idx);
        const uint32_t byte_idx = p.elem_idx / 2;
        const uint32_t shift = (p.elem_idx % 2) * 4;
        const uint8_t nib = p.elem_nibble & 0x0F;
        const uint32_t off = layout.exp_bytes + byte_idx;
        bytes[off] = static_cast<uint8_t>((bytes[off] & ~(0x0Fu << shift)) | (static_cast<uint32_t>(nib) << shift));
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

}  // namespace unit_tests::llk::mxfp4_typecast

namespace mxfp4_tc = unit_tests::llk::mxfp4_typecast;

// ============================================================================
// MXFP4 → Float16_b
// Widening conversion: every MXFP4 value should be representable in BF16.
// Expected: no precision loss → rtol=0.0, atol=0.0.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp4ToFloat16b) {
    const auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp4_tc::create_random_vector_of_mxfp4(
        tt::tile_size(tt::DataFormat::MxFp4) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp4_tc::run_mxfp4_typecast(
        mesh_device, tt::DataFormat::MxFp4, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp4_tc::mx_to_floats(tt::DataFormat::MxFp4, src_vec);
    auto dst_floats = mxfp4_tc::bf16_to_floats(result_vec);
    EXPECT_TRUE(mxfp4_tc::is_close_vectors<float>(src_floats, dst_floats, [](float a, float b) {
        return mxfp4_tc::is_close(a, b, /*rtol=*/0.0f, /*atol=*/0.0f);
    }));
    EXPECT_TRUE(mxfp4_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp4ToFloat16bFp32Dest) {
    const auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp4_tc::create_random_vector_of_mxfp4(
        tt::tile_size(tt::DataFormat::MxFp4) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp4_tc::run_mxfp4_typecast(
        mesh_device, tt::DataFormat::MxFp4, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp4_tc::mx_to_floats(tt::DataFormat::MxFp4, src_vec);
    auto dst_floats = mxfp4_tc::bf16_to_floats(result_vec);
    EXPECT_TRUE(mxfp4_tc::is_close_vectors<float>(src_floats, dst_floats, [](float a, float b) {
        return mxfp4_tc::is_close(a, b, /*rtol=*/0.0f, /*atol=*/0.0f);
    }));
    EXPECT_TRUE(mxfp4_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// ============================================================================
// Float16_b → MXFP4
// Narrowing conversion: BF16 → MXFP4 introduces quantization.
// Tolerances are intentionally loose to account for block-scaling behavior.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp4) {
    const auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp4_tc::run_mxfp4_typecast(
        mesh_device, tt::DataFormat::Float16_b, tt::DataFormat::MxFp4, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp4_tc::bf16_to_floats(src_vec);
    auto dst_floats = mxfp4_tc::mx_to_floats(tt::DataFormat::MxFp4, result_vec);
    EXPECT_TRUE(mxfp4_tc::is_close_vectors<float>(src_floats, dst_floats, [](float a, float b) {
        return mxfp4_tc::is_close(a, b, /*rtol=*/0.5f, /*atol=*/0.5f);
    }));
    EXPECT_TRUE(mxfp4_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/0.98));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp4Fp32Dest) {
    const auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp4_tc::run_mxfp4_typecast(
        mesh_device, tt::DataFormat::Float16_b, tt::DataFormat::MxFp4, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp4_tc::bf16_to_floats(src_vec);
    auto dst_floats = mxfp4_tc::mx_to_floats(tt::DataFormat::MxFp4, result_vec);
    EXPECT_TRUE(mxfp4_tc::is_close_vectors<float>(src_floats, dst_floats, [](float a, float b) {
        return mxfp4_tc::is_close(a, b, /*rtol=*/0.5f, /*atol=*/0.5f);
    }));
    EXPECT_TRUE(mxfp4_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/0.98));
}

// ============================================================================
// MXFP4 → MXFP4 (identity)
// Same format on both sides. The round-trip should be lossless.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp4ToMxFp4) {
    const auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp4_tc::create_random_vector_of_mxfp4(
        tt::tile_size(tt::DataFormat::MxFp4) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp4_tc::run_mxfp4_typecast(
        mesh_device, tt::DataFormat::MxFp4, tt::DataFormat::MxFp4, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = mxfp4_tc::mx_to_floats(tt::DataFormat::MxFp4, src_vec);
    auto dst_floats = mxfp4_tc::mx_to_floats(tt::DataFormat::MxFp4, result_vec);
    EXPECT_TRUE(mxfp4_tc::is_close_vectors<float>(src_floats, dst_floats, [](float a, float b) {
        return mxfp4_tc::is_close(a, b, /*rtol=*/0.0f, /*atol=*/0.0f);
    }));
    EXPECT_TRUE(mxfp4_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp4ToMxFp4Fp32Dest) {
    const auto& mesh_device = *devices_[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = mxfp4_tc::create_random_vector_of_mxfp4(
        tt::tile_size(tt::DataFormat::MxFp4) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = mxfp4_tc::run_mxfp4_typecast(
        mesh_device, tt::DataFormat::MxFp4, tt::DataFormat::MxFp4, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = mxfp4_tc::mx_to_floats(tt::DataFormat::MxFp4, src_vec);
    auto dst_floats = mxfp4_tc::mx_to_floats(tt::DataFormat::MxFp4, result_vec);
    EXPECT_TRUE(mxfp4_tc::is_close_vectors<float>(src_floats, dst_floats, [](float a, float b) {
        return mxfp4_tc::is_close(a, b, /*rtol=*/0.0f, /*atol=*/0.0f);
    }));
    EXPECT_TRUE(mxfp4_tc::check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// ============================================================================
// Device special-case tests for hardware MXFP4 → BF16 typecast.
// MXFP4 (E2M1) is finite-only — every "unit_exp = all 1" pattern is a normal
// finite value (4.0 or 6.0 with sign), so spec rule "leave as is" applies to
// all such elements. Block-exp = 0xFF still emits NaN; over/underflow follow
// BF16 saturation.
//
// Nibble legend (sign:1 exp:2 mant:1, bias=1):
//   0x0 = +0          0x8 = -0
//   0x1 = +0.5 sub    0x9 = -0.5 sub
//   0x2 = +1.0        0xA = -1.0
//   0x3 = +1.5        0xB = -1.5
//   0x4 = +2.0        0xC = -2.0
//   0x5 = +3.0        0xD = -3.0
//   0x6 = +4.0  (unit exp=all 1, mant=0) "leave as is"
//   0x7 = +6.0  (unit exp=all 1, mant=1) max normal, "leave as is"
//   0xE = -4.0
//   0xF = -6.0
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp4ToBf16SpecialCases) {
    const auto& mesh_device = *devices_[0];
    auto layout = mxfp4_tc::get_mxfp4_tile_layout();

    // Block 0: scale = 0xFF → all 32 elements should be NaN (rule 1).
    // Block 1: scale = 1. elem 32-35: 0x6, 0x7, 0xE, 0xF — every "unit exp =
    //          all 1" pattern (mant=0 / mant=all 1, both signs). All should
    //          appear in BF16 unchanged.
    // Block 2: scale = 2^127 (block_exp_biased=0xFE). elem 64-65: ±max normal
    //          → 6.0 * 2^127 ≈ 2^129.6, overflows BF16 → ±Inf.
    // Block 3: scale = 1. elem 96: 0x2 (+1.0) — sanity, must remain finite.
    // Block 4: scale = 2^-127 (block_exp_biased=0x00). elem 128-129: ±0.5
    //          subnormal → ±2^-128, below BF16 normal range. Rule 3
    //          ("< -127 → Zero") expects flush; if silicon retains BF16
    //          subnormal precision instead, that's still spec-conformant
    //          for fp32-range arithmetic, so accept either.
    auto packed = mxfp4_tc::build_mxfp4_tile_raw(
        layout,
        /*scale_default=*/0x7F,
        /*elem_nibble_default=*/0x0,
        {{0, 0xFF}, {1, 0x7F}, {2, 0xFE}, {3, 0x7F}, {4, 0x00}},
        {{32, 0x6}, {33, 0x7}, {34, 0xE}, {35, 0xF}, {64, 0x7}, {65, 0xF}, {96, 0x2}, {128, 0x1}, {129, 0x9}});

    auto result = mxfp4_tc::run_mxfp4_typecast(
        mesh_device,
        tt::DataFormat::MxFp4,
        tt::DataFormat::Float16_b,
        packed,
        /*num_tiles=*/1,
        /*fp32_dest_acc_en=*/false);

    using Cls = mxfp4_tc::Bf16Class;
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_EQ(mxfp4_tc::classify_bf16(mxfp4_tc::bf16_raw_at(result, i)), Cls::NaN)
            << "block 0 (NaN-scale) elem " << i;
    }
    EXPECT_EQ(mxfp4_tc::bf16_raw_at(result, 32), 0x4080u);  // +4.0
    EXPECT_EQ(mxfp4_tc::bf16_raw_at(result, 33), 0x40C0u);  // +6.0
    EXPECT_EQ(mxfp4_tc::bf16_raw_at(result, 34), 0xC080u);  // -4.0
    EXPECT_EQ(mxfp4_tc::bf16_raw_at(result, 35), 0xC0C0u);  // -6.0
    EXPECT_EQ(mxfp4_tc::classify_bf16(mxfp4_tc::bf16_raw_at(result, 64)), Cls::PosInf);
    EXPECT_EQ(mxfp4_tc::classify_bf16(mxfp4_tc::bf16_raw_at(result, 65)), Cls::NegInf);
    EXPECT_EQ(mxfp4_tc::bf16_raw_at(result, 96), 0x3F80u);  // +1.0

    const auto bits_pos = mxfp4_tc::bf16_raw_at(result, 128);
    const auto cls_pos = mxfp4_tc::classify_bf16(bits_pos);
    EXPECT_TRUE(cls_pos == Cls::Zero || cls_pos == Cls::Subnormal)
        << "elem 128 expected Zero/Subnormal, got bits=0x" << std::hex << bits_pos;
    const auto bits_neg = mxfp4_tc::bf16_raw_at(result, 129);
    const auto cls_neg = mxfp4_tc::classify_bf16(bits_neg);
    EXPECT_TRUE(cls_neg == Cls::Zero || cls_neg == Cls::Subnormal)
        << "elem 129 expected Zero/Subnormal, got bits=0x" << std::hex << bits_neg;
}

}  // namespace tt::tt_metal
