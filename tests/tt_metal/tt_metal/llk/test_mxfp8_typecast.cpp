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
#include <tt-metalium/mxfp8.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include <tt-logger/tt-logger.hpp>

#include "llk_device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/float8_utils.hpp"
#include "tt_metal/test_utils/mx_utils.hpp"

namespace tt::tt_metal {

using std::vector;

namespace unit_tests::llk::mxfp8_typecast {

using tt::test_utils::bf16_to_floats;
using tt::test_utils::check_pcc;
using tt::test_utils::is_close;
using tt::test_utils::is_close_vectors;
using tt::test_utils::mx_to_floats;
using tt::test_utils::pack_as_mx_tiles;

// Run a datacopy kernel with different input/output formats.
// For Quasar, data is moved via DataflowBuffers (DFBs) and the hardware
// unpacker/packer performs the format conversion implicitly.
static vector<uint32_t> run_mxfp8_typecast(
    distributed::MeshDevice& mesh_device,
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
        .name = "mxfp8_typecast",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    detail::WriteToBuffer(src_buffer, src_vec);
    // Pass aligned DRAM page stride so the reader/writer advance the DRAM
    // pointer by the allocator's aligned_page_size while the DFB streams the
    // native tile size (e.g. 1056 bytes for MxFp8 on Quasar; the allocator
    // rounds up to 1088 due to 64B DRAM alignment).
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

// Data generators follow the fp8_typecast tests' convention: generate
// row-major floats in U(0, rand_max_float) + offset, then pack into tiles.
static vector<uint32_t> create_random_vector_of_mxfp8(
    tt::DataFormat fmt, uint32_t num_bytes, int rand_max_float, int seed, float offset = 0.0f) {
    TT_FATAL(
        fmt == tt::DataFormat::MxFp8R || fmt == tt::DataFormat::MxFp8P,
        "Unsupported MXFP8 DataFormat: {}",
        static_cast<int>(fmt));
    uint32_t single_tile_size = tt::tile_size(fmt);
    TT_FATAL(
        num_bytes % single_tile_size == 0,
        "num_bytes {} must be divisible by MXFP8 tile_size {}",
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
        "MXFP8 packed size {} bytes does not match expected {} bytes",
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
        case tt::DataFormat::MxFp8R:
        case tt::DataFormat::MxFp8P: return create_random_vector_of_mxfp8(fmt, bytes, kRandMaxFloat, kSeed, kOffset);
        case tt::DataFormat::Float16_b: return create_random_vector_of_bfloat16(bytes, kRandMaxFloat, kSeed, kOffset);
        default: TT_THROW("Unsupported source DataFormat for mxfp8 typecast test: {}", static_cast<int>(fmt));
    }
}

static vector<float> unpack_to_floats(tt::DataFormat fmt, const vector<uint32_t>& packed) {
    switch (fmt) {
        case tt::DataFormat::MxFp8R:
        case tt::DataFormat::MxFp8P: return mx_to_floats(fmt, packed);
        case tt::DataFormat::Float16_b: return bf16_to_floats(packed);
        default: TT_THROW("Unsupported DataFormat for mxfp8 unpack: {}", static_cast<int>(fmt));
    }
}

// Drive one random typecast test: generate U(-10, 10) data in input_fmt, run
// the device, unpack both sides to floats, and check element-wise tolerance
// + PCC. Used by all random-input TEST_F bodies below.
static void run_random_typecast_test(
    distributed::MeshDevice& mesh_device,
    tt::DataFormat input_fmt,
    tt::DataFormat output_fmt,
    float rtol,
    float atol,
    double min_pcc,
    bool fp32_dest_acc_en) {
    auto src_vec = generate_random_src(input_fmt, kDefaultNumTiles);
    auto result_vec =
        run_mxfp8_typecast(mesh_device, input_fmt, output_fmt, src_vec, kDefaultNumTiles, fp32_dest_acc_en);
    auto src_floats = unpack_to_floats(input_fmt, src_vec);
    auto dst_floats = unpack_to_floats(output_fmt, result_vec);
    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [rtol, atol](float a, float b) { return is_close(a, b, rtol, atol); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, min_pcc));
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

static TileLayout get_e5m2_tile_layout() {
    constexpr uint32_t kTileHW = 1024;
    std::vector<float> zeros(kTileHW, 0.0f);
    auto packed = pack_as_mx_tiles(tt::DataFormat::MxFp8R, ttsl::make_const_span(zeros), /*row_major_input=*/true);
    const size_t elem_words = kTileHW / 4;
    const size_t exp_words = packed.size() - elem_words;
    return TileLayout{.total_words = packed.size(), .exp_bytes = exp_words * 4};
}

static TileLayout get_e4m3_tile_layout() {
    constexpr uint32_t kTileHW = 1024;
    std::vector<float> zeros(kTileHW, 0.0f);
    auto packed = pack_as_mx_tiles(tt::DataFormat::MxFp8P, ttsl::make_const_span(zeros), /*row_major_input=*/true);
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

// MXFP8 element-byte classifier. Values reflect the *element-only* class
// (ignoring the per-block scale). Combine with the scale byte via
// effective_*_class below: a NaN-scale block reads as NaN regardless of the
// element bits per the OCP MX rule.
enum class MxFp8Class { Zero, Subnormal, Normal, MaxNormalPos, MaxNormalNeg, PosInf, NegInf, NaN };

// E5M2: 1 sign / 5 exp / 2 mantissa, IEEE-style Inf/NaN.
//   Inf:        0_11111_00 / 1_11111_00 = 0x7C / 0xFC
//   NaN:        S_11111_xx with mant != 0
//   Max normal: 0_11110_11 / 1_11110_11 = 0x7B / 0xFB
static MxFp8Class classify_e5m2(uint8_t bits) {
    uint8_t sign = (bits >> 7) & 0x1u;
    uint8_t exp = (bits >> 2) & 0x1Fu;
    uint8_t mant = bits & 0x3u;
    if (exp == 0x1F) {
        if (mant == 0) {
            return sign ? MxFp8Class::NegInf : MxFp8Class::PosInf;
        }
        return MxFp8Class::NaN;
    }
    if (exp == 0) {
        return mant == 0 ? MxFp8Class::Zero : MxFp8Class::Subnormal;
    }
    if (exp == 0x1E && mant == 0x3) {
        return sign ? MxFp8Class::MaxNormalNeg : MxFp8Class::MaxNormalPos;
    }
    return MxFp8Class::Normal;
}

// E4M3FN (finite): 1 sign / 4 exp / 3 mantissa. No Inf encoding.
//   NaN:        S_1111_111 = 0x7F / 0xFF
//   Max normal: 0_1111_110 / 1_1111_110 = 0x7E / 0xFE
static MxFp8Class classify_e4m3(uint8_t bits) {
    uint8_t sign = (bits >> 7) & 0x1u;
    uint8_t exp = (bits >> 3) & 0xFu;
    uint8_t mant = bits & 0x7u;
    if (exp == 0xF && mant == 0x7) {
        return MxFp8Class::NaN;
    }
    if (exp == 0) {
        return mant == 0 ? MxFp8Class::Zero : MxFp8Class::Subnormal;
    }
    if (exp == 0xF && mant == 0x6) {
        return sign ? MxFp8Class::MaxNormalNeg : MxFp8Class::MaxNormalPos;
    }
    return MxFp8Class::Normal;
}

static uint8_t mxfp8_elem_byte_at(const vector<uint32_t>& packed, const TileLayout& layout, uint32_t i) {
    return reinterpret_cast<const uint8_t*>(packed.data())[layout.exp_bytes + i];
}

static uint8_t mxfp8_scale_byte_at(const vector<uint32_t>& packed, uint32_t block_idx) {
    return reinterpret_cast<const uint8_t*>(packed.data())[block_idx];
}

// Apply the OCP MX block-level rule: a block with scale = 0xFF reads as NaN
// for every element. Otherwise the element class stands.
static MxFp8Class effective_e5m2_class(uint8_t scale_byte, uint8_t elem_byte) {
    if (scale_byte == 0xFF) {
        return MxFp8Class::NaN;
    }
    return classify_e5m2(elem_byte);
}

static MxFp8Class effective_e4m3_class(uint8_t scale_byte, uint8_t elem_byte) {
    if (scale_byte == 0xFF) {
        return MxFp8Class::NaN;
    }
    return classify_e4m3(elem_byte);
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
    auto& mesh_device = *devices_[0];
    mxfp8_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp8R,
        tt::DataFormat::Float16_b,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/false);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8RToFloat16bFp32Dest) {
    auto& mesh_device = *devices_[0];
    mxfp8_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp8R,
        tt::DataFormat::Float16_b,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// MXFP8 (E4M3) → Float16_b
// Widening conversion: every MXFP8 E4M3 value (with a power-of-two block
// scale) is exactly representable in BF16.
// Expected: no precision loss → rtol=0.0, atol=0.0.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8PToFloat16b) {
    auto& mesh_device = *devices_[0];
    mxfp8_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp8P,
        tt::DataFormat::Float16_b,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/false);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8PToFloat16bFp32Dest) {
    auto& mesh_device = *devices_[0];
    mxfp8_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp8P,
        tt::DataFormat::Float16_b,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// Float16_b → MXFP8 (E5M2)
// Narrowing conversion: BF16 → MXFP8 E5M2 introduces quantization. E5M2
// has 2 mantissa bits. Tolerances are loose to account for block-scaling behavior.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp8R) {
    auto& mesh_device = *devices_[0];
    mxfp8_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::MxFp8R,
        /*rtol=*/0.25f,
        /*atol=*/0.1f,
        /*min_pcc=*/0.995,
        /*fp32_dest_acc_en=*/false);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp8RFp32Dest) {
    auto& mesh_device = *devices_[0];
    mxfp8_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::MxFp8R,
        /*rtol=*/0.25f,
        /*atol=*/0.1f,
        /*min_pcc=*/0.995,
        /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// Float16_b → MXFP8 (E4M3)
// Narrowing conversion: BF16 → MXFP8 E4M3 introduces quantization. E4M3
// has 3 mantissa bits. Tolerances are tighter than E5M2.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp8P) {
    auto& mesh_device = *devices_[0];
    mxfp8_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::MxFp8P,
        /*rtol=*/0.125f,
        /*atol=*/0.05f,
        /*min_pcc=*/0.999,
        /*fp32_dest_acc_en=*/false);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp8PFp32Dest) {
    auto& mesh_device = *devices_[0];
    mxfp8_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::MxFp8P,
        /*rtol=*/0.125f,
        /*atol=*/0.05f,
        /*min_pcc=*/0.999,
        /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// MXFP8 (E5M2) → MXFP8 (E5M2) (identity)
// Same format on both sides. The round-trip should be lossless.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8RToMxFp8R) {
    auto& mesh_device = *devices_[0];
    mxfp8_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp8R,
        tt::DataFormat::MxFp8R,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/false);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8RToMxFp8RFp32Dest) {
    auto& mesh_device = *devices_[0];
    mxfp8_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp8R,
        tt::DataFormat::MxFp8R,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// MXFP8 (E4M3) → MXFP8 (E4M3) (identity)
// Same format on both sides. The round-trip should be lossless.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8PToMxFp8P) {
    auto& mesh_device = *devices_[0];
    mxfp8_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp8P,
        tt::DataFormat::MxFp8P,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/false);
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8PToMxFp8PFp32Dest) {
    auto& mesh_device = *devices_[0];
    mxfp8_tc::run_random_typecast_test(
        mesh_device,
        tt::DataFormat::MxFp8P,
        tt::DataFormat::MxFp8P,
        /*rtol=*/0.0f,
        /*atol=*/0.0f,
        /*min_pcc=*/1.0,
        /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// Device special-case tests for hardware MXFP8 → BF16 typecast.
// Each test bundles multiple edge cases into a single tile (one rule per
// block, since each block has its own scale byte) to keep simulator runtime
// low. Output BF16 bits are inspected directly — is_close() can't be used
// for NaN/Inf comparisons.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxFp8RToBf16SpecialCases) {
    auto& mesh_device = *devices_[0];
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
        {{32, 0x7C}, {33, 0xFC}, {34, 0x7D}, {35, 0x7E}, {36, 0x7F}, {64, 0x7B}, {65, 0xFB}, {96, 0x3C}});

    auto result = mxfp8_tc::run_mxfp8_typecast(
        mesh_device,
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
    auto& mesh_device = *devices_[0];
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
        {{32, 0x7F}, {33, 0x78}, {34, 0x7B}, {35, 0x7E}, {64, 0x7E}, {65, 0xFE}, {96, 0x38}});

    auto result = mxfp8_tc::run_mxfp8_typecast(
        mesh_device,
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

// ============================================================================
// Device special-case tests for hardware BF16 → MXFP8 typecast (the narrowing
// direction). Each block of 32 BF16 inputs is filled with a single edge-case
// value so the per-block scale derivation is well defined; we then inspect
// the raw MXFP8 element bytes (combined with the block scale) since the
// random typecast tests above never exercise NaN, Inf, or block-scale rules.
// ============================================================================

namespace unit_tests::llk::mxfp8_typecast {

// Build a 1-tile BF16 input as raw bytes. Each block of 32 face-major BF16
// elements is filled with `block_values[b]`; defaults to BF16 +0 elsewhere.
// BF16 ordering matches the unpacker: two values per uint32, LSB = lower
// face-major index. Returns the packed uint32 vector ready for WriteToBuffer.
static vector<uint32_t> build_bf16_tile_with_block_values(std::initializer_list<uint16_t> block_values) {
    constexpr uint32_t kBf16PerTile = 1024;
    constexpr uint32_t kBlockSize = 32;
    uint32_t bf16_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
    TT_FATAL(bf16_tile_bytes == kBf16PerTile * sizeof(uint16_t), "Unexpected BF16 tile size {} bytes", bf16_tile_bytes);
    vector<uint32_t> packed(bf16_tile_bytes / sizeof(uint32_t), 0);
    auto* bf16 = reinterpret_cast<uint16_t*>(packed.data());
    uint32_t block = 0;
    for (uint16_t val : block_values) {
        TT_FATAL(block < kBf16PerTile / kBlockSize, "Too many block values");
        for (uint32_t i = 0; i < kBlockSize; ++i) {
            bf16[block * kBlockSize + i] = val;
        }
        ++block;
    }
    return packed;
}

}  // namespace unit_tests::llk::mxfp8_typecast

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp8RSpecialCases) {
    auto& mesh_device = *devices_[0];

    // Block layout (32 BF16 elements per block):
    //   0: all +NaN  → block must read as NaN (NaN propagation).
    //   1: all +Inf  → +Inf, NaN, or +max-normal (saturation when ovf_en=0).
    //   2: all -Inf  → symmetric.
    //   3: all +1.0  → sanity, must round-trip exactly to E5M2 1.0.
    constexpr uint16_t kBf16PosNaN = 0x7FC0;
    constexpr uint16_t kBf16PosInf = 0x7F80;
    constexpr uint16_t kBf16NegInf = 0xFF80;
    constexpr uint16_t kBf16PosOne = 0x3F80;

    auto src = mxfp8_tc::build_bf16_tile_with_block_values({kBf16PosNaN, kBf16PosInf, kBf16NegInf, kBf16PosOne});

    auto result = mxfp8_tc::run_mxfp8_typecast(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::MxFp8R,
        src,
        /*num_tiles=*/1,
        /*fp32_dest_acc_en=*/false);

    auto layout = mxfp8_tc::get_e5m2_tile_layout();
    using Cls = mxfp8_tc::MxFp8Class;

    // Block 0: NaN BF16 in → NaN out (per OCP MX, NaN forces scale=0xFF which
    // makes the entire block read as NaN).
    uint8_t scale0 = mxfp8_tc::mxfp8_scale_byte_at(result, 0);
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_EQ(mxfp8_tc::effective_e5m2_class(scale0, mxfp8_tc::mxfp8_elem_byte_at(result, layout, i)), Cls::NaN)
            << "block 0 (BF16 NaN in) elem " << i;
    }

    // Block 1: +Inf BF16 in → +Inf, NaN, or +max-normal. E5M2 has Inf encoding,
    // but the OCP MX overflow policy is configurable (ovf_en): with ovf_en=0
    // (the hardware default) Inf saturates to +max-normal; with ovf_en=1 it
    // round-trips as +Inf. NaN-scale fallback also accepted.
    uint8_t scale1 = mxfp8_tc::mxfp8_scale_byte_at(result, 1);
    for (uint32_t i = 32; i < 64; ++i) {
        auto cls = mxfp8_tc::effective_e5m2_class(scale1, mxfp8_tc::mxfp8_elem_byte_at(result, layout, i));
        EXPECT_TRUE(cls == Cls::PosInf || cls == Cls::NaN || cls == Cls::MaxNormalPos)
            << "block 1 (BF16 +Inf in) elem " << i << " cls=" << static_cast<int>(cls);
    }

    // Block 2: -Inf BF16 in → -Inf, NaN, or -max-normal (symmetric to block 1).
    uint8_t scale2 = mxfp8_tc::mxfp8_scale_byte_at(result, 2);
    for (uint32_t i = 64; i < 96; ++i) {
        auto cls = mxfp8_tc::effective_e5m2_class(scale2, mxfp8_tc::mxfp8_elem_byte_at(result, layout, i));
        EXPECT_TRUE(cls == Cls::NegInf || cls == Cls::NaN || cls == Cls::MaxNormalNeg)
            << "block 2 (BF16 -Inf in) elem " << i << " cls=" << static_cast<int>(cls);
    }

    // Block 3: +1.0 sanity — verify each element round-trips exactly via the
    // float unpack (1.0 is exactly representable in E5M2 with scale=2^0).
    auto floats = mxfp8_tc::mx_to_floats(tt::DataFormat::MxFp8R, result);
    for (uint32_t i = 96; i < 128; ++i) {
        EXPECT_EQ(floats[i], 1.0f) << "block 3 (BF16 +1.0 in) elem " << i;
    }
}

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxFp8PSpecialCases) {
    auto& mesh_device = *devices_[0];

    // Block layout, mirroring the E5M2 test. E4M3 has no Inf encoding, so the
    // ±Inf blocks must produce either NaN or saturate to ±max-normal.
    //   0: all +NaN  → block must read as NaN.
    //   1: all +Inf  → NaN or +max-normal (E4M3 cannot represent Inf).
    //   2: all -Inf  → NaN or -max-normal.
    //   3: all +1.0  → sanity, must round-trip exactly to E4M3 1.0.
    constexpr uint16_t kBf16PosNaN = 0x7FC0;
    constexpr uint16_t kBf16PosInf = 0x7F80;
    constexpr uint16_t kBf16NegInf = 0xFF80;
    constexpr uint16_t kBf16PosOne = 0x3F80;

    auto src = mxfp8_tc::build_bf16_tile_with_block_values({kBf16PosNaN, kBf16PosInf, kBf16NegInf, kBf16PosOne});

    auto result = mxfp8_tc::run_mxfp8_typecast(
        mesh_device,
        tt::DataFormat::Float16_b,
        tt::DataFormat::MxFp8P,
        src,
        /*num_tiles=*/1,
        /*fp32_dest_acc_en=*/false);

    auto layout = mxfp8_tc::get_e4m3_tile_layout();
    using Cls = mxfp8_tc::MxFp8Class;

    // Block 0: NaN BF16 in → NaN out.
    uint8_t scale0 = mxfp8_tc::mxfp8_scale_byte_at(result, 0);
    for (uint32_t i = 0; i < 32; ++i) {
        EXPECT_EQ(mxfp8_tc::effective_e4m3_class(scale0, mxfp8_tc::mxfp8_elem_byte_at(result, layout, i)), Cls::NaN)
            << "block 0 (BF16 NaN in) elem " << i;
    }

    // Block 1: +Inf BF16 in → NaN or +max-normal (saturation, since E4M3 has
    // sat_supported=true and elem_sat_pos_bits=0x7E).
    uint8_t scale1 = mxfp8_tc::mxfp8_scale_byte_at(result, 1);
    for (uint32_t i = 32; i < 64; ++i) {
        auto cls = mxfp8_tc::effective_e4m3_class(scale1, mxfp8_tc::mxfp8_elem_byte_at(result, layout, i));
        EXPECT_TRUE(cls == Cls::NaN || cls == Cls::MaxNormalPos)
            << "block 1 (BF16 +Inf in) elem " << i << " cls=" << static_cast<int>(cls);
    }

    // Block 2: -Inf → NaN or -max-normal.
    uint8_t scale2 = mxfp8_tc::mxfp8_scale_byte_at(result, 2);
    for (uint32_t i = 64; i < 96; ++i) {
        auto cls = mxfp8_tc::effective_e4m3_class(scale2, mxfp8_tc::mxfp8_elem_byte_at(result, layout, i));
        EXPECT_TRUE(cls == Cls::NaN || cls == Cls::MaxNormalNeg)
            << "block 2 (BF16 -Inf in) elem " << i << " cls=" << static_cast<int>(cls);
    }

    // Block 3: +1.0 sanity.
    auto floats = mxfp8_tc::mx_to_floats(tt::DataFormat::MxFp8P, result);
    for (uint32_t i = 96; i < 128; ++i) {
        EXPECT_EQ(floats[i], 1.0f) << "block 3 (BF16 +1.0 in) elem " << i;
    }
}

}  // namespace tt::tt_metal
