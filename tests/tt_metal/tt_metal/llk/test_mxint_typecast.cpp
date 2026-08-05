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
#include <tt-metalium/mxint.hpp>
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

namespace unit_tests::llk::mxint_typecast {

// Run a datacopy kernel with different input/output formats.
// For Quasar, data is moved via DataflowBuffers (DFBs) and the hardware
// unpacker/packer performs the format conversion implicitly. Identical to the
// MXFP4/MXFP8 typecast drivers — only the format(s) under test differ.
static vector<uint32_t> run_mxint_typecast(
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
        .page_size = num_tiles * input_tile_size,
        .buffer_type = BufferType::DRAM};
    auto src_buffer = CreateBuffer(src_config);

    InterleavedBufferConfig dst_config{
        .device = dev,
        .size = num_tiles * output_tile_size,
        .page_size = num_tiles * output_tile_size,
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
        .dfb_bindings = {experimental::ProducerOf(INPUT_DFB, "out")},
        .runtime_arg_schema = {.runtime_arg_names = {"src_addr", "src_bank_id", "num_tiles", "dram_page_stride"}},
        .hw_config = experimental::DataMovementGen2Config{},
    };

    experimental::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary_2_0.cpp",
        .num_threads = 1,
        .dfb_bindings = {experimental::ConsumerOf(OUTPUT_DFB, "in")},
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
                .enable_32_bit_dest = fp32_dest_acc_en,
            },
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "mxint_typecast",
        .kernels = {reader_spec, writer_spec, compute_spec},
        .dataflow_buffers = {input_dfb_spec, output_dfb_spec},
        .work_units = {wu},
    };

    Program program = experimental::MakeProgramFromSpec(mesh_device, spec);

    detail::WriteToBuffer(src_buffer, src_vec);
    // These simple test kernels take an explicit bank id and linear address,
    // so keep each buffer as one DRAM page and walk tiles contiguously within
    // bank 0 instead of using interleaved per-tile pages.
    uint32_t src_dram_stride = input_tile_size;
    uint32_t dst_dram_stride = output_tile_size;

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

// --- Format-aware pack / unpack dispatch ---
// Shared across all MX typecast tests; lives in tt_metal/test_utils/mx_utils.hpp.
using tt::test_utils::mx_to_floats;
using tt::test_utils::pack_as_mx_tiles;

// Generate random MxInt source tiles: U(0, rand_max) + offset floats packed
// into the format. Follows the MXFP4/MXFP8 typecast convention.
static vector<uint32_t> create_random_vector_of_mxint(
    tt::DataFormat fmt, uint32_t num_bytes, int rand_max_float, int seed, float offset = 0.0f) {
    uint32_t single_tile_size = tt::tile_size(fmt);
    TT_FATAL(
        num_bytes % single_tile_size == 0,
        "num_bytes {} must be divisible by MxInt tile_size {}",
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

    vector<uint32_t> packed = pack_as_mx_tiles(fmt, ttsl::make_const_span(fp32_vec), /*row_major_input=*/true);
    TT_FATAL(
        packed.size() * sizeof(uint32_t) == num_tiles * single_tile_size,
        "MxInt packed size {} bytes does not match expected {} bytes",
        packed.size() * sizeof(uint32_t),
        num_tiles * single_tile_size);
    return packed;
}

// bf16_to_floats lives in tt_metal/test_utils/float8_utils.hpp; expose it here.
using tt::test_utils::bf16_to_floats;

// --- Validation ---
using tt::test_utils::check_pcc;
using tt::test_utils::is_close;
using tt::test_utils::is_close_vectors;

// Shared random-test parameters (centralized).
constexpr uint32_t kDefaultNumTiles = 64;
constexpr int kRandMaxFloat = 20;
constexpr int kSeed = 42;
constexpr float kOffset = -10.0f;  // U(0, 20) + (-10) = U(-10, 10)

// Widening (MxInt -> BF16) and identity (MxInt -> MxInt): the source MxInt
// values are exactly representable in BF16 (signed-int magnitude with a
// power-of-two block scale), so the conversion is lossless. We compare the
// device output against the host-decoded source — both must be bit-identical.
static void run_widening_or_identity_test(
    distributed::MeshDevice& mesh_device,
    tt::DataFormat input_fmt,   // an MxInt format
    tt::DataFormat output_fmt,  // Float16_b or the same MxInt format
    bool fp32_dest_acc_en) {
    uint32_t bytes = tt::tile_size(input_fmt) * kDefaultNumTiles;
    auto src_vec = create_random_vector_of_mxint(input_fmt, bytes, kRandMaxFloat, kSeed, kOffset);
    auto result_vec =
        run_mxint_typecast(mesh_device, input_fmt, output_fmt, src_vec, kDefaultNumTiles, fp32_dest_acc_en);

    auto src_floats = mx_to_floats(input_fmt, src_vec);
    auto dst_floats =
        (output_fmt == tt::DataFormat::Float16_b) ? bf16_to_floats(result_vec) : mx_to_floats(output_fmt, result_vec);

    EXPECT_TRUE(is_close_vectors<float>(
        src_floats, dst_floats, [](float a, float b) { return is_close(a, b, /*rtol=*/0.0f, /*atol=*/0.0f); }));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// Narrowing (BF16 -> MxInt): quantization is lossy, so comparing the device
// output against the original BF16 input is only meaningful for the finest
// format (MxInt2 has just three representable levels). Instead we compare the
// device-quantized result against the host *reference* packer's result: this
// validates that the hardware quantizer matches the OCP MX golden bit-for-bit
// (modulo rare rounding ties absorbed by `atol`).
static void run_narrowing_test(
    distributed::MeshDevice& mesh_device,
    tt::DataFormat output_fmt,  // an MxInt format
    float atol,
    bool fp32_dest_acc_en) {
    uint32_t bytes = tt::tile_size(tt::DataFormat::Float16_b) * kDefaultNumTiles;
    auto src_vec = create_random_vector_of_bfloat16(bytes, kRandMaxFloat, kSeed, kOffset);
    auto result_vec = run_mxint_typecast(
        mesh_device, tt::DataFormat::Float16_b, output_fmt, src_vec, kDefaultNumTiles, fp32_dest_acc_en);

    auto src_floats = bf16_to_floats(src_vec);  // storage (tile-major) order
    auto ref_packed = pack_as_mx_tiles(output_fmt, ttsl::make_const_span(src_floats), /*row_major_input=*/false);

    auto ref_floats = mx_to_floats(output_fmt, ref_packed);
    auto hw_floats = mx_to_floats(output_fmt, result_vec);

    EXPECT_TRUE(is_close_vectors<float>(
        ref_floats, hw_floats, [atol](float a, float b) { return is_close(a, b, /*rtol=*/0.0f, atol); }));
    EXPECT_TRUE(check_pcc(ref_floats, hw_floats, /*min_pcc=*/0.999));
}

}  // namespace unit_tests::llk::mxint_typecast

namespace mxint_tc = unit_tests::llk::mxint_typecast;

// ============================================================================
// MxInt -> Float16_b (widening, lossless).
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxInt8ToFloat16b) {
    mxint_tc::run_widening_or_identity_test(
        *devices_[0], tt::DataFormat::MxInt8, tt::DataFormat::Float16_b, /*fp32_dest_acc_en=*/false);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxInt8ToFloat16bFp32Dest) {
    mxint_tc::run_widening_or_identity_test(
        *devices_[0], tt::DataFormat::MxInt8, tt::DataFormat::Float16_b, /*fp32_dest_acc_en=*/true);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxInt4ToFloat16b) {
    mxint_tc::run_widening_or_identity_test(
        *devices_[0], tt::DataFormat::MxInt4, tt::DataFormat::Float16_b, /*fp32_dest_acc_en=*/false);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxInt4ToFloat16bFp32Dest) {
    mxint_tc::run_widening_or_identity_test(
        *devices_[0], tt::DataFormat::MxInt4, tt::DataFormat::Float16_b, /*fp32_dest_acc_en=*/true);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxInt2ToFloat16b) {
    mxint_tc::run_widening_or_identity_test(
        *devices_[0], tt::DataFormat::MxInt2, tt::DataFormat::Float16_b, /*fp32_dest_acc_en=*/false);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxInt2ToFloat16bFp32Dest) {
    mxint_tc::run_widening_or_identity_test(
        *devices_[0], tt::DataFormat::MxInt2, tt::DataFormat::Float16_b, /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// Float16_b -> MxInt (narrowing). atol is one quantization step at the coarsest
// block scale exercised by U(-10, 10): step = 2^floor(log2(10)) / elem_int_scale
// = 8 / {64, 4, 1} for MxInt8/4/2.
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxInt8) {
    mxint_tc::run_narrowing_test(*devices_[0], tt::DataFormat::MxInt8, /*atol=*/0.125f, /*fp32_dest_acc_en=*/false);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxInt8Fp32Dest) {
    mxint_tc::run_narrowing_test(*devices_[0], tt::DataFormat::MxInt8, /*atol=*/0.125f, /*fp32_dest_acc_en=*/true);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxInt4) {
    mxint_tc::run_narrowing_test(*devices_[0], tt::DataFormat::MxInt4, /*atol=*/2.0f, /*fp32_dest_acc_en=*/false);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxInt4Fp32Dest) {
    mxint_tc::run_narrowing_test(*devices_[0], tt::DataFormat::MxInt4, /*atol=*/2.0f, /*fp32_dest_acc_en=*/true);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxInt2) {
    mxint_tc::run_narrowing_test(*devices_[0], tt::DataFormat::MxInt2, /*atol=*/8.0f, /*fp32_dest_acc_en=*/false);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixFloat16bToMxInt2Fp32Dest) {
    mxint_tc::run_narrowing_test(*devices_[0], tt::DataFormat::MxInt2, /*atol=*/8.0f, /*fp32_dest_acc_en=*/true);
}

// ============================================================================
// MxInt -> MxInt (identity, lossless round-trip).
// ============================================================================

TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxInt8ToMxInt8) {
    mxint_tc::run_widening_or_identity_test(
        *devices_[0], tt::DataFormat::MxInt8, tt::DataFormat::MxInt8, /*fp32_dest_acc_en=*/false);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxInt8ToMxInt8Fp32Dest) {
    mxint_tc::run_widening_or_identity_test(
        *devices_[0], tt::DataFormat::MxInt8, tt::DataFormat::MxInt8, /*fp32_dest_acc_en=*/true);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxInt4ToMxInt4) {
    mxint_tc::run_widening_or_identity_test(
        *devices_[0], tt::DataFormat::MxInt4, tt::DataFormat::MxInt4, /*fp32_dest_acc_en=*/false);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxInt4ToMxInt4Fp32Dest) {
    mxint_tc::run_widening_or_identity_test(
        *devices_[0], tt::DataFormat::MxInt4, tt::DataFormat::MxInt4, /*fp32_dest_acc_en=*/true);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxInt2ToMxInt2) {
    mxint_tc::run_widening_or_identity_test(
        *devices_[0], tt::DataFormat::MxInt2, tt::DataFormat::MxInt2, /*fp32_dest_acc_en=*/false);
}
TEST_F(QuasarMeshDeviceSingleCardFixture, TensixMxInt2ToMxInt2Fp32Dest) {
    mxint_tc::run_widening_or_identity_test(
        *devices_[0], tt::DataFormat::MxInt2, tt::DataFormat::MxInt2, /*fp32_dest_acc_en=*/true);
}

}  // namespace tt::tt_metal
