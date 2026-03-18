// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <vector>

#include <tt-metalium/float8.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/span.hpp>
#include "device_fixture.hpp"

namespace tt::tt_metal {

using std::vector;

namespace unit_tests::llk::fp8_typecast {

// Run a datacopy kernel with different input/output CB formats.
// The hardware unpacker reads input_fmt and the packer writes output_fmt,
// performing the format conversion implicitly. fp32_dest_acc_en controls
// whether the Dest register operates in 32-bit mode.
static vector<uint32_t> run_fp8_typecast(
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

    std::map<std::string, std::string> defines = {};
    if (!fp32_dest_acc_en and input_fmt == tt::DataFormat::Fp8_e4m3) {
        if (output_fmt == tt::DataFormat::Float16_b) {
            defines["PACK_A_TO_B"] = "DataFormat::Float16";
        } else if (output_fmt == tt::DataFormat::Bfp8_b) {
            defines["PACK_A_TO_B"] = "DataFormat::Bfp8";
        } else {
            throw std::runtime_error("Invalid output format");
        }
    }
    if (!fp32_dest_acc_en and output_fmt == tt::DataFormat::Fp8_e4m3) {
        if (input_fmt == tt::DataFormat::Float16_b or input_fmt == tt::DataFormat::Bfp8_b) {
            defines["PACK_B_TO_A"] = "DataFormat::Float16_b";
        } else {
            throw std::runtime_error("Invalid input format");
        }
    }

    auto reader = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_fp8.cpp",
        core,
        ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {num_tiles}, .defines = defines});

    detail::WriteToBuffer(src_buffer, src_vec);
    SetRuntimeArgs(program, reader, core, {src_buffer->address(), 0, num_tiles});
    SetRuntimeArgs(program, writer, core, {dst_buffer->address(), 0, num_tiles});

    detail::LaunchProgram(dev, program);

    vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_buffer, result_vec);
    return result_vec;
}

// Data generators use the existing uniform-distribution helpers:
//   create_random_vector_of_bfloat16  (bfloat16.hpp)
//   tt::test_utils::create_random_vector_of_bfp8  (bfloat_utils.hpp)
// Both generate U(0, rand_max_float) + offset, so passing rand_max_float=20
// and offset=-10 yields U(-10, 10).

// --- Format-to-float unpackers ---

static vector<float> fp8_to_floats(const vector<uint32_t>& packed) {
    auto fp8_vec = unpack_uint32_vec_into_float8_e4m3_vec(packed);
    vector<float> floats;
    floats.reserve(fp8_vec.size());
    for (const auto& v : fp8_vec) {
        floats.push_back(static_cast<float>(v));
    }
    return floats;
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

static vector<float> bfp8_to_floats(const vector<uint32_t>& packed) {
    return unpack_bfp8_tiles_into_float_vec(
        tt::stl::make_const_span(packed), /*row_major_output=*/false, /*is_exp_a=*/false);
}

// --- Validation ---

static bool check_floats_close(const vector<float>& a, const vector<float>& b, float rtol, float atol) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (!is_close(a[i], b[i], rtol, atol)) {
            std::cerr << "check_floats_close: mismatch at index " << i << " - a[i] = " << a[i] << ", b[i] = " << b[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

static double compute_pcc(const vector<float>& a, const vector<float>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }
    double mean_a = 0.0, mean_b = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= static_cast<double>(a.size());
    mean_b /= static_cast<double>(b.size());
    double num = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double da = a[i] - mean_a;
        double db = b[i] - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if (denom_a == 0.0 || denom_b == 0.0) {
        return 1.0;
    }
    return num / std::sqrt(denom_a * denom_b);
}

static bool check_pcc(const vector<float>& a, const vector<float>& b, double min_pcc) {
    double pcc = compute_pcc(a, b);
    if (pcc < min_pcc) {
        std::cerr << "check_pcc: PCC = " << pcc << " < min_pcc = " << min_pcc << std::endl;
        return false;
    }
    return true;
}

}  // namespace unit_tests::llk::fp8_typecast

using namespace unit_tests::llk::fp8_typecast;

// ============================================================================
// fp8_e4m3 → Float16_b
// Widening conversion: every fp8 value is exactly representable in BF16.
// Expected: no precision loss → rtol=0.0, atol=0.0.
// ============================================================================

TEST_F(BlackholeSingleCardFixture, TensixFp8e4m3ToFloat16b) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_float8_e4m3(
        tt::tile_size(tt::DataFormat::Fp8_e4m3) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        dev, tt::DataFormat::Fp8_e4m3, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = fp8_to_floats(src_vec);
    auto dst_floats = bf16_to_floats(result_vec);
    EXPECT_TRUE(check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

TEST_F(BlackholeSingleCardFixture, TensixFp8e4m3ToFloat16bFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_float8_e4m3(
        tt::tile_size(tt::DataFormat::Fp8_e4m3) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        dev, tt::DataFormat::Fp8_e4m3, tt::DataFormat::Float16_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = fp8_to_floats(src_vec);
    auto dst_floats = bf16_to_floats(result_vec);
    EXPECT_TRUE(check_floats_close(src_floats, dst_floats, /*rtol=*/0.0f, /*atol=*/0.0f));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/1.0));
}

// ============================================================================
// Float16_b → fp8_e4m3
// Narrowing: BF16 has 7 mantissa bits vs fp8's 3 → precision loss expected.
// rtol=0.125 covers the max relative quantization error of fp8 (~1/8).
// ============================================================================

TEST_F(BlackholeSingleCardFixture, TensixFloat16bToFp8e4m3) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        dev, tt::DataFormat::Float16_b, tt::DataFormat::Fp8_e4m3, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = bf16_to_floats(src_vec);
    auto dst_floats = fp8_to_floats(result_vec);
    EXPECT_TRUE(check_floats_close(src_floats, dst_floats, /*rtol=*/0.125f, /*atol=*/0.015625f));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.999));
}

TEST_F(BlackholeSingleCardFixture, TensixFloat16bToFp8e4m3Fp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_bfloat16(
        tt::tile_size(tt::DataFormat::Float16_b) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        dev, tt::DataFormat::Float16_b, tt::DataFormat::Fp8_e4m3, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = bf16_to_floats(src_vec);
    auto dst_floats = fp8_to_floats(result_vec);
    EXPECT_TRUE(check_floats_close(src_floats, dst_floats, /*rtol=*/0.125f, /*atol=*/0.015625f));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.999));
}

// ============================================================================
// fp8_e4m3 → Bfp8_b
// Widening: Bfp8_b has 8 mantissa bits and a shared exponent per 16-element
// row. For test data within [-10, 10], fp8 values may lose significant
// precision due to the blocking forming process.
// ============================================================================

TEST_F(BlackholeSingleCardFixture, TensixFp8e4m3ToBfp8b) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_float8_e4m3(
        tt::tile_size(tt::DataFormat::Fp8_e4m3) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        dev, tt::DataFormat::Fp8_e4m3, tt::DataFormat::Bfp8_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = fp8_to_floats(src_vec);
    auto dst_floats = bfp8_to_floats(result_vec);
    EXPECT_TRUE(check_floats_close(src_floats, dst_floats, /*rtol=*/0.3f, /*atol=*/0.3f));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.9999));
}

TEST_F(BlackholeSingleCardFixture, TensixFp8e4m3ToBfp8bFp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = create_random_vector_of_float8_e4m3(
        tt::tile_size(tt::DataFormat::Fp8_e4m3) * num_tiles, /*rand_max_float=*/20, /*seed=*/42, /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        dev, tt::DataFormat::Fp8_e4m3, tt::DataFormat::Bfp8_b, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = fp8_to_floats(src_vec);
    auto dst_floats = bfp8_to_floats(result_vec);
    EXPECT_TRUE(check_floats_close(src_floats, dst_floats, /*rtol=*/0.3f, /*atol=*/0.3f));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.9999));
}

// ============================================================================
// Bfp8_b → fp8_e4m3
// Narrowing: Bfp8_b has 8 mantissa bits vs fp8's 3 → precision loss expected.
// ============================================================================

TEST_F(BlackholeSingleCardFixture, TensixBfp8bToFp8e4m3) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = tt::test_utils::create_random_vector_of_bfp8(
        tt::tile_size(tt::DataFormat::Bfp8_b) * num_tiles,
        /*is_exp_a=*/false,
        /*rand_max_float=*/20,
        /*seed=*/42,
        /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        dev, tt::DataFormat::Bfp8_b, tt::DataFormat::Fp8_e4m3, src_vec, num_tiles, /*fp32_dest_acc_en=*/false);
    auto src_floats = bfp8_to_floats(src_vec);
    auto dst_floats = fp8_to_floats(result_vec);
    EXPECT_TRUE(check_floats_close(src_floats, dst_floats, /*rtol=*/0.125f, /*atol=*/0.015625f));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.999));
}

TEST_F(BlackholeSingleCardFixture, TensixBfp8bToFp8e4m3Fp32Dest) {
    IDevice* dev = devices_[0]->get_devices()[0];
    constexpr uint32_t num_tiles = 64;
    auto src_vec = tt::test_utils::create_random_vector_of_bfp8(
        tt::tile_size(tt::DataFormat::Bfp8_b) * num_tiles,
        /*is_exp_a=*/false,
        /*rand_max_float=*/20,
        /*seed=*/42,
        /*offset=*/-10.0f);
    auto result_vec = run_fp8_typecast(
        dev, tt::DataFormat::Bfp8_b, tt::DataFormat::Fp8_e4m3, src_vec, num_tiles, /*fp32_dest_acc_en=*/true);
    auto src_floats = bfp8_to_floats(src_vec);
    auto dst_floats = fp8_to_floats(result_vec);
    EXPECT_TRUE(check_floats_close(src_floats, dst_floats, /*rtol=*/0.125f, /*atol=*/0.015625f));
    EXPECT_TRUE(check_pcc(src_floats, dst_floats, /*min_pcc=*/0.999));
}

}  // namespace tt::tt_metal
