// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/span.hpp>
#include <umd/device/types/arch.hpp>

#include "device_fixture.hpp"
#include "test_golden_impls.hpp"
#include "tt_metal/test_utils/packing.hpp"

namespace tt::tt_metal {

namespace unit_tests::compute::sfpu_binary_bcast {

enum class BcastDim : uint8_t {
    COL = 0,  // replicate col 0 of bcast tile across all 32 cols
    ROW = 1,  // replicate row 0 of bcast tile across all 32 rows
};

enum class BinOp : uint8_t {
    ADD = 0,
    SUB = 1,
    MUL = 2,
};

struct SfpuBcastConfig {
    BcastDim bcast_dim;
    BinOp binop;
    tt::DataFormat data_format;
};

// Format a BcastDim/BinOp/DataFormat for test log output only.
const std::map<BcastDim, std::string> bcast_dim_name = {
    {BcastDim::COL, "BCAST_COL"},
    {BcastDim::ROW, "BCAST_ROW"},
};
const std::map<BinOp, std::string> binop_name = {
    {BinOp::ADD, "ADD"},
    {BinOp::SUB, "SUB"},
    {BinOp::MUL, "MUL"},
};
const std::map<tt::DataFormat, std::string> df_name = {
    {tt::DataFormat::Float32, "FP32"}, {tt::DataFormat::Float16_b, "FP16_B"},
    // {tt::DataFormat::Bfp8_b, "BFP8_B"},
};

constexpr uint32_t kTileHW = 32 * 32;
constexpr uint32_t kNumTiles = 1;  // kernel operates on a single 32x32 tile

// -----------------------------------------------------------------------------
// Stimulus + golden
// -----------------------------------------------------------------------------

std::vector<float> generate_random_tile(uint64_t seed, float lo, float hi) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(kTileHW);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

// Row-major 32x32 FP32 -> reinterpreted as uint32 vector (bitcast, no conversion).
std::vector<uint32_t> reinterpret_fp32_as_u32(const std::vector<float>& in) {
    std::vector<uint32_t> out(in.size());
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::memcpy(out.data(), in.data(), in.size() * sizeof(float));
    return out;
}

// Round a row-major FP32 tile to bfloat16 precision by packing + unpacking.
// This matches the precision the device sees once the unpacker converts the
// input to Float16_b in dest.
std::vector<float> quantize_to_bf16(const std::vector<float>& rm) {
    std::vector<float> out(rm.size());
    for (size_t i = 0; i < rm.size(); ++i) {
        out[i] = static_cast<float>(bfloat16(rm[i]));
    }
    return out;
}

// Round a row-major FP32 tile through a Bfp8_b pack+unpack cycle to obtain
// the quantized fp32 values the device's unpacker would produce.
// std::vector<float> quantize_to_bfp8_b(const std::vector<float>& rm) {
//     auto packed = pack_as_bfp8_tiles<float>(tt::stl::make_const_span(rm), /*row_major_input=*/true,
//     /*is_exp_a=*/false); return unpack_bfp8_tiles_into_float_vec(
//         tt::stl::make_const_span(packed), /*row_major_output=*/true, /*is_exp_a=*/false);
// }

// Apply the selected row/col broadcast + binary op in FP32 to a 32x32 row-major tile.
std::vector<float> compute_golden(
    const std::vector<float>& a_rm, const std::vector<float>& b_rm, const SfpuBcastConfig& cfg) {
    std::vector<float> golden(kTileHW);
    for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 32; ++c) {
            const float a = a_rm[r * 32 + c];
            const float b = (cfg.bcast_dim == BcastDim::COL) ? b_rm[r * 32 + 0] : b_rm[0 * 32 + c];
            float g = 0.0f;
            switch (cfg.binop) {
                case BinOp::ADD: g = a + b; break;
                case BinOp::SUB: g = a - b; break;
                case BinOp::MUL: g = a * b; break;
            }
            golden[r * 32 + c] = g;
        }
    }
    return golden;
}

// Pack a row-major FP32 tile into the device-side tile-layout uint32 vector for
// the requested data format (matches what the packer writes to L1).
std::vector<uint32_t> pack_rm_fp32_as_tile(const std::vector<float>& rm, tt::DataFormat df) {
    using ::unit_tests::compute::gold_standard_tilize;
    using ::unit_tests::compute::GoldenConfig;
    switch (df) {
        case tt::DataFormat::Float32: {
            GoldenConfig gc{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 4};
            return gold_standard_tilize(reinterpret_fp32_as_u32(rm), gc);
        }
        case tt::DataFormat::Float16_b: {
            std::vector<bfloat16> bf(rm.size());
            for (size_t i = 0; i < rm.size(); ++i) {
                bf[i] = bfloat16(rm[i]);
            }
            auto packed = tt::test_utils::pack_vector<uint32_t, bfloat16>(bf);
            GoldenConfig gc{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 2};
            return gold_standard_tilize(packed, gc);
        }
        // case tt::DataFormat::Bfp8_b: {
        //     return pack_as_bfp8_tiles<float>(
        //         tt::stl::make_const_span(rm), /*row_major_input=*/true, /*is_exp_a=*/false);
        // }
        default: TT_THROW("pack_rm_fp32_as_tile: unsupported data format {}", static_cast<int>(df));
    }
}

// Unpack a device-side tile-layout uint32 vector back to row-major FP32 for comparison.
std::vector<float> unpack_tile_to_rm_fp32(const std::vector<uint32_t>& tile_u32, tt::DataFormat df) {
    using ::unit_tests::compute::gold_standard_untilize;
    using ::unit_tests::compute::GoldenConfig;
    switch (df) {
        case tt::DataFormat::Float32: {
            GoldenConfig gc{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 4};
            auto rm_u32 = gold_standard_untilize(tile_u32, gc);
            std::vector<float> rm(rm_u32.size());
            std::memcpy(rm.data(), rm_u32.data(), rm_u32.size() * sizeof(uint32_t));
            return rm;
        }
        case tt::DataFormat::Float16_b: {
            GoldenConfig gc{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 2};
            auto rm_u32 = gold_standard_untilize(tile_u32, gc);
            auto bf = tt::test_utils::unpack_vector<bfloat16, uint32_t>(rm_u32);
            std::vector<float> rm(bf.size());
            for (size_t i = 0; i < bf.size(); ++i) {
                rm[i] = static_cast<float>(bf[i]);
            }
            return rm;
        }
        // case tt::DataFormat::Bfp8_b: {
        //     return unpack_bfp8_tiles_into_float_vec(
        //         tt::stl::make_const_span(tile_u32), /*row_major_output=*/true, /*is_exp_a=*/false);
        // }
        default: TT_THROW("unpack_tile_to_rm_fp32: unsupported data format {}", static_cast<int>(df));
    }
}

// -----------------------------------------------------------------------------
// Test driver
// -----------------------------------------------------------------------------

bool run_sfpu_binary_bcast(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SfpuBcastConfig& cfg) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);

    distributed::MeshWorkload workload;
    Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    CoreCoord core = {0, 0};
    const tt::DataFormat df = cfg.data_format;
    const uint32_t tile_bytes = tt::tile_size(df);
    const uint32_t byte_size = tile_bytes * kNumTiles;
    // FP32 dest is required when any input CB is FP32 (SFPU compute is always FP32 in LRegs;
    // for FP32 inputs we bypass SrcA/B via unpack-to-dest which requires fp32_dest_acc_en).
    const bool fp32_dest_acc_en = (df == tt::DataFormat::Float32);

    // DRAM buffers
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = byte_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

    auto src_data_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto src_bcast_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto dst_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // CBs: c_0 = data, c_1 = bcast, c_16 = out, all in the selected format.
    tt_metal::CircularBufferConfig cb_data_cfg =
        tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_0, df}}).set_page_size(tt::CBIndex::c_0, tile_bytes);
    tt_metal::CreateCircularBuffer(program_, core, cb_data_cfg);

    tt_metal::CircularBufferConfig cb_bcast_cfg =
        tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_1, df}}).set_page_size(tt::CBIndex::c_1, tile_bytes);
    tt_metal::CreateCircularBuffer(program_, core, cb_bcast_cfg);

    tt_metal::CircularBufferConfig cb_out_cfg = tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_16, df}})
                                                    .set_page_size(tt::CBIndex::c_16, tile_bytes);
    tt_metal::CreateCircularBuffer(program_, core, cb_out_cfg);

    // Reader: 2 DRAM buffers -> c_0, c_1 (one tile each, single-bank)
    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    // Writer: c_16 -> DRAM
    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    // Compute: call the new SFPU bcast LLK via the high-level API
    std::map<std::string, std::string> compute_defines = {
        {"BCAST_DIM_VAL", std::to_string(static_cast<int>(cfg.bcast_dim))},
        {"BINOP_VAL", std::to_string(static_cast<int>(cfg.binop))},
    };

    // For FP32 inputs, request unpack-to-dest so the SFPU sees FP32 values
    // without going through SrcA/B (which is 19-bit). Other float formats go
    // through srcA + MATH datacopy into dest and don't need this flag.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (df == tt::DataFormat::Float32) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[tt::CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;
    }

    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/sfpu_binary_bcast.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .compile_args = {},
            .defines = compute_defines});

    // Stimulus: two random row-major FP32 tiles.
    const uint64_t seed_a = std::chrono::system_clock::now().time_since_epoch().count();
    const uint64_t seed_b = seed_a ^ 0x9E3779B97F4A7C15ull;
    auto src_a_rm_fp32 = generate_random_tile(seed_a, -2.0f, 2.0f);
    auto src_b_rm_fp32 = generate_random_tile(seed_b, -2.0f, 2.0f);

    // Quantize stimulus to the selected format's precision *before* computing the
    // golden, so the golden accounts for input quantization the device will see.
    std::vector<float> src_a_rm = src_a_rm_fp32;
    std::vector<float> src_b_rm = src_b_rm_fp32;
    if (df == tt::DataFormat::Float16_b) {
        src_a_rm = quantize_to_bf16(src_a_rm_fp32);
        src_b_rm = quantize_to_bf16(src_b_rm_fp32);
    }
    // } else if (df == tt::DataFormat::Bfp8_b) {
    //     // Bfp8_b quantization on the inputs, then further rounded by the packer's
    //     // Bfp8_b -> Float16_b conversion on the unpack path (unpack-to-dest for
    //     // Bfp8_b produces Float16_b values in dest).
    //     src_a_rm = quantize_to_bf16(quantize_to_bfp8_b(src_a_rm_fp32));
    //     src_b_rm = quantize_to_bf16(quantize_to_bfp8_b(src_b_rm_fp32));
    // }

    // Golden in row-major FP32, then convert to the device tile layout.
    auto golden_rm = compute_golden(src_a_rm, src_b_rm, cfg);

    auto src_a_tiled = pack_rm_fp32_as_tile(src_a_rm_fp32, df);
    auto src_b_tiled = pack_rm_fp32_as_tile(src_b_rm_fp32, df);

    distributed::WriteShard(cq, src_data_buffer, src_a_tiled, zero_coord);
    distributed::WriteShard(cq, src_bcast_buffer, src_b_tiled, zero_coord);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {
            static_cast<uint32_t>(src_data_buffer->address()),
            0u,  // src0 bank id
            static_cast<uint32_t>(src_bcast_buffer->address()),
            0u,  // src1 bank id
            kNumTiles,
        });
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {
            static_cast<uint32_t>(dst_buffer->address()),
            0u,  // dst bank id
            kNumTiles,
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> device_tiled;
    distributed::ReadShard(cq, device_tiled, dst_buffer, zero_coord);

    // Untilize and convert device output back to row-major FP32 for comparison.
    auto device_rm = unpack_tile_to_rm_fp32(device_tiled, df);
    if (device_rm.size() != golden_rm.size()) {
        log_error(tt::LogTest, "Size mismatch: device={} golden={}", device_rm.size(), golden_rm.size());
        return false;
    }

    // Per-format tolerance. FP32 with fp32 dest acc is bit-exact; BF16-class
    // formats need a relative tolerance proportional to their mantissa width.
    float atol = 0.0f;
    float rtol = 0.0f;
    switch (df) {
        case tt::DataFormat::Float32:
            atol = 0.0f;
            rtol = 1e-6f;  // tiny slack in case of any packer rounding nuance
            break;
        case tt::DataFormat::Float16_b:
            // bfloat16: 7 mantissa bits -> ~8e-3 relative; add a small absolute floor.
            atol = 1e-2f;
            rtol = 1e-2f;
            break;
        // case tt::DataFormat::Bfp8_b:
        //     // Bfp8_b: 7-bit shared-exponent mantissa; effectively ~bf16 after the
        //     // unpacker converts to Float16_b, plus per-subblock quantization.
        //     // For magnitudes near 1 the output-repack step has a ~1/32 ≈ 3.1% quant
        //     // step, so single-ULP crossings around the golden value can drift just
        //     // over 2%. Allow ~one ULP of slack; this still catches real bugs.
        //     atol = 4e-2f;
        //     rtol = 4e-2f;
        //     break;
        default: TT_THROW("unsupported format");
    }

    size_t num_mismatches = 0;
    const size_t max_report = 8;
    for (size_t i = 0; i < device_rm.size(); ++i) {
        const float g = golden_rm[i];
        const float d = device_rm[i];
        const float diff = std::fabs(g - d);
        const float tol = std::max(atol, rtol * std::fabs(g));
        if (diff > tol) {
            if (num_mismatches < max_report) {
                log_error(tt::LogTest, "Mismatch at flat idx {}: golden={} device={} diff={}", i, g, d, diff);
            }
            ++num_mismatches;
        }
    }
    if (num_mismatches != 0) {
        log_error(tt::LogTest, "Total mismatches: {}/{}", num_mismatches, device_rm.size());
        return false;
    }
    return true;
}

}  // namespace unit_tests::compute::sfpu_binary_bcast

using unit_tests::compute::sfpu_binary_bcast::bcast_dim_name;
using unit_tests::compute::sfpu_binary_bcast::BcastDim;
using unit_tests::compute::sfpu_binary_bcast::BinOp;
using unit_tests::compute::sfpu_binary_bcast::binop_name;
using unit_tests::compute::sfpu_binary_bcast::df_name;
using unit_tests::compute::sfpu_binary_bcast::SfpuBcastConfig;

class SfpuBinaryBcastFixture : public MeshDeviceFixture, public ::testing::WithParamInterface<SfpuBcastConfig> {};

TEST_P(SfpuBinaryBcastFixture, TensixSfpuBinaryBcast) {
    // These SFPU bcast LLKs are only defined on Wormhole B0 / Blackhole.
    if (this->arch_ != tt::ARCH::WORMHOLE_B0 && this->arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "SFPU binary-bcast kernels are only supported on Wormhole B0 and Blackhole";
    }

    const SfpuBcastConfig cfg = GetParam();
    log_info(
        tt::LogTest,
        "Testing SFPU binary bcast: dim={} op={} df={}",
        bcast_dim_name.at(cfg.bcast_dim),
        binop_name.at(cfg.binop),
        df_name.at(cfg.data_format));

    for (unsigned int id = 0; id < this->num_devices_; ++id) {
        ASSERT_TRUE(unit_tests::compute::sfpu_binary_bcast::run_sfpu_binary_bcast(this->devices_.at(id), cfg));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SfpuBinaryBcastAllVariants,
    SfpuBinaryBcastFixture,
    ::testing::Values(
        // FP32 sweep
        SfpuBcastConfig{BcastDim::COL, BinOp::ADD, tt::DataFormat::Float32},
        SfpuBcastConfig{BcastDim::COL, BinOp::SUB, tt::DataFormat::Float32},
        SfpuBcastConfig{BcastDim::COL, BinOp::MUL, tt::DataFormat::Float32},
        SfpuBcastConfig{BcastDim::ROW, BinOp::ADD, tt::DataFormat::Float32},
        SfpuBcastConfig{BcastDim::ROW, BinOp::SUB, tt::DataFormat::Float32},
        SfpuBcastConfig{BcastDim::ROW, BinOp::MUL, tt::DataFormat::Float32},
        // Float16_b sweep
        SfpuBcastConfig{BcastDim::COL, BinOp::ADD, tt::DataFormat::Float16_b},
        SfpuBcastConfig{BcastDim::COL, BinOp::SUB, tt::DataFormat::Float16_b},
        SfpuBcastConfig{BcastDim::COL, BinOp::MUL, tt::DataFormat::Float16_b},
        SfpuBcastConfig{BcastDim::ROW, BinOp::ADD, tt::DataFormat::Float16_b},
        SfpuBcastConfig{BcastDim::ROW, BinOp::SUB, tt::DataFormat::Float16_b},
        SfpuBcastConfig{BcastDim::ROW, BinOp::MUL, tt::DataFormat::Float16_b}),
    // Bfp8_b sweep
    // SfpuBcastConfig{BcastDim::COL, BinOp::ADD, tt::DataFormat::Bfp8_b},
    // SfpuBcastConfig{BcastDim::COL, BinOp::SUB, tt::DataFormat::Bfp8_b},
    // SfpuBcastConfig{BcastDim::COL, BinOp::MUL, tt::DataFormat::Bfp8_b},
    // SfpuBcastConfig{BcastDim::ROW, BinOp::ADD, tt::DataFormat::Bfp8_b},
    // SfpuBcastConfig{BcastDim::ROW, BinOp::SUB, tt::DataFormat::Bfp8_b},
    // SfpuBcastConfig{BcastDim::ROW, BinOp::MUL, tt::DataFormat::Bfp8_b}),
    [](const ::testing::TestParamInfo<SfpuBcastConfig>& info) {
        return std::string(bcast_dim_name.at(info.param.bcast_dim)) + "_" +
               std::string(binop_name.at(info.param.binop)) + "_" + std::string(df_name.at(info.param.data_format));
    });

}  // namespace tt::tt_metal
