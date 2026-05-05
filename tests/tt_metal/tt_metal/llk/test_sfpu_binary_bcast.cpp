// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
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
#include <umd/device/types/arch.hpp>

#include "device_fixture.hpp"
#include "test_golden_impls.hpp"

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
    tt::DataFormat df;  // Float32 or Float16_b
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
    {tt::DataFormat::Float32, "FP32"},
    {tt::DataFormat::Float16_b, "BF16"},
};

constexpr uint32_t kTileHW = 32 * 32;
constexpr uint32_t kNumTiles = 1;  // kernel operates on a single 32x32 tile

// Bytes per element for the CB data format.
constexpr uint32_t elem_bytes(tt::DataFormat df) {
    return (df == tt::DataFormat::Float32) ? 4u : 2u;
}

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

// Quantize each float to bfloat16 (tie-to-even) and return a parallel float vector
// holding the exactly-representable-as-bfloat16 values. Used both for feeding the
// device and for computing a host-side golden that matches the device's input
// precision.
std::vector<float> quantize_to_bf16_float(const std::vector<float>& in) {
    std::vector<float> out(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = static_cast<float>(bfloat16(in[i]));
    }
    return out;
}

// Row-major 32x32 vector of floats (already quantized to bfloat16) -> packed uint32
// vector with two bfloat16 lanes per uint32 (low 16 = even idx, high 16 = odd idx).
// This is the layout gold_standard_tilize expects when datum_bytes == 2.
std::vector<uint32_t> pack_bf16_as_u32(const std::vector<float>& in) {
    assert(in.size() % 2 == 0);
    std::vector<uint32_t> out(in.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = pack_two_bfloat16_into_uint32({bfloat16(in[2 * i]), bfloat16(in[2 * i + 1])});
    }
    return out;
}

// Compute the row-major 32x32 golden for the selected broadcast + binop.
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
    const tt::DataFormat df = cfg.df;
    const uint32_t tile_bytes = kTileHW * elem_bytes(df);
    const uint32_t byte_size = tile_bytes * kNumTiles;
    const bool is_fp32 = (df == tt::DataFormat::Float32);

    // DRAM buffers
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = byte_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

    auto src_data_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto src_bcast_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto dst_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // CBs: c_0 = data, c_1 = bcast, c_16 = out, all in `df`.
    tt_metal::CircularBufferConfig cb_data_cfg = tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_0, df}})
                                                     .set_page_size(tt::CBIndex::c_0, tile_bytes);
    tt_metal::CreateCircularBuffer(program_, core, cb_data_cfg);

    tt_metal::CircularBufferConfig cb_bcast_cfg = tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_1, df}})
                                                      .set_page_size(tt::CBIndex::c_1, tile_bytes);
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

    // For FP32 we ask the unpacker to land the tile directly in DEST (bypassing srcA/srcB);
    // for Float16_b we leave the default, which routes the tile through srcA/srcB before
    // the SFPU consumes it. That exercises the other unpack path.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (is_fp32) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[tt::CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;
    }

    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/sfpu_binary_bcast.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .compile_args = {},
            .defines = compute_defines});

    // Stimulus (row-major FP32). For bfloat16 we quantize inputs up-front so the host-side
    // golden sees the same bit-exact values the device will see after the unpacker.
    const uint64_t seed_a = std::chrono::system_clock::now().time_since_epoch().count();
    const uint64_t seed_b = seed_a ^ 0x9E3779B97F4A7C15ull;
    auto src_a_rm = generate_random_tile(seed_a, -2.0f, 2.0f);
    auto src_b_rm = generate_random_tile(seed_b, -2.0f, 2.0f);
    if (!is_fp32) {
        src_a_rm = quantize_to_bf16_float(src_a_rm);
        src_b_rm = quantize_to_bf16_float(src_b_rm);
    }

    // Golden (row-major), then tilize to match device-side tile layout.
    auto golden_rm = compute_golden(src_a_rm, src_b_rm, cfg);

    ::unit_tests::compute::GoldenConfig gc{
        .num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = elem_bytes(df)};

    auto to_tile_input = [&](const std::vector<float>& rm) {
        return is_fp32 ? reinterpret_fp32_as_u32(rm) : pack_bf16_as_u32(rm);
    };
    auto src_a_tiled = ::unit_tests::compute::gold_standard_tilize(to_tile_input(src_a_rm), gc);
    auto src_b_tiled = ::unit_tests::compute::gold_standard_tilize(to_tile_input(src_b_rm), gc);
    auto golden_tiled = ::unit_tests::compute::gold_standard_tilize(to_tile_input(golden_rm), gc);

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

    if (device_tiled.size() != golden_tiled.size()) {
        log_error(tt::LogTest, "Size mismatch: device={} golden={}", device_tiled.size(), golden_tiled.size());
        return false;
    }

    // Widen device + golden tiles to flat float vectors for a format-agnostic compare.
    auto widen_to_floats = [&](const std::vector<uint32_t>& tiled) {
        std::vector<float> out;
        out.reserve(tiled.size() * (is_fp32 ? 1 : 2));
        if (is_fp32) {
            for (uint32_t w : tiled) {
                out.push_back(std::bit_cast<float>(w));
            }
        } else {
            for (uint32_t w : tiled) {
                const bfloat16 lo = std::bit_cast<bfloat16>(static_cast<uint16_t>(w & 0xffff));
                const bfloat16 hi = std::bit_cast<bfloat16>(static_cast<uint16_t>(w >> 16));
                out.push_back(static_cast<float>(lo));
                out.push_back(static_cast<float>(hi));
            }
        }
        return out;
    };
    const auto golden_f = widen_to_floats(golden_tiled);
    const auto device_f = widen_to_floats(device_tiled);

    // FP32 path should be bit-exact; bfloat16 path has ~8 bits of mantissa, so allow
    // a ~1-ulp relative tolerance (2^-8 ≈ 3.9e-3) with a small absolute floor for values
    // near zero.
    const float atol = is_fp32 ? 0.0f : 1.0f / 128.0f;
    const float rtol = is_fp32 ? 1e-6f : 1.0f / 128.0f;

    size_t num_mismatches = 0;
    const size_t max_report = 8;
    for (size_t i = 0; i < device_f.size(); ++i) {
        const float g = golden_f[i];
        const float d = device_f[i];
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
        log_error(tt::LogTest, "Total mismatches: {}/{}", num_mismatches, device_f.size());
        return false;
    }
    return true;
}

}  // namespace unit_tests::compute::sfpu_binary_bcast

using unit_tests::compute::sfpu_binary_bcast::bcast_dim_name;
using unit_tests::compute::sfpu_binary_bcast::BcastDim;
using unit_tests::compute::sfpu_binary_bcast::BinOp;
using unit_tests::compute::sfpu_binary_bcast::binop_name;
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
        unit_tests::compute::sfpu_binary_bcast::df_name.at(cfg.df));

    for (unsigned int id = 0; id < this->num_devices_; ++id) {
        ASSERT_TRUE(unit_tests::compute::sfpu_binary_bcast::run_sfpu_binary_bcast(this->devices_.at(id), cfg));
    }
}

using unit_tests::compute::sfpu_binary_bcast::df_name;

INSTANTIATE_TEST_SUITE_P(
    SfpuBinaryBcastAllVariants,
    SfpuBinaryBcastFixture,
    ::testing::Values(
        // Float32: unpacker writes directly to DEST (bypasses srcA/srcB).
        SfpuBcastConfig{BcastDim::COL, BinOp::ADD, tt::DataFormat::Float32},
        SfpuBcastConfig{BcastDim::COL, BinOp::SUB, tt::DataFormat::Float32},
        SfpuBcastConfig{BcastDim::COL, BinOp::MUL, tt::DataFormat::Float32},
        SfpuBcastConfig{BcastDim::ROW, BinOp::ADD, tt::DataFormat::Float32},
        SfpuBcastConfig{BcastDim::ROW, BinOp::SUB, tt::DataFormat::Float32},
        SfpuBcastConfig{BcastDim::ROW, BinOp::MUL, tt::DataFormat::Float32},
        // Float16_b: data is routed through srcA/srcB before the SFPU consumes it.
        SfpuBcastConfig{BcastDim::COL, BinOp::ADD, tt::DataFormat::Float16_b},
        SfpuBcastConfig{BcastDim::COL, BinOp::SUB, tt::DataFormat::Float16_b},
        SfpuBcastConfig{BcastDim::COL, BinOp::MUL, tt::DataFormat::Float16_b},
        SfpuBcastConfig{BcastDim::ROW, BinOp::ADD, tt::DataFormat::Float16_b},
        SfpuBcastConfig{BcastDim::ROW, BinOp::SUB, tt::DataFormat::Float16_b},
        SfpuBcastConfig{BcastDim::ROW, BinOp::MUL, tt::DataFormat::Float16_b}),
    [](const ::testing::TestParamInfo<SfpuBcastConfig>& info) {
        return std::string(bcast_dim_name.at(info.param.bcast_dim)) + "_" +
               std::string(binop_name.at(info.param.binop)) + "_" + std::string(df_name.at(info.param.df));
    });

}  // namespace tt::tt_metal
