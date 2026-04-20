// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <chrono>
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
};

// Format a BcastDim/BinOp for test log output only.
const std::map<BcastDim, std::string> bcast_dim_name = {
    {BcastDim::COL, "BCAST_COL"},
    {BcastDim::ROW, "BCAST_ROW"},
};
const std::map<BinOp, std::string> binop_name = {
    {BinOp::ADD, "ADD"},
    {BinOp::SUB, "SUB"},
    {BinOp::MUL, "MUL"},
};

constexpr uint32_t kTileHW = 32 * 32;
constexpr uint32_t kFp32TileBytes = kTileHW * sizeof(float);
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
    constexpr tt::DataFormat df = tt::DataFormat::Float32;
    const uint32_t byte_size = kFp32TileBytes * kNumTiles;

    // DRAM buffers
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = byte_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::ReplicatedBufferConfig buffer_config{.size = byte_size};

    auto src_data_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto src_bcast_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto dst_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // CBs: c_0 = data, c_1 = bcast, c_16 = out, all FP32
    tt_metal::CircularBufferConfig cb_data_cfg = tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_0, df}})
                                                     .set_page_size(tt::CBIndex::c_0, kFp32TileBytes);
    tt_metal::CreateCircularBuffer(program_, core, cb_data_cfg);

    tt_metal::CircularBufferConfig cb_bcast_cfg = tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_1, df}})
                                                      .set_page_size(tt::CBIndex::c_1, kFp32TileBytes);
    tt_metal::CreateCircularBuffer(program_, core, cb_bcast_cfg);

    tt_metal::CircularBufferConfig cb_out_cfg = tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_16, df}})
                                                    .set_page_size(tt::CBIndex::c_16, kFp32TileBytes);
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

    // Request FP32 through the unpacker for both input CBs so the SFPU sees FP32 values.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[tt::CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;

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

    // Stimulus
    const uint64_t seed_a = std::chrono::system_clock::now().time_since_epoch().count();
    const uint64_t seed_b = seed_a ^ 0x9E3779B97F4A7C15ull;
    auto src_a_rm = generate_random_tile(seed_a, -2.0f, 2.0f);
    auto src_b_rm = generate_random_tile(seed_b, -2.0f, 2.0f);

    // Golden (row-major), then tilize to match device-side tile layout.
    auto golden_rm = compute_golden(src_a_rm, src_b_rm, cfg);

    ::unit_tests::compute::GoldenConfig gc{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 4};

    auto src_a_tiled = ::unit_tests::compute::gold_standard_tilize(reinterpret_fp32_as_u32(src_a_rm), gc);
    auto src_b_tiled = ::unit_tests::compute::gold_standard_tilize(reinterpret_fp32_as_u32(src_b_rm), gc);
    auto golden_tiled = ::unit_tests::compute::gold_standard_tilize(reinterpret_fp32_as_u32(golden_rm), gc);

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

    // FP32 add/sub/mul with fp32 dest acc should be bit-exact. Use a tiny
    // relative tolerance to be safe against any packer rounding nuance.
    if (device_tiled.size() != golden_tiled.size()) {
        log_error(tt::LogTest, "Size mismatch: device={} golden={}", device_tiled.size(), golden_tiled.size());
        return false;
    }

    constexpr float atol = 0.0f;
    constexpr float rtol = 1e-6f;
    size_t num_mismatches = 0;
    const size_t max_report = 8;
    for (size_t i = 0; i < device_tiled.size(); ++i) {
        const float g = std::bit_cast<float>(golden_tiled[i]);
        const float d = std::bit_cast<float>(device_tiled[i]);
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
        log_error(tt::LogTest, "Total mismatches: {}/{}", num_mismatches, device_tiled.size());
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
        "Testing SFPU binary bcast: dim={} op={}",
        bcast_dim_name.at(cfg.bcast_dim),
        binop_name.at(cfg.binop));

    for (unsigned int id = 0; id < this->num_devices_; ++id) {
        ASSERT_TRUE(unit_tests::compute::sfpu_binary_bcast::run_sfpu_binary_bcast(this->devices_.at(id), cfg));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SfpuBinaryBcastAllVariants,
    SfpuBinaryBcastFixture,
    ::testing::Values(
        SfpuBcastConfig{BcastDim::COL, BinOp::ADD},
        SfpuBcastConfig{BcastDim::COL, BinOp::SUB},
        SfpuBcastConfig{BcastDim::COL, BinOp::MUL},
        SfpuBcastConfig{BcastDim::ROW, BinOp::ADD},
        SfpuBcastConfig{BcastDim::ROW, BinOp::SUB},
        SfpuBcastConfig{BcastDim::ROW, BinOp::MUL}),
    [](const ::testing::TestParamInfo<SfpuBcastConfig>& info) {
        return std::string(bcast_dim_name.at(info.param.bcast_dim)) + "_" +
               std::string(binop_name.at(info.param.binop));
    });

}  // namespace tt::tt_metal
