// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>
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

#include "llk_device_fixture.hpp"
#include "test_golden_impls.hpp"

// LLK integration test for llk_math_gdn_inv_rms_diag (Blackhole only): the inverse-RMS diagonal of a Gram tile,
// D = eye * ([scale *] rsqrt(G * eye + eps)), as the chunk_gated_delta_rule prep computes it for its in-kernel q/k
// L2 norm. One core runs reader_binary -> gdn_inv_rms_diag compute -> writer_unary; G (c_0) and the identity (c_1)
// reach DST through unpack-to-dest, so the SFPU sees the fp32 values the host built.
//
// Two checks. The gathered pass visits only the two diagonal faces and runs one rsqrt body per face with the 16
// diagonal entries in 16 distinct lanes; it claims the same fp32 ops in the same order as the per-tile chain
// (mul_binary(eye), add_unary(eps), rsqrt, [mul_unary(scale)], mul_binary(eye)), so (1) its tile must equal the
// chain's bit for bit, off-diagonal +0 included, and (2) its diagonal must match a double-precision
// scale / sqrt(G_ii + eps) to the rsqrt body's accuracy.

namespace tt::tt_metal {

namespace unit_tests::compute::sfpu::gdn_inv_rms_diag {

constexpr uint32_t kTileDim = 32;
constexpr uint32_t kTileHW = kTileDim * kTileDim;
constexpr uint32_t kFp32TileBytes = kTileHW * sizeof(float);
constexpr uint32_t kDramBankId = 0;
constexpr uint32_t kCbTiles = 2;
constexpr uint32_t kRowDim = 128;  // K of the Gram product: G = X X^T with X [32, 128]
constexpr size_t kMaxReportedMismatches = 8;

struct Case {
    std::string name;
    uint32_t num_tiles = 4;
    bool do_scale = true;
    float eps = 1e-6f;
    float scale = 0.08838834764831845f;  // 1 / sqrt(128)
    float row_scale_min = 1e-2f;         // row norms of X span [min, max] log-uniformly: rsqrt over a wide range
    float row_scale_max = 1e2f;
    uint32_t seed = 1;
};

// G tiles as the device receives them (fp32, row-major), from X rows with log-uniform scales.
std::vector<float> make_gram(const Case& c, uint32_t tile, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> log_scale(std::log(c.row_scale_min), std::log(c.row_scale_max));
    std::vector<double> x(kTileDim * kRowDim);
    for (uint32_t r = 0; r < kTileDim; ++r) {
        const double s = std::exp(log_scale(rng)) * (tile % 2 == 0 ? 1.0 : 0.37);
        for (uint32_t k = 0; k < kRowDim; ++k) {
            x[r * kRowDim + k] = s * normal(rng);
        }
    }
    std::vector<float> g(kTileHW);
    for (uint32_t i = 0; i < kTileDim; ++i) {
        for (uint32_t j = 0; j < kTileDim; ++j) {
            double acc = 0.0;
            for (uint32_t k = 0; k < kRowDim; ++k) {
                acc += x[i * kRowDim + k] * x[j * kRowDim + k];
            }
            g[i * kTileDim + j] = static_cast<float>(acc);
        }
    }
    return g;
}

std::vector<uint32_t> fp32_as_u32(const std::vector<float>& in) {
    std::vector<uint32_t> out(in.size());
    std::memcpy(out.data(), in.data(), in.size() * sizeof(float));
    return out;
}

// One run of the kernel in `mode`; returns D row-major per tile and the G tiles it was given.
struct RunResult {
    std::vector<float> d_dev;
    std::vector<float> g;
};

RunResult run(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const Case& c, uint32_t mode) {
    std::mt19937 rng(c.seed);
    const ::unit_tests::compute::GoldenConfig cfg{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 4};
    std::vector<float> eye(kTileHW, 0.0f);
    for (uint32_t i = 0; i < kTileDim; ++i) {
        eye[i * kTileDim + i] = 1.0f;
    }
    const auto eye_tiled = ::unit_tests::compute::gold_standard_tilize(fp32_as_u32(eye), cfg);
    std::vector<uint32_t> g_tiled, eye_all;
    std::vector<float> g_all;
    for (uint32_t t = 0; t < c.num_tiles; ++t) {
        const auto g = make_gram(c, t, rng);
        g_all.insert(g_all.end(), g.begin(), g.end());
        const auto tiled = ::unit_tests::compute::gold_standard_tilize(fp32_as_u32(g), cfg);
        g_tiled.insert(g_tiled.end(), tiled.begin(), tiled.end());
        eye_all.insert(eye_all.end(), eye_tiled.begin(), eye_tiled.end());
    }

    auto& cq = mesh_device->mesh_command_queue();
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    const auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    const CoreCoord core = {0, 0};

    auto make_dram = [&](uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{
                .page_size = bytes, .buffer_type = BufferType::DRAM, .bottom_up = false},
            mesh_device.get());
    };
    const uint32_t bytes = c.num_tiles * kFp32TileBytes;
    auto g_buffer = make_dram(bytes);
    auto eye_buffer = make_dram(bytes);
    auto d_buffer = make_dram(bytes);

    for (const auto cb : {CBIndex::c_0, CBIndex::c_1, CBIndex::c_16}) {
        CreateCircularBuffer(
            program_,
            core,
            CircularBufferConfig(kCbTiles * kFp32TileBytes, {{cb, tt::DataFormat::Float32}})
                .set_page_size(cb, kFp32TileBytes));
    }

    auto reader_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    auto writer_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Both inputs through unpack-to-dest: a plain Float32 CB is lowered to Tf32 in SrcA under fp32_dest_acc_en.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;
    auto compute_kernel = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/gdn_inv_rms_diag.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .compile_args = {c.num_tiles, mode, c.do_scale ? 1u : 0u}});

    distributed::WriteShard(cq, g_buffer, g_tiled, zero_coord);
    distributed::WriteShard(cq, eye_buffer, eye_all, zero_coord);
    SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {static_cast<uint32_t>(g_buffer->address()),
         kDramBankId,
         static_cast<uint32_t>(eye_buffer->address()),
         kDramBankId,
         c.num_tiles});
    SetRuntimeArgs(
        program_, writer_kernel, core, {static_cast<uint32_t>(d_buffer->address()), kDramBankId, c.num_tiles});
    SetRuntimeArgs(program_, compute_kernel, core, {std::bit_cast<uint32_t>(c.eps), std::bit_cast<uint32_t>(c.scale)});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> d_tiled;
    distributed::ReadShard(cq, d_tiled, d_buffer, zero_coord);
    RunResult res;
    res.g = std::move(g_all);
    for (uint32_t t = 0; t < c.num_tiles && (t + 1) * kTileHW <= d_tiled.size(); ++t) {
        const std::vector<uint32_t> tile(d_tiled.begin() + t * kTileHW, d_tiled.begin() + (t + 1) * kTileHW);
        for (const uint32_t word : ::unit_tests::compute::gold_standard_untilize(tile, cfg)) {
            res.d_dev.push_back(std::bit_cast<float>(word));
        }
    }
    return res;
}

// (1) the gathered pass against the chain, bit for bit; (2) the gathered pass's diagonal against double.
void run_case(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const Case& c) {
    const RunResult gathered = run(mesh_device, c, 0);
    const RunResult chain = run(mesh_device, c, 1);
    ASSERT_EQ(gathered.d_dev.size(), c.num_tiles * kTileHW) << c.name;
    ASSERT_EQ(chain.d_dev.size(), c.num_tiles * kTileHW) << c.name;

    size_t bit_mismatches = 0;
    for (size_t i = 0; i < gathered.d_dev.size(); ++i) {
        if (std::bit_cast<uint32_t>(gathered.d_dev[i]) != std::bit_cast<uint32_t>(chain.d_dev[i])) {
            if (bit_mismatches < kMaxReportedMismatches) {
                log_error(
                    tt::LogTest,
                    "{}: gathered != chain at tile {} row {} col {}: {:.9g} vs {:.9g}",
                    c.name,
                    i / kTileHW,
                    (i % kTileHW) / kTileDim,
                    i % kTileDim,
                    gathered.d_dev[i],
                    chain.d_dev[i]);
            }
            ++bit_mismatches;
        }
    }
    EXPECT_EQ(bit_mismatches, 0u) << c.name << ": the gathered pass is not bit-identical to the per-tile chain";

    double max_rel_err = 0.0;
    size_t off_diag_nonzero = 0;
    size_t tol_mismatches = 0;
    for (uint32_t t = 0; t < c.num_tiles; ++t) {
        for (uint32_t i = 0; i < kTileDim; ++i) {
            for (uint32_t j = 0; j < kTileDim; ++j) {
                const size_t idx = t * kTileHW + i * kTileDim + j;
                const float d = gathered.d_dev[idx];
                if (i != j) {
                    off_diag_nonzero += std::bit_cast<uint32_t>(d) != 0u;  // +0 exactly
                    continue;
                }
                const double gold =
                    (c.do_scale ? c.scale : 1.0) / std::sqrt(static_cast<double>(gathered.g[idx]) + c.eps);
                const double rel = std::isnan(d) ? INFINITY : std::fabs(d - gold) / gold;
                max_rel_err = std::max(max_rel_err, rel);
                if (rel > 4e-6) {
                    if (tol_mismatches < kMaxReportedMismatches) {
                        log_error(
                            tt::LogTest,
                            "{}: tile {} diag {}: device {:.9g} golden {:.9g} rel {:.3e}",
                            c.name,
                            t,
                            i,
                            d,
                            gold,
                            rel);
                    }
                    ++tol_mismatches;
                }
            }
        }
    }
    EXPECT_EQ(off_diag_nonzero, 0u) << c.name << ": off-diagonal entries are not +0";
    EXPECT_EQ(tol_mismatches, 0u) << c.name;
    log_info(tt::LogTest, "gdn_inv_rms_diag {}: max rel err of the diagonal vs double {:.3e}", c.name, max_rel_err);
}

}  // namespace unit_tests::compute::sfpu::gdn_inv_rms_diag

using unit_tests::compute::sfpu::gdn_inv_rms_diag::Case;

TEST_F(LLKBlackholeSingleCardFixture, TensixGdnInvRmsDiagScaled) {
    unit_tests::compute::sfpu::gdn_inv_rms_diag::run_case(
        this->devices_.at(0), Case{.name = "scaled_q", .num_tiles = 4, .do_scale = true, .seed = 11});
}

TEST_F(LLKBlackholeSingleCardFixture, TensixGdnInvRmsDiagUnscaled) {
    unit_tests::compute::sfpu::gdn_inv_rms_diag::run_case(
        this->devices_.at(0), Case{.name = "unscaled_k", .num_tiles = 4, .do_scale = false, .seed = 12});
}

// Row norms near the eps floor and far above it: rsqrt over its whole useful range, and the eps term visible.
TEST_F(LLKBlackholeSingleCardFixture, TensixGdnInvRmsDiagRange) {
    unit_tests::compute::sfpu::gdn_inv_rms_diag::run_case(
        this->devices_.at(0),
        Case{
            .name = "wide_range",
            .num_tiles = 2,
            .do_scale = true,
            .row_scale_min = 1e-4f,
            .row_scale_max = 1e3f,
            .seed = 13});
}

}  // namespace tt::tt_metal
