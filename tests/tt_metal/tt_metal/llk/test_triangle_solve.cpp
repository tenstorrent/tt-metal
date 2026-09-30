// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <tt_stl/assert.hpp>
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

#include "llk_device_fixture.hpp"
#include "test_golden_impls.hpp"

// LLK integration test for triangle_solve_tile: the SFPU forward substitution L X = RHS of one 32x32 tile with L
// unit lower-triangular (Blackhole only). One core runs reader_binary -> triangle_solve compute -> writer_unary.
// L (c_0, Float32 or Float16_b) is read by the solve in place from L1; RHS (c_1, Float32 on the unpack-to-dest
// path, so it reaches DST bit-exact) is copied to DST 0; X is solved into DST 1 and packed to c_16 (Float32).
//
// The host golden is a double-precision forward substitution over exactly the L values the device sees (a bf16 L
// is rounded before both the upload and the golden; a negated L is negated only in the upload). Every uploaded L
// tile carries random bit patterns (NaNs, infs and denormals included) on and above the diagonal, so any read of
// those entries corrupts X.

namespace tt::tt_metal {

namespace unit_tests::compute::sfpu::triangle_solve {

constexpr uint32_t kTileDim = 32;
constexpr uint32_t kTileHW = kTileDim * kTileDim;
constexpr uint32_t kFp32TileBytes = kTileHW * sizeof(float);
constexpr uint32_t kBf16TileBytes = kTileHW * sizeof(uint16_t);
// The reader pushes one L and one RHS tile per iteration, so the RHS CB must hold at least l_tiles_per_block tiles
// for the compute kernel's L block wait to complete.
constexpr uint32_t kRhsCbTiles = 2;
constexpr uint32_t kOutCbTiles = 2;
constexpr size_t kMaxReportedMismatches = 8;

enum class LKind {
    Identity,   // strict-lower zero: X == RHS
    Uniform,    // strict-lower uniform in [-l_scale, l_scale]
    MinusOnes,  // strict-lower -1: X[r] = RHS[r] + sum_{c<r} X[c] grows like 2^r
    Ones,       // strict-lower +1: telescopes to X[r] = RHS[r] - RHS[r-1], a cancellation case
};

enum class RhsKind {
    Identity,        // the identity tile
    Random,          // uniform in [-1, 1]
    RandomPositive,  // uniform in [0.5, 1.5]; keeps every partial sum of the MinusOnes solve away from zero
};

struct TriangleSolveCase {
    std::string name;
    tt::DataFormat l_format = tt::DataFormat::Float32;
    bool l_negated = false;
    LKind l_kind = LKind::Uniform;
    float l_scale = 0.1f;
    RhsKind rhs_kind = RhsKind::Random;
    uint32_t num_tiles = 1;
    uint32_t l_tiles_per_block = 1;  // L tiles the kernel front-waits together (used last-first)
    uint32_t l_cb_tiles = 1;         // L CB capacity in tiles
    bool require_exact = false;      // bit-exact X == RHS instead of the tolerance check
    float atol = 1e-6f;
    float rtol = 1e-5f;
    uint32_t seed = 1;
};

struct CaseResult {
    bool pass = false;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    double max_abs_golden = 0.0;
    size_t num_mismatches = 0;
};

// L tile pairing that mirrors the compute kernel: within a block of l_tiles_per_block L tiles, RHS tiles use the
// block's L tiles last-first.
uint32_t l_tile_for_rhs(uint32_t rhs_tile, uint32_t l_tiles_per_block) {
    const uint32_t block = rhs_tile / l_tiles_per_block;
    return block * l_tiles_per_block + (l_tiles_per_block - 1 - rhs_tile % l_tiles_per_block);
}

// Strict-lower entries of one L tile (row-major, everything else zero), already rounded to l_format: exactly the
// values the device reads.
std::vector<double> make_l_strict(const TriangleSolveCase& c, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-c.l_scale, c.l_scale);
    std::vector<double> l(kTileHW, 0.0);
    for (uint32_t r = 0; r < kTileDim; ++r) {
        for (uint32_t col = 0; col < r; ++col) {
            float v = 0.0f;
            switch (c.l_kind) {
                case LKind::Identity: v = 0.0f; break;
                case LKind::Uniform: v = dist(rng); break;
                case LKind::MinusOnes: v = -1.0f; break;
                case LKind::Ones: v = 1.0f; break;
            }
            if (c.l_format == tt::DataFormat::Float16_b) {
                v = static_cast<float>(bfloat16(v));
            }
            l[r * kTileDim + col] = v;
        }
    }
    return l;
}

// The L tile as uploaded: row-major, l_format bits packed into uint32 words (bf16 element 2k in the low half).
// Strict-lower entries are negated when the case supplies -L; the diagonal and upper triangle hold random bits.
std::vector<uint32_t> make_l_words(const TriangleSolveCase& c, const std::vector<double>& l_strict, std::mt19937& rng) {
    const bool bf16 = c.l_format == tt::DataFormat::Float16_b;
    std::vector<uint32_t> words(bf16 ? kTileHW / 2 : kTileHW, 0u);
    for (uint32_t r = 0; r < kTileDim; ++r) {
        for (uint32_t col = 0; col < kTileDim; ++col) {
            const uint32_t i = r * kTileDim + col;
            uint32_t bits = rng();  // garbage on and above the diagonal: the solve never reads it
            if (col < r) {
                const auto v = static_cast<float>(c.l_negated ? -l_strict[i] : l_strict[i]);
                bits = bf16 ? std::bit_cast<uint16_t>(bfloat16(v)) : std::bit_cast<uint32_t>(v);
            }
            if (bf16) {
                words[i / 2] |= (bits & 0xFFFFu) << (16 * (i % 2));
            } else {
                words[i] = bits;
            }
        }
    }
    return words;
}

std::vector<float> make_rhs(const TriangleSolveCase& c, std::mt19937& rng) {
    std::uniform_real_distribution<float> signed_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> positive_dist(0.5f, 1.5f);
    std::vector<float> rhs(kTileHW, 0.0f);
    for (uint32_t i = 0; i < kTileHW; ++i) {
        switch (c.rhs_kind) {
            case RhsKind::Identity: rhs[i] = (i / kTileDim == i % kTileDim) ? 1.0f : 0.0f; break;
            case RhsKind::Random: rhs[i] = signed_dist(rng); break;
            case RhsKind::RandomPositive: rhs[i] = positive_dist(rng); break;
        }
    }
    return rhs;
}

// Forward substitution with unit diagonal in double: X[r] = RHS[r] - sum_{c<r} L[r][c] X[c] for every column.
std::vector<double> solve_golden(const std::vector<double>& l_strict, const std::vector<float>& rhs) {
    std::vector<double> x(kTileHW, 0.0);
    for (uint32_t r = 0; r < kTileDim; ++r) {
        for (uint32_t j = 0; j < kTileDim; ++j) {
            double acc = rhs[r * kTileDim + j];
            for (uint32_t col = 0; col < r; ++col) {
                acc -= l_strict[r * kTileDim + col] * x[col * kTileDim + j];
            }
            x[r * kTileDim + j] = acc;
        }
    }
    return x;
}

std::vector<uint32_t> fp32_as_u32(const std::vector<float>& in) {
    std::vector<uint32_t> out(in.size());
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::memcpy(out.data(), in.data(), in.size() * sizeof(float));
    return out;
}

// |x_dev - x_gold| <= atol + rtol * |x_gold| per element (bit-exact for require_exact), with the max abs / rel error
// over the case. A NaN on the device side counts as an infinite error.
CaseResult compare(const TriangleSolveCase& c, const std::vector<float>& x_dev, const std::vector<double>& x_gold) {
    CaseResult res;
    if (x_dev.size() != x_gold.size()) {
        log_error(tt::LogTest, "{}: size mismatch device={} golden={}", c.name, x_dev.size(), x_gold.size());
        return res;
    }
    for (size_t i = 0; i < x_gold.size(); ++i) {
        const double g = x_gold[i];
        const float d = x_dev[i];
        const double abs_err = std::isnan(d) ? std::numeric_limits<double>::infinity() : std::fabs(d - g);
        const bool ok = c.require_exact ? std::bit_cast<uint32_t>(d) == std::bit_cast<uint32_t>(static_cast<float>(g))
                                        : abs_err <= c.atol + c.rtol * std::fabs(g);
        res.max_abs_err = std::max(res.max_abs_err, abs_err);
        res.max_abs_golden = std::max(res.max_abs_golden, std::fabs(g));
        if (g != 0.0) {
            res.max_rel_err = std::max(res.max_rel_err, abs_err / std::fabs(g));
        }
        if (!ok) {
            if (res.num_mismatches < kMaxReportedMismatches) {
                log_error(
                    tt::LogTest,
                    "{}: mismatch tile {} row {} col {}: device={:.9g} golden={:.9g} abs_err={:.3e}",
                    c.name,
                    i / kTileHW,
                    (i % kTileHW) / kTileDim,
                    i % kTileDim,
                    d,
                    g,
                    abs_err);
            }
            ++res.num_mismatches;
        }
    }
    res.pass = res.num_mismatches == 0;
    log_info(
        tt::LogTest,
        "triangle_solve {}: max abs err {:.3e}, max rel err {:.3e}, max |x_gold| {:.3e}, mismatches {}/{} ({})",
        c.name,
        res.max_abs_err,
        res.max_rel_err,
        res.max_abs_golden,
        res.num_mismatches,
        x_gold.size(),
        c.require_exact ? "bit-exact" : "atol " + std::to_string(c.atol) + " rtol " + std::to_string(c.rtol));
    return res;
}

CaseResult run_case(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TriangleSolveCase& c) {
    TT_FATAL(
        c.num_tiles % c.l_tiles_per_block == 0 && c.l_cb_tiles >= c.l_tiles_per_block &&
            kRhsCbTiles >= c.l_tiles_per_block,
        "{}: inconsistent tile counts",
        c.name);
    const bool l_bf16 = c.l_format == tt::DataFormat::Float16_b;
    const uint32_t l_tile_bytes = l_bf16 ? kBf16TileBytes : kFp32TileBytes;

    // Stimulus and golden, row-major per tile, tilized for the device.
    std::mt19937 rng(c.seed);
    const ::unit_tests::compute::GoldenConfig l_cfg{
        .num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = l_bf16 ? 2u : 4u};
    const ::unit_tests::compute::GoldenConfig fp32_cfg{.num_tiles_r_dim = 1, .num_tiles_c_dim = 1, .datum_bytes = 4};
    std::vector<std::vector<double>> l_strict(c.num_tiles);
    std::vector<uint32_t> l_tiled;
    for (uint32_t t = 0; t < c.num_tiles; ++t) {
        l_strict[t] = make_l_strict(c, rng);
        const auto tiled = ::unit_tests::compute::gold_standard_tilize(make_l_words(c, l_strict[t], rng), l_cfg);
        l_tiled.insert(l_tiled.end(), tiled.begin(), tiled.end());
    }
    std::vector<uint32_t> rhs_tiled;
    std::vector<double> x_golden;
    for (uint32_t t = 0; t < c.num_tiles; ++t) {
        const auto rhs = make_rhs(c, rng);
        const auto tiled = ::unit_tests::compute::gold_standard_tilize(fp32_as_u32(rhs), fp32_cfg);
        rhs_tiled.insert(rhs_tiled.end(), tiled.begin(), tiled.end());
        const auto x = solve_golden(l_strict[l_tile_for_rhs(t, c.l_tiles_per_block)], rhs);
        x_golden.insert(x_golden.end(), x.begin(), x.end());
    }

    auto& cq = mesh_device->mesh_command_queue();
    const auto zero_coord = distributed::MeshCoordinate(0, 0);
    const auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    const CoreCoord core = {0, 0};

    // Each buffer is a single DRAM page so the tiles sit back to back in one bank.
    auto make_dram = [&](uint32_t bytes) {
        return distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = bytes},
            distributed::DeviceLocalBufferConfig{
                .page_size = bytes, .buffer_type = BufferType::DRAM, .bottom_up = false},
            mesh_device.get());
    };
    auto l_buffer = make_dram(c.num_tiles * l_tile_bytes);
    auto rhs_buffer = make_dram(c.num_tiles * kFp32TileBytes);
    auto x_buffer = make_dram(c.num_tiles * kFp32TileBytes);

    CreateCircularBuffer(
        program_,
        core,
        CircularBufferConfig(c.l_cb_tiles * l_tile_bytes, {{CBIndex::c_0, c.l_format}})
            .set_page_size(CBIndex::c_0, l_tile_bytes));
    CreateCircularBuffer(
        program_,
        core,
        CircularBufferConfig(kRhsCbTiles * kFp32TileBytes, {{CBIndex::c_1, tt::DataFormat::Float32}})
            .set_page_size(CBIndex::c_1, kFp32TileBytes));
    CreateCircularBuffer(
        program_,
        core,
        CircularBufferConfig(kOutCbTiles * kFp32TileBytes, {{CBIndex::c_16, tt::DataFormat::Float32}})
            .set_page_size(CBIndex::c_16, kFp32TileBytes));

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

    // RHS goes through unpack-to-dest: a plain Float32 CB is lowered to Tf32 in SrcA under fp32_dest_acc_en and
    // would round the right-hand side before the solve sees it.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[CBIndex::c_1] = UnpackToDestMode::UnpackToDestFp32;
    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/triangle_solve.cpp",
        core,
        ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .compile_args = {c.num_tiles, c.l_tiles_per_block, l_bf16 ? 1u : 0u, c.l_negated ? 1u : 0u}});

    distributed::WriteShard(cq, l_buffer, l_tiled, zero_coord);
    distributed::WriteShard(cq, rhs_buffer, rhs_tiled, zero_coord);

    SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {static_cast<uint32_t>(l_buffer->address()),
         0u,
         static_cast<uint32_t>(rhs_buffer->address()),
         0u,
         c.num_tiles});
    SetRuntimeArgs(program_, writer_kernel, core, {static_cast<uint32_t>(x_buffer->address()), 0u, c.num_tiles});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    std::vector<uint32_t> x_tiled;
    distributed::ReadShard(cq, x_tiled, x_buffer, zero_coord);

    std::vector<float> x_dev;
    x_dev.reserve(x_golden.size());
    for (uint32_t t = 0; t < c.num_tiles && (t + 1) * kTileHW <= x_tiled.size(); ++t) {
        const std::vector<uint32_t> tile(x_tiled.begin() + t * kTileHW, x_tiled.begin() + (t + 1) * kTileHW);
        for (const uint32_t word : ::unit_tests::compute::gold_standard_untilize(tile, fp32_cfg)) {
            x_dev.push_back(std::bit_cast<float>(word));
        }
    }
    return compare(c, x_dev, x_golden);
}

void run_cases(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const std::vector<TriangleSolveCase>& cases) {
    if (mesh_device->arch() != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "triangle_solve_tile is implemented on Blackhole only";
    }
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        EXPECT_TRUE(run_case(mesh_device, c).pass);
    }
}

}  // namespace unit_tests::compute::sfpu::triangle_solve

using unit_tests::compute::sfpu::triangle_solve::LKind;
using unit_tests::compute::sfpu::triangle_solve::RhsKind;
using unit_tests::compute::sfpu::triangle_solve::TriangleSolveCase;

// L = I (garbage on and above the diagonal): the solve must pass a random fp32 RHS through bit-exact.
TEST_F(LLKBlackholeSingleCardFixture, TensixTriangleSolveIdentityExact) {
    unit_tests::compute::sfpu::triangle_solve::run_cases(
        this->devices_.at(0),
        {TriangleSolveCase{
            .name = "identity_fp32_l",
            .l_kind = LKind::Identity,
            .rhs_kind = RhsKind::Random,
            .require_exact = true,
            .seed = 11}});
}

// Float32 L: well-conditioned and larger random strict-lower parts, the 2^r growth case and the telescoping case.
TEST_F(LLKBlackholeSingleCardFixture, TensixTriangleSolveFp32L) {
    unit_tests::compute::sfpu::triangle_solve::run_cases(
        this->devices_.at(0),
        {TriangleSolveCase{
             .name = "fp32_l_uniform_0.1_rhs_identity", .l_scale = 0.1f, .rhs_kind = RhsKind::Identity, .seed = 21},
         TriangleSolveCase{.name = "fp32_l_uniform_0.1_rhs_random", .l_scale = 0.1f, .seed = 22},
         TriangleSolveCase{.name = "fp32_l_uniform_0.5_rhs_random", .l_scale = 0.5f, .seed = 23},
         TriangleSolveCase{
             .name = "fp32_l_minus_ones_growth",
             .l_kind = LKind::MinusOnes,
             .rhs_kind = RhsKind::RandomPositive,
             .seed = 24},
         TriangleSolveCase{.name = "fp32_l_ones_telescoping", .l_kind = LKind::Ones, .seed = 25}});
}

// Float16_b L (rounded on the host before upload and golden); the arithmetic stays fp32.
TEST_F(LLKBlackholeSingleCardFixture, TensixTriangleSolveBf16L) {
    unit_tests::compute::sfpu::triangle_solve::run_cases(
        this->devices_.at(0),
        {TriangleSolveCase{
             .name = "bf16_l_uniform_0.1_rhs_identity",
             .l_format = tt::DataFormat::Float16_b,
             .l_scale = 0.1f,
             .rhs_kind = RhsKind::Identity,
             .seed = 31},
         TriangleSolveCase{
             .name = "bf16_l_uniform_0.1_rhs_random",
             .l_format = tt::DataFormat::Float16_b,
             .l_scale = 0.1f,
             .seed = 32},
         TriangleSolveCase{
             .name = "bf16_l_uniform_0.5_rhs_random",
             .l_format = tt::DataFormat::Float16_b,
             .l_scale = 0.5f,
             .seed = 33},
         TriangleSolveCase{
             .name = "bf16_l_minus_ones_growth",
             .l_format = tt::DataFormat::Float16_b,
             .l_kind = LKind::MinusOnes,
             .rhs_kind = RhsKind::RandomPositive,
             .seed = 34}});
}

// L_NEGATED: the uploaded tile holds -L below the diagonal; the golden uses L.
TEST_F(LLKBlackholeSingleCardFixture, TensixTriangleSolveNegatedL) {
    unit_tests::compute::sfpu::triangle_solve::run_cases(
        this->devices_.at(0),
        {TriangleSolveCase{
             .name = "fp32_negated_l_uniform_0.1_rhs_identity",
             .l_negated = true,
             .l_scale = 0.1f,
             .rhs_kind = RhsKind::Identity,
             .seed = 41},
         TriangleSolveCase{
             .name = "fp32_negated_l_uniform_0.1_rhs_random", .l_negated = true, .l_scale = 0.1f, .seed = 42},
         TriangleSolveCase{
             .name = "fp32_negated_l_uniform_0.5_rhs_random", .l_negated = true, .l_scale = 0.5f, .seed = 43},
         TriangleSolveCase{
             .name = "bf16_negated_l_uniform_0.5_rhs_random",
             .l_format = tt::DataFormat::Float16_b,
             .l_negated = true,
             .l_scale = 0.5f,
             .seed = 44}});
}

// Several tiles per run: an L CB of one tile lands every L tile at the same L1 address, so each solve depends on the
// L1-cache invalidate and on the pop of the previous L tile waiting for MATH to finish reading it; an L block of two
// tiles solved last-first exercises get_tile_address(l_tile_idx = 1).
TEST_F(LLKBlackholeSingleCardFixture, TensixTriangleSolveMultiTile) {
    unit_tests::compute::sfpu::triangle_solve::run_cases(
        this->devices_.at(0),
        {TriangleSolveCase{
             .name = "fp32_l_4_tiles_l_cb_1", .l_scale = 0.5f, .num_tiles = 4, .l_cb_tiles = 1, .seed = 51},
         TriangleSolveCase{
             .name = "bf16_l_4_tiles_l_cb_1",
             .l_format = tt::DataFormat::Float16_b,
             .l_scale = 0.5f,
             .num_tiles = 4,
             .l_cb_tiles = 1,
             .seed = 52},
         TriangleSolveCase{
             .name = "fp32_l_2_tiles_l_block_2",
             .l_scale = 0.5f,
             .num_tiles = 2,
             .l_tiles_per_block = 2,
             .l_cb_tiles = 2,
             .seed = 53}});
}

}  // namespace tt::tt_metal
