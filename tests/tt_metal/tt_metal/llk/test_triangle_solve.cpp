// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LLK unit test for the per-tile forward-substitution triangle solve  L X = RHS  (SFPU,
// Blackhole-only). Exercises the triangle_solve_tile compute API + ckernel_sfpu_triangle_solve
// microcode directly at the tt_metal level: a single 32x32 bf16 tile through a
// reader -> triangle_solve compute -> writer pipeline on one core, compared against a host
// forward-substitution golden with PCC >= 0.99 (bf16 accumulation tolerance).
//
// Input convention (matches the ckernel): L is unit lower-triangular (diagonal an implicit 1) and
// is supplied with its plain (non-negated) strict-lower entries; the solve subtracts them directly
//   X[row] = RHS[row] - sum_{col<row} L[row][col] * X[col]
// via an SFPMAD that negates the L[row][col] * X[col] product.

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>

#include "llk_device_fixture.hpp"
#include "test_golden_impls.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::sfpu::triangle_solve {

constexpr uint32_t N = 32;                    // tile is 32x32
constexpr uint32_t TILE_ELEMS = N * N;        // 1024
constexpr uint32_t SINGLE_TILE_SIZE = TILE_ELEMS * 2;  // bf16 => 2048 bytes

// Round a float through bf16 (the format the device inputs and DST intermediates use).
inline float to_bf16(float x) { return static_cast<float>(bfloat16(x)); }

struct SolveInputs {
    std::vector<bfloat16> l_rm;      // row-major NxN, unit-lower-tri (diagonal = 1)
    std::vector<bfloat16> rhs_rm;    // row-major NxN
    std::vector<float> x_golden_rm;  // row-major NxN, fp32
};

// Build device inputs and the host golden for one case.
//   strict_lower_scale : magnitude of the random strict-lower entries of L
//   identity_only      : if true, L == I (strict-lower all zero) => X must equal RHS
SolveInputs make_inputs(uint32_t seed, float strict_lower_scale, bool identity_only) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Strict-lower part of L (below the diagonal); diagonal is an implicit 1.
    std::vector<float> l_strict(TILE_ELEMS, 0.0f);
    if (!identity_only) {
        for (uint32_t r = 0; r < N; r++) {
            for (uint32_t c = 0; c < r; c++) {
                l_strict[r * N + c] = dist(gen) * strict_lower_scale;
            }
        }
    }

    std::vector<float> rhs(TILE_ELEMS);
    for (auto& v : rhs) {
        v = dist(gen);
    }

    // L = identity + strict-lower, as bf16 (exactly what the device reads).
    std::vector<bfloat16> l(TILE_ELEMS, bfloat16(0.0f));
    for (uint32_t r = 0; r < N; r++) {
        l[r * N + r] = bfloat16(1.0f);
        for (uint32_t c = 0; c < r; c++) {
            l[r * N + c] = bfloat16(l_strict[r * N + c]);
        }
    }
    std::vector<bfloat16> rhs_bf(TILE_ELEMS);
    for (uint32_t i = 0; i < TILE_ELEMS; i++) {
        rhs_bf[i] = bfloat16(rhs[i]);
    }

    // Golden forward substitution over the bf16-rounded inputs. Each solved row element is rounded
    // to bf16 before later rows consume it, mirroring the device re-reading X[col] as bf16 from DST.
    std::vector<float> x(TILE_ELEMS, 0.0f);
    for (uint32_t row = 0; row < N; row++) {
        for (uint32_t j = 0; j < N; j++) {
            float acc = static_cast<float>(rhs_bf[row * N + j]);
            for (uint32_t col = 0; col < row; col++) {
                acc -= static_cast<float>(l[row * N + col]) * x[col * N + j];
            }
            x[row * N + j] = to_bf16(acc);
        }
    }

    return {std::move(l), std::move(rhs_bf), std::move(x)};
}

// Run one solve on device and compare against the golden. Combines two checks so a systematic
// magnitude error cannot slip through (PCC is invariant to affine scaling, e.g. 2*golden):
//   - check_pcc  : structural-correctness backstop (PCC >= 0.99).
//   - value check: an elementwise magnitude check. For identity L the solve is exact, so X must
//                  equal RHS bit-for-bit (require_exact); otherwise a bf16-appropriate is_close
//                  with tolerance scaled by the forward-substitution accumulation depth.
bool run_triangle_solve(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const SolveInputs& in, bool require_exact) {
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* const device = mesh_device->get_devices()[0];

    constexpr CoreCoord core = {0, 0};

    // ---- DRAM buffers: two inputs (L, RHS), one output (X). One tile each. ----
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = SINGLE_TILE_SIZE,
        .page_size = SINGLE_TILE_SIZE,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<Buffer> l_dram = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> rhs_dram = CreateBuffer(dram_config);
    std::shared_ptr<Buffer> x_dram = CreateBuffer(dram_config);

    // ---- CBs: c_0 = L, c_1 = RHS, c_2 = X. One bf16 tile each. ----
    auto make_cb = [&](uint32_t cb_index) {
        CircularBufferConfig cfg =
            CircularBufferConfig(SINGLE_TILE_SIZE, {{cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb_index, SINGLE_TILE_SIZE);
        tt_metal::CreateCircularBuffer(program_, core, cfg);
    };
    make_cb(CBIndex::c_0);
    make_cb(CBIndex::c_1);
    make_cb(CBIndex::c_2);

    // ---- Reader: L -> c_0, RHS -> c_1. CT args are the two TensorAccessors. ----
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(l_dram).append_to(reader_ct_args);
    TensorAccessorArgs(rhs_dram).append_to(reader_ct_args);
    KernelHandle reader_kernel_id = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_triangle_solve.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    // ---- Writer: c_2 -> X. ----
    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(x_dram).append_to(writer_ct_args);
    KernelHandle writer_kernel_id = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_triangle_solve.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    // ---- Compute: SFPU forward-substitution solve for one tile. ----
    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/triangle_solve.cpp",
        core,
        ComputeConfig{.math_approx_mode = false});

    // ---- Tilize row-major inputs and stage them in DRAM. ----
    ::unit_tests::compute::GoldenConfig golden_config = {.num_tiles_r_dim = 1, .num_tiles_c_dim = 1};
    std::vector<uint32_t> l_tilized =
        ::unit_tests::compute::gold_standard_tilize(pack_vector<uint32_t, bfloat16>(in.l_rm), golden_config);
    std::vector<uint32_t> rhs_tilized =
        ::unit_tests::compute::gold_standard_tilize(pack_vector<uint32_t, bfloat16>(in.rhs_rm), golden_config);

    tt_metal::detail::WriteToBuffer(l_dram, l_tilized);
    tt_metal::detail::WriteToBuffer(rhs_dram, rhs_tilized);

    SetRuntimeArgs(program_, reader_kernel_id, core, {l_dram->address(), rhs_dram->address()});
    SetRuntimeArgs(program_, writer_kernel_id, core, {x_dram->address()});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // ---- Read back, untilize to row-major, unpack to fp32, compare. ----
    std::vector<uint32_t> x_tilized;
    tt_metal::detail::ReadFromBuffer(x_dram, x_tilized);
    std::vector<uint32_t> x_rm_packed = ::unit_tests::compute::gold_standard_untilize(x_tilized, golden_config);
    std::vector<bfloat16> x_bf = unpack_vector<bfloat16, uint32_t>(x_rm_packed);

    std::vector<float> x_dev(TILE_ELEMS);
    for (uint32_t i = 0; i < TILE_ELEMS; i++) {
        x_dev[i] = static_cast<float>(x_bf[i]);
    }

    // Structural backstop.
    bool pass = check_pcc(in.x_golden_rm, x_dev, /*min_pcc=*/0.99);
    if (!pass) {
        log_error(tt::LogTest, "triangle_solve PCC check failed");
    }

    // Magnitude check — pins the scale that PCC alone cannot.
    if (require_exact) {
        // Identity L => X == RHS exactly; both sides are bf16 values, so require bit-exact equality.
        bool exact = is_close_vectors<float>(
            in.x_golden_rm, x_dev, [](float a, float b) { return a == b; });
        if (!exact) {
            log_error(tt::LogTest, "triangle_solve identity case is not bit-exact (X != RHS)");
        }
        pass &= exact;
    } else {
        // bf16 forward substitution accumulates up to N terms per element; scale atol with that depth.
        const float rtol = 0.05f;
        const float atol = 0.05f + 0.02f * static_cast<float>(N);
        bool close = is_close_vectors<float>(
            in.x_golden_rm, x_dev, [&](float a, float b) { return is_close(a, b, rtol, atol); });
        if (!close) {
            log_error(tt::LogTest, "triangle_solve value check (is_close) failed");
        }
        pass &= close;
    }
    return pass;
}

}  // namespace unit_tests::compute::sfpu::triangle_solve

// Identity L => the solve is a pass-through: X must equal RHS. Isolates the RHS
// load/transpose/store path from the substitution math.
TEST_F(LLKBlackholeSingleCardFixture, TensixTriangleSolveIdentity) {
    namespace ts = unit_tests::compute::sfpu::triangle_solve;
    auto in = ts::make_inputs(/*seed=*/1, /*strict_lower_scale=*/0.0f, /*identity_only=*/true);
    EXPECT_TRUE(ts::run_triangle_solve(this->devices_.at(0), in, /*require_exact=*/true));
}

// Well-conditioned random unit lower-triangular L (small strict-lower entries). Main correctness
// case; mirrors the previous ttnn pytest.
TEST_F(LLKBlackholeSingleCardFixture, TensixTriangleSolveWellConditioned) {
    namespace ts = unit_tests::compute::sfpu::triangle_solve;
    auto in = ts::make_inputs(/*seed=*/0, /*strict_lower_scale=*/0.1f, /*identity_only=*/false);
    EXPECT_TRUE(ts::run_triangle_solve(this->devices_.at(0), in, /*require_exact=*/false));
}

// Larger strict-lower magnitude: exercises real accumulation across many columns.
TEST_F(LLKBlackholeSingleCardFixture, TensixTriangleSolveLargeStrictLower) {
    namespace ts = unit_tests::compute::sfpu::triangle_solve;
    auto in = ts::make_inputs(/*seed=*/7, /*strict_lower_scale=*/0.5f, /*identity_only=*/false);
    EXPECT_TRUE(ts::run_triangle_solve(this->devices_.at(0), in, /*require_exact=*/false));
}

}  // namespace tt::tt_metal
