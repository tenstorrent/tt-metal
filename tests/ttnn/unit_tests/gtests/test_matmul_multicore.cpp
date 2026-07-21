// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for MatmulMultiCoreProgramFactory (the simplest "fallback of last resort"
// factory selected when no optimised mcast variant fits the device grid).
// The test bypasses the auto-selection logic by explicitly passing a non-default
// MatmulMultiCoreProgramConfig{} as the program_config, so it works regardless of
// the device grid size or memory configuration.

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <optional>
#include <random>
#include <sstream>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/shape.hpp>
#include "common_test_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::operations::matmul::test {

namespace {

// Compute the CPU float32 reference: C = A * B (M x K) * (K x N) -> (M x N).
// Uses i-k-j loop order for good cache locality on the innermost j loop.
void cpu_matmul_f32(
    const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c, int M, int K, int N) {
    std::fill(c.begin(), c.end(), 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float aik = a[i * K + k];
            for (int j = 0; j < N; ++j) {
                c[i * N + j] += aik * b[k * N + j];
            }
        }
    }
}

}  // namespace

struct MatmulMulticoreParam {
    int M;
    int K;
    int N;
};

class MatmulMulticoreFixture : public TTNNFixtureWithSuiteDevice<MatmulMulticoreFixture>,
                               public testing::WithParamInterface<MatmulMulticoreParam> {};

// Verify that MatmulMultiCoreProgramFactory produces accurate results.
TEST_P(MatmulMulticoreFixture, MatmulMulticoreAccuracy) {
    auto param = GetParam();
    const int M = param.M;
    const int K = param.K;
    const int N = param.N;

    // Generate random float32 inputs using a fixed seed (similar to torch.randn behaviour
    // from Python regtests).
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (float& v : data_a) {
        v = dist(rng);
    }
    for (float& v : data_b) {
        v = dist(rng);
    }

    // --- CPU float32 reference ---
    std::vector<float> ref_c(M * N);
    cpu_matmul_f32(data_a, data_b, ref_c, M, K, N);

    // --- Device matmul ---
    auto& device = *device_;

    const MemoryConfig mem_cfg{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    const TensorLayout layout(DataType::FLOAT32, PageConfig(Layout::TILE), mem_cfg);
    const ttnn::QueueId cq = ttnn::QueueId(0);

    // Create input tensors on device via the host → device path used in other gtests.
    const Tensor host_a = Tensor::from_vector(data_a, TensorSpec(ttnn::Shape({M, K}), layout));
    const Tensor host_b = Tensor::from_vector(data_b, TensorSpec(ttnn::Shape({K, N}), layout));
    const Tensor input_a = host_a.to_device(&device, mem_cfg, cq);
    const Tensor input_b = host_b.to_device(&device, mem_cfg, cq);

    // HiFi3 + fp32_dest_acc_en to avoid known Wormhole HW bug #38306.
    const ttnn::ComputeKernelConfig compute_cfg{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi3,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
    };

    // Bypass the auto-selection logic and force MatmulMultiCoreProgramFactory,
    // regardless of the device grid size or memory configuration.
    const MatmulProgramConfig program_cfg = MatmulMultiCoreProgramConfig{};

    const Tensor output = ttnn::matmul(
        input_a,
        input_b,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        mem_cfg,
        /*dtype=*/std::nullopt,
        program_cfg,
        /*activation=*/std::nullopt,
        compute_cfg);

    const std::vector<float> device_c = output.to_vector<float>();

    // --- Validation ---
    // Helper used by both the non-finite pre-check and the worst-element diagnostic below
    // to map flat element indices back to (row, col) for easier correlation with tile
    // positions when debugging.
    auto row_col = [&](size_t idx) {
        return std::make_pair(idx / static_cast<size_t>(N), idx % static_cast<size_t>(N));
    };

    // Pre-check: surface unexpected NaN/Inf cleanly before running the validation
    // metrics below.  Mirrors models/common/utility_functions.py::_comp_nonfinite.
    //
    // Two complementary actions on the result, both aimed at preventing a cascade of
    // confusing secondary failures from the metrics:
    //   - If non-finite POSITIONS disagree, FAIL with a dedicated message.  Without
    //     this, allclose_report would still flag these as failures (finite vs Inf is
    //     out of tolerance), but as part of its generic "N elements failed allclose"
    //     message rather than naming the actual symptom (a non-finite leaked).
    //   - If at least one non-finite is present (even when positions match), skip
    //     pcc and relative_frobenius below.  Both NaN-poison on any non-finite input
    //     (e.g. pcc evaluates inf - inf = NaN in the mean-deviation step) and would
    //     return NaN even when allclose_report correctly reports the result as close.
    //     allclose_report itself runs unconditionally because it correctly treats
    //     matching same-sign infinities as "close" per torch.allclose semantics.
    const ttnn::test_utils::NonfiniteReport nf = ttnn::test_utils::check_nonfinite_positions(device_c, ref_c);
    if (!nf.positions_match) {
        const auto [r, c] = row_col(nf.first_mismatch_index);
        FAIL() << "Non-finite values do not appear at the same positions in device output and CPU reference. "
               << "First mismatch at flat index " << nf.first_mismatch_index << " (row=" << r << ", col=" << c
               << "): device=" << nf.first_mismatch_actual << " ref=" << nf.first_mismatch_expected;
    }

    constexpr float rtol = 0.1f;
    constexpr float atol = 0.108f;
    // Relative Frobenius catches global error magnitude that individual
    // allclose checks might miss when errors are widely distributed.
    constexpr float frobenius_threshold = 0.002f;
    constexpr float pcc_threshold = 0.9999f;
    const ttnn::test_utils::AllcloseReport report = ttnn::test_utils::allclose_report(device_c, ref_c, rtol, atol);

    // pcc and relative_frobenius are skipped when non-finites are present: see the
    // pre-check comment above.
    std::optional<float> frobenius_opt;
    std::optional<float> pcc_opt;
    bool expected_norm_is_zero = false;
    if (!nf.any_nonfinite) {
        frobenius_opt = ttnn::test_utils::relative_frobenius(device_c, ref_c, expected_norm_is_zero);
        pcc_opt = ttnn::test_utils::pcc(device_c, ref_c);
    }

    // Format a metric value for inclusion in an EXPECT_* failure message: numeric
    // when computed, or a clear skip notice when omitted.
    auto fmt_metric = [](const std::optional<float>& v) {
        std::ostringstream os;
        if (v) {
            os << *v;
        } else {
            os << "(skipped: non-finite values present)";
        }
        return os.str();
    };

    // Format a diagnostic identifying the worst elements by absolute error,
    // relative error and allclose margin.  Indexes are reported both as flat
    // offsets and 2D (row, col) coordinates to make it easy to correlate with
    // tile positions when debugging.
    auto worst_summary = [&]() {
        std::ostringstream os;
        const auto [ar, ac] = row_col(report.worst_atol_index);
        os << "\n  worst abs     : [" << ar << "," << ac << "] device=" << report.worst_atol_actual
           << " ref=" << report.worst_atol_expected << " diff=" << report.worst_atol_diff;
        const auto [rr, rc] = row_col(report.worst_rtol_index);
        os << "\n  worst rel     : [" << rr << "," << rc << "] device=" << report.worst_rtol_actual
           << " ref=" << report.worst_rtol_expected << " diff=" << report.worst_rtol_diff
           << " rel_err=" << report.worst_rtol_rel;
        const auto [mr, mc] = row_col(report.worst_margin_index);
        os << "\n  worst allclose: [" << mr << "," << mc << "] device=" << report.worst_margin_actual
           << " ref=" << report.worst_margin_expected << " diff=" << report.worst_margin_diff
           << " tol=" << report.worst_margin_tol << " margin=" << report.worst_margin;
        return os.str();
    };

    EXPECT_EQ(report.failures, 0u) << report.failures << " element(s) failed allclose(atol=" << atol
                                   << ", rtol=" << rtol << "). Result had pcc=" << fmt_metric(pcc_opt)
                                   << ", relative_frobenius=" << fmt_metric(frobenius_opt) << ";" << worst_summary();
    if (frobenius_opt) {
        EXPECT_LE(*frobenius_opt, frobenius_threshold)
            << (expected_norm_is_zero ? "Absolute" : "Relative") << " Frobenius norm " << *frobenius_opt
            << " exceeds threshold " << frobenius_threshold << ". Result had pcc=" << fmt_metric(pcc_opt)
            << worst_summary();
    }
    if (pcc_opt) {
        EXPECT_GE(*pcc_opt, pcc_threshold)
            << "PCC " << *pcc_opt << " below threshold " << pcc_threshold
            << ". Result had relative_frobenius=" << fmt_metric(frobenius_opt) << worst_summary();
    }
}

// Shapes:
//   * {2048, 2048, 2048}: 64*64 = 4096 output tiles -- evenly divides any reasonable
//     compute grid (8x8, 8x10, ...), so split_work_to_cores produces a single non-empty
//     core_group_1 and an empty core_group_2.  Exercises the "primary" CreateKernel call.
//   * {2080, 2048, 64}:   65*2  = 130 output tiles (2080 = 65*32 forces an odd Mt).
//     130 = 2*5*13, so it cannot evenly divide grid sizes like 64 or 80, which forces
//     split_work_to_cores to produce a non-empty core_group_2 and exercise the second
//     CreateKernel call (where the same compute_kernel_config and throttle/stagger
//     defines are forwarded to a different per-core tile count).  Keeps K=2048 so the
//     accumulation depth -- and therefore the precision tolerances below -- match the
//     primary shape.
INSTANTIATE_TEST_SUITE_P(
    MatmulMulticoreTests,
    MatmulMulticoreFixture,
    ::testing::Values(MatmulMulticoreParam{2048, 2048, 2048}, MatmulMulticoreParam{2080, 2048, 64}),
    [](const testing::TestParamInfo<MatmulMulticoreParam>& info) {
        return fmt::format("M{}_K{}_N{}", info.param.M, info.param.K, info.param.N);
    });

}  // namespace ttnn::operations::matmul::test
