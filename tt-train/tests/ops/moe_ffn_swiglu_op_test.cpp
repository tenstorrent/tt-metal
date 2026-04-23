// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/moe_ffn_swiglu_op.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/sparse_matmul/sparse_matmul.hpp"

namespace {

constexpr uint32_t kTile = 32;

class MoeFfnSwigluForwardTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// CPU reference: for each expert slice X_e, compute
// (silu(X_e @ W_gate_e) * (X_e @ W_up_e)) @ W_down_e and splat into
// rows [offsets[e], offsets[e]+counts[e]) of a zero-initialized output.
xt::xarray<float> moe_ffn_swiglu_reference(
    const xt::xarray<float>& grouped,  // [T_cap, H]
    const std::vector<uint32_t>& offsets,
    const std::vector<uint32_t>& counts,  // active rows per expert
    const xt::xarray<float>& w_gate,      // [E, H, I]
    const xt::xarray<float>& w_up,        // [E, H, I]
    const xt::xarray<float>& w_down) {    // [E, I, H]
    const std::size_t T_cap = grouped.shape()[0];
    const std::size_t H = grouped.shape()[1];
    xt::xarray<float> out = xt::zeros<float>(std::vector<std::size_t>{T_cap, H});

    const std::size_t E = counts.size();
    for (std::size_t e = 0; e < E; ++e) {
        const std::size_t row_lo = offsets[e];
        const std::size_t n_rows = counts[e];
        if (n_rows == 0U) {
            continue;
        }
        const std::size_t row_hi = row_lo + n_rows;

        xt::xarray<float> X = xt::view(grouped, xt::range(row_lo, row_hi), xt::all());
        xt::xarray<float> Wg = xt::view(w_gate, e, xt::all(), xt::all());
        xt::xarray<float> Wu = xt::view(w_up, e, xt::all(), xt::all());
        xt::xarray<float> Wd = xt::view(w_down, e, xt::all(), xt::all());

        xt::xarray<float> G = xt::linalg::tensordot(X, Wg, {1}, {0});  // [n, I]
        xt::xarray<float> U = xt::linalg::tensordot(X, Wu, {1}, {0});  // [n, I]
        xt::xarray<float> sigmoid_G = 1.0f / (1.0f + xt::exp(-G));
        xt::xarray<float> A = (G * sigmoid_G) * U;                     // [n, I]
        xt::xarray<float> Y = xt::linalg::tensordot(A, Wd, {1}, {0});  // [n, H]

        xt::view(out, xt::range(row_lo, row_hi), xt::all()) = Y;
    }
    return out;
}

float relative_l2(const xt::xarray<float>& a, const xt::xarray<float>& b) {
    const auto diff = a - b;
    const float diff_l2 = std::sqrt(xt::sum(xt::square(diff))());
    const float ref_l2 = std::sqrt(xt::sum(xt::square(b))());
    return diff_l2 / (ref_l2 + 1e-12f);
}

struct FfnCase {
    uint32_t E;
    uint32_t H;
    uint32_t I;
    std::vector<uint32_t> counts;
};

// Mirrors the fast-path predicate in sparse_matmul.cpp. Used to compute the
// expected split between fast- and slow-path dispatches on the current chip.
bool sparse_matmul_fast_path_eligible(uint32_t H, uint32_t K, uint32_t N, uint32_t num_banks) {
    constexpr uint64_t kTileH = 32U;
    constexpr uint64_t kTileHW = 32U * 32U;
    const uint64_t P = static_cast<uint64_t>(num_banks) * kTileHW;
    return (kTileH * H) % P == 0U && (kTileH * N) % P == 0U && (static_cast<uint64_t>(K) * N) % P == 0U;
}

void RunCase(const FfnCase& c) {
    using namespace ttml;

    // offsets[e+1] = offsets[e] + round_up(counts[e], TILE); last is T_cap.
    std::vector<uint32_t> offsets(c.E + 1, 0U);
    for (uint32_t e = 0; e < c.E; ++e) {
        const uint32_t padded = ((c.counts[e] + kTile - 1U) / kTile) * kTile;
        offsets[e + 1U] = offsets[e] + padded;
    }
    const uint32_t T_cap = offsets.back();
    ASSERT_GT(T_cap, 0U);

    auto& rng = autograd::ctx().get_generator();
    auto gen = [&]() { return std::uniform_real_distribution<float>(0.0f, 1.0f); };

    // 2D grouped ref, zeros everywhere except active per-expert slices.
    xt::xarray<float> grouped =
        xt::zeros<float>(std::vector<std::size_t>{static_cast<std::size_t>(T_cap), static_cast<std::size_t>(c.H)});
    for (uint32_t e = 0; e < c.E; ++e) {
        if (c.counts[e] == 0U) {
            continue;
        }
        std::vector<std::size_t> slice_shape{static_cast<std::size_t>(c.counts[e]), static_cast<std::size_t>(c.H)};
        xt::xarray<float> slice = xt::empty<float>(slice_shape);
        core::parallel_generate<float>(slice, gen, rng());
        xt::view(grouped, xt::range(offsets[e], offsets[e] + c.counts[e]), xt::all()) = slice;
    }

    std::vector<std::size_t> w_gate_up_shape{
        static_cast<std::size_t>(c.E), static_cast<std::size_t>(c.H), static_cast<std::size_t>(c.I)};
    std::vector<std::size_t> w_down_shape{
        static_cast<std::size_t>(c.E), static_cast<std::size_t>(c.I), static_cast<std::size_t>(c.H)};
    xt::xarray<float> w_gate = xt::empty<float>(w_gate_up_shape);
    xt::xarray<float> w_up = xt::empty<float>(w_gate_up_shape);
    xt::xarray<float> w_down = xt::empty<float>(w_down_shape);
    core::parallel_generate<float>(w_gate, gen, rng());
    core::parallel_generate<float>(w_up, gen, rng());
    core::parallel_generate<float>(w_down, gen, rng());

    auto* device = &autograd::ctx().get_device();
    metal::reset_sparse_matmul_counters();
    const uint32_t num_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    // The FFN dispatches 3 sparse_matmul calls: gate (K=H, N=I), up (K=H, N=I), down (K=I, N=H).
    const uint32_t non_empty_experts =
        static_cast<uint32_t>(std::count_if(c.counts.begin(), c.counts.end(), [](uint32_t k) { return k > 0; }));
    const bool gate_fast = sparse_matmul_fast_path_eligible(c.H, c.H, c.I, num_banks);
    const bool down_fast = sparse_matmul_fast_path_eligible(c.I, c.I, c.H, num_banks);
    const uint64_t expected_fast =
        (gate_fast ? 2U : 0U) + (down_fast ? 1U : 0U);  // per op call; gate and up share the eligibility check
    const uint64_t expected_slow = 3U - expected_fast;
    (void)non_empty_experts;

    // Materialize the rank-4 [1,1,T_cap,H] device tensor for `grouped`.
    std::vector<std::size_t> grouped_4d_shape{1U, 1U, static_cast<std::size_t>(T_cap), static_cast<std::size_t>(c.H)};
    xt::xarray<float> grouped_4d = xt::xarray<float>::from_shape(grouped_4d_shape);
    std::copy(grouped.begin(), grouped.end(), grouped_4d.begin());
    auto t_grouped = core::from_xtensor(grouped_4d, device);

    auto t_offsets = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets, ttnn::Shape({static_cast<uint32_t>(offsets.size())}), device, ttnn::Layout::ROW_MAJOR);

    auto t_wg = core::from_xtensor(w_gate, device);
    auto t_wu = core::from_xtensor(w_up, device);
    auto t_wd = core::from_xtensor(w_down, device);

    auto t_out = ops::moe_ffn_swiglu_fw(t_grouped, t_offsets, t_wg, t_wu, t_wd);

    const auto counters = metal::get_sparse_matmul_counters();
    EXPECT_EQ(counters.fast_path_calls, expected_fast)
        << "fast-path dispatches differ from expectation for num_banks=" << num_banks;
    EXPECT_EQ(counters.slow_path_calls, expected_slow)
        << "slow-path dispatches differ from expectation for num_banks=" << num_banks;
    xt::xarray<float> out_xt = core::to_xtensor(t_out);  // [1,1,T_cap,H]

    // Flatten to [T_cap, H] for comparison.
    std::vector<std::size_t> out_2d_shape{static_cast<std::size_t>(T_cap), static_cast<std::size_t>(c.H)};
    xt::xarray<float> out_2d = xt::xarray<float>::from_shape(out_2d_shape);
    std::copy(out_xt.begin(), out_xt.end(), out_2d.begin());

    xt::xarray<float> ref = moe_ffn_swiglu_reference(grouped, offsets, c.counts, w_gate, w_up, w_down);

    ASSERT_EQ(out_2d.shape(), ref.shape());
    EXPECT_TRUE(xt::all(xt::isfinite(out_2d))) << "non-finite values in output";

    const float rl2 = relative_l2(out_2d, ref);
    EXPECT_LT(rl2, 1e-2f) << "relative L2 too large: " << rl2;

    // Per-expert pad rows must be zero in the output.
    for (uint32_t e = 0; e < c.E; ++e) {
        const uint32_t pad_lo = offsets[e] + c.counts[e];
        const uint32_t pad_hi = offsets[e + 1U];
        if (pad_hi == pad_lo) {
            continue;
        }
        xt::xarray<float> pad_slice = xt::view(out_2d, xt::range(pad_lo, pad_hi), xt::all());
        const float max_abs = xt::amax(xt::abs(pad_slice))();
        EXPECT_NEAR(max_abs, 0.0f, 1e-4f) << "pad rows for expert " << e << " not zero";
    }
}

}  // namespace

TEST_F(MoeFfnSwigluForwardTest, Small_E2_H64_I128) {
    RunCase({/*E*/ 2U, /*H*/ 64U, /*I*/ 128U, /*counts*/ {48U, 16U}});
}

TEST_F(MoeFfnSwigluForwardTest, EmptyExpert) {
    RunCase({/*E*/ 3U, /*H*/ 64U, /*I*/ 128U, /*counts*/ {32U, 0U, 40U}});
}

TEST_F(MoeFfnSwigluForwardTest, Medium_E4_H512_I1408) {
    RunCase({/*E*/ 4U, /*H*/ 512U, /*I*/ 1408U, /*counts*/ {200U, 150U, 100U, 180U}});
}

// Dimensions both multiples of 256: takes the narrow fast path on P150 (8 banks),
// the slow path on P100 (7 banks). Same assertions work on either chip.
TEST_F(MoeFfnSwigluForwardTest, AlignedShapes_E4_H512_I1024) {
    RunCase({/*E*/ 4U, /*H*/ 512U, /*I*/ 1024U, /*counts*/ {64U, 96U, 32U, 128U}});
}

// Production-like H (4096) with a small intermediate dim. Fast path on P150,
// slow on P100 (4096 has no factor of 7).
TEST_F(MoeFfnSwigluForwardTest, AlignedShapes_E4_H4096_I512) {
    RunCase({/*E*/ 4U, /*H*/ 4096U, /*I*/ 512U, /*counts*/ {64U, 32U, 96U, 64U}});
}
