// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/moe_ffn_swiglu_op.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "test_utils/random_data.hpp"

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
// (silu(X_e @ W_gate_e^T) * (X_e @ W_up_e^T)) @ W_down_e^T and splat into
// rows [offsets[e], offsets[e]+counts[e]) of a zero-initialized output.
// Weights are in [out, in] layout (LinearLayer convention).
xt::xarray<float> moe_ffn_swiglu_reference(
    const xt::xarray<float>& grouped,  // [T_cap, H]
    const std::vector<uint32_t>& offsets,
    const std::vector<uint32_t>& counts,  // active rows per expert
    const xt::xarray<float>& w_gate,      // [E, I, H]
    const xt::xarray<float>& w_up,        // [E, I, H]
    const xt::xarray<float>& w_down) {    // [E, H, I]
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

        const xt::xarray<float> X = xt::view(grouped, xt::range(row_lo, row_hi), xt::all());
        const xt::xarray<float> Wg = xt::view(w_gate, e, xt::all(), xt::all());
        const xt::xarray<float> Wu = xt::view(w_up, e, xt::all(), xt::all());
        const xt::xarray<float> Wd = xt::view(w_down, e, xt::all(), xt::all());

        const xt::xarray<float> G = xt::linalg::tensordot(X, Wg, {1}, {1});  // [n, I]  X @ Wg^T
        const xt::xarray<float> U = xt::linalg::tensordot(X, Wu, {1}, {1});  // [n, I]  X @ Wu^T
        const xt::xarray<float> sigmoid_G = 1.0f / (1.0f + xt::exp(-G));
        const xt::xarray<float> A = (G * sigmoid_G) * U;                     // [n, I]
        const xt::xarray<float> Y = xt::linalg::tensordot(A, Wd, {1}, {1});  // [n, H]  A @ Wd^T

        xt::view(out, xt::range(row_lo, row_hi), xt::all()) = Y;
    }
    return out;
}

struct FfnCase {
    uint32_t E;
    uint32_t H;
    uint32_t I;
    std::vector<uint32_t> counts;
    // Extra tile-rows of trailing slack between offsets[-1] and T_cap,
    // mirroring moe_group's worst-case T_cap upper-bound. The op pads this
    // region with zeros to produce an output of shape [1,1,T_cap,H].
    uint32_t tail_pad_tiles = 0U;
};

// Build a list of per-expert rank-4 TensorPtrs from a stacked rank-3 xtensor
// [E, K, N]. The list is what the op consumes; the stacked xtensor stays
// around for the CPU reference to use.
std::vector<ttml::autograd::TensorPtr> make_expert_weight_list(
    const xt::xarray<float>& w3d, ttnn::distributed::MeshDevice* device) {
    const std::size_t E = w3d.shape()[0];
    const std::size_t K = w3d.shape()[1];
    const std::size_t N = w3d.shape()[2];
    std::vector<ttml::autograd::TensorPtr> out;
    out.reserve(E);
    const std::vector<std::size_t> shape4d{1U, 1U, K, N};
    for (std::size_t e = 0; e < E; ++e) {
        xt::xarray<float> w4d = xt::xarray<float>::from_shape(shape4d);
        const auto src = xt::view(w3d, e, xt::all(), xt::all());
        std::copy(src.cbegin(), src.cend(), w4d.begin());
        out.push_back(ttml::autograd::create_tensor(ttml::core::from_xtensor(w4d, device), /*requires_grad=*/true));
    }
    return out;
}

void RunCase(const FfnCase& c) {
    using namespace ttml;

    std::vector<uint32_t> offsets(c.E + 1, 0U);
    for (uint32_t e = 0; e < c.E; ++e) {
        const uint32_t padded = ((c.counts[e] + kTile - 1U) / kTile) * kTile;
        offsets[e + 1U] = offsets[e] + padded;
    }
    const uint32_t used_rows = offsets.back();
    const uint32_t T_cap = used_rows + c.tail_pad_tiles * kTile;
    ASSERT_GT(T_cap, 0U);

    auto& rng = autograd::ctx().get_generator();

    // 2D grouped ref, zeros everywhere except active per-expert slices.
    xt::xarray<float> grouped =
        xt::zeros<float>(std::vector<std::size_t>{static_cast<std::size_t>(T_cap), static_cast<std::size_t>(c.H)});
    for (uint32_t e = 0; e < c.E; ++e) {
        if (c.counts[e] == 0U) {
            continue;
        }
        const std::array<std::size_t, 2U> slice_shape{
            static_cast<std::size_t>(c.counts[e]), static_cast<std::size_t>(c.H)};
        const auto slice = test_utils::make_uniform_xarray<float>(slice_shape, 0.0f, 1.0f, rng());
        xt::view(grouped, xt::range(offsets[e], offsets[e] + c.counts[e]), xt::all()) = slice;
    }

    // [out, in] layout: w_gate/w_up are [E, I, H], w_down is [E, H, I].
    const std::array<std::size_t, 3U> w_gate_up_shape{
        static_cast<std::size_t>(c.E), static_cast<std::size_t>(c.I), static_cast<std::size_t>(c.H)};
    const std::array<std::size_t, 3U> w_down_shape{
        static_cast<std::size_t>(c.E), static_cast<std::size_t>(c.H), static_cast<std::size_t>(c.I)};
    const auto w_gate = test_utils::make_uniform_xarray<float>(w_gate_up_shape, 0.0f, 1.0f, rng());
    const auto w_up = test_utils::make_uniform_xarray<float>(w_gate_up_shape, 0.0f, 1.0f, rng());
    const auto w_down = test_utils::make_uniform_xarray<float>(w_down_shape, 0.0f, 1.0f, rng());

    auto* device = &autograd::ctx().get_device();

    // Materialize the rank-4 [1,1,T_cap,H] device tensor for `grouped`.
    const std::vector<std::size_t> grouped_4d_shape{
        1U, 1U, static_cast<std::size_t>(T_cap), static_cast<std::size_t>(c.H)};
    xt::xarray<float> grouped_4d = xt::xarray<float>::from_shape(grouped_4d_shape);
    std::copy(grouped.begin(), grouped.end(), grouped_4d.begin());
    const auto t_grouped = autograd::create_tensor(core::from_xtensor(grouped_4d, device), /*requires_grad=*/true);

    const auto t_offsets = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets, ttnn::Shape({static_cast<uint32_t>(offsets.size())}), device, ttnn::Layout::ROW_MAJOR);

    const auto t_wg = make_expert_weight_list(w_gate, device);
    const auto t_wu = make_expert_weight_list(w_up, device);
    const auto t_wd = make_expert_weight_list(w_down, device);

    const auto t_out = ops::moe_ffn_swiglu_fw(t_grouped, t_offsets, t_wg, t_wu, t_wd);
    const xt::xarray<float> out_xt = core::to_xtensor(t_out->get_value());  // [1,1,T_cap,H]

    // Flatten to [T_cap, H] for comparison.
    const std::vector<std::size_t> out_2d_shape{static_cast<std::size_t>(T_cap), static_cast<std::size_t>(c.H)};
    xt::xarray<float> out_2d = xt::xarray<float>::from_shape(out_2d_shape);
    std::copy(out_xt.begin(), out_xt.end(), out_2d.begin());

    const xt::xarray<float> ref = moe_ffn_swiglu_reference(grouped, offsets, c.counts, w_gate, w_up, w_down);

    ASSERT_EQ(out_2d.shape(), ref.shape());
    EXPECT_TRUE(xt::all(xt::isfinite(out_2d))) << "non-finite values in output";

    constexpr float kRtol = 1e-2f;
    constexpr float kAtol = 1e-2f;
    EXPECT_TRUE(xt::allclose(out_2d, ref, kRtol, kAtol))
        << "allclose failed (rtol=" << kRtol << " atol=" << kAtol << ")";

    // Per-expert pad rows must be zero in the output.
    for (uint32_t e = 0; e < c.E; ++e) {
        const uint32_t pad_lo = offsets[e] + c.counts[e];
        const uint32_t pad_hi = offsets[e + 1U];
        if (pad_hi == pad_lo) {
            continue;
        }
        const xt::xarray<float> pad_slice = xt::view(out_2d, xt::range(pad_lo, pad_hi), xt::all());
        const float max_abs = xt::amax(xt::abs(pad_slice))();
        EXPECT_NEAR(max_abs, 0.0f, 1e-4f) << "pad rows for expert " << e << " not zero";
    }

    // Trailing slack [used_rows, T_cap) must be exactly zero
    if (T_cap > used_rows) {
        const xt::xarray<float> tail = xt::view(out_2d, xt::range(used_rows, T_cap), xt::all());
        EXPECT_EQ(xt::amax(xt::abs(tail))(), 0.0f) << "trailing slack rows not zero";
    }
}

}  // namespace

TEST_F(MoeFfnSwigluForwardTest, Small_E2_H64_I128) {
    RunCase({/*E*/ 2U, /*H*/ 64U, /*I*/ 128U, /*counts*/ {48U, 16U}});
}

TEST_F(MoeFfnSwigluForwardTest, NIGHTLY_EmptyExpert) {
    RunCase({/*E*/ 3U, /*H*/ 64U, /*I*/ 128U, /*counts*/ {32U, 0U, 40U}});
}

TEST_F(MoeFfnSwigluForwardTest, NIGHTLY_Medium_E4_H512_I1408) {
    RunCase({/*E*/ 4U, /*H*/ 512U, /*I*/ 1408U, /*counts*/ {200U, 150U, 100U, 180U}});
}

TEST_F(MoeFfnSwigluForwardTest, NIGHTLY_AlignedShapes_E4_H512_I1024) {
    RunCase({/*E*/ 4U, /*H*/ 512U, /*I*/ 1024U, /*counts*/ {64U, 96U, 32U, 128U}});
}

TEST_F(MoeFfnSwigluForwardTest, NIGHTLY_AlignedShapes_E4_H4096_I512) {
    RunCase({/*E*/ 4U, /*H*/ 4096U, /*I*/ 512U, /*counts*/ {64U, 32U, 96U, 64U}});
}

TEST_F(MoeFfnSwigluForwardTest, NIGHTLY_TrailingPad_E3_H64_I128) {
    RunCase({/*E*/ 3U, /*H*/ 64U, /*I*/ 128U, /*counts*/ {32U, 16U, 48U}, /*tail_pad_tiles*/ 5U});
}

// Backward sanity: gradients populate, are finite, have the right shapes,
// and are zero on per-expert pad rows of `grouped` (those rows contribute
// nothing to the loss). Numerical correctness is left to a downstream
// reference test; this guards the op wiring and graph dependencies.
class MoeFfnSwigluBackwardTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(MoeFfnSwigluBackwardTest, NIGHTLY_GradientsRunAndShapesMatch) {
    using namespace ttml;

    constexpr uint32_t E = 3U;
    constexpr uint32_t H = 64U;
    constexpr uint32_t I = 128U;
    const std::vector<uint32_t> counts = {32U, 16U, 48U};

    std::vector<uint32_t> offsets(E + 1U, 0U);
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t padded = ((counts[e] + 31U) / 32U) * 32U;
        offsets[e + 1U] = offsets[e] + padded;
    }
    const uint32_t T_cap = offsets.back();

    auto& rng = autograd::ctx().get_generator();

    xt::xarray<float> grouped_4d = xt::zeros<float>(
        std::vector<std::size_t>{1U, 1U, static_cast<std::size_t>(T_cap), static_cast<std::size_t>(H)});
    for (uint32_t e = 0; e < E; ++e) {
        if (counts[e] == 0U) {
            continue;
        }
        const std::array<std::size_t, 4U> slice_shape{
            1U, 1U, static_cast<std::size_t>(counts[e]), static_cast<std::size_t>(H)};
        const auto slice = test_utils::make_uniform_xarray<float>(slice_shape, 0.0f, 1.0f, rng());
        xt::view(grouped_4d, 0, 0, xt::range(offsets[e], offsets[e] + counts[e]), xt::all()) = xt::view(slice, 0, 0);
    }

    // [out, in] layout: w_gate/w_up are [E, I, H], w_down is [E, H, I].
    const auto w_gate = test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 3U>{static_cast<std::size_t>(E), I, H}, 0.0f, 1.0f, rng());
    const auto w_up = test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 3U>{static_cast<std::size_t>(E), I, H}, 0.0f, 1.0f, rng());
    const auto w_down = test_utils::make_uniform_xarray<float>(
        std::array<std::size_t, 3U>{static_cast<std::size_t>(E), H, I}, 0.0f, 1.0f, rng());

    auto* device = &autograd::ctx().get_device();

    const auto t_grouped = autograd::create_tensor(core::from_xtensor(grouped_4d, device), /*requires_grad=*/true);
    const auto t_offsets = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        offsets, ttnn::Shape({static_cast<uint32_t>(offsets.size())}), device, ttnn::Layout::ROW_MAJOR);
    const auto t_wg = make_expert_weight_list(w_gate, device);
    const auto t_wu = make_expert_weight_list(w_up, device);
    const auto t_wd = make_expert_weight_list(w_down, device);

    const auto t_out = ops::moe_ffn_swiglu_fw(t_grouped, t_offsets, t_wg, t_wu, t_wd);
    t_out->set_grad(core::ones_like(t_out->get_value()));
    t_out->backward();

    const auto dgrouped = core::to_xtensor(t_grouped->get_grad());
    EXPECT_EQ(dgrouped.shape(), grouped_4d.shape());
    EXPECT_TRUE(xt::all(xt::isfinite(dgrouped))) << "non-finite dgrouped";

    auto check_per_expert_grads = [&](const std::vector<autograd::TensorPtr>& list,
                                      const std::vector<std::size_t>& expected_per_expert_shape,
                                      const std::string& name) {
        ASSERT_EQ(list.size(), E);
        for (uint32_t e = 0; e < E; ++e) {
            const auto g = core::to_xtensor(list[e]->get_grad());
            EXPECT_EQ(g.shape(), expected_per_expert_shape) << name << "[" << e << "] shape mismatch";
            EXPECT_TRUE(xt::all(xt::isfinite(g))) << "non-finite " << name << "[" << e << "]";
        }
    };
    check_per_expert_grads(t_wg, {1U, 1U, I, H}, "dW_gate");
    check_per_expert_grads(t_wu, {1U, 1U, I, H}, "dW_up");
    check_per_expert_grads(t_wd, {1U, 1U, H, I}, "dW_down");

    // Loss = sum(Y); inputs are positive, weights are positive — every active row
    // contributes a non-zero gradient. Active-row dgrouped should be non-trivial.
    bool any_active_nonzero = false;
    for (uint32_t e = 0; e < E; ++e) {
        if (counts[e] == 0U) {
            continue;
        }
        const auto active = xt::view(dgrouped, 0, 0, xt::range(offsets[e], offsets[e] + counts[e]), xt::all());
        if (xt::amax(xt::abs(active))() > 0.0f) {
            any_active_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(any_active_nonzero) << "all active-row gradients are zero — backward likely not connected";

    // Per-expert pad rows of `grouped` had zero input → must have zero gradient.
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t pad_lo = offsets[e] + counts[e];
        const uint32_t pad_hi = offsets[e + 1U];
        if (pad_hi == pad_lo) {
            continue;
        }
        const auto pad = xt::view(dgrouped, 0, 0, xt::range(pad_lo, pad_hi), xt::all());
        EXPECT_NEAR(xt::amax(xt::abs(pad))(), 0.0f, 1e-3f) << "non-zero dgrouped on pad rows for expert " << e;
    }

    autograd::ctx().reset_graph();
}
