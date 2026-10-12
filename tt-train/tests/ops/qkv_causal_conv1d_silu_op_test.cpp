// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/qkv_causal_conv1d_silu_op.hpp"

#include <gtest/gtest.h>

#include <array>
#include <random>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"

namespace {

using ttml::autograd::TensorPtr;

constexpr uint32_t T = 64;
constexpr uint32_t Q = 96;
constexpr uint32_t K = 64;
constexpr uint32_t V = 32;
constexpr uint32_t C = Q + K + V;

xt::xarray<float> random_tensor(const std::vector<size_t>& shape, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    xt::xarray<float> out = xt::empty<float>(shape);
    for (auto& value : out) {
        value = dist(gen);
    }
    return out;
}

struct Reference {
    xt::xarray<float> q, k, v, grad_input;
    std::array<xt::xarray<float>, 4> grad_taps;
};

// Mirrors loss = sum(y * y) / 2, i.e. grad_y = y.
Reference reference(
    const xt::xarray<float>& x, const xt::xarray<float>& h, const std::array<xt::xarray<float>, 4>& taps) {
    xt::xarray<float> window = xt::concatenate(xt::xtuple(h, x), 1);
    xt::xarray<float> preact = xt::zeros<float>({size_t{1}, size_t{T}, size_t{C}});
    for (size_t j = 0; j < 4; ++j) {
        preact += taps[j] * xt::view(window, xt::all(), xt::range(j, j + T), xt::all());
    }
    xt::xarray<float> sig = 1.0F / (1.0F + xt::exp(-preact));
    xt::xarray<float> y = preact * sig;
    xt::xarray<float> grad_preact = y * sig * (1.0F + preact * (1.0F - sig));
    xt::xarray<float> grad_window = xt::zeros_like(window);
    Reference ref;
    for (size_t j = 0; j < 4; ++j) {
        xt::view(grad_window, xt::all(), xt::range(j, j + T), xt::all()) += taps[j] * grad_preact;
        ref.grad_taps[j] =
            xt::sum(grad_preact * xt::view(window, xt::all(), xt::range(j, j + T), xt::all()), {0, 1}, xt::keep_dims);
    }
    ref.grad_input = xt::view(grad_window, xt::all(), xt::range(3, 3 + T), xt::all());
    ref.q = xt::view(y, xt::all(), xt::all(), xt::range(0, Q));
    ref.k = xt::view(y, xt::all(), xt::all(), xt::range(Q, Q + K));
    ref.v = xt::view(y, xt::all(), xt::all(), xt::range(Q + K, C));
    return ref;
}

void expect_close(const xt::xarray<float>& expected, const xt::xarray<float>& actual, const char* name) {
    ASSERT_EQ(expected.shape(), actual.shape()) << name;
    const float scale = std::max(1e-3F, static_cast<float>(xt::amax(xt::abs(expected))()));
    EXPECT_TRUE(xt::allclose(expected, actual, 1e-2F, 3e-2F * scale)) << name;
}

class QkvCausalConv1dSiluOpTest : public ::testing::TestWithParam<uint32_t> {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

TEST_P(QkvCausalConv1dSiluOpTest, MatchesReference) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const uint32_t channel_chunk_size = GetParam();

    auto x = random_tensor({1, T, C}, 1);
    auto h = random_tensor({1, 3, C}, 2);
    std::array<xt::xarray<float>, 4> taps;
    for (size_t j = 0; j < 4; ++j) {
        taps[j] = random_tensor({1, 1, C}, 3 + j);
    }
    const auto ref = reference(x, h, taps);

    auto input = autograd::create_tensor(core::from_xtensor(x, device), /* requires_grad */ true);
    auto history = core::from_xtensor(h, device, ttnn::Layout::ROW_MAJOR);
    std::array<TensorPtr, 4> tap_tensors;
    for (size_t j = 0; j < 4; ++j) {
        tap_tensors[j] = autograd::create_tensor(core::from_xtensor(taps[j], device), /* requires_grad */ true);
    }

    auto [q, k, v] = ops::qkv_causal_conv1d_silu(
        input, history, tap_tensors[0], tap_tensors[1], tap_tensors[2], tap_tensors[3], Q, K, V, channel_chunk_size);
    expect_close(ref.q, core::to_xtensor(q->get_value()), "q");
    expect_close(ref.k, core::to_xtensor(k->get_value()), "k");
    expect_close(ref.v, core::to_xtensor(v->get_value()), "v");

    for (const auto& out : {q, k, v}) {
        out->set_grad(out->get_value());
    }
    q->backward();

    expect_close(ref.grad_input, core::to_xtensor(input->get_grad()), "grad_input");
    for (size_t j = 0; j < 4; ++j) {
        expect_close(ref.grad_taps[j], core::to_xtensor(tap_tensors[j]->get_grad()), "grad_tap");
    }
}

INSTANTIATE_TEST_SUITE_P(ChannelChunk, QkvCausalConv1dSiluOpTest, ::testing::Values(32U, 64U, 192U));

}  // namespace
