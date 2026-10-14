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
#include "metal/operations.hpp"

namespace {

using ttml::autograd::TensorPtr;

struct Shape {
    uint32_t T, Q, K, V;
    uint32_t C() const {
        return Q + K + V;
    }
};

xt::xarray<float> random_tensor(const std::vector<size_t>& shape, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    xt::xarray<float> out = xt::empty<float>(shape);
    for (auto& value : out) {
        value = dist(gen);
    }
    return out;
}

struct Problem {
    xt::xarray<float> x, h, grad_y;
    std::array<xt::xarray<float>, 4> taps;
};

Problem random_problem(const Shape& s, uint32_t seed) {
    Problem p;
    p.x = random_tensor({1, s.T, s.C()}, seed);
    p.h = random_tensor({1, 3, s.C()}, seed + 1);
    p.grad_y = random_tensor({1, s.T, s.C()}, seed + 2);
    for (size_t j = 0; j < 4; ++j) {
        p.taps[j] = random_tensor({1, 1, s.C()}, seed + 3 + j);
    }
    return p;
}

struct Reference {
    xt::xarray<float> y, grad_input;
    std::array<xt::xarray<float>, 4> grad_taps;
};

Reference reference(const Shape& s, const Problem& p) {
    const size_t T = s.T;
    xt::xarray<float> window = xt::concatenate(xt::xtuple(p.h, p.x), 1);
    xt::xarray<float> preact = xt::zeros<float>({size_t{1}, T, size_t{s.C()}});
    for (size_t j = 0; j < 4; ++j) {
        preact += p.taps[j] * xt::view(window, xt::all(), xt::range(j, j + T), xt::all());
    }
    xt::xarray<float> sig = 1.0F / (1.0F + xt::exp(-preact));
    xt::xarray<float> grad_preact = p.grad_y * sig * (1.0F + preact * (1.0F - sig));
    xt::xarray<float> grad_window = xt::zeros_like(window);
    Reference ref;
    ref.y = preact * sig;
    for (size_t j = 0; j < 4; ++j) {
        xt::view(grad_window, xt::all(), xt::range(j, j + T), xt::all()) += p.taps[j] * grad_preact;
        ref.grad_taps[j] =
            xt::sum(grad_preact * xt::view(window, xt::all(), xt::range(j, j + T), xt::all()), {0, 1}, xt::keep_dims);
    }
    ref.grad_input = xt::view(grad_window, xt::all(), xt::range(3, 3 + T), xt::all());
    return ref;
}

void expect_close(const xt::xarray<float>& expected, const xt::xarray<float>& actual, const std::string& name) {
    ASSERT_EQ(expected.shape(), actual.shape()) << name;
    const float scale = static_cast<float>(xt::amax(xt::abs(expected))());
    EXPECT_TRUE(xt::allclose(expected, actual, 1e-2, 3e-2 * scale))
        << name << " max|diff| = " << xt::amax(xt::abs(expected - actual))();
}

xt::xarray<float> slice_cols(const xt::xarray<float>& t, size_t begin, size_t end) {
    return xt::view(t, xt::all(), xt::all(), xt::range(begin, end));
}

class QkvCausalConv1dSiluTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// Runs the fused backward directly and checks every gradient against the xtensor reference.
void check_metal_bw(const Shape& s, uint32_t chunk, uint32_t splits, uint32_t seed, const std::string& tag) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const auto p = random_problem(s, seed);
    const auto ref = reference(s, p);

    auto input = core::from_xtensor(p.x, device, ttnn::Layout::ROW_MAJOR);
    auto history = core::from_xtensor(p.h, device, ttnn::Layout::ROW_MAJOR);
    std::array<ttnn::Tensor, 4> taps;
    for (size_t j = 0; j < 4; ++j) {
        taps[j] = core::from_xtensor(p.taps[j], device);
    }
    auto grad_q = core::from_xtensor(slice_cols(p.grad_y, 0, s.Q), device);
    auto grad_k = core::from_xtensor(slice_cols(p.grad_y, s.Q, s.Q + s.K), device);
    auto grad_v = core::from_xtensor(slice_cols(p.grad_y, s.Q + s.K, s.C()), device);

    auto [grad_input, g0, g1, g2, g3] = metal::qkv_causal_conv1d_silu_bw(
        grad_q, grad_k, grad_v, input, history, taps[0], taps[1], taps[2], taps[3], s.Q, s.K, s.V, chunk, splits);
    expect_close(ref.grad_input, core::to_xtensor(grad_input), tag + " grad_input");
    const std::array<ttnn::Tensor, 4> grads = {g0, g1, g2, g3};
    for (size_t j = 0; j < 4; ++j) {
        expect_close(ref.grad_taps[j], core::to_xtensor(grads[j]), tag + " grad_tap" + std::to_string(j));
    }
}

struct MetalCase {
    Shape shape;
    uint32_t chunk;
    uint32_t splits;
};

class QkvCausalConv1dSiluMetalTest : public QkvCausalConv1dSiluTest, public ::testing::WithParamInterface<MetalCase> {};

TEST_P(QkvCausalConv1dSiluMetalTest, MatchesReference) {
    const auto& c = GetParam();
    check_metal_bw(c.shape, c.chunk, c.splits, 7, "case");
}

INSTANTIATE_TEST_SUITE_P(
    Cases,
    QkvCausalConv1dSiluMetalTest,
    ::testing::Values(
        MetalCase{{32, 32, 32, 32}, 32, 1},
        MetalCase{{64, 96, 64, 32}, 32, 1},
        MetalCase{{64, 96, 64, 32}, 64, 2},
        MetalCase{{96, 96, 64, 32}, 192, 2},
        MetalCase{{96, 32, 32, 32}, 96, 3},
        MetalCase{{256, 128, 64, 64}, 128, 1},
        MetalCase{{256, 128, 64, 64}, 256, 8},
        MetalCase{{256, 128, 64, 64}, 32, 5},
        MetalCase{{512, 256, 128, 128}, 256, 0},
        MetalCase{{512, 256, 128, 128}, 32, 0},
        MetalCase{{256, 128, 64, 64}, 0, 0},  // auto chunk: 256
        MetalCase{{96, 96, 64, 32}, 0, 2}));  // auto chunk: 192 (C = 192 is not a multiple of 256)

TEST_F(QkvCausalConv1dSiluTest, RepeatedRunsAreStable) {
    for (uint32_t i = 0; i < 10; ++i) {
        check_metal_bw({128, 64, 64, 64}, 64, 2 + (i % 3), 100 + i, "repeat" + std::to_string(i));
    }
}

TEST_F(QkvCausalConv1dSiluTest, RandomShapes) {
    std::mt19937 gen(42);
    const auto pick = [&](uint32_t lo, uint32_t hi) { return std::uniform_int_distribution<uint32_t>(lo, hi)(gen); };
    for (uint32_t i = 0; i < 12; ++i) {
        const Shape s{32 * pick(1, 8), 32 * pick(1, 4), 32 * pick(1, 4), 32 * pick(1, 4)};
        std::vector<uint32_t> chunks;
        for (uint32_t c = 32; c <= std::min(s.C(), 256U); c += 32) {
            if (s.C() % c == 0) {
                chunks.push_back(c);
            }
        }
        const uint32_t chunk = chunks[pick(0, chunks.size() - 1)];
        const uint32_t splits = pick(1, s.T / 32);
        check_metal_bw(
            s,
            chunk,
            splits,
            1000 + i,
            "random T=" + std::to_string(s.T) + " QKV=" + std::to_string(s.Q) + "," + std::to_string(s.K) + "," +
                std::to_string(s.V) + " chunk=" + std::to_string(chunk) + " splits=" + std::to_string(splits));
    }
}

class QkvCausalConv1dSiluOpTest : public QkvCausalConv1dSiluTest, public ::testing::WithParamInterface<uint32_t> {};

// Autograd wrapper: forward matches, and grads seeded with grad_y = y match (loss = sum(y * y) / 2).
TEST_P(QkvCausalConv1dSiluOpTest, MatchesReference) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const Shape s{64, 96, 64, 32};
    const uint32_t channel_chunk_size = GetParam();
    auto p = random_problem(s, 1);
    auto ref = reference(s, p);
    p.grad_y = ref.y;
    ref = reference(s, p);

    auto input = autograd::create_tensor(core::from_xtensor(p.x, device), /* requires_grad */ true);
    auto history = core::from_xtensor(p.h, device, ttnn::Layout::ROW_MAJOR);
    std::array<TensorPtr, 4> tap_tensors;
    for (size_t j = 0; j < 4; ++j) {
        tap_tensors[j] = autograd::create_tensor(core::from_xtensor(p.taps[j], device), /* requires_grad */ true);
    }

    auto [q, k, v] = ops::qkv_causal_conv1d_silu(
        input,
        history,
        tap_tensors[0],
        tap_tensors[1],
        tap_tensors[2],
        tap_tensors[3],
        s.Q,
        s.K,
        s.V,
        channel_chunk_size);
    expect_close(slice_cols(ref.y, 0, s.Q), core::to_xtensor(q->get_value()), "q");
    expect_close(slice_cols(ref.y, s.Q, s.Q + s.K), core::to_xtensor(k->get_value()), "k");
    expect_close(slice_cols(ref.y, s.Q + s.K, s.C()), core::to_xtensor(v->get_value()), "v");

    for (const auto& out : {q, k, v}) {
        out->set_grad(out->get_value());
    }
    q->backward();

    expect_close(ref.grad_input, core::to_xtensor(input->get_grad()), "grad_input");
    for (size_t j = 0; j < 4; ++j) {
        expect_close(ref.grad_taps[j], core::to_xtensor(tap_tensors[j]->get_grad()), "grad_tap");
    }
}

INSTANTIATE_TEST_SUITE_P(ChannelChunk, QkvCausalConv1dSiluOpTest, ::testing::Values(0U, 32U, 64U, 192U));

}  // namespace
