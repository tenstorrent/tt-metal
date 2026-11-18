// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/grouped_dynamic_causal_conv_op.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <tt-metalium/bfloat16.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"

namespace {

struct Case {
    uint32_t batch, seq, channels, kernel_size, block, group;

    [[nodiscard]] uint32_t rows() const {
        return batch * seq;
    }
    [[nodiscard]] uint32_t groups() const {
        return channels / group;
    }
};

xt::xarray<float> random_tensor(const std::vector<size_t>& shape, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    xt::xarray<float> out = xt::empty<float>(shape);
    for (auto& value : out) {
        value = static_cast<float>(bfloat16(dist(gen)));  // exact on device, so the reference sees the same inputs
    }
    return out;
}

struct Problem {
    xt::xarray<float> x, dynamic_kernel, base_kernel, grad_out;
};

Problem random_problem(const Case& c, uint32_t seed) {
    return {
        random_tensor({c.batch, 1, c.seq, c.channels}, seed),
        random_tensor({c.batch, 1, c.seq, c.kernel_size * c.groups()}, seed + 1),
        random_tensor({1, 1, c.kernel_size, c.channels}, seed + 2),
        random_tensor({c.batch, 1, c.seq, c.channels}, seed + 3),
    };
}

struct Reference {
    xt::xarray<float> out, grad_input, grad_dynamic_kernel, grad_base_kernel;
};

// z-lab `_grouped_dynamic_convolve` and its gradients over flattened rows; row t's block starts at t - t % block.
Reference reference(const Case& c, const Problem& p) {
    const size_t N = c.rows(), C = c.channels, K = c.kernel_size, G = c.groups(), L = c.block, gs = c.group;
    const float* x = p.x.data();
    const float* dynamic_kernel = p.dynamic_kernel.data();
    const float* base_kernel = p.base_kernel.data();
    const float* gy = p.grad_out.data();
    Reference r{
        xt::zeros_like(p.x),
        xt::zeros_like(p.x),
        xt::zeros_like(p.dynamic_kernel),
        xt::zeros<float>(p.base_kernel.shape())};
    for (size_t t = 0; t < N; ++t) {
        for (size_t o = 0; o < K && o <= t % L; ++o) {
            const size_t s = t - o;  // the source row, same block
            for (size_t ch = 0; ch < C; ++ch) {
                const size_t d = t * K * G + o * G + ch / gs;
                const float w = base_kernel[o * C + ch] + dynamic_kernel[d];
                r.out.data()[t * C + ch] += w * x[s * C + ch];
                r.grad_input.data()[s * C + ch] += w * gy[t * C + ch];
                r.grad_dynamic_kernel.data()[d] += gy[t * C + ch] * x[s * C + ch];
                r.grad_base_kernel.data()[o * C + ch] += gy[t * C + ch] * x[s * C + ch];
            }
        }
    }
    return r;
}

// bf16 inputs and outputs: an element's error scales with the tensor's range, so bounds are relative to its max.
// grad_base_kernel sums `rows` products per element, hence its looser per-element bound.
void expect_close(
    const xt::xarray<float>& expected, const ttnn::Tensor& actual_tensor, const std::string& name, double max_rel) {
    const xt::xarray<float> actual = ttml::core::to_xtensor(actual_tensor);
    ASSERT_EQ(expected.shape(), actual.shape()) << name;
    const double scale = xt::amax(xt::abs(expected))();
    const xt::xarray<double> diff = xt::abs(xt::cast<double>(actual) - xt::cast<double>(expected));
    EXPECT_LE(xt::amax(diff)(), max_rel * scale) << name << " max|diff| of max " << scale;
    EXPECT_LE(xt::mean(diff)(), 2e-3 * scale) << name << " mean|diff| of max " << scale;
}

class GroupedDynamicCausalConvTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }

    static void run(const Case& c, uint32_t seed = 7) {
        auto* device = &ttml::autograd::ctx().get_device();
        const Problem p = random_problem(c, seed);
        const Reference ref = reference(c, p);
        auto input = ttml::autograd::create_tensor(ttml::core::from_xtensor(p.x, device));
        auto dynamic_kernel = ttml::autograd::create_tensor(ttml::core::from_xtensor(p.dynamic_kernel, device));
        auto base_kernel = ttml::autograd::create_tensor(ttml::core::from_xtensor(p.base_kernel, device));

        auto out = ttml::ops::grouped_dynamic_causal_conv(input, dynamic_kernel, base_kernel, c.block, c.group);
        expect_close(ref.out, out->get_value(), "out", 2e-2);

        out->set_grad(ttml::core::from_xtensor(p.grad_out, device));
        out->backward();
        expect_close(ref.grad_input, input->get_grad(), "grad_input", 2e-2);
        expect_close(ref.grad_dynamic_kernel, dynamic_kernel->get_grad(), "grad_dynamic_kernel", 2e-2);
        expect_close(ref.grad_base_kernel, base_kernel->get_grad(), "grad_base_kernel", 3e-2);
    }
};

// DFlash 2 drafters: kernel 2, group 16, blocks of 8.
TEST_F(GroupedDynamicCausalConvTest, FlatRows) {
    run({1, 64, 512, 2, 8, 16});
}

TEST_F(GroupedDynamicCausalConvTest, OneBlockPerBatchEntry) {
    run({8, 8, 512, 2, 8, 16});
}

TEST_F(GroupedDynamicCausalConvTest, KernelThreeBlockFour) {
    run({1, 96, 1024, 3, 4, 32});
}

TEST_F(GroupedDynamicCausalConvTest, KernelOne) {
    run({2, 32, 512, 1, 8, 16});
}

// 512 blocks of 8 at the GLM-5.3 / MiniMax-M3.1 DFlash 2 width.
TEST_F(GroupedDynamicCausalConvTest, Batch512Hidden6144) {
    run({1, 512 * 8, 6144, 2, 8, 16});
}

TEST_F(GroupedDynamicCausalConvTest, Batch512AsBatchDim) {
    run({512, 8, 512, 2, 8, 16});
}

TEST_F(GroupedDynamicCausalConvTest, RejectsBadShapes) {
    auto* device = &ttml::autograd::ctx().get_device();
    const auto make = [&](std::vector<size_t> shape) {
        return ttml::autograd::create_tensor(
            ttml::core::from_xtensor(xt::xarray<float>(xt::zeros<float>(shape)), device));
    };
    auto x = make({1, 1, 64, 512});
    auto dynamic_kernel_o = make({1, 1, 64, 64});
    auto base_kernel = make({1, 1, 2, 512});
    // seq not a multiple of block
    EXPECT_THROW(
        ttml::ops::grouped_dynamic_causal_conv(x, dynamic_kernel_o, base_kernel, 24, 16), std::invalid_argument);
    // dynamic_kernel width != K * G
    EXPECT_THROW(
        ttml::ops::grouped_dynamic_causal_conv(x, make({1, 1, 64, 32}), base_kernel, 8, 16), std::invalid_argument);
    // groups not tile-aligned (512 / 32 = 16)
    EXPECT_THROW(
        ttml::ops::grouped_dynamic_causal_conv(x, make({1, 1, 64, 32}), base_kernel, 8, 32), std::invalid_argument);
    // kernel longer than the block
    EXPECT_THROW(
        ttml::ops::grouped_dynamic_causal_conv(x, make({1, 1, 64, 96}), make({1, 1, 3, 512}), 2, 16),
        std::invalid_argument);
}

}  // namespace
