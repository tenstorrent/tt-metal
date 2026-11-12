// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/conv3d_op.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "test_utils/random_data.hpp"
#include "test_utils/relative_l2.hpp"

namespace {

using Dims3 = ttml::ops::Conv3dDims;

struct Conv3dCase {
    uint32_t N = 1;
    Dims3 in_size{4, 5, 6};  // D, H, W
    uint32_t C_in = 32;
    uint32_t C_out = 32;
    Dims3 kernel{3, 3, 3};
    Dims3 stride{1, 1, 1};
    Dims3 padding{0, 0, 0};
    Dims3 dilation{1, 1, 1};
    bool with_bias = true;
    bool bias_rank1 = false;  // pass the bias as [C_out] instead of [1, 1, 1, C_out]
    bool input_requires_grad = true;
    bool weight_requires_grad = true;
    bool bias_requires_grad = true;
    ttnn::Layout layout = ttnn::Layout::ROW_MAJOR;
    float relative_l2_tolerance = 0.F;  // > 0 compares by relative L2 instead of elementwise

    [[nodiscard]] Dims3 out_size() const {
        Dims3 out{};
        for (size_t i = 0; i < 3; ++i) {
            const uint32_t effective_kernel = dilation[i] * (kernel[i] - 1) + 1;
            out[i] = (in_size[i] + 2 * padding[i] - effective_kernel) / stride[i] + 1;
        }
        return out;
    }
};

// Plain-float CPU reference for channels-last conv3d and its gradients.
struct Conv3dReference {
    std::vector<float> output;       // [N, D_out, H_out, W_out, C_out]
    std::vector<float> grad_input;   // [N, D, H, W, C_in]
    std::vector<float> grad_weight;  // [C_out, C_in, kD, kH, kW]
    std::vector<float> grad_bias;    // [C_out]
};

Conv3dReference compute_reference(
    const Conv3dCase& c,
    const xt::xarray<float>& input,
    const xt::xarray<float>& weight,
    const xt::xarray<float>& bias,
    const xt::xarray<float>& grad_output) {
    const auto out = c.out_size();
    const auto [D, H, W] = c.in_size;
    const auto [kD, kH, kW] = c.kernel;

    Conv3dReference ref;
    ref.output.assign(static_cast<size_t>(c.N) * out[0] * out[1] * out[2] * c.C_out, 0.F);
    ref.grad_input.assign(input.size(), 0.F);
    ref.grad_weight.assign(weight.size(), 0.F);
    ref.grad_bias.assign(c.C_out, 0.F);

    auto input_index = [&](uint32_t n, uint32_t d, uint32_t h, uint32_t w, uint32_t ci) {
        return (((static_cast<size_t>(n) * D + d) * H + h) * W + w) * c.C_in + ci;
    };
    auto output_index = [&](uint32_t n, uint32_t d, uint32_t h, uint32_t w, uint32_t co) {
        return (((static_cast<size_t>(n) * out[0] + d) * out[1] + h) * out[2] + w) * c.C_out + co;
    };
    auto weight_index = [&](uint32_t co, uint32_t ci, uint32_t kd, uint32_t kh, uint32_t kw) {
        return (((static_cast<size_t>(co) * c.C_in + ci) * kD + kd) * kH + kh) * kW + kw;
    };

    for (uint32_t n = 0; n < c.N; ++n) {
        for (uint32_t od = 0; od < out[0]; ++od) {
            for (uint32_t oh = 0; oh < out[1]; ++oh) {
                for (uint32_t ow = 0; ow < out[2]; ++ow) {
                    for (uint32_t co = 0; co < c.C_out; ++co) {
                        const size_t o_idx = output_index(n, od, oh, ow, co);
                        const float dy = grad_output.flat(o_idx);
                        float acc = c.with_bias ? bias.flat(co) : 0.F;
                        if (c.with_bias) {
                            ref.grad_bias[co] += dy;
                        }
                        for (uint32_t kd = 0; kd < kD; ++kd) {
                            const int64_t id = static_cast<int64_t>(od) * c.stride[0] + kd * c.dilation[0] -
                                               static_cast<int64_t>(c.padding[0]);
                            if (id < 0 || id >= static_cast<int64_t>(D)) {
                                continue;
                            }
                            for (uint32_t kh = 0; kh < kH; ++kh) {
                                const int64_t ih = static_cast<int64_t>(oh) * c.stride[1] + kh * c.dilation[1] -
                                                   static_cast<int64_t>(c.padding[1]);
                                if (ih < 0 || ih >= static_cast<int64_t>(H)) {
                                    continue;
                                }
                                for (uint32_t kw = 0; kw < kW; ++kw) {
                                    const int64_t iw = static_cast<int64_t>(ow) * c.stride[2] + kw * c.dilation[2] -
                                                       static_cast<int64_t>(c.padding[2]);
                                    if (iw < 0 || iw >= static_cast<int64_t>(W)) {
                                        continue;
                                    }
                                    for (uint32_t ci = 0; ci < c.C_in; ++ci) {
                                        const size_t i_idx = input_index(
                                            n,
                                            static_cast<uint32_t>(id),
                                            static_cast<uint32_t>(ih),
                                            static_cast<uint32_t>(iw),
                                            ci);
                                        const size_t w_idx = weight_index(co, ci, kd, kh, kw);
                                        acc += input.flat(i_idx) * weight.flat(w_idx);
                                        ref.grad_input[i_idx] += dy * weight.flat(w_idx);
                                        ref.grad_weight[w_idx] += dy * input.flat(i_idx);
                                    }
                                }
                            }
                        }
                        ref.output[o_idx] = acc;
                    }
                }
            }
        }
    }
    return ref;
}

std::vector<size_t> xshape(const ttnn::Shape& shape) {
    return {shape.cbegin(), shape.cend()};
}

class Conv3dOpTest : public ::testing::Test {
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
};

void run_case(const Conv3dCase& c) {
    const auto out = c.out_size();
    const auto [D, H, W] = c.in_size;
    const auto [kD, kH, kW] = c.kernel;

    const ttnn::Shape input_shape({c.N, D, H, W, c.C_in});
    const ttnn::Shape weight_shape({c.C_out, c.C_in, kD, kH, kW});
    const ttnn::Shape bias_shape = c.bias_rank1 ? ttnn::Shape({c.C_out}) : ttnn::Shape({1U, 1U, 1U, c.C_out});
    const ttnn::Shape output_shape({c.N, out[0], out[1], out[2], c.C_out});
    auto* device = &ttml::autograd::ctx().get_device();

    auto input = ttml::autograd::create_tensor(
        ttml::core::from_xtensor(
            ttml::test_utils::make_uniform_xarray<float>(input_shape, -1.F, 1.F, /*seed=*/1), device, c.layout),
        c.input_requires_grad);
    auto weight = ttml::autograd::create_tensor(
        ttml::core::from_xtensor(
            ttml::test_utils::make_uniform_xarray<float>(weight_shape, -0.5F, 0.5F, /*seed=*/2), device, c.layout),
        c.weight_requires_grad);
    ttml::autograd::TensorPtr bias;
    if (c.with_bias) {
        // A rank-1 bias cannot be tilized on its own; the op accepts it in ROW_MAJOR.
        const auto bias_layout = c.bias_rank1 ? ttnn::Layout::ROW_MAJOR : c.layout;
        bias = ttml::autograd::create_tensor(
            ttml::core::from_xtensor(
                ttml::test_utils::make_uniform_xarray<float>(bias_shape, -1.F, 1.F, /*seed=*/3), device, bias_layout),
            c.bias_requires_grad);
    }
    auto grad_output = ttml::core::from_xtensor(
        ttml::test_utils::make_uniform_xarray<float>(output_shape, -1.F, 1.F, /*seed=*/4), device, c.layout);

    // Build the reference from the bf16-rounded values that actually live on device.
    const auto ref = compute_reference(
        c,
        ttml::core::to_xtensor(input->get_value()),
        ttml::core::to_xtensor(weight->get_value()),
        c.with_bias ? ttml::core::to_xtensor(bias->get_value()) : xt::xarray<float>{},
        ttml::core::to_xtensor(grad_output));

    const auto expect_matches = [&](const ttnn::Tensor& actual,
                                    const std::vector<float>& reference,
                                    const ttnn::Shape& shape,
                                    const char* what) {
        const auto actual_x = ttml::core::to_xtensor(actual);
        const xt::xarray<float> expected = xt::adapt(reference, xshape(shape));
        if (c.relative_l2_tolerance > 0.F) {
            EXPECT_LT(ttml::test_utils::relative_l2(actual_x, expected), c.relative_l2_tolerance) << what;
        } else {
            EXPECT_TRUE(xt::allclose(actual_x, expected, /*rtol=*/1e-2, /*atol=*/2e-2)) << what;
        }
    };

    auto result = ttml::ops::conv3d(input, weight, bias, c.stride, c.padding, c.dilation);

    ASSERT_EQ(result->get_value().logical_shape(), output_shape);
    EXPECT_EQ(result->get_value().layout(), c.layout);
    expect_matches(result->get_value(), ref.output, output_shape, "forward");

    const bool any_grad = c.input_requires_grad || c.weight_requires_grad || (c.with_bias && c.bias_requires_grad);
    if (!any_grad) {
        EXPECT_FALSE(result->get_requires_grad());
        return;
    }
    result->set_grad(grad_output);
    result->backward();

    if (c.input_requires_grad) {
        ASSERT_TRUE(input->is_grad_initialized());
        EXPECT_EQ(input->get_grad().logical_shape(), input_shape);
        EXPECT_EQ(input->get_grad().layout(), c.layout);
        expect_matches(input->get_grad(), ref.grad_input, input_shape, "grad_input");
    } else {
        EXPECT_FALSE(input->is_grad_initialized()) << "frozen input received a gradient";
    }
    if (c.weight_requires_grad) {
        ASSERT_TRUE(weight->is_grad_initialized());
        EXPECT_EQ(weight->get_grad().logical_shape(), weight_shape);
        EXPECT_EQ(weight->get_grad().layout(), c.layout);
        expect_matches(weight->get_grad(), ref.grad_weight, weight_shape, "grad_weight");
    } else {
        EXPECT_FALSE(weight->is_grad_initialized()) << "frozen weight received a gradient";
    }
    if (c.with_bias) {
        if (c.bias_requires_grad) {
            ASSERT_TRUE(bias->is_grad_initialized());
            EXPECT_EQ(bias->get_grad().logical_shape(), bias_shape);
            EXPECT_EQ(bias->get_grad().layout(), bias->get_value().layout());
            expect_matches(bias->get_grad(), ref.grad_bias, bias_shape, "grad_bias");
        } else {
            EXPECT_FALSE(bias->is_grad_initialized()) << "frozen bias received a gradient";
        }
    }
}

}  // namespace

TEST_F(Conv3dOpTest, ForwardBackwardWithBias) {
    Conv3dCase c;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, ForwardBackwardNoBias) {
    Conv3dCase c;
    c.with_bias = false;
    run_case(c);
}

TEST_F(Conv3dOpTest, BatchStrideAndPadding) {
    Conv3dCase c;
    c.N = 2;
    c.in_size = {5, 6, 7};
    c.stride = {2, 2, 2};
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, NIGHTLY_AsymmetricStrideAndPadding) {
    Conv3dCase c;
    c.in_size = {6, 7, 8};
    c.stride = {1, 2, 3};
    c.padding = {0, 1, 2};
    run_case(c);
}

TEST_F(Conv3dOpTest, Dilation) {
    Conv3dCase c;
    c.in_size = {6, 7, 8};
    c.dilation = {2, 2, 2};
    c.padding = {2, 2, 2};
    run_case(c);
}

TEST_F(Conv3dOpTest, NIGHTLY_DilationWithStrideAndPadding) {
    Conv3dCase c;
    c.N = 2;
    c.in_size = {7, 8, 9};
    c.dilation = {2, 1, 2};
    c.stride = {1, 2, 1};
    c.padding = {2, 1, 2};
    run_case(c);
}

TEST_F(Conv3dOpTest, NIGHTLY_PerAxisDilation) {
    Conv3dCase c;
    c.in_size = {6, 8, 10};
    c.dilation = {1, 2, 3};
    c.padding = {1, 2, 3};
    run_case(c);
}

// span = dil * (k - 1) = 4 on every axis; padding 5 exceeds it, so dX takes the crop path with a dilated kernel.
TEST_F(Conv3dOpTest, NIGHTLY_DilationWithPaddingBeyondSpan) {
    Conv3dCase c;
    c.in_size = {5, 6, 7};
    c.dilation = {2, 2, 2};
    c.padding = {5, 5, 5};
    run_case(c);
}

TEST_F(Conv3dOpTest, PaddingLargerThanKernelSpanPointwise) {
    Conv3dCase c;
    c.kernel = {1, 1, 1};
    c.padding = {1, 1, 1};
    run_case(c);
}

// span 2 with padding above it: the dX crop path, unreachable from the pointwise case (span 0).
TEST_F(Conv3dOpTest, PaddingLargerThanKernelSpan) {
    Conv3dCase c;
    c.padding = {3, 4, 3};
    run_case(c);
}

TEST_F(Conv3dOpTest, NIGHTLY_PaddingLargerThanKernelSpanWithStride) {
    Conv3dCase c;
    c.N = 2;
    c.stride = {2, 2, 2};
    c.padding = {3, 3, 3};
    run_case(c);
}

TEST_F(Conv3dOpTest, NIGHTLY_OutputChannelsNotTileAligned) {
    Conv3dCase c;
    c.C_out = 40;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, NIGHTLY_InputChannelsNotTileAligned) {
    Conv3dCase c;
    c.C_in = 12;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, InputAndOutputChannelsNotTileAligned) {
    Conv3dCase c;
    c.C_in = 3;
    c.C_out = 40;
    c.N = 2;
    c.stride = {2, 1, 2};
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, PointwiseKernelWithStride) {
    Conv3dCase c;
    c.kernel = {1, 1, 1};
    c.stride = {2, 2, 2};
    c.C_in = 64;
    run_case(c);
}

// Even kernel volume: C_in block 16. The 16-channel input runs unpadded; C_out 40 pads to 48 as the dX
// activation, so re-blocking runs there.
TEST_F(Conv3dOpTest, PatchEmbedKernel) {
    Conv3dCase c;
    c.C_in = 16;
    c.C_out = 40;
    c.kernel = {1, 2, 2};
    c.stride = {1, 2, 2};
    run_case(c);
}

TEST_F(Conv3dOpTest, NIGHTLY_AsymmetricOddKernel) {
    Conv3dCase c;
    c.kernel = {1, 3, 3};
    c.padding = {0, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, NIGHTLY_EvenCubicKernelWithStride) {
    Conv3dCase c;
    c.kernel = {2, 2, 2};
    c.stride = {2, 2, 2};
    run_case(c);
}

// dW and db reduce over M = 2 * 8 * 48 * 48 = 36864 output positions, the accumulation length of a video layer;
// sums that long are compared by relative L2, since elementwise bounds are unstable on them.
TEST_F(Conv3dOpTest, NIGHTLY_VideoSizedReduction) {
    Conv3dCase c;
    c.N = 2;
    c.in_size = {8, 48, 48};
    c.padding = {1, 1, 1};
    c.relative_l2_tolerance = 1e-2F;
    run_case(c);
}

TEST_F(Conv3dOpTest, TileLayoutTensors) {
    Conv3dCase c;
    c.layout = ttnn::Layout::TILE;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, OneDimensionalBias) {
    Conv3dCase c;
    c.bias_rank1 = true;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, NIGHTLY_LargerChannelCounts) {
    Conv3dCase c;
    c.C_in = 128;
    c.C_out = 128;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, TileLayoutUnalignedChannelsWithBatch) {
    Conv3dCase c;
    c.layout = ttnn::Layout::TILE;
    c.N = 2;
    c.C_in = 24;
    c.C_out = 40;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, FrozenWeightAndBias) {
    Conv3dCase c;
    c.weight_requires_grad = false;
    c.bias_requires_grad = false;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, FrozenInput) {
    Conv3dCase c;
    c.input_requires_grad = false;
    c.padding = {1, 1, 1};
    run_case(c);
}

TEST_F(Conv3dOpTest, FrozenWeightTrainableBias) {
    Conv3dCase c;
    c.input_requires_grad = false;
    c.weight_requires_grad = false;
    run_case(c);
}

TEST_F(Conv3dOpTest, NoGradientsRequested) {
    Conv3dCase c;
    c.input_requires_grad = false;
    c.weight_requires_grad = false;
    c.bias_requires_grad = false;
    run_case(c);
}

// Every training step must hit the program cache: a miss in any of the composed ttnn ops would recompile per step.
TEST_F(Conv3dOpTest, ProgramCacheStableAcrossSteps) {
    auto* device = &ttml::autograd::ctx().get_device();
    Conv3dCase c;
    c.N = 2;
    c.C_out = 40;
    c.stride = {1, 2, 2};
    c.padding = {1, 1, 1};
    const auto out = c.out_size();
    const auto [D, H, W] = c.in_size;
    const auto [kD, kH, kW] = c.kernel;
    const ttnn::Shape input_shape({c.N, D, H, W, c.C_in});
    const ttnn::Shape weight_shape({c.C_out, c.C_in, kD, kH, kW});
    const ttnn::Shape bias_shape({1U, 1U, 1U, c.C_out});
    const ttnn::Shape output_shape({c.N, out[0], out[1], out[2], c.C_out});

    auto step = [&](uint32_t seed) {
        auto input = ttml::autograd::create_tensor(
            ttml::core::from_xtensor(
                ttml::test_utils::make_uniform_xarray<float>(input_shape, -1.F, 1.F, seed), device, c.layout),
            true);
        auto weight = ttml::autograd::create_tensor(
            ttml::core::from_xtensor(
                ttml::test_utils::make_uniform_xarray<float>(weight_shape, -0.5F, 0.5F, seed + 1), device, c.layout),
            true);
        auto bias = ttml::autograd::create_tensor(
            ttml::core::from_xtensor(
                ttml::test_utils::make_uniform_xarray<float>(bias_shape, -1.F, 1.F, seed + 2), device, c.layout),
            true);
        auto result = ttml::ops::conv3d(input, weight, bias, c.stride, c.padding, c.dilation);
        result->set_grad(ttml::core::from_xtensor(
            ttml::test_utils::make_uniform_xarray<float>(output_shape, -1.F, 1.F, seed + 3), device, c.layout));
        result->backward();
        ttml::autograd::ctx().reset_graph();
    };

    step(31);
    const auto entries_after_first = device->num_program_cache_entries();
    // Guard against a vacuous pass with the cache disabled.
    EXPECT_GT(entries_after_first, 0U);
    step(41);
    EXPECT_EQ(device->num_program_cache_entries(), entries_after_first)
        << "second identical step compiled new programs";
}

TEST_F(Conv3dOpTest, GradientsAccumulateAcrossTwoUses) {
    const Conv3dCase c{.layout = ttnn::Layout::TILE};
    const auto [D, H, W] = c.in_size;
    const auto [kD, kH, kW] = c.kernel;
    const ttnn::Shape input_shape({c.N, D, H, W, c.C_in});
    const ttnn::Shape weight_shape({c.C_out, c.C_in, kD, kH, kW});

    auto* device = &ttml::autograd::ctx().get_device();

    auto input = ttml::autograd::create_tensor(
        ttml::core::from_xtensor(
            ttml::test_utils::make_uniform_xarray<float>(input_shape, -1.F, 1.F, 11), device, c.layout),
        /*requires_grad=*/true);
    auto weight = ttml::autograd::create_tensor(
        ttml::core::from_xtensor(
            ttml::test_utils::make_uniform_xarray<float>(weight_shape, -0.5F, 0.5F, 12), device, c.layout),
        /*requires_grad=*/true);

    auto single = ttml::ops::conv3d(input, weight);
    single->backward();
    const auto single_input_grad = ttml::core::to_xtensor(input->get_grad());
    const auto single_weight_grad = ttml::core::to_xtensor(weight->get_grad());

    input->set_grad(ttnn::Tensor());
    weight->set_grad(ttnn::Tensor());
    ttml::autograd::ctx().reset_graph();

    auto sum = ttml::ops::conv3d(input, weight) + ttml::ops::conv3d(input, weight);
    sum->backward();
    const auto doubled_input_grad = ttml::core::to_xtensor(input->get_grad());
    const auto doubled_weight_grad = ttml::core::to_xtensor(weight->get_grad());

    EXPECT_TRUE(xt::allclose(doubled_input_grad, 2.F * single_input_grad, /*rtol=*/1e-2, /*atol=*/2e-2))
        << "accumulated grad_input";
    EXPECT_TRUE(xt::allclose(doubled_weight_grad, 2.F * single_weight_grad, /*rtol=*/1e-2, /*atol=*/2e-2))
        << "accumulated grad_weight";
}

TEST_F(Conv3dOpTest, RejectsUnsupportedArguments) {
    const Conv3dCase c;
    const auto [D, H, W] = c.in_size;
    const auto [kD, kH, kW] = c.kernel;
    const ttnn::Shape input_shape({c.N, D, H, W, c.C_in});
    const ttnn::Shape weight_shape({c.C_out, c.C_in, kD, kH, kW});
    auto* device = &ttml::autograd::ctx().get_device();
    const auto zeros = [device](const ttnn::Shape& shape) {
        return ttml::autograd::create_tensor(
            ttml::core::from_xtensor<float>(xt::zeros<float>(xshape(shape)), device, ttnn::Layout::ROW_MAJOR));
    };
    auto input = zeros(input_shape);
    auto weight = zeros(weight_shape);

    const Dims3 unit{1, 1, 1};
    const Dims3 none{0, 0, 0};

    EXPECT_THROW(ttml::ops::conv3d(input, weight, nullptr, /*stride=*/{0, 1, 1}), std::invalid_argument);
    EXPECT_THROW(
        ttml::ops::conv3d(input, weight, nullptr, /*stride=*/unit, /*padding=*/none, /*dilation=*/{1, 0, 1}),
        std::invalid_argument);
    // dilation 3 makes the effective kernel 7 > D = 4 with no padding
    EXPECT_THROW(
        ttml::ops::conv3d(input, weight, nullptr, /*stride=*/unit, /*padding=*/none, /*dilation=*/{3, 1, 1}),
        std::invalid_argument);
    EXPECT_THROW(ttml::ops::conv3d(zeros(ttnn::Shape({D, H, W, c.C_in})), weight), std::invalid_argument);
    EXPECT_THROW(ttml::ops::conv3d(input, zeros(ttnn::Shape({c.C_out, c.C_in, kD, kH}))), std::invalid_argument);
    EXPECT_THROW(ttml::ops::conv3d(input, zeros(ttnn::Shape({c.C_out, 64U, kD, kH, kW}))), std::invalid_argument);
    // float tensors are read at bf16, so only a non-float dtype can mismatch
    {
        auto weight_u32 = ttml::autograd::create_tensor(ttml::core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
            xt::zeros<uint32_t>(xshape(weight_shape)), device, ttnn::Layout::ROW_MAJOR));
        EXPECT_THROW(ttml::ops::conv3d(input, weight_u32), std::invalid_argument);
    }
    EXPECT_THROW(ttml::ops::conv3d(input, weight, zeros(ttnn::Shape({1U, c.C_out + 1U}))), std::invalid_argument);
    EXPECT_THROW(ttml::ops::conv3d(input, weight, zeros(ttnn::Shape({c.C_out, 1U}))), std::invalid_argument);
}

// Every public entry point rejects host, deallocated and non-float tensors before they reach ttnn.
TEST_F(Conv3dOpTest, RejectsHostDeallocatedAndMistypedTensors) {
    const Conv3dCase c;
    const auto [D, H, W] = c.in_size;
    const auto [kD, kH, kW] = c.kernel;
    const ttnn::Shape input_shape({c.N, D, H, W, c.C_in});
    const ttnn::Shape weight_shape({c.C_out, c.C_in, kD, kH, kW});
    const ttnn::Shape bias_shape({1U, 1U, 1U, c.C_out});
    auto* device = &ttml::autograd::ctx().get_device();

    const auto zeros_value = [device](const ttnn::Shape& shape) {
        return ttml::core::from_xtensor<float>(xt::zeros<float>(xshape(shape)), device, ttnn::Layout::ROW_MAJOR);
    };
    const auto wrap = [](const ttnn::Tensor& value) { return ttml::autograd::create_tensor(value); };
    // Copies share the buffer, so each deallocated tensor is built fresh.
    const auto freed_value = [&](const ttnn::Shape& shape) {
        auto value = zeros_value(shape);
        value.deallocate();
        return value;
    };
    const auto u32_value = [&](const ttnn::Shape& shape, ttnn::Layout layout) {
        return ttml::core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
            xt::zeros<uint32_t>(xshape(shape)), device, layout);
    };

    auto input = wrap(zeros_value(input_shape));
    auto weight = wrap(zeros_value(weight_shape));
    auto bias = wrap(zeros_value(bias_shape));
    EXPECT_NO_THROW(ttml::ops::conv3d(input, weight, bias));

    EXPECT_THROW(ttml::ops::conv3d(wrap(input->get_value().cpu()), weight, bias), std::invalid_argument);
    EXPECT_THROW(ttml::ops::conv3d(input, wrap(weight->get_value().cpu()), bias), std::invalid_argument);
    EXPECT_THROW(ttml::ops::conv3d(input, weight, wrap(bias->get_value().cpu())), std::invalid_argument);

    EXPECT_THROW(ttml::ops::conv3d(wrap(freed_value(input_shape)), weight, bias), std::invalid_argument);
    EXPECT_THROW(ttml::ops::conv3d(input, wrap(freed_value(weight_shape)), bias), std::invalid_argument);
    EXPECT_THROW(ttml::ops::conv3d(input, weight, wrap(freed_value(bias_shape))), std::invalid_argument);

    EXPECT_THROW(
        ttml::ops::conv3d(input, weight, wrap(u32_value(bias_shape, ttnn::Layout::ROW_MAJOR))), std::invalid_argument);
}
