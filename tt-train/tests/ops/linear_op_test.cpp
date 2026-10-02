// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/linear_op.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "init/tensor_initializers.hpp"

class LinearOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

ttml::autograd::TensorPtr make_tensor(const ttnn::Shape& shape) {
    auto tensor = ttml::autograd::create_tensor();
    ttml::init::uniform_init(tensor, shape, ttml::init::UniformRange{-0.1F, 0.1F});
    return tensor;
}

void expect_linear_contract_error(
    const ttnn::Shape& input_shape,
    const ttnn::Shape& weight_shape,
    const std::optional<ttnn::Shape>& bias_shape,
    const std::string& expected_message) {
    auto input = make_tensor(input_shape);
    auto weight = make_tensor(weight_shape);
    auto bias = bias_shape.has_value() ? make_tensor(*bias_shape) : nullptr;

    try {
        (void)ttml::ops::linear_op(input, weight, bias);
        FAIL() << "linear_op accepted an unsupported parameter shape";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find(expected_message), std::string::npos) << error.what();
    }
    ttml::autograd::ctx().reset_graph();
}

}  // namespace

void compare_tensors(const ttnn::Tensor& t1, const ttnn::Tensor& t2, float eps) {
    ASSERT_EQ(t1.logical_shape(), t2.logical_shape());
    auto t1_vec = ttml::core::to_vector(t1);
    auto t2_vec = ttml::core::to_vector(t2);
    ASSERT_EQ(t1_vec.size(), t2_vec.size());
    auto it = std::mismatch(
        t1_vec.begin(), t1_vec.end(), t2_vec.begin(), [eps](float a, float b) { return std::abs(a - b) <= eps; });
    if (it.first != t1_vec.end()) {
        EXPECT_NEAR(*it.first, *it.second, eps);
    }
    EXPECT_TRUE(it.first == t1_vec.end());
}

bool compare_tensors_for_broken(const ttnn::Tensor& t1, const ttnn::Tensor& t2, float eps) {
    if (t1.logical_shape() != t2.logical_shape()) {
        return false;
    }
    auto t1_vec = ttml::core::to_vector(t1);
    auto t2_vec = ttml::core::to_vector(t2);
    return std::equal(
        t1_vec.begin(), t1_vec.end(), t2_vec.begin(), [eps](float a, float b) { return std::abs(a - b) <= eps; });
}

// Disabled: non-deterministic accuracy failures — https://github.com/tenstorrent/tt-metal/issues/46121
TEST_F(LinearOpTest, DISABLED_TTNNBackwardGoodShape) {
    auto tensor = ttml::autograd::create_tensor();
    ttml::init::uniform_init(tensor, ttnn::Shape({64, 1, 256, 64}), ttml::init::UniformRange{-0.1F, 0.1F});
    tensor->set_requires_grad(true);

    auto weight = ttml::autograd::create_tensor();
    ttml::init::uniform_init(weight, ttnn::Shape({1, 1, 64, 64}), ttml::init::UniformRange{-0.1F, 0.1F});
    weight->set_requires_grad(true);

    auto bias = ttml::autograd::create_tensor();
    ttml::init::uniform_init(bias, ttnn::Shape({1, 1, 1, 64}), ttml::init::UniformRange{-0.1F, 0.1F});
    bias->set_requires_grad(true);

    auto out = ttml::autograd::create_tensor();
    ttml::init::uniform_init(out, ttnn::Shape({64, 1, 256, 64}), ttml::init::UniformRange{-0.1F, 0.1F});
    out->set_grad(out->get_value());
    out->set_requires_grad(true);

    ttml::ops::ttnn_linear_backward(tensor, weight, bias, out);
    auto ttnn_tensor_grad = tensor->get_grad();
    auto ttnn_weight_grad = weight->get_grad();
    auto ttnn_bias_grad = bias->get_grad();
    tensor->set_grad(ttnn::Tensor());
    weight->set_grad(ttnn::Tensor());
    bias->set_grad(ttnn::Tensor());

    ttml::ops::moreh_linear_backward(tensor, weight, bias, out);
    auto moreh_tensor_grad = tensor->get_grad();
    auto moreh_weight_grad = weight->get_grad();
    auto moreh_bias_grad = bias->get_grad();

    const float eps = 3125e-5F;
    compare_tensors(ttnn_tensor_grad, moreh_tensor_grad, eps);
    compare_tensors(ttnn_weight_grad, moreh_weight_grad, eps);
    compare_tensors(ttnn_bias_grad, moreh_bias_grad, eps);
}
void test_linear(uint32_t batch, uint32_t emb_dim) {
    std::cout << "dim: " << emb_dim << std::endl;
    auto tensor = ttml::autograd::create_tensor();
    ttml::init::uniform_init(tensor, ttnn::Shape({batch, 1, 1024, 768}), ttml::init::UniformRange{-0.1F, 0.1F});

    auto weight = ttml::autograd::create_tensor();
    ttml::init::uniform_init(weight, ttnn::Shape({1, 1, emb_dim, 768}), ttml::init::UniformRange{-0.1F, 0.1F});

    auto bias = ttml::autograd::create_tensor();
    ttml::init::uniform_init(bias, ttnn::Shape({1, 1, 1, emb_dim}), ttml::init::UniformRange{-0.1F, 0.1F});

    ttml::ops::linear_op(tensor, weight, bias);
}
TEST_F(LinearOpTest, TTNNLargeLinearOpWithBias) {
    uint32_t dim = 4096;
    uint32_t batch = 32;  // it works with batch = 1, please try to check from 4 to 64
    EXPECT_NO_FATAL_FAILURE(test_linear(batch, 4 * dim));
}

TEST_F(LinearOpTest, SharedRank4ParametersRemainSupported) {
    auto input = make_tensor(ttnn::Shape({2, 1, 32, 32}));
    auto weight = make_tensor(ttnn::Shape({1, 1, 64, 32}));
    auto bias = make_tensor(ttnn::Shape({1, 1, 1, 64}));

    auto output_with_bias = ttml::ops::linear_op(input, weight, bias);
    EXPECT_EQ(output_with_bias->get_value().logical_shape(), ttnn::Shape({2, 1, 32, 64}));

    auto output_without_bias = ttml::ops::linear_op(input, weight, nullptr);
    EXPECT_EQ(output_without_bias->get_value().logical_shape(), ttnn::Shape({2, 1, 32, 64}));
    ttml::autograd::ctx().reset_graph();
}

TEST_F(LinearOpTest, RejectsNonSingletonWeightPrefixes) {
    const auto shared_bias = ttnn::Shape({1, 1, 1, 64});

    expect_linear_contract_error(
        ttnn::Shape({2, 1, 32, 32}),
        ttnn::Shape({2, 1, 64, 32}),
        std::nullopt,
        "weight to have singleton leading dimensions");
    expect_linear_contract_error(
        ttnn::Shape({2, 1, 32, 32}),
        ttnn::Shape({2, 1, 64, 32}),
        shared_bias,
        "weight to have singleton leading dimensions");
    expect_linear_contract_error(
        ttnn::Shape({1, 2, 32, 32}),
        ttnn::Shape({1, 2, 64, 32}),
        shared_bias,
        "weight to have singleton leading dimensions");
}

TEST_F(LinearOpTest, RejectsNonSharedBias) {
    expect_linear_contract_error(
        ttnn::Shape({2, 1, 32, 32}),
        ttnn::Shape({1, 1, 64, 32}),
        ttnn::Shape({2, 1, 1, 64}),
        "bias to have singleton leading dimensions");
    expect_linear_contract_error(
        ttnn::Shape({2, 1, 32, 32}),
        ttnn::Shape({1, 1, 64, 32}),
        ttnn::Shape({1, 1, 32, 64}),
        "bias to have singleton leading dimensions");
}

// Currently raises SEGFAULT

// TEST_F(LinearOpTest, TTNNBackwardBadShape_BROKEN) {
//     auto* device = &ttml::autograd::ctx().get_device();
//     auto tensor = ttml::autograd::create_tensor();
//     ttml::init::uniform_init(tensor, ttnn::Shape({128, 1, 1,
//     128}), ttml::init::UniformRange{-0.1F, 0.1F});

//     auto weight = ttml::autograd::create_tensor();
//     ttml::init::uniform_init(weight, ttnn::Shape({1, 1, 256,
//     128}), ttml::init::UniformRange{-0.1F, 0.1F});

//     auto bias = ttml::autograd::create_tensor();
//     ttml::init::uniform_init(bias, ttnn::Shape({1, 1, 1, 256}),
//     ttml::init::UniformRange{-0.1F, 0.1F});

//     auto out = ttml::autograd::create_tensor();
//     ttml::init::uniform_init(out, ttnn::Shape({128, 1, 1, 256}),
//     ttml::init::UniformRange{-0.1F, 0.1F}); out->set_grad(out->get_value());

//     ttml::ops::ttnn_linear_backward(tensor, weight, bias, out);
//     auto ttnn_tensor_grad = tensor->get_grad();
//     auto ttnn_weight_grad = weight->get_grad();
//     auto ttnn_bias_grad = bias->get_grad();
//     tensor->set_grad(ttnn::Tensor());
//     weight->set_grad(ttnn::Tensor());
//     bias->set_grad(ttnn::Tensor());

//     ttml::ops::moreh_linear_backward(tensor, weight, bias, out);
//     auto moreh_tensor_grad = tensor->get_grad();
//     auto moreh_weight_grad = weight->get_grad();
//     auto moreh_bias_grad = bias->get_grad();

//     const float eps = 2e-2F;
//     bool success = compare_tensors_for_broken(ttnn_tensor_grad,
//     moreh_tensor_grad, eps) &&
//                    compare_tensors_for_broken(ttnn_weight_grad,
//                    moreh_weight_grad, eps) &&
//                    compare_tensors_for_broken(ttnn_bias_grad,
//                    moreh_bias_grad, eps);
//     EXPECT_FALSE(success);
// }
