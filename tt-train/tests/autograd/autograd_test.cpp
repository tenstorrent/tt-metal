// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

class AutogradTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

protected:
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

TEST_F(AutogradTest, TestSum) {
    using namespace ttml::ops;
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> test_data1 = {1.F, 2.F, 3.F, 4.F};
    std::vector<float> test_data2 = {4.F, 3.F, 2.F, 1.F};
    auto shape = ttnn::Shape({1, 1, 1, 4});
    auto tensor1 = ttml::core::from_vector(test_data1, shape, device);
    auto tensor2 = ttml::core::from_vector(test_data2, shape, device);

    auto t1 = ttml::autograd::create_tensor(tensor1, /* requires_grad */ true);
    auto t2 = ttml::autograd::create_tensor(tensor2, /* requires_grad */ true);

    auto res = t1 + t2;
    res->backward();
    auto res_back = ttml::core::to_vector(res->get_grad());
    auto t1_back = ttml::core::to_vector(t1->get_grad());
    auto t2_back = ttml::core::to_vector(t2->get_grad());

    for (float it : res_back) {
        EXPECT_EQ(it, 1.0F);
    }
    for (float it : t1_back) {
        EXPECT_EQ(it, 1.0F);
    }
    for (float it : t2_back) {
        EXPECT_EQ(it, 1.0F);
    }
}

TEST_F(AutogradTest, TestMul) {
    using namespace ttml::ops;
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> test_data1 = {1.F, 2.F, 3.F, 4.F};
    std::vector<float> test_data2 = {4.F, 3.F, 2.F, 1.F};
    auto shape = ttnn::Shape({1, 1, 1, 4});
    auto tensor1 = ttml::core::from_vector(test_data1, shape, device);
    auto tensor2 = ttml::core::from_vector(test_data2, shape, device);

    auto t1 = ttml::autograd::create_tensor(tensor1, /* requires_grad */ true);
    auto t2 = ttml::autograd::create_tensor(tensor2, /* requires_grad */ true);

    auto res = t1 * t2;
    res->backward();
    auto res_back = ttml::core::to_vector(res->get_grad());
    auto t1_back = ttml::core::to_vector(t1->get_grad());
    auto t2_back = ttml::core::to_vector(t2->get_grad());

    for (float it : res_back) {
        EXPECT_EQ(it, 1.0F);
    }
    EXPECT_EQ(t2_back, test_data1);
    EXPECT_EQ(t1_back, test_data2);
}

// Regression test for #57751: freezing a parameter must drop its previously accumulated
// gradient. Otherwise add_grad() keeps skipping (it no-ops while !requires_grad) and the stale
// gradient from before the freeze survives is_grad_initialized() checks in every optimizer's
// zero_grad()/step(), so the parameter keeps getting updated after being frozen.
TEST_F(AutogradTest, SetRequiresGradFalseClearsGrad) {
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> test_data = {1.F, 2.F, 3.F, 4.F};
    auto shape = ttnn::Shape({1, 1, 1, 4});
    auto tensor = ttml::core::from_vector(test_data, shape, device);
    auto t = ttml::autograd::create_tensor(tensor, /* requires_grad */ true);

    t->set_grad(ttml::core::from_vector(test_data, shape, device));
    EXPECT_TRUE(t->is_grad_initialized());

    t->set_requires_grad(false);
    EXPECT_FALSE(t->is_grad_initialized());

    // Freezing does not create a gradient out of nothing.
    t->set_requires_grad(true);
    EXPECT_FALSE(t->is_grad_initialized());
}

TEST_F(AutogradTest, BroadCastBatchTest) {
    using namespace ttml::ops;
    auto* device = &ttml::autograd::ctx().get_device();
    std::vector<float> test_data1 = {1.F, 2.F, 3.F, 4.F};
    auto shape = ttnn::Shape({1, 1, 1, 4});
    auto tensor1 = ttml::core::from_vector(test_data1, shape, device);
    auto t1 = ttml::autograd::create_tensor(tensor1, /* requires_grad */ true);
    uint32_t new_batch = 4;
    auto res = ttml::ops::broadcast_batch(t1, new_batch);
    res->backward();
    auto t1_back = ttml::core::to_vector(t1->get_grad());
    auto batch_shape = ttnn::Shape({4, 1, 1, 4});
    auto new_shape = res->get_value().logical_shape();
    auto back_shape = t1->get_grad().logical_shape();

    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(new_shape[i], batch_shape[i]);
    }
    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(back_shape[i], shape[i]);
    }
    for (size_t i = 0; i < 4; i++) {
        EXPECT_EQ(t1_back[i], new_batch);
    }
}

namespace {

const ttnn::Shape kGradShape({1, 1, 32, 32});

// Random gradient values in [-1, 1) as bf16 on the device, with `first` as element 0.
ttnn::Tensor bf16_gradient(uint32_t seed, float first) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> values(kGradShape.volume());
    for (auto& value : values) {
        value = dist(gen);
    }
    values[0] = first;
    return ttml::core::from_vector(values, kGradShape, &ttml::autograd::ctx().get_device());
}

// The values a gradient holds, as floats (bf16 and fp32 values are exact in float).
std::vector<float> values_of(const ttnn::Tensor& tensor) {
    return ttml::core::to_vector(tensor);
}

// IEEE fp32 sums of the gradients' values, added one at a time: what an exact fp32 accumulation gives.
std::vector<float> fp32_sums(const std::vector<ttnn::Tensor>& gradients) {
    auto sums = values_of(gradients.front());
    for (size_t i = 1; i < gradients.size(); ++i) {
        const auto values = values_of(gradients[i]);
        for (size_t j = 0; j < sums.size(); ++j) {
            sums[j] += values[j];
        }
    }
    return sums;
}

ttml::autograd::TensorPtr trainable_tensor() {
    return ttml::autograd::create_tensor(bf16_gradient(1U, 0.5F), /* requires_grad */ true);
}

}  // namespace

// With an fp32 accumulation dtype, the gradient turns fp32 at its second add and holds the IEEE fp32 sum: 1 + 2^-9 is
// below half a bf16 step at 1, so a bf16 sum would stay 1.
TEST_F(AutogradTest, AccumulatesGradientsInFp32FromTheSecondAdd) {
    auto tensor = trainable_tensor();
    tensor->set_grad_accumulation_dtype(ttnn::DataType::FLOAT32);
    const std::vector<ttnn::Tensor> gradients = {
        bf16_gradient(2U, 1.0F), bf16_gradient(3U, std::ldexp(1.0F, -9)), bf16_gradient(4U, 0.25F)};

    tensor->add_grad(gradients[0]);
    tensor->add_grad(gradients[1]);
    ASSERT_EQ(tensor->get_grad().dtype(), ttnn::DataType::FLOAT32);
    const auto two = values_of(tensor->get_grad());
    EXPECT_EQ(two, fp32_sums({gradients[0], gradients[1]}));
    EXPECT_EQ(two[0], 1.0F + std::ldexp(1.0F, -9));

    tensor->add_grad(gradients[2]);
    EXPECT_EQ(tensor->get_grad().dtype(), ttnn::DataType::FLOAT32);
    EXPECT_EQ(values_of(tensor->get_grad()), fp32_sums(gradients));
}

// A gradient that is never added to stays the tensor it came as: no cast, no copy.
TEST_F(AutogradTest, KeepsTheFirstGradientAsItCame) {
    auto tensor = trainable_tensor();
    tensor->set_grad_accumulation_dtype(ttnn::DataType::FLOAT32);
    const auto gradient = bf16_gradient(2U, 1.0F);

    tensor->add_grad(gradient);

    EXPECT_EQ(tensor->get_grad().dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_EQ(tensor->get_grad().buffer()->address(), gradient.buffer()->address());
}

// Without an accumulation dtype, bf16 gradients add in bf16 as before: the same bits as one ttnn::add.
TEST_F(AutogradTest, AddsBf16GradientsInBf16WithoutAnAccumulationDtype) {
    auto tensor = trainable_tensor();
    EXPECT_EQ(tensor->get_grad_accumulation_dtype(), std::nullopt);
    const auto first = bf16_gradient(2U, 1.0F);
    const auto second = bf16_gradient(3U, std::ldexp(1.0F, -9));

    tensor->add_grad(first);
    tensor->add_grad(second);

    ASSERT_EQ(tensor->get_grad().dtype(), ttnn::DataType::BFLOAT16);
    EXPECT_EQ(values_of(tensor->get_grad()), values_of(ttnn::add(first, second)));
    EXPECT_EQ(values_of(tensor->get_grad())[0], 1.0F);
}

// A gradient that is already fp32 (an FSDP shard summed in fp32, say) takes a bf16 one exactly with no accumulation
// dtype set too. Added as they are, the fp32 operand would lose its low 13 bits.
TEST_F(AutogradTest, AddsBf16GradientsIntoAnFp32GradientExactly) {
    auto tensor = trainable_tensor();
    std::vector<float> fp32_values(kGradShape.volume());
    std::mt19937 gen(5U);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& value : fp32_values) {
        value = dist(gen);
    }
    fp32_values[0] = 1.0F + std::ldexp(1.0F, -20);
    const auto accumulated = ttml::core::from_vector<float, ttnn::DataType::FLOAT32>(
        fp32_values, kGradShape, &ttml::autograd::ctx().get_device());
    const auto gradient = bf16_gradient(2U, 1.0F);

    tensor->set_grad(accumulated);
    tensor->add_grad(gradient);

    ASSERT_EQ(tensor->get_grad().dtype(), ttnn::DataType::FLOAT32);
    EXPECT_EQ(values_of(tensor->get_grad()), fp32_sums({accumulated, gradient}));
    EXPECT_EQ(values_of(tensor->get_grad())[0], 2.0F + std::ldexp(1.0F, -20));
}

TEST_F(AutogradTest, GradAccumulationDtypeIsFp32OrUnset) {
    auto tensor = trainable_tensor();
    EXPECT_ANY_THROW(tensor->set_grad_accumulation_dtype(ttnn::DataType::BFLOAT16));
    tensor->set_grad_accumulation_dtype(ttnn::DataType::FLOAT32);
    EXPECT_EQ(tensor->get_grad_accumulation_dtype(), ttnn::DataType::FLOAT32);
    tensor->set_grad_accumulation_dtype(std::nullopt);
    EXPECT_EQ(tensor->get_grad_accumulation_dtype(), std::nullopt);
}
