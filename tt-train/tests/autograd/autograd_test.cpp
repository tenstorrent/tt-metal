// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
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
