// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
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

TEST_F(AutogradTest, RetainedGraphDoesNotReuseIntermediateGradients) {
    using namespace ttml::ops;
    auto* device = &ttml::autograd::ctx().get_device();
    auto shape = ttnn::Shape({1, 1, 1, 4});
    auto value = ttml::core::from_vector(std::vector<float>{1.F, 2.F, 3.F, 4.F}, shape, device);
    auto constant = ttml::core::from_vector(std::vector<float>(4, 1.F), shape, device);

    auto input = ttml::autograd::create_tensor(value, /* requires_grad */ true);
    auto constant_tensor = ttml::autograd::create_tensor(constant);
    auto intermediate = input + constant_tensor;
    auto output = intermediate + constant_tensor;

    output->backward(/* retain_graph */ true);
    output->backward(/* retain_graph */ true);

    EXPECT_EQ(ttml::core::to_vector(output->get_grad()), std::vector<float>(4, 1.F));
    EXPECT_EQ(ttml::core::to_vector(intermediate->get_grad()), std::vector<float>(4, 1.F));
    EXPECT_EQ(ttml::core::to_vector(input->get_grad()), std::vector<float>(4, 2.F));
}

TEST_F(AutogradTest, RetainedMultiOutputGraphClearsUnreachableSiblingGradient) {
    using namespace ttml::ops;
    auto* device = &ttml::autograd::ctx().get_device();
    auto shape = ttnn::Shape({1, 1, 1, 4});
    auto value = ttml::core::from_vector(std::vector<float>{1.F, 2.F, 3.F, 4.F}, shape, device);

    auto input = ttml::autograd::create_tensor(value, /* requires_grad */ true);
    auto out1 = ttml::autograd::create_tensor(value);
    auto out2 = ttml::autograd::create_tensor(value);
    auto out3 = ttml::autograd::create_tensor(value);

    ttml::autograd::GradFunction grad = [input, outputs = std::array{out1, out2, out3}]() {
        for (const auto& output : outputs) {
            if (output->is_grad_initialized()) {
                input->add_grad(output->get_grad());
            }
        }
    };

    auto primary = ttml::autograd::add_backward_node_for_outputs(std::move(grad), {out1, out2, out3}, input);
    out1->set_node(primary);
    out2->set_node(ttml::autograd::add_backward_node_always([]() {}, out2, input, out1));
    out3->set_node(ttml::autograd::add_backward_node_always([]() {}, out3, input, out1));

    // out2's traversal reaches the primary closure but not out3's dummy node.
    // The primary output group must still clear out3's stale root gradient.
    out3->backward(/* retain_graph */ true);
    out2->backward(/* retain_graph */ true);

    EXPECT_FALSE(out3->is_grad_initialized());
    EXPECT_EQ(ttml::core::to_vector(input->get_grad()), std::vector<float>(4, 2.F));
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
