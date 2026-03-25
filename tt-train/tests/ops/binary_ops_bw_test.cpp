// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"

namespace ttml::ops::tests {

class BinaryOpsBackwardTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().reset_graph();
        autograd::ctx().close_device();
    }
};

// ============================================================================
// Same-shape backward tests
// ============================================================================

TEST_F(BinaryOpsBackwardTest, AddSameShape) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{1.F, 2.F, 3.F, 4.F}}}};
    xt::xarray<float> data_b = {{{{5.F, 6.F, 7.F, 8.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), /* requires_grad */ true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), /* requires_grad */ true);

    auto out = a + b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a+b)/da = 1, d(a+b)/db = 1
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a)));
    EXPECT_TRUE(xt::allclose(b_grad, xt::ones_like(data_b)));
}

TEST_F(BinaryOpsBackwardTest, SubSameShape) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{1.F, 2.F, 3.F, 4.F}}}};
    xt::xarray<float> data_b = {{{{5.F, 6.F, 7.F, 8.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), /* requires_grad */ true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), /* requires_grad */ true);

    auto out = a - b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a-b)/da = 1, d(a-b)/db = -1
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a)));
    EXPECT_TRUE(xt::allclose(b_grad, -xt::ones_like(data_b)));
}

TEST_F(BinaryOpsBackwardTest, MulSameShape) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{1.F, 2.F, 3.F, 4.F}}}};
    xt::xarray<float> data_b = {{{{4.F, 3.F, 2.F, 1.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), /* requires_grad */ true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), /* requires_grad */ true);

    auto out = a * b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a*b)/da = b, d(a*b)/db = a
    EXPECT_TRUE(xt::allclose(a_grad, data_b));
    EXPECT_TRUE(xt::allclose(b_grad, data_a));
}

TEST_F(BinaryOpsBackwardTest, MulScalar) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{1.F, 2.F, 3.F, 4.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), /* requires_grad */ true);

    auto out = a * 3.0F;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());

    // d(a*c)/da = c
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a) * 3.0F));
}

TEST_F(BinaryOpsBackwardTest, DivSameShape) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{4.F, 6.F, 8.F, 10.F}}}};
    xt::xarray<float> data_b = {{{{2.F, 3.F, 4.F, 5.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), /* requires_grad */ true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), /* requires_grad */ true);

    auto out = a / b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    xt::xarray<float> expected_a_grad = 1.0F / data_b;
    xt::xarray<float> expected_b_grad = -data_a / (data_b * data_b);
    EXPECT_TRUE(xt::allclose(a_grad, expected_a_grad, /* rtol */ 1e-2F, /* atol */ 1e-2F));
    EXPECT_TRUE(xt::allclose(b_grad, expected_b_grad, /* rtol */ 1e-2F, /* atol */ 1e-2F));
}

// ============================================================================
// Broadcast backward tests
// ============================================================================

TEST_F(BinaryOpsBackwardTest, AddBroadcastDim0) {
    auto* device = &autograd::ctx().get_device();
    // a: [2, 1, 1, 4], b: [1, 1, 1, 4] -> b is broadcast along dim 0
    xt::xarray<float> data_a = {1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F};
    data_a.reshape({2, 1, 1, 4});
    xt::xarray<float> data_b = {{{{10.F, 20.F, 30.F, 40.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), true);

    auto out = a + b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // a_grad: same shape as a, all 1s
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a)));

    // b_grad: summed over dim 0 -> [1,1,1,4], each element = 2 (two batches)
    EXPECT_TRUE(xt::allclose(b_grad, xt::ones_like(data_b) * 2.0F));
}

TEST_F(BinaryOpsBackwardTest, SubBroadcastDim0) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F};
    data_a.reshape({2, 1, 1, 4});
    xt::xarray<float> data_b = {{{{10.F, 20.F, 30.F, 40.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), true);

    auto out = a - b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // a_grad: all 1s
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a)));

    // b_grad: summed -1s over dim 0 -> each element = -2
    EXPECT_TRUE(xt::allclose(b_grad, -xt::ones_like(data_b) * 2.0F));
}

TEST_F(BinaryOpsBackwardTest, MulBroadcastDim0) {
    auto* device = &autograd::ctx().get_device();
    // a: [2, 1, 1, 4], b: [1, 1, 1, 4]
    xt::xarray<float> data_a = {1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F};
    data_a.reshape({2, 1, 1, 4});
    xt::xarray<float> data_b = {{{{2.F, 2.F, 2.F, 2.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), true);

    auto out = a * b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a*b)/da = b (broadcast) -> each element of a_grad = 2
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a) * 2.0F));

    // d(a*b)/db = a, summed over dim 0 -> [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
    xt::xarray<float> expected_b_grad = {{{{6.F, 8.F, 10.F, 12.F}}}};
    EXPECT_TRUE(xt::allclose(b_grad, expected_b_grad));
}

TEST_F(BinaryOpsBackwardTest, MulBroadcastDim3) {
    auto* device = &autograd::ctx().get_device();
    // a: [1, 1, 1, 4], b: [1, 1, 1, 1] -> b is broadcast along dim 3
    xt::xarray<float> data_a = {{{{1.F, 2.F, 3.F, 4.F}}}};
    xt::xarray<float> data_b = {3.F};
    data_b.reshape({1, 1, 1, 1});

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), true);

    auto out = a * b;
    out->backward();

    auto a_grad = core::to_xtensor(a->get_grad());
    auto b_grad = core::to_xtensor(b->get_grad());

    // d(a*b)/da = b (broadcast) -> each element = 3
    EXPECT_TRUE(xt::allclose(a_grad, xt::ones_like(data_a) * 3.0F));

    // d(a*b)/db = a, summed over dim 3 -> [1+2+3+4] = [10]
    xt::xarray<float> expected_b_grad = {10.F};
    expected_b_grad.reshape({1, 1, 1, 1});
    EXPECT_TRUE(xt::allclose(b_grad, expected_b_grad));
}

TEST_F(BinaryOpsBackwardTest, DivBroadcastThrows) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {4.F, 6.F, 8.F, 10.F, 12.F, 14.F, 16.F, 18.F};
    data_a.reshape({2, 1, 1, 4});
    xt::xarray<float> data_b = {{{{2.F, 2.F, 2.F, 2.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device), true);
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device), true);

    auto out = a / b;
    EXPECT_THROW(out->backward(), std::runtime_error);
}

}  // namespace ttml::ops::tests
