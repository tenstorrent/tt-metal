// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"

namespace ttml::ops::tests {

class BinaryOpsForwardTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().reset_graph();
        autograd::ctx().close_device();
    }
};

TEST_F(BinaryOpsForwardTest, AddSameShape) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{1.F, 2.F, 3.F, 4.F}}}};
    xt::xarray<float> data_b = {{{{5.F, 6.F, 7.F, 8.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device));
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device));

    auto out = a + b;
    auto result = core::to_xtensor(out->get_value());

    xt::xarray<float> expected = {{{{6.F, 8.F, 10.F, 12.F}}}};
    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST_F(BinaryOpsForwardTest, AddBroadcast) {
    auto* device = &autograd::ctx().get_device();
    // a: [2, 1, 1, 4], b: [1, 1, 1, 4] -> broadcast b along dim 0
    xt::xarray<float> data_a = {1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F};
    data_a.reshape({2, 1, 1, 4});
    xt::xarray<float> data_b = {{{{10.F, 20.F, 30.F, 40.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device));
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device));

    auto out = a + b;
    auto result = core::to_xtensor(out->get_value());

    xt::xarray<float> expected = {11.F, 22.F, 33.F, 44.F, 15.F, 26.F, 37.F, 48.F};
    expected.reshape({2, 1, 1, 4});
    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST_F(BinaryOpsForwardTest, SubSameShape) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{5.F, 6.F, 7.F, 8.F}}}};
    xt::xarray<float> data_b = {{{{1.F, 2.F, 3.F, 4.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device));
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device));

    auto out = a - b;
    auto result = core::to_xtensor(out->get_value());

    xt::xarray<float> expected = {{{{4.F, 4.F, 4.F, 4.F}}}};
    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST_F(BinaryOpsForwardTest, SubBroadcast) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {10.F, 20.F, 30.F, 40.F, 50.F, 60.F, 70.F, 80.F};
    data_a.reshape({2, 1, 1, 4});
    xt::xarray<float> data_b = {{{{1.F, 2.F, 3.F, 4.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device));
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device));

    auto out = a - b;
    auto result = core::to_xtensor(out->get_value());

    xt::xarray<float> expected = {9.F, 18.F, 27.F, 36.F, 49.F, 58.F, 67.F, 76.F};
    expected.reshape({2, 1, 1, 4});
    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST_F(BinaryOpsForwardTest, MulSameShape) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{1.F, 2.F, 3.F, 4.F}}}};
    xt::xarray<float> data_b = {{{{4.F, 3.F, 2.F, 1.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device));
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device));

    auto out = a * b;
    auto result = core::to_xtensor(out->get_value());

    xt::xarray<float> expected = {{{{4.F, 6.F, 6.F, 4.F}}}};
    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST_F(BinaryOpsForwardTest, MulBroadcast) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F};
    data_a.reshape({2, 1, 1, 4});
    xt::xarray<float> data_b = {{{{2.F, 3.F, 4.F, 5.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device));
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device));

    auto out = a * b;
    auto result = core::to_xtensor(out->get_value());

    xt::xarray<float> expected = {2.F, 6.F, 12.F, 20.F, 10.F, 18.F, 28.F, 40.F};
    expected.reshape({2, 1, 1, 4});
    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST_F(BinaryOpsForwardTest, MulScalar) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{1.F, 2.F, 3.F, 4.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device));

    auto out = a * 3.0F;
    auto result = core::to_xtensor(out->get_value());

    xt::xarray<float> expected = {{{{3.F, 6.F, 9.F, 12.F}}}};
    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST_F(BinaryOpsForwardTest, DivSameShape) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {{{{4.F, 6.F, 8.F, 10.F}}}};
    xt::xarray<float> data_b = {{{{2.F, 3.F, 4.F, 5.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device));
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device));

    auto out = a / b;
    auto result = core::to_xtensor(out->get_value());

    xt::xarray<float> expected = {{{{2.F, 2.F, 2.F, 2.F}}}};
    EXPECT_TRUE(xt::allclose(result, expected));
}

TEST_F(BinaryOpsForwardTest, DivBroadcast) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data_a = {4.F, 6.F, 8.F, 10.F, 12.F, 14.F, 16.F, 18.F};
    data_a.reshape({2, 1, 1, 4});
    xt::xarray<float> data_b = {{{{2.F, 2.F, 2.F, 2.F}}}};

    auto a = autograd::create_tensor(core::from_xtensor(data_a, device));
    auto b = autograd::create_tensor(core::from_xtensor(data_b, device));

    auto out = a / b;
    auto result = core::to_xtensor(out->get_value());

    xt::xarray<float> expected = {2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F, 9.F};
    expected.reshape({2, 1, 1, 4});
    EXPECT_TRUE(xt::allclose(result, expected));
}

}  // namespace ttml::ops::tests
