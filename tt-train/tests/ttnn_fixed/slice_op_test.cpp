// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

class SliceOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(SliceOpTest, Slice_BROKEN) {
    auto* device = &ttml::autograd::ctx().get_device();

    // Use xarray of shape {1, 1, 32, 65}
    constexpr uint32_t N = 1U, C = 1U, H = 32U, W = 65U;
    xt::xarray<float>::shape_type shape = {N, C, H, W};
    xt::xarray<float> a = xt::arange<float>(0, N * C * H * W).reshape(shape);

    // Example: slice out the last row, columns 1 to 5

    ttnn::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};
    ttnn::SmallVector<uint32_t> start_index = {0U, 0U, H - 1, 1U};
    ttnn::SmallVector<uint32_t> end_index = {N, C, H, 5U};

    // Expected output: shape {1, 1, 1, 4}, values from a(0,0,H-1,1) to a(0,0,H-1,4)
    xt::xarray<float> b = {{{{a(0, 0, H - 1, 1), a(0, 0, H - 1, 2), a(0, 0, H - 1, 3), a(0, 0, H - 1, 4)}}}};

    auto tensor_a = ttml::core::from_xtensor(a, device);
    auto tensor_b = ttnn::slice(tensor_a, start_index, end_index, step);
    auto tensor_b_xtensor = ttml::core::to_xtensor(tensor_b);

    std::cerr << "Sliced tensor:\n" << tensor_b_xtensor << "\n";
    std::cerr << "Expected tensor:\n" << b << "\n";

    EXPECT_FALSE(xt::allclose(tensor_b_xtensor, b, 1e-5F, 1e-8F));
}
