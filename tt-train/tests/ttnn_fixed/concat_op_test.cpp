// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

class ConcatOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(ConcatOpTest, TestConcatLastDim) {
    auto* device = &ttml::autograd::ctx().get_device();
    auto N = 1;
    auto C = 1;
    auto H = 12;
    auto W = 50;
    auto prod = N * C * H * W;
    xt::xarray<float> xtensor_a = xt::arange<float>(0.F, prod).reshape({N, C, H, W});
    xt::xarray<float> xtensor_b = xt::arange<float>(prod, 2 * prod).reshape({N, C, H, W});

    xt::xarray<float> expected = xt::concatenate(xt::xtuple(xtensor_a, xtensor_b), 3);

    auto tensor_a = ttml::core::from_xtensor(xtensor_a, device);
    auto tensor_b = ttml::core::from_xtensor(xtensor_b, device);

    auto ttnn_concat = ttnn::concat(std::vector<ttnn::Tensor>{tensor_a, tensor_b}, 3);
    auto ttnn_concat_xtensor = ttml::core::to_xtensor(ttnn_concat);
    EXPECT_TRUE(xt::allclose(ttnn_concat_xtensor, expected, 7e-3F, 1e-6F));
}
