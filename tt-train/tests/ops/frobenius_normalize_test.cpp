// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "metal/operations.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

class FrobeniusNormalizeTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        xt::random::seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

xt::xarray<float> frobenius_normalize_ref(const xt::xarray<float>& X, float eps) {
    auto squares = X * X;
    float sum_sq = xt::sum(squares)();
    float norm = std::sqrt(sum_sq) + eps;
    return X / norm;
}

}  // namespace

TEST_F(FrobeniusNormalizeTest, SingleTile) {
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 32, 32};
    xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    auto diff = xt::abs(result - expected);
    float max_abs = xt::amax(diff)();
    float max_rel = xt::amax(diff / (xt::abs(expected) + 1e-10f))();
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f))
        << "max_abs=" << max_abs << " max_rel=" << max_rel
        << " result[0]=" << result(0,0,0,0) << " expected[0]=" << expected(0,0,0,0);
}

TEST_F(FrobeniusNormalizeTest, SmallMatrix) {
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 64, 64};
    xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f));
}

TEST_F(FrobeniusNormalizeTest, MediumMatrix) {
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 256, 320};
    xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f));
}

TEST_F(FrobeniusNormalizeTest, ProductionSize) {
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 2048, 5632};
    xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f));
}
