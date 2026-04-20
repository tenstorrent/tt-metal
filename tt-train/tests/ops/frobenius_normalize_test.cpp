// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

namespace {

struct FrobeniusCase {
    std::array<uint32_t, 4> shape;
    std::string name;
};

xt::xarray<float> frobenius_normalize_ref(const xt::xarray<float>& X, float eps) {
    const auto squares = X * X;
    const float sum_sq = xt::sum(squares)();
    const float norm = std::sqrt(sum_sq) + eps;
    return X / norm;
}

}  // namespace

class FrobeniusNormalizeTest : public ::testing::TestWithParam<FrobeniusCase> {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        xt::random::seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_P(FrobeniusNormalizeTest, MatchesCpuReference) {
    using namespace ttml;

    const auto& c = GetParam();
    constexpr float kEps = 1e-7f;

    const xt::xarray<float> data = xt::random::randn<float>(c.shape, 0.0f, 1.0f);
    const auto input_tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &autograd::ctx().get_device());

    const auto bf16_data = core::to_xtensor(input_tensor);
    const auto expected = frobenius_normalize_ref(bf16_data, kEps);

    const auto result_tensor = metal::frobenius_normalize(input_tensor, kEps);
    const auto result = core::to_xtensor(result_tensor);

    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f));
}

static std::string CaseName(const ::testing::TestParamInfo<FrobeniusCase>& info) {
    return info.param.name;
}

static const FrobeniusCase kCases[] = {
    {{1, 1, 32, 32}, "SingleTile"},
    {{1, 1, 32, 64}, "TwoTiles"},
    {{1, 1, 32, 96}, "ThreeTiles"},
    {{1, 1, 64, 64}, "SmallMatrix"},
    {{1, 1, 256, 320}, "MediumMatrix"},
    {{1, 1, 2048, 5632}, "ProductionSize"},
};

INSTANTIATE_TEST_SUITE_P(All, FrobeniusNormalizeTest, ::testing::ValuesIn(kCases), CaseName);
