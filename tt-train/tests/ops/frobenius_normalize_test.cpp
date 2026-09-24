// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "test_utils/random_data.hpp"

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
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

TEST_P(FrobeniusNormalizeTest, MatchesCpuReference) {
    using namespace ttml;

    const auto& c = GetParam();
    constexpr float kEps = 1e-7f;

    const uint32_t seed = static_cast<uint32_t>(std::hash<std::string>{}(c.name));
    const auto data = ttml::test_utils::make_uniform_xarray<float>(c.shape, -1.0f, 1.0f, seed);
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

class FrobeniusNormalizeCacheTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().get_device().enable_program_cache();
    }
    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(FrobeniusNormalizeCacheTest, RepeatedMulticoreCacheHitsCompleteAndUseOneGlobalNorm) {
    using namespace ttml;

    constexpr std::array<uint32_t, 4> kShape = {1, 1, 64, 64};
    constexpr uint32_t kIterations = 64;
    constexpr std::array<float, 2> kEpsilons = {4.0F, 32.0F};
    constexpr std::array<std::array<float, 4>, 2> kTileValues = {
        std::array<float, 4>{1.0F, 2.0F, -3.0F, 4.0F}, std::array<float, 4>{8.0F, -1.0F, 0.5F, -4.0F}};

    auto& device = autograd::ctx().get_device();
    for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
        const uint32_t variant = iteration % static_cast<uint32_t>(kTileValues.size());
        xt::xarray<float> data = xt::zeros<float>(kShape);
        for (uint32_t row = 0; row < kShape[2]; ++row) {
            for (uint32_t col = 0; col < kShape[3]; ++col) {
                const uint32_t tile_index = (row / 32U) * 2U + col / 32U;
                data(0, 0, row, col) = kTileValues[variant][tile_index];
            }
        }

        const auto input = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &device);
        const auto expected = frobenius_normalize_ref(data, kEpsilons[variant]);

        const auto entries_before = device.num_program_cache_entries();
        const auto output = metal::frobenius_normalize(input, kEpsilons[variant]);
        const auto entries_after = device.num_program_cache_entries();

        if (iteration == 0U) {
            EXPECT_GT(entries_after, entries_before) << "first call did not populate the program cache";
        } else {
            EXPECT_EQ(entries_after, entries_before)
                << "frobenius_normalize compiled a new program on iteration " << iteration;
        }

        const auto result = core::to_xtensor(output);
        EXPECT_TRUE(xt::all(xt::isfinite(result))) << "non-finite output on iteration " << iteration;
        EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/2e-2F, /*atol=*/2e-4F))
            << "global norm mismatch on iteration " << iteration;
    }
}
