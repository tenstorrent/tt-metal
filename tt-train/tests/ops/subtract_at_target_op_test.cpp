// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <random>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

class SubtractAtTargetTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

static xt::xarray<float> make_random_4d(uint32_t N, uint32_t S, uint32_t V, float low, float high, std::mt19937& gen) {
    xt::xarray<float> t = xt::empty<float>({N, 1U, S, V});
    std::uniform_real_distribution<float> float_dist(low, high);
    for (auto& v : t) {
        v = float_dist(gen);
    }
    return t;
}

static xt::xarray<uint32_t> make_random_targets(uint32_t N, uint32_t S, uint32_t vocab_size, std::mt19937& gen) {
    std::uniform_int_distribution<uint32_t> idx_dist(0U, vocab_size - 1U);
    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({N, S});
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t s = 0; s < S; ++s) {
            target_t(n, s) = idx_dist(gen);
        }
    }
    return target_t;
}

static xt::xarray<float> subtract_at_target_reference(
    const xt::xarray<float>& input,      // [N, 1, S, local_V]
    const xt::xarray<uint32_t>& target,  // [N, S]
    uint32_t first_v,
    uint32_t last_v,
    float subtract_value = 1.0F) {
    xt::xarray<float> result = input;
    const uint32_t N = static_cast<uint32_t>(target.shape(0));
    const uint32_t S = static_cast<uint32_t>(target.shape(1));
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t s = 0; s < S; ++s) {
            const uint32_t c = target(n, s);
            if (c >= first_v && c < last_v) {
                result(n, 0U, s, c - first_v) -= subtract_value;
            }
        }
    }
    return result;
}

TEST_F(SubtractAtTargetTest, SmallFullVocab) {
    using namespace ttml;

    xt::xarray<float> input_t = {{{{0.1F, 0.2F, 0.3F, 0.4F, 0.0F, 0.0F, 0.0F, 0.0F}}}};
    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({1U, 1U});
    target_t(0, 0) = 3U;

    auto input_dev = core::from_xtensor(input_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::subtract_at_target(input_dev, target_dev, /*local_V=*/8U);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = subtract_at_target_reference(input_t, target_t, 0U, 8U);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
    EXPECT_NEAR(result_xt(0, 0, 0, 3), 0.4F - 1.0F, 1e-2F);
    EXPECT_NEAR(result_xt(0, 0, 0, 0), 0.1F, 1e-2F);
}

TEST_F(SubtractAtTargetTest, SmallPartialVocab) {
    using namespace ttml;

    xt::xarray<float> input_t = {{{{0.5F, 0.6F, 0.7F, 0.8F}, {-0.5F, -0.6F, -0.7F, -0.8F}}}};
    xt::xarray<uint32_t> target_t = xt::zeros<uint32_t>({1U, 2U});
    target_t(0, 0) = 2U;  // out of shard [4, 8) → no modification
    target_t(0, 1) = 5U;  // in shard, local col = 5-4 = 1

    auto input_dev = core::from_xtensor(input_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result =
        metal::subtract_at_target(input_dev, target_dev, /*local_V=*/4U, /*cluster_axis=*/std::nullopt, /*first_v=*/4U);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = subtract_at_target_reference(input_t, target_t, 4U, 8U);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));

    EXPECT_NEAR(result_xt(0, 0, 0, 0), 0.5F, 1e-2F);          // row 0: target out of shard, unchanged
    EXPECT_NEAR(result_xt(0, 0, 1, 1), -0.6F - 1.0F, 1e-2F);  // row 1: local col 1 modified
}

TEST_F(SubtractAtTargetTest, BatchedNonAlignedShape) {
    using namespace ttml;

    const uint32_t N = 2U, S = 91U, V = 157U;

    std::mt19937 gen(42);
    auto input_t = make_random_4d(N, S, V, -1.F, 1.F, gen);
    auto target_t = make_random_targets(N, S, V, gen);

    auto input_dev = core::from_xtensor(input_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::subtract_at_target(input_dev, target_dev, /*local_V=*/V);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = subtract_at_target_reference(input_t, target_t, 0U, V);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

TEST_F(SubtractAtTargetTest, BatchedPartialVocabShard) {
    using namespace ttml;

    const uint32_t N = 2U, S = 64U, global_V = 128U;
    const uint32_t first_v = global_V / 2;
    const uint32_t last_v = global_V;
    const uint32_t local_V = last_v - first_v;

    std::mt19937 gen(7);
    auto input_t = make_random_4d(N, S, local_V, -1.F, 1.F, gen);
    auto target_t = make_random_targets(N, S, global_V, gen);

    auto input_dev = core::from_xtensor(input_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::subtract_at_target(
        input_dev, target_dev, /*local_V=*/local_V, /*cluster_axis=*/std::nullopt, /*first_v=*/first_v);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = subtract_at_target_reference(input_t, target_t, first_v, last_v);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

TEST_F(SubtractAtTargetTest, CustomSubtractValue) {
    using namespace ttml;

    const uint32_t N = 2U, S = 64U, V = 128U;
    const float subtract_value = 1.0F / 128.0F;

    std::mt19937 gen(99);
    auto input_t = make_random_4d(N, S, V, 0.F, 0.01F, gen);
    auto target_t = make_random_targets(N, S, V, gen);

    auto input_dev = core::from_xtensor(input_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::subtract_at_target(
        input_dev,
        target_dev,
        /*local_V=*/V,
        /*cluster_axis=*/std::nullopt,
        /*first_v=*/0U,
        subtract_value);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = subtract_at_target_reference(input_t, target_t, 0U, V, subtract_value);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}

TEST_F(SubtractAtTargetTest, CustomSubtractValuePartialVocab) {
    using namespace ttml;

    const uint32_t N = 2U, S = 64U, global_V = 256U;
    const uint32_t first_v = 64U;
    const uint32_t last_v = 192U;
    const uint32_t local_V = last_v - first_v;
    const float subtract_value = 1.0F / static_cast<float>(N * S);

    std::mt19937 gen(123);
    auto input_t = make_random_4d(N, S, local_V, 0.F, 0.01F, gen);
    auto target_t = make_random_targets(N, S, global_V, gen);

    auto input_dev = core::from_xtensor(input_t, &autograd::ctx().get_device());
    auto target_dev = core::from_xtensor<uint32_t, ttnn::DataType::UINT32>(
        target_t, &autograd::ctx().get_device(), ttnn::Layout::ROW_MAJOR);

    auto result = metal::subtract_at_target(
        input_dev, target_dev, /*local_V=*/local_V, /*cluster_axis=*/std::nullopt, /*first_v=*/first_v, subtract_value);

    auto result_xt = core::to_xtensor(result);
    auto expected_xt = subtract_at_target_reference(input_t, target_t, first_v, last_v, subtract_value);

    ASSERT_EQ(result_xt.shape(), expected_xt.shape());
    EXPECT_TRUE(xt::allclose(result_xt, expected_xt, /*rtol=*/3e-2F, /*atol=*/1e-2F));
}
