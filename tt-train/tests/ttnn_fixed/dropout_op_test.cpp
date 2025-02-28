// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/device.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

class DropoutTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(DropoutTest, TestSeed) {
    uint32_t dropout_seed1 = 42;
    uint32_t dropout_seed2 = 32;
    float scale = 2.0F;
    float prob = 0.5F;
    xt::random::seed(42);
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    auto shapes = {std::vector<int>{64, 1, 256, 384}, std::vector<int>{1, 1, 32, 32}};
    for (auto& shape : shapes) {
        fmt::println("Testing shape: {}", shape);
        xt::xarray<float> xtensor_a = xt::random::rand(shape, -0.5, 0.5);

        auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);
        auto num_cache_before = device->num_program_cache_entries();
        auto result01 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed1);
        auto result02 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed2);
        auto result11 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed1);
        auto result12 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed2);
        auto num_cache_after = device->num_program_cache_entries();

        auto result01_vec = ttml::core::to_xtensor(result01);
        auto result02_vec = ttml::core::to_xtensor(result02);
        auto result11_vec = ttml::core::to_xtensor(result11);
        auto result12_vec = ttml::core::to_xtensor(result12);

        EXPECT_TRUE(xt::allclose(result01_vec, result11_vec, /*rtol=*/1e-4, /*atol=*/1e-3));
        EXPECT_TRUE(xt::allclose(result02_vec, result12_vec, /*rtol=*/1e-4, /*atol=*/1e-3));
        EXPECT_FALSE(xt::allclose(result01_vec, result02_vec, /*rtol=*/1e-4, /*atol=*/1e-3));
        EXPECT_EQ(num_cache_before, num_cache_after - 1);
    }
}

TEST_F(DropoutTest, TestProb) {
    uint32_t dropout_seed = 42;
    float scale = 1.0F;
    float prob = 0.2F;
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    xt::xarray<float> xtensor_a = xt::ones<float>({64, 1, 256, 384});
    std::vector<float> ratios;
    ratios.reserve(100);
    auto xtensor_a_tensor = ttml::core::from_xtensor(xtensor_a, device);
    for (int i = 0; i < 100; i++) {
        auto result01 = ttnn::experimental::dropout(xtensor_a_tensor, prob, scale, dropout_seed);
        auto result01_vec = ttml::core::to_xtensor(result01);
        float ratio = xt::sum(result01_vec)() / xt::sum(xtensor_a)();
        ratios.push_back(ratio);
    }
    auto xt_ratios = xt::adapt(ratios);
    auto mean_ratio = xt::mean(xt_ratios)();
    auto std_ratio = xt::stddev(xt_ratios)();
    EXPECT_NEAR(mean_ratio, 1.0F - prob, 0.05);
    EXPECT_NEAR(std_ratio, 0.05, 0.05);
}

namespace {
xt::xarray<float> golden_dropout(
    const xt::xarray<float>& input, float p = 0.5f, bool scale = true, uint64_t seed = 42ULL) {
    // 1) Create a random engine seeded for reproducibility
    std::mt19937_64 rng(seed);

    auto rand_vals = xt::random::rand<float>(input.shape(), 0.0f, 1.0f, rng);
    auto mask = xt::cast<float>(rand_vals >= p);

    float scale_factor = (scale && (1.0f - p) > 1e-7f) ? (1.0f / (1.0f - p)) : 1.0f;
    auto output = input * mask * scale_factor;

    return output;
}
}  // namespace

TEST_F(DropoutTest, TestKeepRatioApproximatelyNormal) {
    GTEST_SKIP() << "Currently random number generator using in the WH is not perfect. This Test show that "
                    "distribution is not normal.";
    // -------------------------------------------------------------------
    // 1) Configuration
    // -------------------------------------------------------------------
    uint32_t dropout_seed = 42;
    float dropout_prob = 0.2F;
    float scale = 1.0F;
    int num_runs = 200;

    xt::xarray<float> xtensor_a = xt::ones<float>({1, 1, 64, 64});

    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    auto input_tensor = ttml::core::from_xtensor(xtensor_a, device);

    xt::xarray<int> keep_count = xt::zeros<int>(xtensor_a.shape());

    for (int i = 0; i < num_runs; ++i) {
        auto dropped_tensor = ttnn::experimental::dropout(input_tensor, dropout_prob, scale, dropout_seed + i);
        auto dropped_x = ttml::core::to_xtensor(dropped_tensor);

        /*
        Commented out golden dropout. It passes the test, but it's not needed.
        It's here for reference to test my test.
        */
        // auto dropped_x = golden_dropout(xtensor_a, dropout_prob, false, dropout_seed + i);

        // 1 where kept, 0 where dropped
        auto mask_kept = xt::not_equal(dropped_x, 0.0F);
        keep_count += xt::cast<int>(mask_kept);
    }

    xt::xarray<float> keep_ratio = xt::cast<float>(keep_count) / float(num_runs);

    // -------------------------------------------------------------------
    // 4) Basic checks (mean and std dev)
    // -------------------------------------------------------------------
    float mean_keep_ratio = xt::mean(keep_ratio)();
    float std_keep_ratio = xt::stddev(keep_ratio)();

    // Check that mean ~ (1 - dropout_prob)
    EXPECT_NEAR(mean_keep_ratio, 1.0F - dropout_prob, 0.05F);

    // Check standard deviation across elements is not too large. var = p(1-p)/N
    float expected_variance = dropout_prob * (1.0F - dropout_prob) / num_runs;
    EXPECT_NEAR(std_keep_ratio, std::sqrt(expected_variance), 0.01F);

    // -------------------------------------------------------------------
    // 5) Approximate normality checks via skewness & kurtosis
    // -------------------------------------------------------------------
    // Flatten keep_ratio so we can compute distribution stats over all elements
    auto flattened = xt::ravel(keep_ratio);

    // The usual formulas:
    //     skewness = E[(X - mu)^3] / sigma^3
    //     kurtosis = E[(X - mu)^4] / sigma^4 (the "Fisher" definition)
    //     excess_kurtosis = kurtosis - 3
    //
    // For a perfect normal distribution:
    //     skewness = 0
    //     excess kurtosis = 0

    double mu = xt::mean(flattened)();
    double var = xt::variance(flattened)();
    double sigma = std::sqrt(var);
    double skew = 0.0;
    double kurt = 0.0;

    // Compute skew & kurtosis in a single pass
    auto size = flattened.size();
    for (size_t i = 0; i < size; ++i) {
        double x = flattened.data()[i];
        double dx = (x - mu);

        skew += std::pow(dx, 3);
        kurt += std::pow(dx, 4);
    }

    // Divide by N, then by sigma^3 or sigma^4
    skew = (skew / size) / std::pow(sigma, 3);

    // "Raw" kurtosis
    kurt = (kurt / size) / (var * var);
    // For a normal distribution, raw kurtosis = 3. Let's shift to "excess" kurtosis:
    double excess_kurt = kurt - 3.0;

    // We'll just check approximate closeness to 0 for both
    // The tolerances below are somewhat arbitrary and might need tuning.
    EXPECT_NEAR(skew, 0.0, 0.2) << "Skewness is too far from 0 for a normal-like distribution.";
    EXPECT_NEAR(excess_kurt, 0.0, 0.5) << "Excess kurtosis is too far from 0 for a normal-like distribution.";
}
