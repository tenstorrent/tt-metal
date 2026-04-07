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
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::FLOAT32>(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    auto diff = xt::abs(result - expected);
    float max_abs = xt::amax(diff)();
    float max_rel = xt::amax(diff / (xt::abs(expected) + 1e-10f))();
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-4f, /*atol=*/1e-4f))
        << "max_abs=" << max_abs << " max_rel=" << max_rel
        << " result[0]=" << result(0,0,0,0) << " expected[0]=" << expected(0,0,0,0);
}

TEST_F(FrobeniusNormalizeTest, ThreeTiles) {
    // 3 tiles = 3 cores → 2 chain hops in column
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 32, 96};
    xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
    float eps = 1e-7f;

    // Print expected per-tile sums for comparison with DPRINT
    for (int t = 0; t < 3; ++t) {
        auto tile = xt::view(data, 0, 0, xt::all(), xt::range(t * 32, (t + 1) * 32));
        double tile_sum = xt::sum(tile * tile)();
        printf("HOST tile %d sum_sq = %.6f\n", t, (float)tile_sum);
    }
    double total = xt::sum(data * data)();
    float norm = std::sqrt((float)total) + eps;
    printf("HOST total_sum_sq = %.6f, norm = %.6f, 1/norm = %.6f\n",
           (float)total, norm, 1.0f / norm);

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::FLOAT32>(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    auto d1 = xt::abs(result - expected);
    float max_abs = xt::amax(d1)();
    float max_rel = xt::amax(d1 / (xt::abs(expected) + 1e-10f))();
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-4f, /*atol=*/1e-4f))
        << "max_abs=" << max_abs << " max_rel=" << max_rel;
}

TEST_F(FrobeniusNormalizeTest, FourTilesAllOnes) {
    // 4 tiles of all-ones → norm = sqrt(4096) = 64, result = 1/64 = 0.015625
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 64, 64};
    xt::xarray<float> data = xt::ones<float>(shape);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::FLOAT32>(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    float r0 = result(0,0,0,0);
    float e0 = expected(0,0,0,0);
    auto d1 = xt::abs(result - expected);
    float max_abs = xt::amax(d1)();
    float max_rel = xt::amax(d1 / (xt::abs(expected) + 1e-10f))();
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-4f, /*atol=*/1e-4f))
        << "max_abs=" << max_abs << " max_rel=" << max_rel
        << " result[0]=" << r0 << " expected=" << e0;
}

TEST_F(FrobeniusNormalizeTest, TwoTiles) {
    // 2 tiles = 2 cores → single chain hop. Isolates chain add error.
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 32, 64};
    xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::FLOAT32>(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    auto d1 = xt::abs(result - expected);
    float max_abs = xt::amax(d1)();
    float max_rel = xt::amax(d1 / (xt::abs(expected) + 1e-10f))();
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-4f, /*atol=*/1e-4f))
        << "max_abs=" << max_abs << " max_rel=" << max_rel;
}

TEST_F(FrobeniusNormalizeTest, SmallMatrix) {
    // 4 tiles = 4 cores → 3 chain hops in column
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 64, 64};
    xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::FLOAT32>(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    auto d1 = xt::abs(result - expected);
    float max_abs = xt::amax(d1)();
    float max_rel = xt::amax(d1 / (xt::abs(expected) + 1e-10f))();
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-4f, /*atol=*/1e-4f))
        << "max_abs=" << max_abs << " max_rel=" << max_rel;
}

TEST_F(FrobeniusNormalizeTest, MediumMatrix) {
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 256, 320};
    xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::FLOAT32>(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-4f, /*atol=*/1e-4f));
}

TEST_F(FrobeniusNormalizeTest, ProductionSize) {
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 2048, 5632};
    xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::FLOAT32>(data, &autograd::ctx().get_device());

    // Warmup
    auto warmup = metal::frobenius_normalize(input_tensor, eps);
    (void)core::to_xtensor(warmup);

    // Timed runs (with and without readback)
    constexpr int N = 10;

    // Timed: dispatch only (no readback)
    auto t0 = std::chrono::high_resolution_clock::now();
    ttnn::Tensor last;
    for (int i = 0; i < N; ++i) {
        last = metal::frobenius_normalize(input_tensor, eps);
    }
    // Sync by reading back the last result
    (void)core::to_xtensor(last);
    auto t1 = std::chrono::high_resolution_clock::now();
    double avg_dispatch_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)N;
    printf("ProductionSize (2048x5632): %.0f us avg over %d dispatches (incl 1 readback)\n", avg_dispatch_us, N);

    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-4f, /*atol=*/1e-4f));
}
