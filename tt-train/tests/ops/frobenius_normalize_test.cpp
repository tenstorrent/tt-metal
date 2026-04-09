// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

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

    auto input_tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &autograd::ctx().get_device());

    // Sum-of-squares comparison: composite vs reference
    auto bf16_data = core::to_xtensor(input_tensor);
    double ref_sum_sq = xt::sum(bf16_data * bf16_data)();
    auto comp_sum_sq_tensor = ttnn::sum(ttnn::square(input_tensor), ttsl::SmallVector<int>{-2, -1}, true);
    float comp_sum_sq = core::to_xtensor(comp_sum_sq_tensor)(0, 0, 0, 0);
    printf(
        "SingleTile sum_sq: ref=%.6f  composite=%.6f (rel err=%.2e)\n",
        (float)ref_sum_sq,
        comp_sum_sq,
        std::abs((double)comp_sum_sq - ref_sum_sq) / ref_sum_sq);
    printf("  (fused sum_sq from DPRINT: check FUSED_SUM_SQ=<uint32>, reinterpret as float)\n");

    auto expected = frobenius_normalize_ref(bf16_data, eps);
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    auto diff = xt::abs(result - expected);
    float max_abs = xt::amax(diff)();
    float max_rel = xt::amax(diff / (xt::abs(expected) + 1e-10f))();
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f))
        << "max_abs=" << max_abs << " max_rel=" << max_rel << " result[0]=" << result(0, 0, 0, 0)
        << " expected[0]=" << expected(0, 0, 0, 0);
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
    printf("HOST total_sum_sq = %.6f, norm = %.6f, 1/norm = %.6f\n", (float)total, norm, 1.0f / norm);

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    auto d1 = xt::abs(result - expected);
    float max_abs = xt::amax(d1)();
    float max_rel = xt::amax(d1 / (xt::abs(expected) + 1e-10f))();
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f))
        << "max_abs=" << max_abs << " max_rel=" << max_rel;
}

TEST_F(FrobeniusNormalizeTest, FourTilesAllOnes) {
    // 4 tiles of all-ones → norm = sqrt(4096) = 64, result = 1/64 = 0.015625
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 64, 64};
    xt::xarray<float> data = xt::ones<float>(shape);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    float r0 = result(0, 0, 0, 0);
    float e0 = expected(0, 0, 0, 0);
    auto d1 = xt::abs(result - expected);
    float max_abs = xt::amax(d1)();
    float max_rel = xt::amax(d1 / (xt::abs(expected) + 1e-10f))();
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f))
        << "max_abs=" << max_abs << " max_rel=" << max_rel << " result[0]=" << r0 << " expected=" << e0;
}

TEST_F(FrobeniusNormalizeTest, TwoTiles) {
    // 2 tiles = 2 cores → single chain hop. Isolates chain add error.
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 32, 64};
    xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    auto d1 = xt::abs(result - expected);
    float max_abs = xt::amax(d1)();
    float max_rel = xt::amax(d1 / (xt::abs(expected) + 1e-10f))();
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f))
        << "max_abs=" << max_abs << " max_rel=" << max_rel;
}

TEST_F(FrobeniusNormalizeTest, SmallMatrix) {
    // 4 tiles = 4 cores → 3 chain hops in column
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 64, 64};
    xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &autograd::ctx().get_device());
    auto result_tensor = metal::frobenius_normalize(input_tensor, eps);
    auto result = core::to_xtensor(result_tensor);

    auto d1 = xt::abs(result - expected);
    float max_abs = xt::amax(d1)();
    float max_rel = xt::amax(d1 / (xt::abs(expected) + 1e-10f))();
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f))
        << "max_abs=" << max_abs << " max_rel=" << max_rel;
}

TEST_F(FrobeniusNormalizeTest, MediumMatrix) {
    using namespace ttml;

    std::array<uint32_t, 4> shape = {1, 1, 256, 320};
    xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
    float eps = 1e-7f;

    auto expected = frobenius_normalize_ref(data, eps);
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &autograd::ctx().get_device());
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
    auto input_tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &autograd::ctx().get_device());

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
    auto diff = xt::abs(result - expected);
    auto rel = diff / (xt::abs(expected) + 1e-10f);
    printf("  max_abs=%.6f max_rel=%.6f\n", xt::amax(diff)(), xt::amax(rel)());
    // Find first mismatching tile
    for (uint32_t r = 0; r < 2048; r += 32) {
        for (uint32_t c = 0; c < 5632; c += 32) {
            auto tile_rel = xt::view(rel, 0, 0, xt::range(r, r + 32), xt::range(c, c + 32));
            float tile_max = xt::amax(tile_rel)();
            if (tile_max > 1e-4f) {
                uint32_t tile_idx = (r / 32) * (5632 / 32) + (c / 32);
                printf("  First bad tile: row=%u col=%u tile_idx=%u max_rel=%.6f\n", r, c, tile_idx, tile_max);
                printf(
                    "  result[%u,%u]=%.8f expected=%.8f\n",
                    r,
                    c,
                    (float)result(0, 0, r, c),
                    (float)expected(0, 0, r, c));
                goto done;
            }
        }
    }
done:
    EXPECT_TRUE(xt::allclose(result, expected, /*rtol=*/1e-2f, /*atol=*/1e-2f));
}

TEST_F(FrobeniusNormalizeTest, CompositeVsFused) {
    using namespace ttml;

    float eps = 1e-7f;
    auto& device = autograd::ctx().get_device();
    auto& cq = device.mesh_command_queue();
    constexpr int N = 10;

    struct Shape {
        uint32_t r, c;
    };
    Shape shapes[] = {{2048, 5632}, {8192, 8192}};

    for (auto [rows, cols] : shapes) {
        std::array<uint32_t, 4> shape = {1, 1, rows, cols};
        xt::xarray<float> data = xt::random::randn<float>(shape, 0.0f, 1.0f);
        auto input_tensor = core::from_xtensor<float, ttnn::DataType::BFLOAT16>(data, &device);

        auto composite_fn = [&]() {
            auto squares = ttnn::square(input_tensor);
            auto sum_squares = ttnn::sum(squares, ttsl::SmallVector<int>{-2, -1}, true);
            auto norm_tensor = ttnn::sqrt(sum_squares);
            auto norm_plus_eps = ttnn::add(norm_tensor, eps);
            return ttnn::divide(input_tensor, norm_plus_eps);
        };

        // Warmup
        for (int i = 0; i < 2; ++i) {
            (void)composite_fn();
            (void)metal::frobenius_normalize(input_tensor, eps);
        }
        tt::tt_metal::distributed::Finish(cq);

        // Benchmark fused
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            (void)metal::frobenius_normalize(input_tensor, eps);
        }
        tt::tt_metal::distributed::Finish(cq);
        auto t1 = std::chrono::high_resolution_clock::now();
        double fused_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / (double)N;

        // Benchmark composite
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) {
            (void)composite_fn();
        }
        tt::tt_metal::distributed::Finish(cq);
        auto t3 = std::chrono::high_resolution_clock::now();
        double composite_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() / (double)N;

        // Accuracy of sum-of-squares (before tile reduction).
        // Compare norm scalar: composite via ttnn::sum(square), fused via back-computing from output.
        auto bf16_data = core::to_xtensor(input_tensor);

        // Reference: FP32 host sum_sq from BF16 data
        double ref_sum_sq = xt::sum(bf16_data * bf16_data)();

        // Composite: ttnn::square → ttnn::sum
        auto comp_sum_sq_tensor = ttnn::sum(ttnn::square(input_tensor), ttsl::SmallVector<int>{-2, -1}, true);
        auto comp_sum_sq_xt = core::to_xtensor(comp_sum_sq_tensor);
        float comp_sum_sq = comp_sum_sq_xt(0, 0, 0, 0);

        // Fused: back-compute sum_sq from output using MULTIPLE elements for better accuracy
        // result[i] = input[i] / norm, so norm = input[i] / result[i]
        // Average over many elements to cancel BF16 rounding noise
        auto fused_result_acc = metal::frobenius_normalize(input_tensor, eps);
        auto fused_out_acc = core::to_xtensor(fused_result_acc);
        double inv_norm_sum = 0;
        int count = 0;
        for (size_t i = 0; i < bf16_data.size(); i += 37) {  // sample every 37th element
            float in_v = bf16_data.flat(i);
            float out_v = fused_out_acc.flat(i);
            if (std::abs(in_v) > 0.1f) {  // skip near-zero to avoid division instability
                inv_norm_sum += (double)out_v / (double)in_v;
                count++;
            }
        }
        double inv_norm_avg = inv_norm_sum / count;
        double norm_fused = 1.0 / inv_norm_avg;
        double fused_sum_sq = (norm_fused - eps) * (norm_fused - eps);

        float comp_err = std::abs((double)comp_sum_sq - ref_sum_sq) / ref_sum_sq;
        float fused_err = std::abs(fused_sum_sq - ref_sum_sq) / ref_sum_sq;

        auto expected = frobenius_normalize_ref(bf16_data, eps);

        auto fused_result2 = metal::frobenius_normalize(input_tensor, eps);
        auto fused_out = core::to_xtensor(fused_result2);

        auto composite_result = composite_fn();
        auto composite_out = core::to_xtensor(composite_result);

        auto fused_diff = xt::abs(fused_out - expected);
        auto comp_diff = xt::abs(composite_out - expected);
        auto abs_expected = xt::abs(expected) + 1e-10f;

        float fused_max_abs = xt::amax(fused_diff)();
        float fused_max_rel = xt::amax(fused_diff / abs_expected)();
        float fused_mean_rel = xt::mean(fused_diff / abs_expected)();

        float comp_max_abs = xt::amax(comp_diff)();
        float comp_max_rel = xt::amax(comp_diff / abs_expected)();
        float comp_mean_rel = xt::mean(comp_diff / abs_expected)();

        // Also compute relative error only for elements where |expected| > 1e-5
        auto mask = xt::abs(expected) > 1e-5f;
        auto fused_rel_masked = xt::filter(fused_diff / xt::abs(expected), mask);
        auto comp_rel_masked = xt::filter(comp_diff / xt::abs(expected), mask);
        float fused_mean_rel_filtered = xt::mean(fused_rel_masked)();
        float comp_mean_rel_filtered = xt::mean(comp_rel_masked)();
        int total_els = expected.size();
        int filtered_els = fused_rel_masked.size();

        printf("=== %ux%u BF16 ===\n", rows, cols);
        printf("  Sum-of-squares accuracy (vs FP32 host ref):\n");
        printf("    Reference:  %.6f\n", (float)ref_sum_sq);
        printf("    Composite:  %.6f  (rel err: %.2e)\n", comp_sum_sq, comp_err);
        printf("    Fused:      %.6f  (rel err: %.2e)\n", fused_sum_sq, fused_err);
        printf("  Normalize output accuracy:\n");
        printf("  %-20s %12s %12s %12s %12s\n", "", "Time (us)", "Max |err|", "Max rel", "Mean rel");
        printf(
            "  %-20s %12s %12s %12s %12s %14s\n",
            "",
            "Time (us)",
            "Max |err|",
            "Max rel",
            "Mean rel",
            "Mean rel (>1e-5)");
        printf(
            "  %-20s %12.0f %12.6f %12.6f %12.6f %14.6f\n",
            "Composite (5 ops)",
            composite_us,
            comp_max_abs,
            comp_max_rel,
            comp_mean_rel,
            comp_mean_rel_filtered);
        printf(
            "  %-20s %12.0f %12.6f %12.6f %12.6f %14.6f\n",
            "Fused kernel",
            fused_us,
            fused_max_abs,
            fused_max_rel,
            fused_mean_rel,
            fused_mean_rel_filtered);
        printf(
            "  Speedup: %.2fx  (%d/%d elements above threshold)\n", composite_us / fused_us, filtered_els, total_els);
    }
}
