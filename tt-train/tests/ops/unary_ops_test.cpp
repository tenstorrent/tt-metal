// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/unary_ops.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <core/ttnn_all_includes.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

namespace ttml::ops::tests {

/**
 * Compute and print tolerance statistics for comparing two tensors
 * This helps determine appropriate tolerances for EXPECT_TRUE(xt::allclose(...))
 *
 * @param computed The computed tensor values
 * @param expected The expected tensor values
 * @param test_name Name of the test for logging purposes
 * @return A pair of (rtol, atol) that would pass the comparison
 */
std::pair<float, float> compute_and_print_tolerances(
    const xt::xarray<float>& computed, const xt::xarray<float>& expected, const std::string& test_name) {
    // Ensure shapes match
    if (computed.shape() != expected.shape()) {
        std::cerr << "ERROR: Shape mismatch in " << test_name << std::endl;
        return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    }

    auto abs_diff = xt::abs(computed - expected);

    // Compute relative difference using the same formula as xt::allclose:
    // rel_diff = abs_diff / max(|computed|, |expected|)
    // This avoids huge relative differences when values are near zero
    auto abs_computed = xt::abs(computed);
    auto abs_expected = xt::abs(expected);
    auto rel_denom = xt::maximum(abs_computed, abs_expected);
    // Only compute relative diff where denominator is meaningful (> threshold)
    const float rel_threshold = 1e-6f;
    auto rel_diff = xt::where(
        rel_denom > rel_threshold,
        abs_diff / (rel_denom + 1e-8f),
        xt::zeros_like(abs_diff));  // Set to 0 for near-zero values

    // Compute statistics using double precision for accuracy
    // Convert to double to avoid accumulation errors, especially for large arrays
    double max_abs_diff = static_cast<double>(xt::amax(abs_diff)());

    // Compute mean using double precision accumulation
    auto abs_diff_flat = xt::flatten(abs_diff);
    double abs_sum = 0.0;
    for (auto val : abs_diff_flat) {
        abs_sum += static_cast<double>(val);
    }
    double mean_abs_diff = abs_sum / abs_diff_flat.size();

    // For relative differences, only consider values where denominator is meaningful
    // Filter out near-zero values for relative difference statistics
    auto rel_diff_valid_mask = rel_denom > rel_threshold;

    // Compute max and mean only on valid values using double precision
    double max_rel_diff = 0.0;
    double mean_rel_diff = 0.0;
    size_t valid_count = 0;
    auto rel_diff_flat = xt::flatten(rel_diff);
    auto mask_flat = xt::flatten(rel_diff_valid_mask);
    for (size_t i = 0; i < rel_diff_flat.size(); ++i) {
        if (mask_flat(i)) {
            double val = static_cast<double>(rel_diff_flat(i));
            max_rel_diff = std::max(max_rel_diff, val);
            mean_rel_diff += val;
            valid_count++;
        }
    }
    if (valid_count > 0) {
        mean_rel_diff /= valid_count;
    }

    // Count how many values are near zero (where relative tolerance doesn't apply)
    auto near_zero_mask = rel_denom <= rel_threshold;
    size_t near_zero_count = xt::sum(near_zero_mask)();
    size_t total_count = computed.size();

    // Find percentiles using double precision for better accuracy
    // Convert to double vector for sorting
    std::vector<double> abs_vec;
    abs_vec.reserve(abs_diff_flat.size());
    for (auto val : abs_diff_flat) {
        abs_vec.push_back(static_cast<double>(val));
    }
    std::sort(abs_vec.begin(), abs_vec.end());

    double abs_p50 = abs_vec[abs_vec.size() / 2];
    double abs_p95 = abs_vec[static_cast<size_t>(abs_vec.size() * 0.95)];
    double abs_p99 = abs_vec[static_cast<size_t>(abs_vec.size() * 0.99)];

    // For relative differences, only consider valid (non-near-zero) values
    std::vector<double> rel_vec;
    rel_vec.reserve(valid_count);
    for (size_t i = 0; i < rel_diff_flat.size(); ++i) {
        if (mask_flat(i)) {
            rel_vec.push_back(static_cast<double>(rel_diff_flat(i)));
        }
    }
    std::sort(rel_vec.begin(), rel_vec.end());

    double rel_p50 = rel_vec.empty() ? 0.0 : rel_vec[rel_vec.size() / 2];
    double rel_p95 = rel_vec.empty() ? 0.0 : rel_vec[static_cast<size_t>(rel_vec.size() * 0.95)];
    double rel_p99 = rel_vec.empty() ? 0.0 : rel_vec[static_cast<size_t>(rel_vec.size() * 0.99)];

    // Print statistics
    std::cout << "\n=== Tolerance Analysis for " << test_name << " ===" << std::endl;
    std::cout << "Values near zero (|value| < " << rel_threshold << "): " << near_zero_count << " / " << total_count
              << " (" << (100.0f * near_zero_count / total_count) << "%)" << std::endl;
    std::cout << "Absolute Differences:" << std::endl;
    std::cout << "  Max:    " << max_abs_diff << std::endl;
    std::cout << "  Mean:   " << mean_abs_diff << std::endl;
    std::cout << "  P50:    " << abs_p50 << std::endl;
    std::cout << "  P95:    " << abs_p95 << std::endl;
    std::cout << "  P99:    " << abs_p99 << std::endl;

    std::cout << "Relative Differences (only for |value| > " << rel_threshold << "):" << std::endl;
    std::cout << "  Max:    " << max_rel_diff << std::endl;
    std::cout << "  Mean:   " << mean_rel_diff << std::endl;
    std::cout << "  P50:    " << rel_p50 << std::endl;
    std::cout << "  P95:    " << rel_p95 << std::endl;
    std::cout << "  P99:    " << rel_p99 << std::endl;

    // Suggest tolerances based on actual xt::allclose formula:
    // |computed - expected| <= atol + rtol * max(|computed|, |expected|)
    // For near-zero values, absolute tolerance dominates
    // For larger values, relative tolerance dominates
    // Use double precision for calculations, convert to float at the end
    double suggested_atol = std::max(abs_p99 * 1.1, 1e-6);
    // For rtol, use the relative difference from non-near-zero values
    double suggested_rtol = rel_vec.empty() ? 1e-2 : std::max(rel_p99 * 1.1, 1e-6);

    std::cout << "Suggested Tolerances:" << std::endl;
    std::cout << "  rtol: " << suggested_rtol << std::endl;
    std::cout << "  atol: " << suggested_atol << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Convert back to float for return value
    return {static_cast<float>(suggested_rtol), static_cast<float>(suggested_atol)};
}

class UnaryOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

TEST_F(UnaryOpsTest, GlobalMean) {
    std::vector<float> test_data = {1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F};

    auto shape = ttnn::Shape({2, 1, 1, 4});
    auto tensor = core::from_vector(test_data, shape, &autograd::ctx().get_device());

    auto tensor_ptr = autograd::create_tensor(tensor);

    auto result = mean(tensor_ptr);
    auto result_data = core::to_vector(result->get_value());

    ASSERT_EQ(result_data.size(), 1);
    EXPECT_FLOAT_EQ(result_data[0], 2.5F);

    result->backward();
    auto tensor_grad = core::to_vector(tensor_ptr->get_grad());
    ASSERT_EQ(tensor_grad.size(), test_data.size());
    for (float it : tensor_grad) {
        EXPECT_FLOAT_EQ(it, 0.125F);
    }
}

TEST_F(UnaryOpsTest, LogSoftmax) {
    auto* device = &autograd::ctx().get_device();
    std::vector<float> test_data = {-0.1F, -0.2F, -0.3F, -0.4F, 0.F, -0.2F, -0.3F, -0.4F};
    auto tensor = core::from_vector(test_data, ttnn::Shape({2, 1, 1, 4}), device);
    auto tensor_ptr = autograd::create_tensor(tensor);
    auto result = log_softmax_moreh(tensor_ptr, 3);
    auto result_data = core::to_vector(result->get_value());
    std::vector<float> expected_data = {
        -1.24253553F, -1.34253553F, -1.44253553F, -1.54253553F, -1.17244159F, -1.37244159F, -1.47244159F, -1.57244159F};
    EXPECT_EQ(result_data.size(), expected_data.size());
    for (uint32_t idx = 0; idx < result_data.size(); ++idx) {
        EXPECT_NEAR(result_data[idx], expected_data[idx], 2e-2F);
    }

    result->backward();
    auto tensor_grad = core::to_vector(tensor_ptr->get_grad());
    std::vector<float> expected_grad = {-0.156F, -0.03906F, 0.05078F, 0.1406F, -0.25F, -0.0156F, 0.07421F, 0.16406F};
    EXPECT_EQ(tensor_grad.size(), expected_grad.size());
    for (uint32_t idx = 0; idx < tensor_grad.size(); ++idx) {
        EXPECT_NEAR(tensor_grad[idx], expected_grad[idx], 2e-2F);
    }
}

// Reference implementation of SiLU forward pass: silu(x) = x * sigmoid(x)
xt::xarray<float> silu_forward_reference(const xt::xarray<float>& x) {
    auto sigmoid_x = 1.0f / (1.0f + xt::exp(-x));
    return x * sigmoid_x;
}

// Reference implementation of SiLU backward pass: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
xt::xarray<float> silu_backward_reference(const xt::xarray<float>& x, const xt::xarray<float>& grad) {
    auto sigmoid_x = 1.0f / (1.0f + xt::exp(-x));
    auto silu_grad = sigmoid_x * (1.0f + x * (1.0f - sigmoid_x));
    return grad * silu_grad;
}

TEST_F(UnaryOpsTest, Silu) {
    auto N = 4;
    auto C = 1;
    auto H = 20;
    auto W = 5;
    xt::xarray<float> a = xt::empty<float>({N, C, H, W});
    ttml::core::parallel_generate(
        std::span{a.data(), a.size()}, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, 42);

    // Compute expected values using reference implementation (tests correctness, not RNG reproducibility)
    xt::xarray<float> expected_silu = silu_forward_reference(a);

    auto a_tensor = autograd::create_tensor(core::from_xtensor(a, &autograd::ctx().get_device()));
    auto computed_silu = silu(a_tensor);
    auto computed_silu_xtensor = core::to_xtensor(computed_silu->get_value());

    // Compare forward pass against reference implementation
    // This tests correctness: that SiLU computes x * sigmoid(x) correctly
    EXPECT_TRUE(xt::allclose(computed_silu_xtensor, expected_silu, 8e-3F, 4e-2F));

    // Compute backward pass
    auto target = autograd::create_tensor(core::zeros_like(computed_silu->get_value()));
    auto result = mse_loss(computed_silu, target);
    result->backward();
    auto computed_silu_grad = core::to_xtensor(computed_silu->get_grad());

    // Compute expected gradient using reference implementation
    // For MSE loss with zero target: upstream_grad = 2/N * silu(x)
    auto total_elements = static_cast<float>(a.size());
    auto upstream_grad = (2.0f / total_elements) * computed_silu_xtensor;
    xt::xarray<float> expected_silu_grad = silu_backward_reference(a, upstream_grad);
    // Compare backward pass against reference implementation
    // This tests correctness: that SiLU gradient computes sigmoid(x) * (1 + x * (1 - sigmoid(x))) correctly
    EXPECT_TRUE(xt::allclose(computed_silu_grad, expected_silu_grad, 8e-3F, 4e-2F));
}

}  // namespace ttml::ops::tests
