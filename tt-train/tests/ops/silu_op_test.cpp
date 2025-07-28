// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cassert>
#include <core/ttnn_all_includes.hpp>
#include <iostream>
#include <numeric>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"
#include "ops/unary_ops.hpp"

class SiLUOpTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// ============================================================================
// Section 1: SiLU Kernel vs Reference Implementation
// ============================================================================
// These tests validate the SiLU kernel implementation against
// a reference implementation using basic operations: silu(x) = x * sigmoid(x)
//
// Test methodology:
// 1. Create test tensor `x` of shape [B,N,S,C] with x.requires_grad = True
// 2. Compute SiLU using both kernel and reference implementations
// 3. Compare forward and backward results for numerical correctness
// ============================================================================

namespace {

/**
 * Reference implementation of SiLU forward pass using xt library
 * SiLU(x) = x * sigmoid(x)
 *
 * @param x Input tensor
 * @return SiLU activation result
 */
xt::xarray<float> silu_forward_reference(const xt::xarray<float>& x) {
    auto sigmoid_x = 1.0f / (1.0f + xt::exp(-x));
    return x * sigmoid_x;
}

/**
 * Reference implementation of SiLU backward pass using xt library
 * SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
 *
 * @param x Input tensor (original input to SiLU)
 * @param grad Gradient from upstream
 * @return Gradient with respect to input
 */
xt::xarray<float> silu_backward_reference(const xt::xarray<float>& x, const xt::xarray<float>& grad) {
    auto sigmoid_x = 1.0f / (1.0f + xt::exp(-x));
    auto silu_grad = sigmoid_x * (1.0f + x * (1.0f - sigmoid_x));
    return grad * silu_grad;
}

/**
 * Helper function to compute precision metrics between two tensors
 * Returns a struct with various precision metrics
 */
struct PrecisionMetrics {
    float max_abs_error;
    float mean_abs_error;
    float relative_error;
    float rmse;
    bool allclose_strict;
    bool allclose_loose;
};

PrecisionMetrics compute_precision_metrics(const xt::xarray<float>& reference, const xt::xarray<float>& test) {
    PrecisionMetrics metrics;

    auto abs_diff = xt::abs(reference - test);
    auto rel_diff = abs_diff / (xt::abs(reference) + 1e-8f);

    metrics.max_abs_error = xt::amax(abs_diff)();
    metrics.mean_abs_error = xt::mean(abs_diff)();
    metrics.relative_error = xt::mean(rel_diff)();
    metrics.rmse = xt::sqrt(xt::mean(xt::square(abs_diff)))();
    metrics.allclose_strict = xt::allclose(reference, test, 1e-4f, 1e-3f);  // Moderate precision
    metrics.allclose_loose = xt::allclose(reference, test, 1e-3f, 3e-2f);   // Same as CompareKernelVsReference

    return metrics;
}

void CompareKernelVsReference(const xt::xarray<float>& input_data) {
    using namespace ttml;

    // Create input tensors for kernel and reference implementations
    auto input_kernel = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
    auto input_reference = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));

    // Forward pass - kernel implementation
    auto result_kernel = ops::silu(input_kernel);
    auto result_kernel_xtensor = core::to_xtensor(result_kernel->get_value());

    // Forward pass - reference implementation
    auto result_reference = ops::silu(input_reference, /*use_composite_bw=*/true);
    auto result_reference_xtensor = core::to_xtensor(result_reference->get_value());

    // Compare forward results
    // NOTE: This comparison does not much make sense for SiLU, because there is no forward SiLU composite if I am not
    // mistaken, but we still do it for consistency.
    EXPECT_TRUE(xt::allclose(result_kernel_xtensor, result_reference_xtensor, 1.0e-3F, 3e-2F));

    // Backward pass - create target and compute gradients
    auto target_kernel = autograd::create_tensor(core::zeros_like(result_kernel->get_value()));
    auto target_reference = autograd::create_tensor(core::zeros_like(result_reference->get_value()));

    auto mse_kernel = ops::mse_loss(result_kernel, target_kernel);
    auto mse_reference = ops::mse_loss(result_reference, target_reference);

    mse_kernel->backward();
    mse_reference->backward();

    // Compare backward results
    auto input_grad_kernel = core::to_xtensor(input_kernel->get_grad());
    auto input_grad_reference = core::to_xtensor(input_reference->get_grad());

    EXPECT_TRUE(xt::allclose(input_grad_kernel, input_grad_reference, 1.0e-3F, 3e-2F));
}

/**
 * Helper function to compare kernel vs reference implementations of SiLU
 *
 * This function tests both forward and backward passes to ensure:
 * 1. Forward pass: kernel and reference produce identical results
 * 2. Backward pass: gradients computed by both implementations match
 * 3. All outputs and gradients are finite (no NaN/Inf values)
 *
 * @param shape Input tensor shape [B, N, S, C] where:
 *   - B: batch size
 *   - N: number of channels (usually 1 for transformers)
 *   - S: sequence length (height for transformers)
 *   - C: feature dimension (width/embedding dimension)
 *
 * Test cases cover various scenarios:
 * - Block size alignment: Wt % block_size patterns (0, 1, 2, 3)
 * - Memory patterns: small vs large tensors
 * - Realistic shapes: transformer model dimensions
 */
static void CompareKernelVsReferenceWithShape(const std::vector<uint32_t>& shape) {
    using namespace ttml;

    // Generate random input data with range [-2, 2] to test SiLU behavior across saturation regions
    xt::random::seed(42);
    xt::xarray<float> input_data = xt::random::rand<float>(shape, -2.0F, 2.0F);

    CompareKernelVsReference(input_data);
}

}  // namespace

// ============================================================================
// Section 2: SiLU Kernel vs Reference Implementation - Comprehensive Tests
// ============================================================================
// These tests systematically compare the SiLU kernel implementation against
// the reference implementation across different scenarios:
//
// - Block size patterns: Wt % block_size = 0, 1, 2, 3 (where block_size = 4)
// - Memory patterns: small tensors vs large tensors
// - Training scenarios: realistic model shapes (NanoLlama, etc.)
// - Batch patterns: single vs multiple batches and sequences
//
// Note: Wt = ceil(C / 32), so we test different C values to cover all
// block_size remainder patterns.
// Shape notation: [B, N, S, C] where C is the feature dimension
// ============================================================================

// Test small tensor - basic functionality
TEST_F(SiLUOpTest, SiLU_Compare_Small) {
    // C=8, Wt=1, Wt%4=1
    CompareKernelVsReferenceWithShape({1, 1, 1, 8});
}

// Test block_size alignment patterns
// Wt % block_size = 0 (perfectly aligned)
TEST_F(SiLUOpTest, SiLU_Compare_BlockSize_Remainder0) {
    // C=128, Wt=4, Wt%4=0
    CompareKernelVsReferenceWithShape({1, 1, 1, 128});
}

// Wt % block_size = 1
TEST_F(SiLUOpTest, SiLU_Compare_BlockSize_Remainder1) {
    // C=160, Wt=5, Wt%4=1
    CompareKernelVsReferenceWithShape({1, 1, 1, 160});
}

// Wt % block_size = 2
TEST_F(SiLUOpTest, SiLU_Compare_BlockSize_Remainder2) {
    // C=192, Wt=6, Wt%4=2
    CompareKernelVsReferenceWithShape({1, 1, 1, 192});
}

// Wt % block_size = 3
TEST_F(SiLUOpTest, SiLU_Compare_BlockSize_Remainder3) {
    // C=224, Wt=7, Wt%4=3
    CompareKernelVsReferenceWithShape({1, 1, 1, 224});
}

// Test large tensor - memory stress test
TEST_F(SiLUOpTest, SiLU_Compare_Large) {
    // Large C dimension to test memory handling
    // C=32768, Wt=1024, Wt%4=0
    CompareKernelVsReferenceWithShape({1, 1, 1, 32768});
}

// Test very large tensor - extreme memory test
TEST_F(SiLUOpTest, SiLU_Compare_VeryLarge) {
    // Very large C dimension ~1M elements
    // C=1048576, Wt=32768, Wt%4=0
    CompareKernelVsReferenceWithShape({1, 1, 1, 1048576});
}

// Test NanoLlama-like shape - realistic transformer model
TEST_F(SiLUOpTest, SiLU_Compare_NanoLlama_Shape) {
    // Typical NanoLlama dimensions: multiple batches and sequences
    // B=2, N=1, S=64, C=512 (hidden dimension)
    // C=512, Wt=16, Wt%4=0
    CompareKernelVsReferenceWithShape({2, 1, 64, 512});
}

// Test batch processing with different sequence lengths
TEST_F(SiLUOpTest, SiLU_Compare_MultiBatch_MultiSeq) {
    // Multiple batches with longer sequences
    // B=4, N=1, S=128, C=768
    // C=768, Wt=24, Wt%4=0
    CompareKernelVsReferenceWithShape({4, 1, 128, 768});
}

// Test smaller model dimensions with unaligned C
TEST_F(SiLUOpTest, SiLU_Compare_SmallModel_Unaligned) {
    // Smaller model with unaligned channel dimension
    // B=2, N=1, S=32, C=100
    // C=100, Wt=4, Wt%4=0 (but C is not multiple of 32)
    CompareKernelVsReferenceWithShape({2, 1, 32, 100});
}

// ============================================================================
// Section 3: Backward Pass Precision Comparison Test
// ============================================================================
// This test compares the precision of kernel vs composite implementations
// against a reference xt implementation for backward pass only to determine
// which is more accurate for gradient computation
// ============================================================================

TEST_F(SiLUOpTest, SiLU_Precision_Comparison) {
    using namespace ttml;

    // Test with 3 different shapes
    std::vector<std::vector<uint32_t>> test_shapes = {
        {1, 1, 1, 8},     // small
        {2, 1, 64, 512},  // nanollama like
        {1, 1, 1, 32768}  // large
    };

    std::vector<std::string> shape_names = {"small", "nanollama-like", "large"};

    // Test range: moderate range that covers typical activation values
    const float min_val = -2.0f;
    const float max_val = 2.0f;

    // Run the test multiple times with different seeds for statistical confidence
    const int num_runs = 10;

    for (size_t shape_idx = 0; shape_idx < test_shapes.size(); ++shape_idx) {
        const auto& shape = test_shapes[shape_idx];
        const auto& shape_name = shape_names[shape_idx];

        // Collect metrics across all runs
        std::vector<float> kernel_rmse_values, kernel_max_error_values, kernel_mean_error_values;
        std::vector<float> composite_rmse_values, composite_max_error_values, composite_mean_error_values;
        int kernel_wins = 0, composite_wins = 0;

        for (int run = 0; run < num_runs; ++run) {
            // Generate test data with different seed for each run
            xt::random::seed(42 + run);
            xt::xarray<float> input_data = xt::random::rand<float>(shape, min_val, max_val);

            // Create input tensors
            auto input_kernel = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
            auto input_composite =
                autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));

            // Forward pass - kernel implementation
            auto result_kernel = ops::silu(input_kernel, /*use_composite_bw=*/false);

            // Forward pass - composite implementation
            auto result_composite = ops::silu(input_composite, /*use_composite_bw=*/true);

            // Backward pass - create target and compute gradients (same as CompareKernelVsReference)
            auto target_kernel = autograd::create_tensor(core::zeros_like(result_kernel->get_value()));
            auto target_composite = autograd::create_tensor(core::zeros_like(result_composite->get_value()));

            auto mse_kernel = ops::mse_loss(result_kernel, target_kernel);
            auto mse_composite = ops::mse_loss(result_composite, target_composite);

            mse_kernel->backward();
            mse_composite->backward();

            // Get computed gradients
            auto input_grad_kernel = core::to_xtensor(input_kernel->get_grad());
            auto input_grad_composite = core::to_xtensor(input_composite->get_grad());

            // Reference backward pass - compute what the gradient should be
            // For MSE loss with zero target: loss = mean((result - 0)^2) = mean(result^2)
            // Gradient: d_loss/d_input = d_loss/d_result * d_result/d_input
            // d_loss/d_result = 2 * result / N (where N is total number of elements)
            // d_result/d_input = silu_derivative
            // So: d_loss/d_input = (2 * result / N) * silu_derivative
            auto result_ref = silu_forward_reference(input_data);
            auto total_elements = static_cast<float>(input_data.size());
            auto mse_grad = (2.0f / total_elements) * result_ref;  // MSE gradient with mean reduction
            auto grad_reference = silu_backward_reference(input_data, mse_grad);

            // Compare backward precision
            auto backward_metrics_kernel = compute_precision_metrics(grad_reference, input_grad_kernel);
            auto backward_metrics_composite = compute_precision_metrics(grad_reference, input_grad_composite);

            // Collect metrics for consolidation
            kernel_rmse_values.push_back(backward_metrics_kernel.rmse);
            kernel_max_error_values.push_back(backward_metrics_kernel.max_abs_error);
            kernel_mean_error_values.push_back(backward_metrics_kernel.mean_abs_error);

            composite_rmse_values.push_back(backward_metrics_composite.rmse);
            composite_max_error_values.push_back(backward_metrics_composite.max_abs_error);
            composite_mean_error_values.push_back(backward_metrics_composite.mean_abs_error);

            // Count wins
            if (backward_metrics_kernel.rmse < backward_metrics_composite.rmse) {
                kernel_wins++;
            } else {
                composite_wins++;
            }

            // Sanity check - both should be reasonably close to reference
            EXPECT_TRUE(backward_metrics_kernel.allclose_loose) << "Kernel backward pass too inaccurate";
            EXPECT_TRUE(backward_metrics_composite.allclose_loose) << "Composite backward pass too inaccurate";
        }

        // Compute consolidated statistics
        auto compute_mean = [](const std::vector<float>& values) {
            return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
        };

        float kernel_mean_rmse = compute_mean(kernel_rmse_values);
        float kernel_mean_max_error = compute_mean(kernel_max_error_values);
        float kernel_mean_mean_error = compute_mean(kernel_mean_error_values);

        float composite_mean_rmse = compute_mean(composite_rmse_values);
        float composite_mean_max_error = compute_mean(composite_max_error_values);
        float composite_mean_mean_error = compute_mean(composite_mean_error_values);

        // Print consolidated results for this shape
        std::cout << "=== CONSOLIDATED RESULTS - " << shape_name << " shape [" << shape[0] << ", " << shape[1] << ", "
                  << shape[2] << ", " << shape[3] << "] (Average across " << num_runs << " runs) ===\n";
        std::cout << "Kernel (4 packs, float36)  - Mean Max Error: " << kernel_mean_max_error
                  << ", Mean Mean Error: " << kernel_mean_mean_error << ", Mean RMSE: " << kernel_mean_rmse << "\n";
        std::cout << "Composite                  - Mean Max Error: " << composite_mean_max_error
                  << ", Mean Mean Error: " << composite_mean_mean_error << ", Mean RMSE: " << composite_mean_rmse
                  << "\n";
        std::cout << "Overall Winner             - "
                  << (kernel_mean_rmse < composite_mean_rmse ? "Kernel" : "Composite") << " (based on mean RMSE)\n";
        std::cout << "Win Count                  - Kernel: " << kernel_wins << "/" << num_runs
                  << ", Composite: " << composite_wins << "/" << num_runs << "\n\n";
    }
}
