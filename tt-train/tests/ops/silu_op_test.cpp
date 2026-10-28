// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cassert>
#include <numeric>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/random.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/silu_bw/silu_bw.hpp"
#include "ops/losses.hpp"
#include "ops/unary_ops.hpp"
#include "test_utils/program_cache.hpp"
#include "test_utils/random_data.hpp"

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
    float rmse;
    bool allclose_strict;
    bool allclose_loose;
};

PrecisionMetrics compute_precision_metrics(const xt::xarray<float>& reference, const xt::xarray<float>& test) {
    PrecisionMetrics metrics;

    auto abs_diff = xt::abs(reference - test);
    metrics.rmse = xt::sqrt(xt::mean(xt::square(abs_diff)))();
    metrics.allclose_strict = xt::allclose(reference, test, 1e-4f, 1e-3f);  // Moderate precision
    metrics.allclose_loose = xt::allclose(reference, test, 1e-3f, 3e-2f);   // Same as CompareKernelVsReference

    return metrics;
}

void CompareKernelVsReference(const xt::xarray<float>& input_data) {
    using namespace ttml;

    // Create input tensors for kernel and reference implementations
    auto input_kernel = autograd::create_tensor(
        core::from_xtensor(input_data, &autograd::ctx().get_device()), /* requires_grad */ true);
    auto input_reference = autograd::create_tensor(
        core::from_xtensor(input_data, &autograd::ctx().get_device()), /* requires_grad */ true);

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
    auto& rng = autograd::ctx().get_generator();
    xt::xarray<float> input_data = xt::empty<float>(shape);
    core::parallel_generate<float>(
        input_data, []() { return std::uniform_real_distribution<float>(-2.0F, 2.0F); }, /* seed */ rng());

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

    // Single test shape: nanollama-like
    std::vector<uint32_t> shape = {2, 1, 64, 512};

    // Test range: moderate range that covers typical activation values
    const float min_val = -2.0f;
    const float max_val = 2.0f;

    // Reduced number of runs for faster testing
    const int num_runs = 3;

    // Collect metrics across all runs
    std::vector<float> kernel_rmse_values(num_runs);
    std::vector<float> composite_rmse_values(num_runs);

    auto& rng = autograd::ctx().get_generator();
    for (int run = 0; run < num_runs; ++run) {
        // Generate test data with different seed for each run
        xt::xarray<float> input_data = xt::empty<float>(shape);
        core::parallel_generate<float>(
            input_data,
            [min_val, max_val]() { return std::uniform_real_distribution<float>(min_val, max_val); },
            /* seed */ rng());

        // Create input tensors
        auto input_kernel = autograd::create_tensor(
            core::from_xtensor(input_data, &autograd::ctx().get_device()), /* requires_grad */ true);
        auto input_composite = autograd::create_tensor(
            core::from_xtensor(input_data, &autograd::ctx().get_device()), /* requires_grad */ true);

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
        auto result_ref = silu_forward_reference(input_data);
        auto total_elements = static_cast<float>(input_data.size());
        auto mse_grad = (2.0f / total_elements) * result_ref;
        auto grad_reference = silu_backward_reference(input_data, mse_grad);

        // Compare backward precision
        auto backward_metrics_kernel = compute_precision_metrics(grad_reference, input_grad_kernel);
        auto backward_metrics_composite = compute_precision_metrics(grad_reference, input_grad_composite);

        // Collect metrics for consolidation
        kernel_rmse_values[run] = backward_metrics_kernel.rmse;
        composite_rmse_values[run] = backward_metrics_composite.rmse;

        // Sanity check - both should be reasonably close to reference
        EXPECT_TRUE(backward_metrics_kernel.allclose_loose) << "Kernel backward pass too inaccurate";
        EXPECT_TRUE(backward_metrics_composite.allclose_loose) << "Composite backward pass too inaccurate";
    }

    // Compute consolidated statistics (no output)
    auto compute_mean = [](const std::vector<float>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
    };

    float kernel_mean_rmse = compute_mean(kernel_rmse_values);
    float composite_mean_rmse = compute_mean(composite_rmse_values);

    // Silent validation - just ensure both implementations work reasonably
    EXPECT_LT(kernel_mean_rmse, 1e-5f) << "Kernel implementation has poor precision";
    EXPECT_LT(composite_mean_rmse, 1e-5f) << "Composite implementation has poor precision";
}

// ============================================================================
// Section 4: metal::silu_bw program-cache hits
// ============================================================================
// A launch whose specs match a cached program reuses it, and only the buffer addresses of the new
// tensors are patched in. Each test clears the cache so that its first launch is the miss, and keeps
// every tensor of the earlier launches alive so that a later launch cannot land on a freed address:
// the hit must read and write the new buffers and leave the earlier ones untouched. The program key
// does not see which buffer an output is, so an in-place launch (preallocated_output == dL_dout) and a
// launch with a separate preallocated output share one program; the tests switch between the two on
// cache hits in both orders.
// ============================================================================

namespace {

using ttml::test_utils::device_with_empty_program_cache;
using ttml::test_utils::expect_new_buffers;
using ttml::test_utils::expect_unchanged;
using ttml::test_utils::two_core_group_rows_shape;
using ttml::test_utils::zeros_on_device;

struct SiluBwInputs {
    ttnn::Tensor input;
    ttnn::Tensor dL_dout;
    xt::xarray<float> input_stored;  // what the kernels read: the values rounded to bf16
    xt::xarray<float> dL_dout_stored;
};

SiluBwInputs make_silu_bw_inputs(const std::vector<uint32_t>& shape, uint32_t seed) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    auto input = core::from_xtensor(test_utils::make_uniform_xarray<float>(shape, -2.0F, 2.0F, seed), device);
    auto dL_dout = core::from_xtensor(test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed + 1U), device);
    auto input_stored = core::to_xtensor(input);
    auto dL_dout_stored = core::to_xtensor(dL_dout);
    return {input, dL_dout, std::move(input_stored), std::move(dL_dout_stored)};
}

void expect_silu_bw_matches_reference(const ttnn::Tensor& dL_da, const SiluBwInputs& in, const std::string& label) {
    const auto result = ttml::core::to_xtensor(dL_da);
    const auto expected = silu_backward_reference(in.input_stored, in.dL_dout_stored);
    EXPECT_TRUE(xt::allclose(result, expected, 1.0e-3F, 3e-2F)) << label;
}

}  // namespace

TEST_F(SiLUOpTest, SiluBw_CacheHitReadsFreshBuffers_TwoCoreGroups) {
    using namespace ttml;
    auto& device = device_with_empty_program_cache();
    const auto shape = two_core_group_rows_shape(64U);

    const auto a = make_silu_bw_inputs(shape, 11U);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto a_dL_da = metal::silu_bw(a.input, a.dL_dout);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "the first launch must compile a program";
    expect_silu_bw_matches_reference(a_dL_da, a, "launch A");
    const auto a_dL_da_result = core::to_xtensor(a_dL_da);

    // Same specs, new values, launch A's tensors still alive.
    const auto b = make_silu_bw_inputs(shape, 21U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto b_dL_da = metal::silu_bw(b.input, b.dL_dout);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch B must reuse launch A's program";
    expect_new_buffers({a.input, a.dL_dout, a_dL_da}, {b.input, b.dL_dout, b_dL_da}, "launch B");
    expect_silu_bw_matches_reference(b_dL_da, b, "launch B (cache hit)");

    expect_unchanged(a.input, a.input_stored, "launch A input");
    expect_unchanged(a.dL_dout, a.dL_dout_stored, "launch A dL_dout");
    expect_unchanged(a_dL_da, a_dL_da_result, "launch A dL_da");
}

// A launch without preallocated_output, then two with one, then one in place: the op must write into the
// preallocated tensor and return it, whether or not the program cache keys on its presence. The in-place
// launch hits the program of the separate preallocated output.
TEST_F(SiLUOpTest, SiluBw_PreallocatedOutputAfterAllocatedOutput_TwoCoreGroups) {
    using namespace ttml;
    auto& device = device_with_empty_program_cache();
    const auto shape = two_core_group_rows_shape(64U);

    const auto a = make_silu_bw_inputs(shape, 31U);
    const auto a_dL_da = metal::silu_bw(a.input, a.dL_dout);
    expect_silu_bw_matches_reference(a_dL_da, a, "launch A (op allocates the output)");
    const auto a_dL_da_result = core::to_xtensor(a_dL_da);

    const auto b = make_silu_bw_inputs(shape, 41U);
    const auto b_preallocated = zeros_on_device(shape);
    const auto b_dL_da = metal::silu_bw(b.input, b.dL_dout, b_preallocated);
    EXPECT_EQ(b_dL_da.buffer()->address(), b_preallocated.buffer()->address()) << "must return the preallocated output";
    expect_silu_bw_matches_reference(b_preallocated, b, "launch B (preallocated)");
    const auto b_dL_da_result = core::to_xtensor(b_preallocated);

    // The pattern of launch B again: a hit under any program-cache key.
    const auto c = make_silu_bw_inputs(shape, 51U);
    const auto c_preallocated = zeros_on_device(shape);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto c_dL_da = metal::silu_bw(c.input, c.dL_dout, c_preallocated);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch C must reuse launch B's program";
    EXPECT_EQ(c_dL_da.buffer()->address(), c_preallocated.buffer()->address()) << "must return the preallocated output";
    expect_new_buffers({b.input, b.dL_dout, b_preallocated}, {c.input, c.dL_dout, c_preallocated}, "launch C");
    expect_silu_bw_matches_reference(c_preallocated, c, "launch C (preallocated, cache hit)");
    const auto c_dL_da_result = core::to_xtensor(c_preallocated);

    // In place over dL_dout: the key of launch C, now with dL_dout's buffer passed twice.
    const auto d = make_silu_bw_inputs(shape, 61U);
    const auto d_dL_da = metal::silu_bw(d.input, d.dL_dout, d.dL_dout);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch D must reuse launch B's program";
    EXPECT_EQ(d_dL_da.buffer()->address(), d.dL_dout.buffer()->address()) << "must write into dL_dout";
    expect_new_buffers({c.input, c.dL_dout, c_preallocated}, {d.input, d.dL_dout, d.dL_dout}, "launch D");
    expect_silu_bw_matches_reference(d.dL_dout, d, "launch D (in place, cache hit)");

    expect_unchanged(a_dL_da, a_dL_da_result, "launch A dL_da");
    expect_unchanged(b_preallocated, b_dL_da_result, "launch B dL_da");
    expect_unchanged(b.input, b.input_stored, "launch B input");
    expect_unchanged(b.dL_dout, b.dL_dout_stored, "launch B dL_dout");
    expect_unchanged(c_preallocated, c_dL_da_result, "launch C dL_da");
    expect_unchanged(c.input, c.input_stored, "launch C input");
    expect_unchanged(c.dL_dout, c.dL_dout_stored, "launch C dL_dout");
}

// preallocated_output == dL_dout: the gradient is overwritten in place. The same buffer appears twice
// among the op's tensors, and each hit must still read and write its own call's buffers. The last launch
// hits the in-place program with a separate preallocated output.
TEST_F(SiLUOpTest, SiluBw_InPlaceIntoDLdout_CacheHit_TwoCoreGroups) {
    using namespace ttml;
    auto& device = device_with_empty_program_cache();
    const auto shape = two_core_group_rows_shape(64U);

    const auto a = make_silu_bw_inputs(shape, 71U);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto a_dL_da = metal::silu_bw(a.input, a.dL_dout, a.dL_dout);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "the first launch must compile a program";
    EXPECT_EQ(a_dL_da.buffer()->address(), a.dL_dout.buffer()->address()) << "must write into dL_dout";
    expect_silu_bw_matches_reference(a.dL_dout, a, "launch A (in place)");
    const auto a_dL_da_result = core::to_xtensor(a.dL_dout);

    const auto b = make_silu_bw_inputs(shape, 81U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto b_dL_da = metal::silu_bw(b.input, b.dL_dout, b.dL_dout);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch B must reuse launch A's program";
    EXPECT_EQ(b_dL_da.buffer()->address(), b.dL_dout.buffer()->address()) << "must write into dL_dout";
    expect_new_buffers({a.input, a.dL_dout}, {b.input, b.dL_dout}, "launch B");
    expect_silu_bw_matches_reference(b.dL_dout, b, "launch B (in place, cache hit)");
    const auto b_dL_da_result = core::to_xtensor(b.dL_dout);

    // A separate preallocated output: the key of the in-place launches, the output now a buffer of its own.
    const auto c = make_silu_bw_inputs(shape, 91U);
    const auto c_preallocated = zeros_on_device(shape);
    const auto entries_before_c = device.num_program_cache_entries();
    const auto c_dL_da = metal::silu_bw(c.input, c.dL_dout, c_preallocated);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_c) << "launch C must reuse launch A's program";
    EXPECT_EQ(c_dL_da.buffer()->address(), c_preallocated.buffer()->address()) << "must return the preallocated output";
    expect_new_buffers({b.input, b.dL_dout, b.dL_dout}, {c.input, c.dL_dout, c_preallocated}, "launch C");
    expect_silu_bw_matches_reference(c_preallocated, c, "launch C (preallocated, cache hit)");

    expect_unchanged(a.input, a.input_stored, "launch A input");
    expect_unchanged(a.dL_dout, a_dL_da_result, "launch A dL_dout (holds dL_da)");
    expect_unchanged(b.input, b.input_stored, "launch B input");
    expect_unchanged(b.dL_dout, b_dL_da_result, "launch B dL_dout (holds dL_da)");
    expect_unchanged(c.dL_dout, c.dL_dout_stored, "launch C dL_dout");
}

// A launch with preallocated_output, then two without: the op must allocate and return its own output
// again, whether or not the program cache keys on the preallocated output's presence.
TEST_F(SiLUOpTest, SiluBw_AllocatedOutputAfterPreallocatedOutput_TwoCoreGroups) {
    using namespace ttml;
    auto& device = device_with_empty_program_cache();
    const auto shape = two_core_group_rows_shape(64U);

    const auto a = make_silu_bw_inputs(shape, 101U);
    const auto a_preallocated = zeros_on_device(shape);
    const auto entries_before_miss = device.num_program_cache_entries();
    const auto a_dL_da = metal::silu_bw(a.input, a.dL_dout, a_preallocated);
    ASSERT_GT(device.num_program_cache_entries(), entries_before_miss) << "the first launch must compile a program";
    EXPECT_EQ(a_dL_da.buffer()->address(), a_preallocated.buffer()->address()) << "must return the preallocated output";
    expect_silu_bw_matches_reference(a_preallocated, a, "launch A (preallocated)");
    const auto a_dL_da_result = core::to_xtensor(a_preallocated);

    const auto b = make_silu_bw_inputs(shape, 111U);
    const auto b_dL_da = metal::silu_bw(b.input, b.dL_dout);
    expect_new_buffers({a.input, a.dL_dout, a_preallocated}, {b.input, b.dL_dout, b_dL_da}, "launch B");
    expect_silu_bw_matches_reference(b_dL_da, b, "launch B (op allocates the output)");
    const auto b_dL_da_result = core::to_xtensor(b_dL_da);

    // The pattern of launch B again: a hit under any program-cache key.
    const auto c = make_silu_bw_inputs(shape, 121U);
    const auto entries_before_hit = device.num_program_cache_entries();
    const auto c_dL_da = metal::silu_bw(c.input, c.dL_dout);
    EXPECT_EQ(device.num_program_cache_entries(), entries_before_hit) << "launch C must reuse launch B's program";
    expect_new_buffers({b.input, b.dL_dout, b_dL_da}, {c.input, c.dL_dout, c_dL_da}, "launch C");
    expect_silu_bw_matches_reference(c_dL_da, c, "launch C (op allocates the output, cache hit)");

    expect_unchanged(a_preallocated, a_dL_da_result, "launch A dL_da");
    expect_unchanged(a.input, a.input_stored, "launch A input");
    expect_unchanged(a.dL_dout, a.dL_dout_stored, "launch A dL_dout");
    expect_unchanged(b_dL_da, b_dL_da_result, "launch B dL_da");
    expect_unchanged(b.input, b.input_stored, "launch B input");
    expect_unchanged(b.dL_dout, b.dL_dout_stored, "launch B dL_dout");
}
