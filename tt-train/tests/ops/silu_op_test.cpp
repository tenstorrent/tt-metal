// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/random.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/silu_bw/silu_bw.hpp"
#include "ops/unary_ops.hpp"
#include "test_utils/bf16_ulp.hpp"
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
// Section 1: ops::silu vs a float64 reference
// ============================================================================
// Like the other activation tests, the backward is seeded with ones (result->backward()), so the input
// gradient is silu'(x) itself and is compared element by element against float64 on the bf16 values the
// device read:
//     silu(x)  = x * s(x)
//     silu'(x) = s(x) * (1 + x * s(-x)),  s(x) = 1 / (1 + exp(-x))
// s(-x) is computed directly rather than as 1 - s(x), which loses precision for large x.
// ============================================================================

namespace {

xt::xarray<double> sigmoid_reference(const xt::xarray<double>& x) {
    return 1.0 / (1.0 + xt::exp(-x));
}

xt::xarray<double> silu_reference(const xt::xarray<double>& x) {
    return x * sigmoid_reference(x);
}

xt::xarray<double> silu_grad_reference(const xt::xarray<double>& x) {
    return sigmoid_reference(x) * (1.0 + x * sigmoid_reference(-x));
}

// Reference implementation of the SiLU backward pass: dL/dx = dL/dout * silu'(x).
xt::xarray<float> silu_backward_reference(const xt::xarray<float>& x, const xt::xarray<float>& grad) {
    const xt::xarray<double> x64 = xt::cast<double>(x);
    return xt::cast<float>(xt::cast<double>(grad) * silu_grad_reference(x64));
}

// ttnn::silu and ttnn::silu_bw (with a gradient of ones) are within 1 bf16 ULP of the exact result in every
// element (measured 0.89 and 0.5002).
constexpr double kSiluMaxUlp = 1.0;
constexpr double kSiluGradMaxUlp = 1.0;
// metal::silu_bw on Blackhole with a gradient of ones over [-8, 8] (Section 3): 1.63 ULP at the largest
// |silu'|, p99 3.38 ULP.
constexpr double kMetalSiluBwMaxUlp = 2.5;
constexpr double kMetalSiluBwMaxUlpP99 = 5.0;

void CompareKernelVsReference(const xt::xarray<float>& input_data) {
    using namespace ttml;

    auto input = autograd::create_tensor(
        core::from_xtensor(input_data, &autograd::ctx().get_device()), /* requires_grad */ true);
    const xt::xarray<double> input_stored = xt::cast<double>(core::to_xtensor(input->get_value()));

    auto result = ops::silu(input);
    test_utils::expect_each_within_bf16_ulp(
        core::to_xtensor(result->get_value()), silu_reference(input_stored), "silu forward", kSiluMaxUlp);

    result->backward();
    test_utils::expect_each_within_bf16_ulp(
        core::to_xtensor(input->get_grad()), silu_grad_reference(input_stored), "silu backward", kSiluGradMaxUlp);
}

/**
 * Helper function to compare ops::silu against the float64 reference
 *
 * This function tests both forward and backward passes to ensure:
 * 1. Forward pass: within kSiluMaxUlp of silu(x) in every element
 * 2. Backward pass (gradient of ones): within kSiluGradMaxUlp of silu'(x) in every element
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
// Section 2: ops::silu vs the float64 reference across shapes
// ============================================================================
// These tests systematically compare ops::silu against the float64
// reference across different scenarios:
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
// Section 3: both backward paths of ops::silu vs the float64 reference
// ============================================================================
// use_composite_bw = true (the default) backpropagates with ttnn::silu_bw, which is within 1 bf16 ULP in every element.
// false runs ttml::metal::silu_bw, which rounds sigmoid and each of four later steps to bf16; it is close
// overall but loses most significant bits where silu'(x) crosses zero (x ~ -1.28), so it is bounded by
// its error at the largest |silu'| and its p99 instead of element by element.
// ============================================================================

TEST_F(SiLUOpTest, SiLU_Backward_BothPaths_vs_Reference) {
    using namespace ttml;

    // nanollama-like shape; [-8, 8] covers the zero of silu' and both saturation tails.
    const std::vector<uint32_t> shape = {2, 1, 64, 512};
    const auto input_data = test_utils::make_uniform_xarray<float>(shape, -8.0F, 8.0F, 7U);

    for (const bool use_composite_bw : {true, false}) {
        const std::string label = use_composite_bw ? "ttnn::silu_bw" : "metal::silu_bw";
        auto input = autograd::create_tensor(
            core::from_xtensor(input_data, &autograd::ctx().get_device()), /* requires_grad */ true);
        const xt::xarray<double> expected = silu_grad_reference(xt::cast<double>(core::to_xtensor(input->get_value())));

        auto result = ops::silu(input, use_composite_bw);
        result->backward();
        const auto grad = core::to_xtensor(input->get_grad());
        if (use_composite_bw) {
            test_utils::expect_each_within_bf16_ulp(grad, expected, label, kSiluGradMaxUlp);
        } else {
            test_utils::expect_within_bf16_ulp(grad, expected, label, kMetalSiluBwMaxUlp, kMetalSiluBwMaxUlpP99);
        }
        autograd::ctx().reset_graph();
    }
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
