// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/swiglu_op.hpp"

#include <gtest/gtest.h>

#include <xtensor-blas/xlinalg.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"

class SwiGLUOpTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// ============================================================================
// Section 1: SwiGLU Forward Pass Test
// ============================================================================
// This test validates the SwiGLU forward pass implementation
//
// Test methodology:
// 1. Create test tensors x, w1, w2, w3
// 2. Compute SwiGLU forward pass
// 3. Verify the result is finite and has expected shape
// ============================================================================

namespace {

/**
 * Reference implementation of SwiGLU forward pass using xtensor
 * SwiGLU(x, w1, w2, w3) = (silu(x @ w1) * (x @ w3)) @ w2
 *
 * @param x  Input tensor [B, N, S, C]
 * @param w1 Weight tensor [1, 1, C, H]
 * @param w2 Weight tensor [1, 1, H, C]
 * @param w3 Weight tensor [1, 1, C, H]
 * @return   SwiGLU activation result [B, N, S, C]
 */
xt::xarray<float> swiglu_forward_reference(
    const xt::xarray<float>& x, const xt::xarray<float>& w1, const xt::xarray<float>& w2, const xt::xarray<float>& w3) {
    // Strip the first two singleton dimensions from weight tensors:
    auto w1_mat = xt::squeeze(w1, {0, 1});  // [C, H]
    auto w2_mat = xt::squeeze(w2, {0, 1});  // [H, C]
    auto w3_mat = xt::squeeze(w3, {0, 1});  // [C, H]

    // x @ w1: -> [B, N, S, H]
    auto xw1 = xt::linalg::tensordot(x, w1_mat, {3}, {0});
    // x @ w3: -> [B, N, S, H]
    auto xw3 = xt::linalg::tensordot(x, w3_mat, {3}, {0});

    // SiLU activation: z * sigmoid(z)
    // Note: Must return xt::xarray to force evaluation, not a lazy expression
    auto silu = [](const xt::xarray<float>& t) -> xt::xarray<float> {
        auto sigmoid = 1.0f / (1.0f + xt::exp(-t));
        return t * sigmoid;
    };

    // Gate: [B, N, S, H]
    auto gated = silu(xw1) * xw3;

    // gated @ w2: -> [B, N, S, C]
    return xt::linalg::tensordot(gated, w2_mat, {3}, {0});
}

void CompareKernelVsReference(
    const xt::xarray<float>& input_data,
    const xt::xarray<float>& w1_data,
    const xt::xarray<float>& w2_data,
    const xt::xarray<float>& w3_data,
    ttml::ops::SwiGLUAlgorithm algorithm = ttml::ops::SwiGLUAlgorithm::AUTO) {
    using namespace ttml;

    // Create input tensors for kernel implementation
    auto input_kernel = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
    auto w1_kernel = autograd::create_tensor(core::from_xtensor(w1_data, &autograd::ctx().get_device()));
    auto w2_kernel = autograd::create_tensor(core::from_xtensor(w2_data, &autograd::ctx().get_device()));
    auto w3_kernel = autograd::create_tensor(core::from_xtensor(w3_data, &autograd::ctx().get_device()));

    // Forward pass - kernel implementation with specified algorithm
    auto result_kernel = ops::swiglu(input_kernel, w1_kernel, w2_kernel, w3_kernel, algorithm);
    result_kernel->get_value();
    auto result_kernel_xtensor = core::to_xtensor(result_kernel->get_value());

    // Forward pass - reference implementation
    auto result_reference = swiglu_forward_reference(input_data, w1_data, w2_data, w3_data);

    // Verify shapes match
    EXPECT_EQ(result_kernel_xtensor.shape(), result_reference.shape())
        << "Shape mismatch between kernel and reference results";
    // Verify both results are finite
    EXPECT_TRUE(xt::all(xt::isfinite(result_kernel_xtensor))) << "SwiGLU kernel result contains NaN or Inf values";
    EXPECT_TRUE(xt::all(xt::isfinite(result_reference))) << "SwiGLU reference result contains NaN or Inf values";

    // We validate using relative L2 error instead of elementwise allclose().
    //
    // SwiGLU involves two matmuls + SiLU, so the output magnitude scales with
    // hidden_dim and tensor size. Combined with BF16 internal math (≈1e-2 relative
    // noise per element), elementwise tolerances become unstable: they may fail on
    // large tensors (outliers) or fail on small ones (tiny reference values).
    //
    // Relative L2 is scale-invariant and reflects the total numerical deviation of
    // the tensor, giving consistent results across all shapes. A threshold of 0.01
    // is standard for BF16 vs FP32 reference comparisons.
    auto diff = result_kernel_xtensor - result_reference;

    // Compute L2 norms.
    float diff_l2 = std::sqrt(xt::sum(xt::square(diff))());
    float ref_l2 = std::sqrt(xt::sum(xt::square(result_reference))());
    const float eps = 1e-12f;
    float rel_l2 = diff_l2 / (ref_l2 + eps);

    // Check the error bound.
    const float tolerance = 1e-2f;
    EXPECT_LT(rel_l2, tolerance) << "Relative L2 error too large: " << rel_l2 << " (expected < " << tolerance << ")";
}

/**
 * Helper function to test SwiGLU forward pass with given input shape
 *
 * @param input_shape Input tensor shape [B, N, S, C] where:
 *   - B: batch size
 *   - N: number of channels (usually 1 for transformers)
 *   - S: sequence length (height for transformers)
 *   - C: feature dimension (width/embedding dimension)
 * @param hidden_dim Hidden dimension for the weight matrices
 * @param algorithm Algorithm to use (default: TRUE_FLASH for testing the optimized path)
 */
static void CompareKernelVsReferenceWithShape(
    const std::vector<uint32_t>& input_shape,
    const uint32_t hidden_dim,
    ttml::ops::SwiGLUAlgorithm algorithm = ttml::ops::SwiGLUAlgorithm::TRUE_FLASH) {
    using namespace ttml;

    // Generate random input data using parallel_generate (following RMSNorm pattern)
    auto& rng = autograd::ctx().get_generator();

    const float bound = 1.0f;

    xt::xarray<float> input_data = xt::empty<float>(input_shape);
    core::parallel_generate<float>(
        input_data, [bound]() { return std::uniform_real_distribution<float>(-bound, bound); }, /* seed */ rng());

    // Create weight tensors: w1, w3 [1, 1, C, H], w2 [1, 1, H, C]
    const uint32_t input_dim = input_shape.back();
    std::vector<uint32_t> w1_w3_shape = {1, 1, input_dim, hidden_dim};
    std::vector<uint32_t> w2_shape = {1, 1, hidden_dim, input_dim};

    xt::xarray<float> w1_data = xt::empty<float>(w1_w3_shape);
    core::parallel_generate<float>(
        w1_data, [bound]() { return std::uniform_real_distribution<float>(-bound, bound); }, /* seed */ rng());

    xt::xarray<float> w2_data = xt::empty<float>(w2_shape);
    core::parallel_generate<float>(
        w2_data, [bound]() { return std::uniform_real_distribution<float>(-bound, bound); }, /* seed */ rng());
    xt::xarray<float> w3_data = xt::empty<float>(w1_w3_shape);
    core::parallel_generate<float>(
        w3_data, [bound]() { return std::uniform_real_distribution<float>(-bound, bound); }, /* seed */ rng());

    CompareKernelVsReference(input_data, w1_data, w2_data, w3_data, algorithm);
}

}  // namespace

// ============================================================================
// Section 2: SwiGLU Kernel vs Reference Implementation Tests
// ============================================================================
// These tests compare the SwiGLU kernel implementation against
// the reference implementation to ensure correctness.
// Only tests where C and hidden_dim are divisible by 32 are included.
//
// TODO: Add tests for masking in C and H (mask_w, mask_hw) when implementation supports it
// ============================================================================

// 1. Basic single tile test: 1x1x32x32, hidden_dim=32
TEST_F(SwiGLUOpTest, SwiGLU_Basic_1x1x32x32) {
    CompareKernelVsReferenceWithShape({1, 1, 32, 32}, 32);
}

// 2. Multi-tile width test: 1x1x32x64, hidden_dim=64 (C > 32)
TEST_F(SwiGLUOpTest, SwiGLU_MultiTile_1x1x32x64) {
    CompareKernelVsReferenceWithShape({1, 1, 32, 64}, 64);
}

// 3. Multi-tile height test: 1x1x64x32, hidden_dim=32 (S > 32)
TEST_F(SwiGLUOpTest, SwiGLU_MultiTile_1x1x64x32) {
    CompareKernelVsReferenceWithShape({1, 1, 64, 32}, 32);
}

// 4. Multi-batch test: 8x1x32x32, hidden_dim=32 (B != 32)
TEST_F(SwiGLUOpTest, SwiGLU_MultiBatch_8x1x32x32) {
    CompareKernelVsReferenceWithShape({8, 1, 32, 32}, 32);
}

// 5. Medium test: 2x1x32x128, hidden_dim=128
TEST_F(SwiGLUOpTest, SwiGLU_Medium_2x1x32x128) {
    CompareKernelVsReferenceWithShape({2, 1, 32, 128}, 128);
}

// 6. Large test: 4x1x64x256, hidden_dim=256
TEST_F(SwiGLUOpTest, SwiGLU_Large_4x1x64x256) {
    CompareKernelVsReferenceWithShape({4, 1, 64, 256}, 256);
}

TEST_F(SwiGLUOpTest, SwiGLU_RepeatedRuns_NoHang) {
    using namespace ttml;

    auto& device = autograd::ctx().get_device();

    // Create a batch size that will force row imbalance:
    // Use enough rows to fill multiple grid rows with uneven distribution
    // This ensures some logical grid rows have cores with different workloads
    const uint32_t batch = 100;

    const std::vector<uint32_t> input_shape = {batch, 1, 32, 64};
    const uint32_t hidden_dim = 128U;

    auto& rng = autograd::ctx().get_generator();
    const float bound = 1.0f;

    xt::xarray<float> input_data = xt::empty<float>(input_shape);
    core::parallel_generate<float>(
        input_data, [bound]() { return std::uniform_real_distribution<float>(-bound, bound); }, rng());

    std::vector<uint32_t> w_shape = {1, 1, input_shape.back(), hidden_dim};
    std::vector<uint32_t> w2_shape = {1, 1, hidden_dim, input_shape.back()};

    xt::xarray<float> w1_data = xt::empty<float>(w_shape);
    core::parallel_generate<float>(
        w1_data, [bound]() { return std::uniform_real_distribution<float>(-bound, bound); }, rng());
    xt::xarray<float> w2_data = xt::empty<float>(w2_shape);
    core::parallel_generate<float>(
        w2_data, [bound]() { return std::uniform_real_distribution<float>(-bound, bound); }, rng());
    xt::xarray<float> w3_data = xt::empty<float>(w_shape);
    core::parallel_generate<float>(
        w3_data, [bound]() { return std::uniform_real_distribution<float>(-bound, bound); }, rng());

    auto input_tensor = autograd::create_tensor(core::from_xtensor(input_data, &device));
    auto w1_tensor = autograd::create_tensor(core::from_xtensor(w1_data, &device));
    auto w2_tensor = autograd::create_tensor(core::from_xtensor(w2_data, &device));
    auto w3_tensor = autograd::create_tensor(core::from_xtensor(w3_data, &device));

    auto reference = swiglu_forward_reference(input_data, w1_data, w2_data, w3_data);

    const float tolerance = 1e-2f;
    for (int iteration = 0; iteration < 3; ++iteration) {
        auto output_tensor =
            ops::swiglu(input_tensor, w1_tensor, w2_tensor, w3_tensor, ops::SwiGLUAlgorithm::TRUE_FLASH);
        auto value = output_tensor->get_value();
        auto output_xtensor = core::to_xtensor(value);

        EXPECT_EQ(output_xtensor.shape(), reference.shape()) << "Unexpected output shape on iteration " << iteration;
        EXPECT_TRUE(xt::all(xt::isfinite(output_xtensor))) << "Non-finite tensor detected on iteration " << iteration;

        auto diff = output_xtensor - reference;
        float diff_l2 = std::sqrt(xt::sum(xt::square(diff))());
        float ref_l2 = std::sqrt(xt::sum(xt::square(reference))());
        const float rel_l2 = diff_l2 / (ref_l2 + 1e-12f);
        EXPECT_LT(rel_l2, tolerance) << "Relative L2 error too large on iteration " << iteration;
    }
}

// 7. Large test: 2x1x128x512, hidden_dim=512
TEST_F(SwiGLUOpTest, SwiGLU_Large_2x1x128x512) {
    CompareKernelVsReferenceWithShape({2, 1, 128, 512}, 512);
}

// 8. Very large dimensions test: 1x1x1024x1024, hidden_dim=1024
TEST_F(SwiGLUOpTest, NIGHTLY_SwiGLU_VeryLarge_1x1x1024x1024) {
    CompareKernelVsReferenceWithShape({1, 1, 1024, 1024}, 1024);
}

// 9. NanoLlama-like shape: 64x1x256x384, hidden_dim=1024
TEST_F(SwiGLUOpTest, NIGHTLY_SwiGLU_NanoLlama_64x1x256x384) {
    CompareKernelVsReferenceWithShape({64, 1, 256, 384}, 1024);
}

// 10. Edge case: Unbalanced workload where some cores get more rows than others
// This tests the padding synchronization mechanism in multicast.
// 57 rows on a 56-core grid means 1 core gets 2 rows, 55 cores get 1 row.
TEST_F(SwiGLUOpTest, SwiGLU_UnbalancedWorkload_57x1x32x32) {
    CompareKernelVsReferenceWithShape({57, 1, 32, 32}, 32);
}

// ============================================================================
// Section 3: Shape Validation Tests
// ============================================================================
// These tests verify that invalid weight shapes are properly rejected.
// Fused SwiGLU expects: W1[embed, hidden], W3[embed, hidden], W2[hidden, embed]
// Using wrong layout (e.g., LinearLayer's [out, in]) should fail with clear error.
// ============================================================================

// 10. Shape mismatch: W1 has wrong layout [hidden, embed] instead of [embed, hidden]
TEST_F(SwiGLUOpTest, SwiGLU_ShapeMismatch_W1WrongLayout) {
    using namespace ttml;

    const size_t embed_dim = 64;
    const size_t hidden_dim = 128;

    // Input: [1, 1, 32, embed_dim]
    std::vector<size_t> input_shape = {1, 1, 32, embed_dim};
    std::vector<size_t> w1_wrong_shape = {1, 1, hidden_dim, embed_dim};  // WRONG: [hidden, embed]
    std::vector<size_t> w2_shape = {1, 1, hidden_dim, embed_dim};        // Correct: [hidden, embed]
    std::vector<size_t> w3_wrong_shape = {1, 1, hidden_dim, embed_dim};  // WRONG: [hidden, embed]

    xt::xarray<float> input_data = xt::ones<float>(input_shape);
    xt::xarray<float> w1_wrong = xt::ones<float>(w1_wrong_shape);
    xt::xarray<float> w2_data = xt::ones<float>(w2_shape);
    xt::xarray<float> w3_wrong = xt::ones<float>(w3_wrong_shape);

    auto input = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
    auto w1 = autograd::create_tensor(core::from_xtensor(w1_wrong, &autograd::ctx().get_device()));
    auto w2 = autograd::create_tensor(core::from_xtensor(w2_data, &autograd::ctx().get_device()));
    auto w3 = autograd::create_tensor(core::from_xtensor(w3_wrong, &autograd::ctx().get_device()));

    // Should throw due to shape validation: W1[-2]=128 != input[-1]=64
    // Capture stdout to suppress expected TT_FATAL critical log messages (tt-logger writes to stdout)
    testing::internal::CaptureStdout();
    EXPECT_THROW(ops::swiglu(input, w1, w2, w3), std::exception);
    testing::internal::GetCapturedStdout();  // Discard captured output
}

// 11. Shape mismatch: W3 doesn't match W1
TEST_F(SwiGLUOpTest, SwiGLU_ShapeMismatch_W3DoesntMatchW1) {
    using namespace ttml;

    const size_t embed_dim = 64;
    const size_t hidden_dim = 128;

    std::vector<size_t> input_shape = {1, 1, 32, embed_dim};
    std::vector<size_t> w1_shape = {1, 1, embed_dim, hidden_dim};
    std::vector<size_t> w2_shape = {1, 1, hidden_dim, embed_dim};
    std::vector<size_t> w3_wrong_shape = {1, 1, embed_dim, hidden_dim * 2};  // WRONG: different hidden

    xt::xarray<float> input_data = xt::ones<float>(input_shape);
    xt::xarray<float> w1_data = xt::ones<float>(w1_shape);
    xt::xarray<float> w2_data = xt::ones<float>(w2_shape);
    xt::xarray<float> w3_wrong = xt::ones<float>(w3_wrong_shape);

    auto input = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
    auto w1 = autograd::create_tensor(core::from_xtensor(w1_data, &autograd::ctx().get_device()));
    auto w2 = autograd::create_tensor(core::from_xtensor(w2_data, &autograd::ctx().get_device()));
    auto w3 = autograd::create_tensor(core::from_xtensor(w3_wrong, &autograd::ctx().get_device()));

    // Should throw due to W3 shape not matching W1
    // Capture stdout to suppress expected TT_FATAL critical log messages (tt-logger writes to stdout)
    testing::internal::CaptureStdout();
    EXPECT_THROW(ops::swiglu(input, w1, w2, w3), std::exception);
    testing::internal::GetCapturedStdout();  // Discard captured output
}

// 12. Shape mismatch: W2 has wrong dimensions
TEST_F(SwiGLUOpTest, SwiGLU_ShapeMismatch_W2WrongDimensions) {
    using namespace ttml;

    const size_t embed_dim = 64;
    const size_t hidden_dim = 128;

    std::vector<size_t> input_shape = {1, 1, 32, embed_dim};
    std::vector<size_t> w1_shape = {1, 1, embed_dim, hidden_dim};
    std::vector<size_t> w2_wrong_shape = {
        1, 1, embed_dim, hidden_dim};  // WRONG: [embed, hidden] instead of [hidden, embed]
    std::vector<size_t> w3_shape = {1, 1, embed_dim, hidden_dim};

    xt::xarray<float> input_data = xt::ones<float>(input_shape);
    xt::xarray<float> w1_data = xt::ones<float>(w1_shape);
    xt::xarray<float> w2_wrong = xt::ones<float>(w2_wrong_shape);
    xt::xarray<float> w3_data = xt::ones<float>(w3_shape);

    auto input = autograd::create_tensor(core::from_xtensor(input_data, &autograd::ctx().get_device()));
    auto w1 = autograd::create_tensor(core::from_xtensor(w1_data, &autograd::ctx().get_device()));
    auto w2 = autograd::create_tensor(core::from_xtensor(w2_wrong, &autograd::ctx().get_device()));
    auto w3 = autograd::create_tensor(core::from_xtensor(w3_data, &autograd::ctx().get_device()));

    // Should throw due to W2 shape validation
    // Capture stdout to suppress expected TT_FATAL critical log messages (tt-logger writes to stdout)
    testing::internal::CaptureStdout();
    EXPECT_THROW(ops::swiglu(input, w1, w2, w3), std::exception);
    testing::internal::GetCapturedStdout();  // Discard captured output
}
