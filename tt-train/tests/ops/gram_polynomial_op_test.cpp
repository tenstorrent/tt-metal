// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

class GramPolynomialTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

namespace {

ttnn::Tensor make_random_tensor(uint32_t N, uint32_t C, uint32_t H, uint32_t W, ttnn::distributed::MeshDevice* device) {
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> data = xt::empty<float>({N, C, H, W});
    ttml::core::parallel_generate(
        std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, seed);
    return ttml::core::from_xtensor(data, device);
}

void print_error_stats(
    const std::string& label, const xt::xarray<float>& result_xt, const xt::xarray<float>& expected_xt) {
    xt::xarray<float> abs_diff = xt::abs(result_xt - expected_xt);
    xt::xarray<float> abs_expected = xt::abs(expected_xt) + 1e-8F;
    xt::xarray<float> rel_diff = abs_diff / abs_expected;

    float max_abs_error = xt::amax(abs_diff)();
    float mean_abs_error = xt::mean(abs_diff)();
    float max_rel_error = xt::amax(rel_diff)();
    float mean_rel_error = xt::mean(rel_diff)();

    auto flat_abs_diff = xt::flatten(abs_diff);
    size_t flat_idx = 0;
    float max_val = 0.0F;
    for (size_t i = 0; i < flat_abs_diff.size(); ++i) {
        if (flat_abs_diff(i) > max_val) {
            max_val = flat_abs_diff(i);
            flat_idx = i;
        }
    }
    auto flat_result = xt::flatten(result_xt);
    auto flat_expected = xt::flatten(expected_xt);

    std::cout << "  [" << label << "]\n"
              << "    Max abs error:  " << max_abs_error << "\n"
              << "    Mean abs error: " << mean_abs_error << "\n"
              << "    Max rel error:  " << max_rel_error << "\n"
              << "    Mean rel error: " << mean_rel_error << "\n"
              << "    At max error (flat=" << flat_idx << "): "
              << "got=" << flat_result(flat_idx) << " ref=" << flat_expected(flat_idx) << "\n";
}

// Compute PCC (Pearson Correlation Coefficient) between two arrays.
float compute_pcc(const xt::xarray<float>& a, const xt::xarray<float>& b) {
    auto a_flat = xt::flatten(a);
    auto b_flat = xt::flatten(b);
    float mean_a = xt::mean(a_flat)();
    float mean_b = xt::mean(b_flat)();
    auto a_centered = a_flat - mean_a;
    auto b_centered = b_flat - mean_b;
    float cov = xt::sum(a_centered * b_centered)();
    float std_a = std::sqrt(xt::sum(a_centered * a_centered)());
    float std_b = std::sqrt(xt::sum(b_centered * b_centered)());
    return cov / (std_a * std_b + 1e-12F);
}

// Test gram_polynomial Phase 2 independently:
// Use the same G (from gram_matmul) for both reference and our implementation.
// Reference: compute b*G + c*G^2 using ttnn ops on the same G.
void run_gram_polynomial_test(uint32_t batch, uint32_t channels, uint32_t rows, uint32_t cols, float b, float c) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    auto x = make_random_tensor(batch, channels, rows, cols, device);

    // gram_polynomial(X, b, c) — fused Phase 1 + Phase 2
    auto H_fused = metal::gram_polynomial(x, b, c);

    // Reference: compute G and H_ref on device using standard ops,
    // then compare on CPU (xtensor). Use gram_matmul for G (same Phase 1).
    auto G = metal::gram_matmul(x);

    // Build reference: H_ref = c * G @ G + b * G using ttnn ops
    auto G_squared = ttnn::matmul(G, G);
    auto c_G_squared = ttnn::multiply(G_squared, c);
    auto b_G = ttnn::multiply(G, b);
    auto H_ref = ttnn::add(c_G_squared, b_G);

    auto ref_xt = core::to_xtensor(H_ref);
    auto result_xt = core::to_xtensor(H_fused);

    ASSERT_EQ(result_xt.shape(), ref_xt.shape());

    std::cout << "Shape: [" << batch << ", " << channels << ", " << rows << ", " << cols << "], b=" << b << ", c=" << c
              << "\n";
    print_error_stats("gram_polynomial vs reference", result_xt, ref_xt);

    // PCC check: correlation > 0.999 ensures outputs are structurally identical,
    // even if absolute values differ due to bf16 matmul precision differences
    // between our custom kernel and ttnn::matmul.
    float pcc = compute_pcc(result_xt, ref_xt);
    std::cout << "    PCC: " << pcc << "\n";
    EXPECT_GT(pcc, 0.999F) << "PCC too low — structural mismatch between gram_polynomial and reference";

    // Also check mean relative error stays bounded (allow for bf16 precision)
    xt::xarray<float> abs_diff = xt::abs(result_xt - ref_xt);
    xt::xarray<float> abs_expected = xt::abs(ref_xt) + 1e-8F;
    float mean_rel_error = xt::mean(abs_diff / abs_expected)();
    EXPECT_LT(mean_rel_error, 0.05F) << "Mean relative error too high";
}

}  // namespace

// Test 1: b=1, c=0 -> H should equal G (exercises only the bG path)
TEST_F(GramPolynomialTest, IdentityG_128) {
    run_gram_polynomial_test(1, 1, 128, 128, 1.0F, 0.0F);
}

// Test 2: b=0, c=1 -> H should equal G^2 (exercises only the matmul path)
TEST_F(GramPolynomialTest, GSquared_128) {
    run_gram_polynomial_test(1, 1, 128, 128, 0.0F, 1.0F);
}

// Test 3: General case with both coefficients
TEST_F(GramPolynomialTest, General_128) {
    run_gram_polynomial_test(1, 1, 128, 128, 0.5F, 0.3F);
}

// Test 4: Rectangular input (K > M)
TEST_F(GramPolynomialTest, Rectangular_128x256) {
    run_gram_polynomial_test(1, 1, 128, 256, 0.5F, 0.3F);
}

// Test 5: Larger size 256x256
TEST_F(GramPolynomialTest, Square256) {
    run_gram_polynomial_test(1, 1, 256, 256, 0.5F, 0.3F);
}

// Test 6: Large 2048x5632 (production size)
TEST_F(GramPolynomialTest, Large2048x5632) {
    run_gram_polynomial_test(1, 1, 2048, 5632, 0.5F, 0.3F);
}
