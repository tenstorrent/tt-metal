// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/experimental/minimal_matmul/minimal_matmul.hpp"

class GramMatmulTest : public ::testing::Test {
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

// Helper: create a random bf16 tensor on device with the given 4D shape.
ttnn::Tensor make_random_tensor(uint32_t N, uint32_t C, uint32_t H, uint32_t W, ttnn::distributed::MeshDevice* device) {
    auto& rng = ttml::autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> data = xt::empty<float>({N, C, H, W});
    ttml::core::parallel_generate(
        std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, seed);
    return ttml::core::from_xtensor(data, device);
}

// Print error statistics between two xtensor arrays.
void print_error_stats(
    const std::string& label, const xt::xarray<float>& result_xt, const xt::xarray<float>& expected_xt) {
    xt::xarray<float> abs_diff = xt::abs(result_xt - expected_xt);
    xt::xarray<float> abs_expected = xt::abs(expected_xt) + 1e-8F;
    xt::xarray<float> rel_diff = abs_diff / abs_expected;

    float max_abs_error = xt::amax(abs_diff)();
    float mean_abs_error = xt::mean(abs_diff)();
    float max_rel_error = xt::amax(rel_diff)();
    float mean_rel_error = xt::mean(rel_diff)();

    // Find position of max absolute error
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

// Run gram_matmul(X) and compare against minimal_matmul(X, X^T) and ttnn::matmul(X, X^T).
void run_gram_matmul_vs_minimal(uint32_t N, uint32_t C, uint32_t H, uint32_t W) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();

    // X has shape [N, C, H, W] where W >= H (cols >= rows).
    auto x = make_random_tensor(N, C, H, W, device);

    // X^T for the reference implementations
    auto xt_tensor = ttnn::transpose(x, -2, -1);

    // Reference: ttnn::matmul(X, X^T)
    auto ref = ttnn::matmul(x, xt_tensor);

    // minimal_matmul(X, X^T)
    auto minimal = ttnn::experimental::minimal_matmul(
        x,
        xt_tensor,
        /*bias_tensor=*/std::nullopt,
        /*fused_activation=*/std::nullopt,
        /*config=*/std::nullopt);

    // gram_matmul(X) -- single tensor, transpose happens internally
    auto gram = metal::gram_matmul(x);

    auto ref_xt = core::to_xtensor(ref);
    auto minimal_xt = core::to_xtensor(minimal);
    auto gram_xt = core::to_xtensor(gram);

    ASSERT_EQ(minimal_xt.shape(), ref_xt.shape());
    ASSERT_EQ(gram_xt.shape(), ref_xt.shape());

    std::cout << "Shape: [" << N << ", " << C << ", " << H << ", " << W << "]\n";
    print_error_stats("minimal_matmul vs ttnn::matmul", minimal_xt, ref_xt);
    print_error_stats("gram_matmul    vs ttnn::matmul", gram_xt, ref_xt);
    print_error_stats("gram_matmul    vs minimal_matmul", gram_xt, minimal_xt);

    // PCC check: different block sizes cause different bf16 accumulation order,
    // so allclose is too strict. PCC > 0.999 ensures structural correctness.
    auto a_flat = xt::flatten(gram_xt);
    auto b_flat = xt::flatten(minimal_xt);
    float mean_a = xt::mean(a_flat)();
    float mean_b = xt::mean(b_flat)();
    auto a_centered = a_flat - mean_a;
    auto b_centered = b_flat - mean_b;
    float cov = xt::sum(a_centered * b_centered)();
    float std_a = std::sqrt(xt::sum(a_centered * a_centered)());
    float std_b = std::sqrt(xt::sum(b_centered * b_centered)());
    float pcc = cov / (std_a * std_b + 1e-12F);
    std::cout << "    PCC (gram vs minimal): " << pcc << "\n";
    EXPECT_GT(pcc, 0.999F) << "PCC too low for shape [" << N << ", " << C << ", " << H << ", " << W << "]";
}

}  // namespace

TEST_F(GramMatmulTest, Square128x128) {
    // X: [1, 1, 128, 128], output: [1, 1, 128, 128]
    run_gram_matmul_vs_minimal(1, 1, 128, 128);
}

TEST_F(GramMatmulTest, Large2048x5632) {
    // X: [1, 1, 2048, 5632], output: [1, 1, 2048, 2048]
    run_gram_matmul_vs_minimal(1, 1, 2048, 5632);
}
