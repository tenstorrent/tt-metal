// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"

class NewtonSchulzIterationTest : public ::testing::Test {
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

void run_newton_schulz_iteration_test(
    uint32_t batch, uint32_t channels, uint32_t rows, uint32_t cols, float a, float b, float c) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    auto x = make_random_tensor(batch, channels, rows, cols, device);

    // newton_schulz_iteration(X, a, b, c) — fused Phase 1 + Phase 2 + Phase 3
    auto X_prime = metal::newton_schulz(x, a, b, c);

    // Reference: compute using standard ops
    // G = X @ X^T
    auto G = metal::gram_matmul(x);

    // H = c*G*G + b*G
    auto G_squared = ttnn::matmul(G, G);
    auto c_G_squared = ttnn::multiply(G_squared, c);
    auto b_G = ttnn::multiply(G, b);
    auto H = ttnn::add(c_G_squared, b_G);

    // X' = H @ X + a*X
    auto HX = ttnn::matmul(H, x);
    auto aX = ttnn::multiply(x, a);
    auto X_prime_ref = ttnn::add(HX, aX);

    auto ref_xt = core::to_xtensor(X_prime_ref);
    auto result_xt = core::to_xtensor(X_prime);

    ASSERT_EQ(result_xt.shape(), ref_xt.shape());

    std::cout << "Shape: [" << batch << ", " << channels << ", " << rows << ", " << cols << "], a=" << a << ", b=" << b
              << ", c=" << c << "\n";
    print_error_stats("newton_schulz vs reference", result_xt, ref_xt);

    float pcc = compute_pcc(result_xt, ref_xt);
    std::cout << "    PCC: " << pcc << "\n";
    EXPECT_GT(pcc, 0.999F) << "PCC too low — structural mismatch between newton_schulz and reference";

    xt::xarray<float> abs_diff = xt::abs(result_xt - ref_xt);
    xt::xarray<float> abs_expected = xt::abs(ref_xt) + 1e-8F;
    float mean_rel_error = xt::mean(abs_diff / abs_expected)();
    // Three chained bf16 matmuls (Phase 1 + Phase 2 + Phase 3) accumulate more
    // precision error at large sizes. Use a relaxed threshold for the full pipeline.
    EXPECT_LT(mean_rel_error, 0.25F) << "Mean relative error too high";
}

}  // namespace

// Test 1: a=1, b=0, c=0 -> X' should equal X
TEST_F(NewtonSchulzIterationTest, IdentityX_128) {
    run_newton_schulz_iteration_test(1, 1, 128, 128, 1.0F, 0.0F, 0.0F);
}

// Test 2: a=0, b=1, c=0 -> X' = GX
TEST_F(NewtonSchulzIterationTest, GX_128) {
    run_newton_schulz_iteration_test(1, 1, 128, 128, 0.0F, 1.0F, 0.0F);
}

// Test 3: a=0, b=0, c=1 -> X' = G^2 X
TEST_F(NewtonSchulzIterationTest, G2X_128) {
    run_newton_schulz_iteration_test(1, 1, 128, 128, 0.0F, 0.0F, 1.0F);
}

// Test 4: General case
TEST_F(NewtonSchulzIterationTest, General_128) {
    run_newton_schulz_iteration_test(1, 1, 128, 128, 0.9F, 0.5F, 0.3F);
}

// Test 5: Rectangular input (K > M)
TEST_F(NewtonSchulzIterationTest, Rectangular_128x256) {
    run_newton_schulz_iteration_test(1, 1, 128, 256, 0.9F, 0.5F, 0.3F);
}

// Test 6: Large 2048x5632 (production size)
TEST_F(NewtonSchulzIterationTest, Large2048x5632) {
    run_newton_schulz_iteration_test(1, 1, 2048, 5632, 0.9F, 0.5F, 0.3F);
}

// Test 7: Traced vs non-traced multi-iteration (5 iterations)
// Uses its own device with trace_region_size > 0.
TEST(NewtonSchulzTracedTest, TracedVsNonTraced_128) {
    using namespace ttml;
    auto device = ttnn::device::open_mesh_device(0, /*l1_small_size=*/200000, /*trace_region_size=*/4 * 1048576);
    device->enable_program_cache();

    auto& rng = autograd::ctx().get_generator();
    uint32_t seed = rng();
    xt::xarray<float> data = xt::empty<float>({(uint32_t)1, (uint32_t)1, (uint32_t)128, (uint32_t)256});
    core::parallel_generate(
        std::span{data.data(), data.size()}, []() { return std::uniform_real_distribution<float>(-1.0F, 1.0F); }, seed);
    auto x = core::from_xtensor(data, device.get());

    // Scale X down to prevent explosion over 4 iterations
    auto x_small = ttnn::multiply(x, 0.01F);
    x.deallocate();

    // Use a=1 (identity), b=0, c=0 so each iteration just copies X
    // This way we can verify traced == non-traced exactly regardless of iteration count.
    const float ta = 1.0F, tb = 0.0F, tc = 0.0F;
    constexpr int n_iters = 4;

    // Non-traced loop
    auto result_no_trace = metal::newton_schulz(x_small, ta, tb, tc, n_iters, /*use_trace=*/false);
    tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
    auto no_trace_xt = core::to_xtensor(result_no_trace);
    std::cout << "  Non-traced " << n_iters << "-iter mean: " << xt::mean(xt::abs(no_trace_xt))() << "\n";

    // Traced loop
    auto result_traced = metal::newton_schulz(x_small, ta, tb, tc, n_iters, /*use_trace=*/true);
    tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
    auto traced_xt = core::to_xtensor(result_traced);
    std::cout << "  Traced " << n_iters << "-iter mean: " << xt::mean(xt::abs(traced_xt))() << "\n";

    ASSERT_EQ(no_trace_xt.shape(), traced_xt.shape());

    float pcc = compute_pcc(no_trace_xt, traced_xt);
    std::cout << "  Traced vs non-traced PCC: " << pcc << "\n";

    xt::xarray<float> abs_diff = xt::abs(no_trace_xt - traced_xt);
    float max_abs_error = xt::amax(abs_diff)();
    std::cout << "  Max abs error: " << max_abs_error << "\n";

    EXPECT_GT(pcc, 0.999F) << "Traced and non-traced should produce matching results";

    result_no_trace.deallocate();
    result_traced.deallocate();
    x_small.deallocate();
    device->close();
}
