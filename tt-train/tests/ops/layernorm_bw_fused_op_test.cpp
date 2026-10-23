// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/layernorm_bw/layernorm_bw.hpp"
#include "metal/ops/layernorm_fw/layernorm_fw.hpp"
#include "test_utils/random_data.hpp"

// Reference implementation using xtensor
struct LayerNormCache {
    xt::xarray<float> x;
    xt::xarray<float> x_hat;
    xt::xarray<float> mu;
    xt::xarray<float> s;
    xt::xarray<float> gamma;
    xt::xarray<float> beta;
    uint32_t batch_size;
    uint32_t features;
    float eps;
};

std::tuple<xt::xarray<float>, LayerNormCache> layernorm_forward_reference(
    const xt::xarray<float>& x,
    const xt::xarray<float>& gamma,
    const xt::xarray<float>& beta,
    uint32_t batch_size,
    uint32_t features,
    float eps = 1e-6f) {
    // Reshape input to (batch_size, features) for easier manipulation
    auto x_reshaped = xt::reshape_view(x, {batch_size, features});

    // Compute mean along features axis
    xt::xarray<float> mu = xt::mean(x_reshaped, {1});

    // Compute variance along features axis
    xt::xarray<float> x_centered = x_reshaped - xt::view(mu, xt::all(), xt::newaxis());
    xt::xarray<float> var = xt::mean(xt::square(x_centered), {1});

    // Compute standard deviation with epsilon
    xt::xarray<float> s = xt::sqrt(var + eps);

    // Normalize
    xt::xarray<float> x_hat = x_centered / xt::view(s, xt::all(), xt::newaxis());

    // Scale and shift
    xt::xarray<float> y = x_hat * xt::view(gamma, xt::newaxis(), xt::all()) + xt::view(beta, xt::newaxis(), xt::all());

    // Flatten outputs back to 1D
    y = xt::flatten(y);
    x_hat = xt::flatten(x_hat);

    LayerNormCache cache{x, x_hat, mu, s, gamma, beta, batch_size, features, eps};
    return std::make_tuple(y, cache);
}

std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>> layernorm_backward_reference(
    const xt::xarray<float>& dy, const LayerNormCache& cache) {
    // Reshape to (batch_size, features)
    auto dy_reshaped = xt::reshape_view(dy, {cache.batch_size, cache.features});
    auto x_hat_reshaped = xt::reshape_view(cache.x_hat, {cache.batch_size, cache.features});

    // Compute dgamma and dbeta - sum over batch dimension
    xt::xarray<float> dgamma = xt::sum(dy_reshaped * x_hat_reshaped, {0});
    xt::xarray<float> dbeta = xt::sum(dy_reshaped, {0});

    // Compute dxhat = dy * gamma
    xt::xarray<float> dxhat = dy_reshaped * xt::view(cache.gamma, xt::newaxis(), xt::all());

    // Compute mean_dxhat - mean along features axis
    xt::xarray<float> mean_dxhat = xt::mean(dxhat, {1});

    // Compute mean_dxhat_xhat - mean along features axis
    xt::xarray<float> mean_dxhat_xhat = xt::mean(dxhat * x_hat_reshaped, {1});

    // Compute final dx
    xt::xarray<float> dx = (dxhat - xt::view(mean_dxhat, xt::all(), xt::newaxis()) -
                            x_hat_reshaped * xt::view(mean_dxhat_xhat, xt::all(), xt::newaxis())) /
                           xt::view(cache.s, xt::all(), xt::newaxis());

    // Flatten dx back to 1D
    dx = xt::flatten(dx);

    return std::make_tuple(dx, dgamma, dbeta);
}

class LayerNormBackwardOpTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// Helper function to compare metal kernel results against xtensor reference
static void CompareKernelVsXArray(
    const uint32_t batch_size,
    const uint32_t seq_len,
    const uint32_t heads,
    const uint32_t features,
    const int num_iterations = 3,
    // dx comes straight from the kernel (<=0.014 max abs error on device), while
    // dgamma/dbeta are host-side sums of bf16 per-row components whose noise grows
    // with rows*features (~0.29 at features~8k on device) - graded separately.
    const float dx_atol = 2e-2F,
    const float dgb_atol = 5e-2F) {
    using namespace ttml;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Generate test data using flattened 1D arrays
        uint32_t total_elements = batch_size * seq_len * heads * features;
        uint32_t combined_batch = batch_size * seq_len * heads;

        auto& rng = autograd::ctx().get_generator();
        uint32_t seed1 = rng();
        xt::xarray<float> x_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{total_elements}, -1.0F, 1.0F, seed1);

        uint32_t seed2 = rng();
        xt::xarray<float> gamma_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{features}, 0.0F, 1.0F, seed2);

        uint32_t seed3 = rng();
        xt::xarray<float> beta_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{features}, 0.0F, 1.0F, seed3);

        uint32_t seed4 = rng();
        xt::xarray<float> dy_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{total_elements}, -1.0F, 1.0F, seed4);

        // Compute reference results
        auto [y_ref, cache] =
            layernorm_forward_reference(x_data, gamma_data, beta_data, combined_batch, features, 1e-6f);
        auto [dx_ref, dgamma_ref, dbeta_ref] = layernorm_backward_reference(dy_data, cache);

        // Copy and reshape data to 4D for device tensors (copy to avoid corrupting reference data)
        xt::xarray<float> x_4d = x_data;
        x_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> gamma_4d = gamma_data;
        gamma_4d.reshape({1, 1, 1, features});
        xt::xarray<float> dy_4d = dy_data;
        dy_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> mu_4d = cache.mu;
        mu_4d.reshape({batch_size, heads, seq_len, 1});

        // Compute rstd from s and reshape
        xt::xarray<float> rstd_data = 1.0f / cache.s;
        rstd_data.reshape({batch_size, heads, seq_len, 1});

        // Create tensors on device using from_xtensor
        auto input_tensor = core::from_xtensor(x_4d, &autograd::ctx().get_device());
        auto gamma_tensor = core::from_xtensor(gamma_4d, &autograd::ctx().get_device());
        auto mean_tensor = core::from_xtensor(mu_4d, &autograd::ctx().get_device());
        auto rstd_tensor = core::from_xtensor(rstd_data, &autograd::ctx().get_device());
        auto dy_tensor = core::from_xtensor(dy_4d, &autograd::ctx().get_device());

        auto output_tensors = metal::layernorm_bw(input_tensor, gamma_tensor, mean_tensor, rstd_tensor, dy_tensor);

        auto metal_dx_xtensor = core::to_xtensor(output_tensors[0].value());
        auto metal_dgamma_xtensor = core::to_xtensor(output_tensors[1].value());
        auto metal_dbeta_xtensor = core::to_xtensor(output_tensors[2].value());

        // Flatten metal results for comparison
        xt::xarray<float> metal_dx_flat = xt::flatten(metal_dx_xtensor);
        xt::xarray<float> metal_dgamma_flat = xt::flatten(metal_dgamma_xtensor);
        xt::xarray<float> metal_dbeta_flat = xt::flatten(metal_dbeta_xtensor);

        // Compare shapes
        ASSERT_EQ(dx_ref.shape(), metal_dx_flat.shape());
        ASSERT_EQ(dgamma_ref.shape(), metal_dgamma_flat.shape());
        ASSERT_EQ(dbeta_ref.shape(), metal_dbeta_flat.shape());

        // Compare values
        EXPECT_TRUE(xt::allclose(metal_dx_flat, dx_ref, 1.0e-3F, dx_atol))
            << "dx max_abs_diff=" << xt::amax(xt::abs(metal_dx_flat - dx_ref))();
        EXPECT_TRUE(xt::allclose(metal_dgamma_flat, dgamma_ref, 1.0e-3F, dgb_atol))
            << "dgamma max_abs_diff=" << xt::amax(xt::abs(metal_dgamma_flat - dgamma_ref))();
        EXPECT_TRUE(xt::allclose(metal_dbeta_flat, dbeta_ref, 1.0e-3F, dgb_atol))
            << "dbeta max_abs_diff=" << xt::amax(xt::abs(metal_dbeta_flat - dbeta_ref))();
    }
}

// ============================================================================
// Test Cases - LayerNorm Backward Metal Kernel vs XArray Reference
// ============================================================================

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_OneTile) {
    CompareKernelVsXArray(1, 13, 1, 20);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_TwoIncompleteTiles) {
    CompareKernelVsXArray(1, 32, 1, 33);
}

TEST_F(LayerNormBackwardOpTest, NIGHTLY_MetalLayerNormBw_LargeFeatures_NoL1Fit) {
    CompareKernelVsXArray(3, 273, 1, 8462, 3, /*dx_atol=*/2e-2F, /*dgb_atol=*/4e-1F);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_DoesNotFitInL1_WtNotDivisibleBy4) {
    CompareKernelVsXArray(3, 100, 1, 8191, 10, /*dx_atol=*/2e-2F, /*dgb_atol=*/4e-1F);
}

TEST_F(LayerNormBackwardOpTest, MetalLayerNormBw_OneTilePerRow) {
    CompareKernelVsXArray(1, 19, 1, 213, 10);
}

// ============================================================================
// Skipping grads in the LayerNorm backward kernel
// ============================================================================
// layernorm_bw takes a flag for each of dx, dgamma and dbeta. A skipped grad is nullopt; the kernel drops its compute
// and its output CB and, for dx, gamma and both row sums. Rows cover dx off on both kernel paths (everything in L1,
// with several rows per core so gamma is read once, and not), dgamma and dbeta each skipped, and each output as the
// one the writer takes its tile size from.
// ============================================================================

namespace {

struct LayerNormBwSkipCase {
    std::string name;
    std::vector<uint32_t> shape;
    bool compute_grad_input = true;
    bool compute_grad_gamma = true;
    bool compute_grad_beta = true;
};

// gtest prints a failing row's parameter through this; without it the row prints as raw bytes.
void PrintTo(const LayerNormBwSkipCase& c, std::ostream* os) {
    *os << c.name;
}

struct LayerNormBwInputs {
    ttnn::Tensor input;
    ttnn::Tensor gamma;
    ttnn::Tensor mean;
    ttnn::Tensor rstd;
    ttnn::Tensor dL_dout;
};

LayerNormBwInputs make_layernorm_bw_inputs(const std::vector<uint32_t>& shape, uint32_t seed) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    const std::vector<uint32_t> gamma_shape{1U, 1U, 1U, shape[3]};
    auto input = core::from_xtensor(test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed), device);
    auto gamma = core::from_xtensor(test_utils::make_uniform_xarray<float>(gamma_shape, 0.0F, 1.0F, seed + 1U), device);
    auto beta = core::from_xtensor(test_utils::make_uniform_xarray<float>(gamma_shape, 0.0F, 1.0F, seed + 2U), device);
    auto dL_dout = core::from_xtensor(test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed + 3U), device);
    const auto fw = metal::layernorm_fw(input, gamma, beta, /* epsilon */ 1e-5F, /* return_mean_rstd */ true);
    return {input, gamma, fw[1].value(), fw[2].value(), dL_dout};
}

}  // namespace

class LayerNormBwSkipTest : public ::testing::TestWithParam<LayerNormBwSkipCase> {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// Each requested grad must equal the one from the full backward bit for bit: the kernel does the same per-tile
// arithmetic either way. The flags are compile-time, so the first skipping run must compile its own program; a second
// run with other live inputs must hit that program and read and write the new buffers.
TEST_P(LayerNormBwSkipTest, RequestedGradsMatchFullBackward) {
    using namespace ttml;
    const LayerNormBwSkipCase& c = GetParam();
    auto* device = &autograd::ctx().get_device();
    device->enable_program_cache();

    // Both sets stay alive, so the second cannot land at the first's addresses.
    const LayerNormBwInputs first = make_layernorm_bw_inputs(c.shape, 1U);
    const LayerNormBwInputs second = make_layernorm_bw_inputs(c.shape, 11U);

    const auto run =
        [](const LayerNormBwInputs& in, bool compute_grad_input, bool compute_grad_gamma, bool compute_grad_beta) {
            return metal::layernorm_bw(
                in.input,
                in.gamma,
                in.mean,
                in.rstd,
                in.dL_dout,
                compute_grad_input,
                compute_grad_gamma,
                compute_grad_beta);
        };
    const std::array<bool, 3> requested{c.compute_grad_input, c.compute_grad_gamma, c.compute_grad_beta};
    // The full backward also compiles the dgamma / dbeta reductions, so the skipping runs below only add their own
    // program.
    const auto to_host = [](const std::vector<std::optional<ttnn::Tensor>>& grads) {
        std::vector<xt::xarray<float>> host;
        for (const auto& grad : grads) {
            host.push_back(core::to_xtensor(grad.value()));
        }
        return host;
    };
    const auto expected_first = to_host(run(first, true, true, true));
    const auto expected_second = to_host(run(second, true, true, true));

    const auto entries_before = device->num_program_cache_entries();
    const auto skip_first = run(first, c.compute_grad_input, c.compute_grad_gamma, c.compute_grad_beta);
    const auto entries_after_first = device->num_program_cache_entries();
    EXPECT_EQ(entries_after_first, entries_before + 1U)
        << c.name << ": the skipping run must compile exactly one program of its own";
    const auto skip_second = run(second, c.compute_grad_input, c.compute_grad_gamma, c.compute_grad_beta);
    EXPECT_EQ(device->num_program_cache_entries(), entries_after_first)
        << c.name << ": the second skipping run at identical shapes missed the program cache";

    const std::array<const char*, 3> grad_names{"dx", "dgamma", "dbeta"};
    for (const auto& [grads, expected, label] :
         {std::tuple{&skip_first, &expected_first, "first"}, std::tuple{&skip_second, &expected_second, "second"}}) {
        ASSERT_EQ(grads->size(), 3U) << c.name;
        for (size_t i = 0; i < 3U; ++i) {
            ASSERT_EQ((*grads)[i].has_value(), requested[i]) << c.name << ": " << grad_names[i] << " presence";
            if (requested[i]) {
                EXPECT_TRUE(xt::all(xt::equal(core::to_xtensor((*grads)[i].value()), (*expected)[i])))
                    << c.name << ": " << grad_names[i] << " on the " << label
                    << " inputs differs from the full backward";
            }
        }
    }
}

// C = 127: Wt = 4, so a partial block of 3 and masking; 256 tile rows give every core more than one row on Wormhole
// and Blackhole. C = 16383: Wt = 512, so input, dL_dout and x_hat alone (3 MB) exceed L1 even without gamma.
INSTANTIATE_TEST_SUITE_P(
    LayerNormBwSkipCases,
    LayerNormBwSkipTest,
    ::testing::Values(
        LayerNormBwSkipCase{
            .name = "InputOnly_FitsInL1",
            .shape = {1U, 1U, 8192U, 127U},
            .compute_grad_gamma = false,
            .compute_grad_beta = false},
        LayerNormBwSkipCase{.name = "NoInput_FitsInL1", .shape = {1U, 1U, 8192U, 127U}, .compute_grad_input = false},
        LayerNormBwSkipCase{
            .name = "BetaOnly_FitsInL1",
            .shape = {1U, 1U, 8192U, 127U},
            .compute_grad_input = false,
            .compute_grad_gamma = false},
        LayerNormBwSkipCase{
            .name = "NoBeta_DoesNotFitInL1", .shape = {1U, 1U, 64U, 16383U}, .compute_grad_beta = false},
        LayerNormBwSkipCase{
            .name = "GammaOnly_DoesNotFitInL1",
            .shape = {1U, 1U, 64U, 16383U},
            .compute_grad_input = false,
            .compute_grad_beta = false}),
    [](const ::testing::TestParamInfo<LayerNormBwSkipCase>& info) { return info.param.name; });
