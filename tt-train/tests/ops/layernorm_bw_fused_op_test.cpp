// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/layernorm_bw/device/layernorm_bw_device_operation.hpp"
#include "metal/ops/layernorm_bw/layernorm_bw.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/device_operation.hpp"

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

namespace {

constexpr uint32_t kCacheTestRows = 32U;
constexpr uint32_t kCacheTestWidth = 64U;
constexpr float kCacheTestEpsilon = 1.0e-3F;

ttnn::Tensor make_cache_test_tensor(
    const std::vector<float>& data,
    const ttnn::Shape& shape,
    ttnn::distributed::MeshDevice* device,
    const tt::tt_metal::Alignment& alignment = {}) {
    const auto layout = tt::tt_metal::TensorLayout(
        ttnn::DataType::BFLOAT16, ttnn::PageConfig(ttnn::Layout::TILE), ttnn::DRAM_MEMORY_CONFIG, alignment);
    return ttnn::Tensor::from_vector(data, tt::tt_metal::TensorSpec(shape, layout), device);
}

std::vector<float> make_cache_test_data(size_t count, float offset, uint32_t multiplier) {
    std::vector<float> data(count);
    for (size_t index = 0; index < count; ++index) {
        data[index] = offset + 0.0078125F * static_cast<float>((index * multiplier) % 31U);
    }
    return data;
}

struct BackwardCacheTestData {
    std::vector<float> input;
    std::vector<float> gamma;
    std::vector<float> mean;
    std::vector<float> rstd;
    std::vector<float> upstream_grad;
    std::vector<float> dx;
    std::vector<float> dgamma_components;
};

BackwardCacheTestData make_backward_cache_test_data(
    const float offset, const uint32_t rows = kCacheTestRows, const uint32_t width = kCacheTestWidth) {
    BackwardCacheTestData data;
    data.input = make_cache_test_data(rows * width, offset, 7U);
    data.gamma = make_cache_test_data(width, 0.75F + offset, 3U);
    const auto beta = make_cache_test_data(width, -0.125F + offset, 5U);
    data.upstream_grad = make_cache_test_data(rows * width, 0.0625F + offset, 11U);

    xt::xarray<float> input_array = xt::adapt(data.input, std::array<size_t, 1>{data.input.size()});
    xt::xarray<float> gamma_array = xt::adapt(data.gamma, std::array<size_t, 1>{data.gamma.size()});
    xt::xarray<float> beta_array = xt::adapt(beta, std::array<size_t, 1>{beta.size()});
    [[maybe_unused]] const auto [unused_output, cache] =
        layernorm_forward_reference(input_array, gamma_array, beta_array, rows, width, kCacheTestEpsilon);
    xt::xarray<float> upstream_grad_array =
        xt::adapt(data.upstream_grad, std::array<size_t, 1>{data.upstream_grad.size()});
    [[maybe_unused]] const auto [dx, unused_dgamma, unused_dbeta] =
        layernorm_backward_reference(upstream_grad_array, cache);

    data.mean.assign(cache.mu.begin(), cache.mu.end());
    data.rstd.resize(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        data.rstd[row] = 1.0F / cache.s[row];
    }
    data.dx.assign(dx.begin(), dx.end());
    data.dgamma_components.resize(data.input.size());
    for (size_t index = 0; index < data.input.size(); ++index) {
        data.dgamma_components[index] = data.upstream_grad[index] * cache.x_hat[index];
    }
    return data;
}

void expect_cache_test_backward_matches(
    const std::vector<ttnn::Tensor>& result, const BackwardCacheTestData& expected) {
    const auto dx = ttml::core::to_vector<float>(result[0]);
    const auto dgamma = ttml::core::to_vector<float>(result[1]);
    const auto dbeta = ttml::core::to_vector<float>(result[2]);
    ASSERT_EQ(dx.size(), expected.dx.size());
    ASSERT_EQ(dgamma.size(), expected.dgamma_components.size());
    ASSERT_EQ(dbeta.size(), expected.upstream_grad.size());
    for (size_t index = 0; index < dx.size(); ++index) {
        EXPECT_NEAR(dx[index], expected.dx[index], 5.0e-2F) << "dx index=" << index;
        EXPECT_NEAR(dgamma[index], expected.dgamma_components[index], 5.0e-2F) << "dgamma index=" << index;
        EXPECT_NEAR(dbeta[index], expected.upstream_grad[index], 2.0e-2F) << "dbeta index=" << index;
    }
}

template <typename Operation>
void expect_validation_accepts(
    const typename Operation::operation_attributes_t& attributes,
    const typename Operation::tensor_args_t& tensor_args) {
    using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Operation>;
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_miss(attributes, tensor_args));
    EXPECT_NO_THROW(Adapter::validate_on_program_cache_hit(attributes, tensor_args));
}

template <typename Operation>
void expect_validation_rejects(
    const typename Operation::operation_attributes_t& attributes,
    const typename Operation::tensor_args_t& tensor_args,
    const std::string_view diagnostic) {
    using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<Operation>;
    EXPECT_THAT(
        ([&] { Adapter::validate_on_program_cache_miss(attributes, tensor_args); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(std::string(diagnostic))));
    EXPECT_THAT(
        ([&] { Adapter::validate_on_program_cache_hit(attributes, tensor_args); }),
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(std::string(diagnostic))));
}

}  // namespace

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

TEST_F(LayerNormBackwardOpTest, ProgramCacheSeparatesPaddingAndRebindsAllAddresses) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    const ttnn::Shape input_shape({1U, 1U, kCacheTestRows, kCacheTestWidth});
    const ttnn::Shape parameter_shape({1U, 1U, 1U, kCacheTestWidth});
    const ttnn::Shape stats_shape({1U, 1U, kCacheTestRows, 1U});
    const tt::tt_metal::Alignment overpadded_alignment({1U, 1U, 64U, 32U});
    const std::vector<float> output_sentinel(input_shape.volume(), -7.0F);

    const auto run = [&](float offset, const tt::tt_metal::Alignment& alignment) {
        auto data = make_backward_cache_test_data(offset);
        auto input = make_cache_test_tensor(data.input, input_shape, device, alignment);
        auto gamma = make_cache_test_tensor(data.gamma, parameter_shape, device);
        auto mean = make_cache_test_tensor(data.mean, stats_shape, device, alignment);
        auto rstd = make_cache_test_tensor(data.rstd, stats_shape, device, alignment);
        auto upstream_grad = make_cache_test_tensor(data.upstream_grad, input_shape, device, alignment);
        auto dx = make_cache_test_tensor(output_sentinel, input_shape, device, alignment);
        auto dgamma = make_cache_test_tensor(output_sentinel, input_shape, device, alignment);
        auto dbeta = make_cache_test_tensor(output_sentinel, input_shape, device, alignment);
        auto result = ttnn::prim::ttml_layernorm_bw(input, gamma, mean, rstd, upstream_grad, dx, dgamma, dbeta);
        return std::make_tuple(
            std::move(result),
            std::move(input),
            std::move(gamma),
            std::move(mean),
            std::move(rstd),
            std::move(upstream_grad),
            std::move(dx),
            std::move(dgamma),
            std::move(dbeta),
            std::move(data));
    };

    const auto entries_before_standard = device->num_program_cache_entries();
    auto
        [standard_result,
         standard_input,
         standard_gamma,
         standard_mean,
         standard_rstd,
         standard_upstream_grad,
         standard_dx,
         standard_dgamma,
         standard_dbeta,
         standard_data] = run(0.25F, {});
    (void)standard_input;
    (void)standard_gamma;
    (void)standard_mean;
    (void)standard_rstd;
    (void)standard_upstream_grad;
    const auto entries_after_standard = device->num_program_cache_entries();
    EXPECT_GT(entries_after_standard, entries_before_standard);
    ASSERT_EQ(standard_result.size(), 3U);
    EXPECT_EQ(standard_result[0].buffer()->address(), standard_dx.buffer()->address());
    EXPECT_EQ(standard_result[1].buffer()->address(), standard_dgamma.buffer()->address());
    EXPECT_EQ(standard_result[2].buffer()->address(), standard_dbeta.buffer()->address());
    expect_cache_test_backward_matches(standard_result, standard_data);

    const auto entries_before_padded = device->num_program_cache_entries();
    auto
        [padded_result,
         padded_input,
         padded_gamma,
         padded_mean,
         padded_rstd,
         padded_upstream_grad,
         padded_dx,
         padded_dgamma,
         padded_dbeta,
         padded_data] = run(0.5F, overpadded_alignment);
    const auto entries_after_padded = device->num_program_cache_entries();
    EXPECT_GT(entries_after_padded, entries_before_padded)
        << "LayerNorm backward reused a program compiled for different padded geometry";
    expect_cache_test_backward_matches(padded_result, padded_data);

    const auto entries_before_replay = device->num_program_cache_entries();
    auto
        [replay_result,
         replay_input,
         replay_gamma,
         replay_mean,
         replay_rstd,
         replay_upstream_grad,
         replay_dx,
         replay_dgamma,
         replay_dbeta,
         replay_data] = run(0.875F, overpadded_alignment);
    const auto entries_after_replay = device->num_program_cache_entries();
    EXPECT_EQ(entries_after_replay, entries_before_replay)
        << "same-spec LayerNorm backward replay should reuse its cached program";
    ASSERT_NE(replay_input.buffer()->address(), padded_input.buffer()->address());
    ASSERT_NE(replay_gamma.buffer()->address(), padded_gamma.buffer()->address());
    ASSERT_NE(replay_mean.buffer()->address(), padded_mean.buffer()->address());
    ASSERT_NE(replay_rstd.buffer()->address(), padded_rstd.buffer()->address());
    ASSERT_NE(replay_upstream_grad.buffer()->address(), padded_upstream_grad.buffer()->address());
    ASSERT_NE(replay_dx.buffer()->address(), padded_dx.buffer()->address());
    ASSERT_NE(replay_dgamma.buffer()->address(), padded_dgamma.buffer()->address());
    ASSERT_NE(replay_dbeta.buffer()->address(), padded_dbeta.buffer()->address());
    expect_cache_test_backward_matches(replay_result, replay_data);
}

TEST_F(LayerNormBackwardOpTest, StatsUseSingleTileWidthForWidthAlignedInput) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    device->clear_program_cache();

    constexpr uint32_t rows = 64U;
    constexpr uint32_t width = 64U;
    const ttnn::Shape input_shape({1U, 1U, rows, width});
    const ttnn::Shape parameter_shape({1U, 1U, 1U, width});
    const ttnn::Shape stats_shape({1U, 1U, rows, 1U});
    const tt::tt_metal::Alignment width_aligned_input({1U, 1U, 32U, 64U});

    const auto run = [&](const float offset) {
        auto data = make_backward_cache_test_data(offset, rows, width);
        auto input = make_cache_test_tensor(data.input, input_shape, device, width_aligned_input);
        auto gamma = make_cache_test_tensor(data.gamma, parameter_shape, device);
        auto mean = make_cache_test_tensor(data.mean, stats_shape, device);
        auto rstd = make_cache_test_tensor(data.rstd, stats_shape, device);
        auto upstream_grad = make_cache_test_tensor(data.upstream_grad, input_shape, device, width_aligned_input);
        auto result = ttnn::prim::ttml_layernorm_bw(input, gamma, mean, rstd, upstream_grad);
        return std::make_pair(std::move(result), std::move(data));
    };

    const auto entries_before_first = device->num_program_cache_entries();
    auto [first_result, first_data] = run(0.25F);
    const auto entries_after_first = device->num_program_cache_entries();
    EXPECT_GT(entries_after_first, entries_before_first);
    ASSERT_EQ(first_result.size(), 3U);
    EXPECT_EQ(first_result[0].padded_shape(), input_shape);
    EXPECT_EQ(first_result[1].padded_shape(), input_shape);
    EXPECT_EQ(first_result[2].padded_shape(), input_shape);
    expect_cache_test_backward_matches(first_result, first_data);

    const auto entries_before_replay = device->num_program_cache_entries();
    auto [replay_result, replay_data] = run(0.5F);
    const auto entries_after_replay = device->num_program_cache_entries();
    EXPECT_EQ(entries_after_replay, entries_before_replay);
    expect_cache_test_backward_matches(replay_result, replay_data);
}

TEST_F(LayerNormBackwardOpTest, ValidatesContractsOnProgramCacheMissAndHit) {
    using Operation = ttml::metal::ops::layernorm_bw::device::LayerNormBackwardDeviceOperation;

    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    device->clear_program_cache();

    const ttnn::Shape input_shape({1U, 1U, kCacheTestRows, kCacheTestWidth});
    const ttnn::Shape parameter_shape({1U, 1U, 1U, kCacheTestWidth});
    const ttnn::Shape stats_shape({1U, 1U, kCacheTestRows, 1U});
    const tt::tt_metal::Alignment overpadded_alignment({1U, 1U, 64U, 32U});
    const auto data = make_backward_cache_test_data(0.25F);

    auto input = make_cache_test_tensor(data.input, input_shape, device, overpadded_alignment);
    auto overwide_input =
        make_cache_test_tensor(data.input, input_shape, device, tt::tt_metal::Alignment({1U, 1U, 32U, 96U}));
    auto gamma = make_cache_test_tensor(data.gamma, parameter_shape, device);
    auto mean = make_cache_test_tensor(data.mean, stats_shape, device, overpadded_alignment);
    auto rstd = make_cache_test_tensor(data.rstd, stats_shape, device, overpadded_alignment);
    auto upstream_grad = make_cache_test_tensor(data.upstream_grad, input_shape, device, overpadded_alignment);
    auto rank3_gamma = make_cache_test_tensor(data.gamma, ttnn::Shape({1U, 1U, kCacheTestWidth}), device);
    auto oversized_mean = make_cache_test_tensor(
        std::vector<float>(2U * stats_shape.volume(), 0.0F),
        ttnn::Shape({1U, 1U, 2U * kCacheTestRows, 1U}),
        device,
        overpadded_alignment);
    auto oversized_upstream_grad = make_cache_test_tensor(
        std::vector<float>(2U * input_shape.volume(), 0.0F),
        ttnn::Shape({1U, 1U, 2U * kCacheTestRows, kCacheTestWidth}),
        device,
        overpadded_alignment);
    auto oversized_dx = make_cache_test_tensor(
        std::vector<float>(2U * input_shape.volume(), -7.0F),
        ttnn::Shape({1U, 1U, 2U * kCacheTestRows, kCacheTestWidth}),
        device,
        overpadded_alignment);
    auto oversized_dgamma = make_cache_test_tensor(
        std::vector<float>(2U * input_shape.volume(), -7.0F),
        ttnn::Shape({1U, 1U, 2U * kCacheTestRows, kCacheTestWidth}),
        device,
        overpadded_alignment);
    auto oversized_dbeta = make_cache_test_tensor(
        std::vector<float>(2U * input_shape.volume(), -7.0F),
        ttnn::Shape({1U, 1U, 2U * kCacheTestRows, kCacheTestWidth}),
        device,
        overpadded_alignment);
    auto overwide_mean =
        make_cache_test_tensor(data.mean, stats_shape, device, tt::tt_metal::Alignment({1U, 1U, 64U, 64U}));

    const auto narrow_tile_layout = tt::tt_metal::TensorLayout(
        ttnn::DataType::BFLOAT16,
        ttnn::PageConfig(ttnn::Layout::TILE, tt::tt_metal::Tile({16U, 16U})),
        ttnn::DRAM_MEMORY_CONFIG);
    auto narrow_tile_input =
        ttnn::Tensor::from_vector(data.input, tt::tt_metal::TensorSpec(input_shape, narrow_tile_layout), device);

    const Operation::operation_attributes_t attributes{};
    const Operation::tensor_args_t valid_args{
        .input = input, .gamma = gamma, .mean = mean, .rstd = rstd, .dL_dout = upstream_grad};
    expect_validation_accepts<Operation>(attributes, valid_args);

    struct ValidationCase {
        Operation::tensor_args_t tensor_args;
        std::string_view diagnostic;
    };

    std::vector<ValidationCase> invalid_cases;
    const auto add_case = [&](const auto& mutate, const std::string_view diagnostic) {
        auto tensor_args = valid_args;
        mutate(tensor_args);
        invalid_cases.push_back({std::move(tensor_args), diagnostic});
    };
    add_case([&](auto& args) { args.gamma = rank3_gamma; }, "Gamma tensor must have shape");
    add_case([&](auto& args) { args.mean = oversized_mean; }, "Mean tensor must have logical shape");
    add_case([&](auto& args) { args.rstd = oversized_mean; }, "Rstd tensor must have logical shape");
    add_case([&](auto& args) { args.mean = overwide_mean; }, "Mean tensor must have logical shape");
    add_case(
        [&](auto& args) { args.dL_dout = oversized_upstream_grad; },
        "dL_dout TensorSpec must match the input TensorSpec");
    add_case(
        [&](auto& args) { args.preallocated_dx = oversized_dx; },
        "Preallocated dx TensorSpec must match the input TensorSpec");
    add_case(
        [&](auto& args) { args.preallocated_dgamma_components = oversized_dgamma; },
        "Preallocated dgamma_components TensorSpec must match the input TensorSpec");
    add_case(
        [&](auto& args) { args.preallocated_dbeta_components = oversized_dbeta; },
        "Preallocated dbeta_components TensorSpec must match the input TensorSpec");
    add_case(
        [&](auto& args) { args.input = narrow_tile_input; },
        "Tensor 'Input' must use the canonical non-transposed 32x32 tile");
    add_case(
        [&](auto& args) { args.input = overwide_input; },
        "Input tensor may be overpadded in height but must use canonical width padding");

    for (const auto& test_case : invalid_cases) {
        SCOPED_TRACE(test_case.diagnostic);
        expect_validation_rejects<Operation>(attributes, test_case.tensor_args, test_case.diagnostic);
    }

    const auto entries_before_warmup = device->num_program_cache_entries();
    const auto valid_result = ttnn::prim::ttml_layernorm_bw(input, gamma, mean, rstd, upstream_grad);
    const auto entries_after_warmup = device->num_program_cache_entries();
    EXPECT_GT(entries_after_warmup, entries_before_warmup);
    ASSERT_EQ(valid_result.size(), 3U);
    EXPECT_EQ(valid_result[0].tensor_spec(), input.tensor_spec());
    EXPECT_EQ(valid_result[1].tensor_spec(), input.tensor_spec());
    EXPECT_EQ(valid_result[2].tensor_spec(), input.tensor_spec());
    expect_cache_test_backward_matches(valid_result, data);
}
