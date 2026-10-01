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
#include "metal/ops/layernorm_fw/device/layernorm_fw_device_operation.hpp"
#include "metal/ops/layernorm_fw/layernorm_fw.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/device_operation.hpp"

// Reference implementation using xtensor
std::tuple<xt::xarray<float>, xt::xarray<float>, xt::xarray<float>> layernorm_forward_reference_(
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

    // Compute reciprocal standard deviation (rstd)
    xt::xarray<float> rstd = 1.0f / xt::sqrt(var + eps);

    // Normalize
    xt::xarray<float> x_hat = x_centered * xt::view(rstd, xt::all(), xt::newaxis());

    // Scale and shift
    xt::xarray<float> y = x_hat * xt::view(gamma, xt::newaxis(), xt::all()) + xt::view(beta, xt::newaxis(), xt::all());

    // Flatten outputs back to 1D
    y = xt::flatten(y);

    return std::make_tuple(y, mu, rstd);
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

void expect_cache_test_forward_matches(
    const std::vector<std::optional<ttnn::Tensor>>& result,
    const std::vector<float>& input,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    const uint32_t rows = kCacheTestRows,
    const uint32_t width = kCacheTestWidth) {
    xt::xarray<float> input_array = xt::adapt(input, std::array<size_t, 1>{input.size()});
    xt::xarray<float> gamma_array = xt::adapt(gamma, std::array<size_t, 1>{gamma.size()});
    xt::xarray<float> beta_array = xt::adapt(beta, std::array<size_t, 1>{beta.size()});
    const auto [output_ref, mean_ref, rstd_ref] =
        layernorm_forward_reference_(input_array, gamma_array, beta_array, rows, width, kCacheTestEpsilon);

    const auto output = xt::flatten(ttml::core::to_xtensor(result[0].value()));
    const auto mean = xt::flatten(ttml::core::to_xtensor(result[1].value()));
    const auto rstd = xt::flatten(ttml::core::to_xtensor(result[2].value()));
    EXPECT_TRUE(xt::allclose(output, output_ref, 1.0e-3F, 5.0e-2F));
    EXPECT_TRUE(xt::allclose(mean, mean_ref, 1.0e-3F, 5.0e-2F));
    EXPECT_TRUE(xt::allclose(rstd, rstd_ref, 1.0e-3F, 5.0e-2F));
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

class LayerNormForwardOpTest : public ::testing::Test {
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
    uint32_t batch_size,
    const uint32_t seq_len,
    const uint32_t heads,
    const uint32_t features,
    const int num_iterations = 3) {
    using namespace ttml;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Generate test data using xtensor
        uint32_t total_elements = batch_size * seq_len * heads * features;
        uint32_t combined_batch = batch_size * seq_len * heads;

        auto& rng = autograd::ctx().get_generator();
        uint32_t seed1 = rng();
        xt::xarray<float> x_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{total_elements}, -1.0F, 1.0F, seed1);

        uint32_t seed2 = rng();
        xt::xarray<float> gamma_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{features}, 0.5F, 1.5F, seed2);

        uint32_t seed3 = rng();
        xt::xarray<float> beta_data =
            test_utils::make_uniform_xarray<float>(std::array<std::size_t, 1>{features}, -0.1F, 0.1F, seed3);

        // Compute reference results
        auto [y_ref, mu_ref, rstd_ref] =
            layernorm_forward_reference_(x_data, gamma_data, beta_data, combined_batch, features, 1e-6f);

        // Copy and reshape data to 4D for device tensors (copy to avoid corrupting reference data)
        xt::xarray<float> x_4d = x_data;
        x_4d.reshape({batch_size, heads, seq_len, features});
        xt::xarray<float> gamma_4d = gamma_data;
        gamma_4d.reshape({1, 1, 1, features});
        xt::xarray<float> beta_4d = beta_data;
        beta_4d.reshape({1, 1, 1, features});

        // Create tensors on device using from_xtensor
        auto input_tensor = core::from_xtensor(x_4d, &autograd::ctx().get_device());
        auto gamma_tensor = core::from_xtensor(gamma_4d, &autograd::ctx().get_device());
        auto beta_tensor = core::from_xtensor(beta_4d, &autograd::ctx().get_device());

        // Run metal kernel
        auto output_tensors =
            metal::layernorm_fw(input_tensor, gamma_tensor, beta_tensor, 1e-6f, /* return_mean_rstd */ true);

        auto metal_y_xtensor = core::to_xtensor(output_tensors[0].value());
        auto metal_mu_xtensor = core::to_xtensor(output_tensors[1].value());
        auto metal_rstd_xtensor = core::to_xtensor(output_tensors[2].value());

        // Flatten metal results for comparison
        xt::xarray<float> metal_y_flat = xt::flatten(metal_y_xtensor);
        xt::xarray<float> metal_mu_flat = xt::flatten(metal_mu_xtensor);
        xt::xarray<float> metal_rstd_flat = xt::flatten(metal_rstd_xtensor);

        // Compare shapes
        ASSERT_EQ(y_ref.shape(), metal_y_flat.shape());
        ASSERT_EQ(mu_ref.shape(), metal_mu_flat.shape());
        ASSERT_EQ(rstd_ref.shape(), metal_rstd_flat.shape());

        // Compare values
        EXPECT_TRUE(xt::allclose(metal_y_flat, y_ref, 1.0e-3F, 5e-2F));
        EXPECT_TRUE(xt::allclose(metal_mu_flat, mu_ref, 1.0e-3F, 5e-2F));
        EXPECT_TRUE(xt::allclose(metal_rstd_flat, rstd_ref, 1.0e-3F, 5e-2F));
    }
}

TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_OneTile) {
    CompareKernelVsXArray(1, 32, 1, 32);
}

TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_OneIncompleteTile) {
    CompareKernelVsXArray(1, 12, 1, 19);
}

TEST_F(LayerNormForwardOpTest, NIGHTLY_MetalLayerNormFw_MediumTensorFitsInL1) {
    CompareKernelVsXArray(2, 182, 1, 2083);
}

TEST_F(LayerNormForwardOpTest, NIGHTLY_MetalLayerNormFw_LargeTensor_DoesNotFitInL1) {
    CompareKernelVsXArray(4, 324, 1, 9132);
}

TEST_F(LayerNormForwardOpTest, MetalLayerNormFw_HeadsDimNot1) {
    CompareKernelVsXArray(2, 8, 4, 512);
}

TEST_F(LayerNormForwardOpTest, ProgramCacheSeparatesPaddingAndRebindsAllAddresses) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    const ttnn::Shape input_shape({1U, 1U, kCacheTestRows, kCacheTestWidth});
    const ttnn::Shape parameter_shape({1U, 1U, 1U, kCacheTestWidth});
    const ttnn::Shape stats_shape({1U, 1U, kCacheTestRows, 1U});
    const tt::tt_metal::Alignment overpadded_alignment({1U, 1U, 64U, 32U});
    const std::vector<float> output_sentinel(input_shape.volume(), -7.0F);
    const std::vector<float> stats_sentinel(stats_shape.volume(), -7.0F);

    const auto run = [&](float offset, const tt::tt_metal::Alignment& alignment) {
        const auto input_data = make_cache_test_data(input_shape.volume(), offset, 7U);
        const auto gamma_data = make_cache_test_data(parameter_shape.volume(), 0.75F + offset, 3U);
        const auto beta_data = make_cache_test_data(parameter_shape.volume(), -0.125F + offset, 5U);
        auto input = make_cache_test_tensor(input_data, input_shape, device, alignment);
        auto gamma = make_cache_test_tensor(gamma_data, parameter_shape, device);
        auto beta = make_cache_test_tensor(beta_data, parameter_shape, device);
        auto output = make_cache_test_tensor(output_sentinel, input_shape, device, alignment);
        auto mean = make_cache_test_tensor(stats_sentinel, stats_shape, device, alignment);
        auto rstd = make_cache_test_tensor(stats_sentinel, stats_shape, device, alignment);
        auto result = ttnn::prim::ttml_layernorm_fw(
            input, gamma, beta, kCacheTestEpsilon, /*return_mean_rstd=*/true, output, mean, rstd);
        return std::make_tuple(
            std::move(result),
            std::move(input),
            std::move(gamma),
            std::move(beta),
            std::move(output),
            std::move(mean),
            std::move(rstd),
            input_data,
            gamma_data,
            beta_data);
    };

    const auto entries_before_standard = device->num_program_cache_entries();
    auto
        [standard_result,
         standard_input,
         standard_gamma,
         standard_beta,
         standard_output,
         standard_mean,
         standard_rstd,
         standard_input_data,
         standard_gamma_data,
         standard_beta_data] = run(0.25F, {});
    (void)standard_input;
    (void)standard_gamma;
    (void)standard_beta;
    const auto entries_after_standard = device->num_program_cache_entries();
    EXPECT_GT(entries_after_standard, entries_before_standard);
    ASSERT_EQ(standard_result.size(), 3U);
    EXPECT_EQ(standard_result[0]->buffer()->address(), standard_output.buffer()->address());
    EXPECT_EQ(standard_result[1]->buffer()->address(), standard_mean.buffer()->address());
    EXPECT_EQ(standard_result[2]->buffer()->address(), standard_rstd.buffer()->address());
    expect_cache_test_forward_matches(standard_result, standard_input_data, standard_gamma_data, standard_beta_data);

    // The logical shapes and dtypes are unchanged, but every row-bearing tensor now has an extra
    // physical tile row. LayerNorm compiles its work split from that padded geometry.
    const auto entries_before_padded = device->num_program_cache_entries();
    auto
        [padded_result,
         padded_input,
         padded_gamma,
         padded_beta,
         padded_output,
         padded_mean,
         padded_rstd,
         padded_input_data,
         padded_gamma_data,
         padded_beta_data] = run(0.5F, overpadded_alignment);
    const auto entries_after_padded = device->num_program_cache_entries();
    EXPECT_GT(entries_after_padded, entries_before_padded)
        << "LayerNorm forward reused a program compiled for different padded geometry";
    expect_cache_test_forward_matches(padded_result, padded_input_data, padded_gamma_data, padded_beta_data);

    const auto entries_before_replay = device->num_program_cache_entries();
    auto
        [replay_result,
         replay_input,
         replay_gamma,
         replay_beta,
         replay_output,
         replay_mean,
         replay_rstd,
         replay_input_data,
         replay_gamma_data,
         replay_beta_data] = run(0.875F, overpadded_alignment);
    const auto entries_after_replay = device->num_program_cache_entries();
    EXPECT_EQ(entries_after_replay, entries_before_replay)
        << "same-spec LayerNorm forward replay should reuse its cached program";
    ASSERT_NE(replay_input.buffer()->address(), padded_input.buffer()->address());
    ASSERT_NE(replay_gamma.buffer()->address(), padded_gamma.buffer()->address());
    ASSERT_NE(replay_beta.buffer()->address(), padded_beta.buffer()->address());
    ASSERT_NE(replay_output.buffer()->address(), padded_output.buffer()->address());
    ASSERT_NE(replay_mean.buffer()->address(), padded_mean.buffer()->address());
    ASSERT_NE(replay_rstd.buffer()->address(), padded_rstd.buffer()->address());
    expect_cache_test_forward_matches(replay_result, replay_input_data, replay_gamma_data, replay_beta_data);
}

TEST_F(LayerNormForwardOpTest, StatsUseSingleTileWidthForWidthAlignedInput) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    device->clear_program_cache();

    constexpr uint32_t rows = 64U;
    constexpr uint32_t width = 64U;
    const ttnn::Shape input_shape({1U, 1U, rows, width});
    const ttnn::Shape parameter_shape({1U, 1U, 1U, width});
    const tt::tt_metal::Alignment width_aligned_input({1U, 1U, 32U, 64U});

    const auto run = [&](const float offset) {
        auto input_data = make_cache_test_data(input_shape.volume(), offset, 7U);
        auto gamma_data = make_cache_test_data(parameter_shape.volume(), 0.75F + offset, 3U);
        auto beta_data = make_cache_test_data(parameter_shape.volume(), -0.125F + offset, 5U);
        auto input = make_cache_test_tensor(input_data, input_shape, device, width_aligned_input);
        auto gamma = make_cache_test_tensor(gamma_data, parameter_shape, device);
        auto beta = make_cache_test_tensor(beta_data, parameter_shape, device);
        auto result = ttnn::prim::ttml_layernorm_fw(input, gamma, beta, kCacheTestEpsilon, /*return_mean_rstd=*/true);
        return std::make_tuple(std::move(result), std::move(input_data), std::move(gamma_data), std::move(beta_data));
    };

    const auto entries_before_first = device->num_program_cache_entries();
    auto [first_result, first_input, first_gamma, first_beta] = run(0.25F);
    const auto entries_after_first = device->num_program_cache_entries();
    EXPECT_GT(entries_after_first, entries_before_first);
    ASSERT_EQ(first_result.size(), 3U);
    ASSERT_TRUE(first_result[0].has_value());
    ASSERT_TRUE(first_result[1].has_value());
    ASSERT_TRUE(first_result[2].has_value());
    EXPECT_EQ(first_result[0]->padded_shape(), input_shape);
    EXPECT_EQ(first_result[1]->logical_shape(), (ttnn::Shape({1U, 1U, rows, 1U})));
    EXPECT_EQ(first_result[1]->padded_shape(), (ttnn::Shape({1U, 1U, rows, 32U})));
    EXPECT_EQ(first_result[2]->tensor_spec(), first_result[1]->tensor_spec());
    expect_cache_test_forward_matches(first_result, first_input, first_gamma, first_beta, rows, width);

    const auto entries_before_replay = device->num_program_cache_entries();
    auto [replay_result, replay_input, replay_gamma, replay_beta] = run(0.5F);
    const auto entries_after_replay = device->num_program_cache_entries();
    EXPECT_EQ(entries_after_replay, entries_before_replay);
    expect_cache_test_forward_matches(replay_result, replay_input, replay_gamma, replay_beta, rows, width);
}

TEST_F(LayerNormForwardOpTest, ValidatesContractsOnProgramCacheMissAndHit) {
    using Operation = ttml::metal::ops::layernorm_fw::device::LayerNormForwardDeviceOperation;

    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    device->clear_program_cache();

    const ttnn::Shape input_shape({1U, 1U, kCacheTestRows, kCacheTestWidth});
    const ttnn::Shape parameter_shape({1U, 1U, 1U, kCacheTestWidth});
    const ttnn::Shape stats_shape({1U, 1U, kCacheTestRows, 1U});
    const tt::tt_metal::Alignment overpadded_alignment({1U, 1U, 64U, 32U});
    const auto input_data = make_cache_test_data(input_shape.volume(), 0.25F, 7U);
    const auto gamma_data = make_cache_test_data(parameter_shape.volume(), 0.75F, 3U);
    const auto beta_data = make_cache_test_data(parameter_shape.volume(), -0.125F, 5U);

    auto input = make_cache_test_tensor(input_data, input_shape, device, overpadded_alignment);
    auto overwide_input =
        make_cache_test_tensor(input_data, input_shape, device, tt::tt_metal::Alignment({1U, 1U, 32U, 96U}));
    auto gamma = make_cache_test_tensor(gamma_data, parameter_shape, device);
    auto beta = make_cache_test_tensor(beta_data, parameter_shape, device);
    auto rank3_gamma = make_cache_test_tensor(gamma_data, ttnn::Shape({1U, 1U, kCacheTestWidth}), device);
    auto rank3_beta = make_cache_test_tensor(beta_data, ttnn::Shape({1U, 1U, kCacheTestWidth}), device);
    auto oversized_output = make_cache_test_tensor(
        std::vector<float>(2U * input_shape.volume(), -7.0F),
        ttnn::Shape({1U, 1U, 2U * kCacheTestRows, kCacheTestWidth}),
        device,
        overpadded_alignment);
    auto overwide_stats = make_cache_test_tensor(
        std::vector<float>(stats_shape.volume(), -7.0F),
        stats_shape,
        device,
        tt::tt_metal::Alignment({1U, 1U, 64U, 64U}));
    auto oversized_stats = make_cache_test_tensor(
        std::vector<float>(2U * stats_shape.volume(), -7.0F),
        ttnn::Shape({1U, 1U, 2U * kCacheTestRows, 1U}),
        device,
        overpadded_alignment);

    const auto narrow_tile_layout = tt::tt_metal::TensorLayout(
        ttnn::DataType::BFLOAT16,
        ttnn::PageConfig(ttnn::Layout::TILE, tt::tt_metal::Tile({16U, 16U})),
        ttnn::DRAM_MEMORY_CONFIG);
    auto narrow_tile_input =
        ttnn::Tensor::from_vector(input_data, tt::tt_metal::TensorSpec(input_shape, narrow_tile_layout), device);

    const Operation::operation_attributes_t with_stats{.epsilon = kCacheTestEpsilon, .return_mean_rstd = true};
    const Operation::operation_attributes_t without_stats{.epsilon = kCacheTestEpsilon, .return_mean_rstd = false};
    const Operation::tensor_args_t valid_args{.input = input, .gamma = gamma, .beta = beta};
    expect_validation_accepts<Operation>(with_stats, valid_args);
    expect_validation_accepts<Operation>(without_stats, valid_args);

    struct ValidationCase {
        Operation::operation_attributes_t attributes;
        Operation::tensor_args_t tensor_args;
        std::string_view diagnostic;
    };

    std::vector<ValidationCase> invalid_cases;
    const auto add_case = [&](const Operation::operation_attributes_t& attributes,
                              const auto& mutate,
                              const std::string_view diagnostic) {
        auto tensor_args = valid_args;
        mutate(tensor_args);
        invalid_cases.push_back({attributes, std::move(tensor_args), diagnostic});
    };
    add_case(with_stats, [&](auto& args) { args.gamma = rank3_gamma; }, "Gamma tensor must have shape");
    add_case(with_stats, [&](auto& args) { args.beta = rank3_beta; }, "Beta tensor must have shape");
    add_case(
        with_stats,
        [&](auto& args) { args.input = narrow_tile_input; },
        "Tensor 'Input' must use the canonical non-transposed 32x32 tile");
    add_case(
        with_stats,
        [&](auto& args) { args.input = overwide_input; },
        "Input tensor may be overpadded in height but must use canonical width padding");
    add_case(
        with_stats,
        [&](auto& args) { args.preallocated_output = oversized_output; },
        "Preallocated output TensorSpec must match the input TensorSpec");
    add_case(
        with_stats,
        [&](auto& args) { args.preallocated_mean = oversized_stats; },
        "Preallocated mean tensor must have logical shape");
    add_case(
        with_stats,
        [&](auto& args) { args.preallocated_rstd = oversized_stats; },
        "Preallocated rstd tensor must have logical shape");
    add_case(
        with_stats,
        [&](auto& args) { args.preallocated_mean = overwide_stats; },
        "Preallocated mean tensor must have logical shape");
    add_case(
        without_stats,
        [&](auto& args) { args.preallocated_mean = oversized_stats; },
        "Preallocated mean/rstd tensors require return_mean_rstd=true");

    for (const auto& test_case : invalid_cases) {
        SCOPED_TRACE(test_case.diagnostic);
        expect_validation_rejects<Operation>(test_case.attributes, test_case.tensor_args, test_case.diagnostic);
    }

    const auto entries_before_warmup = device->num_program_cache_entries();
    const auto valid_result =
        ttnn::prim::ttml_layernorm_fw(input, gamma, beta, kCacheTestEpsilon, /*return_mean_rstd=*/true);
    const auto entries_after_warmup = device->num_program_cache_entries();
    EXPECT_GT(entries_after_warmup, entries_before_warmup);
    ASSERT_EQ(valid_result.size(), 3U);
    EXPECT_EQ(valid_result[0]->tensor_spec(), input.tensor_spec());
    auto expected_stats_shape = input_shape;
    expected_stats_shape[-1] = 1U;
    const auto expected_stats_spec =
        tt::tt_metal::TensorSpec(expected_stats_shape, input.tensor_spec().tensor_layout());
    EXPECT_EQ(valid_result[1]->tensor_spec(), expected_stats_spec);
    EXPECT_EQ(valid_result[2]->tensor_spec(), expected_stats_spec);
    expect_cache_test_forward_matches(valid_result, input_data, gamma_data, beta_data);
}
