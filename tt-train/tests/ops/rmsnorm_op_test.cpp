// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/rmsnorm_op.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <umd/device/cluster.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/ops/rmsnorm_bw/device/rmsnorm_bw_device_operation.hpp"
#include "metal/ops/rmsnorm_fw/device/rmsnorm_fw_device_operation.hpp"
#include "ops/losses.hpp"
#include "test_utils/random_data.hpp"

namespace {

constexpr uint32_t kValidationTestBatches = 2U;
constexpr uint32_t kValidationTestRows = 32U;
constexpr uint32_t kValidationTestWidth = 64U;
constexpr float kValidationTestEpsilon = 1.0e-3F;

ttnn::Tensor make_validation_test_tensor(
    const std::vector<float>& data,
    const ttnn::Shape& shape,
    ttnn::distributed::MeshDevice* device,
    const tt::tt_metal::Alignment& alignment = {},
    const tt::tt_metal::Tile& tile = {}) {
    const auto layout = tt::tt_metal::TensorLayout(
        ttnn::DataType::BFLOAT16, ttnn::PageConfig(ttnn::Layout::TILE, tile), ttnn::DRAM_MEMORY_CONFIG, alignment);
    return ttnn::Tensor::from_vector(data, tt::tt_metal::TensorSpec(shape, layout), device);
}

std::vector<float> make_validation_test_input(float offset) {
    constexpr uint32_t logical_rows = kValidationTestBatches * kValidationTestRows;
    std::vector<float> data(logical_rows * kValidationTestWidth);
    for (uint32_t row = 0; row < logical_rows; ++row) {
        for (uint32_t col = 0; col < kValidationTestWidth; ++col) {
            data[row * kValidationTestWidth + col] =
                offset + 0.015625F * static_cast<float>((row * 7U + col * 3U) % 41U) - 0.25F;
        }
    }
    return data;
}

std::vector<float> make_validation_test_gamma(float offset) {
    std::vector<float> data(kValidationTestWidth);
    for (uint32_t col = 0; col < kValidationTestWidth; ++col) {
        data[col] = offset + 0.0078125F * static_cast<float>(col % 17U);
    }
    return data;
}

std::vector<float> make_validation_test_upstream_grad(float offset) {
    constexpr uint32_t logical_rows = kValidationTestBatches * kValidationTestRows;
    std::vector<float> data(logical_rows * kValidationTestWidth);
    for (uint32_t row = 0; row < logical_rows; ++row) {
        for (uint32_t col = 0; col < kValidationTestWidth; ++col) {
            data[row * kValidationTestWidth + col] =
                offset + 0.00390625F * static_cast<float>((row * 5U + col * 11U) % 29U);
        }
    }
    return data;
}

std::vector<float> validation_test_rms_reference(const std::vector<float>& input) {
    constexpr uint32_t logical_rows = kValidationTestBatches * kValidationTestRows;
    std::vector<float> rms(logical_rows);
    for (uint32_t row = 0; row < logical_rows; ++row) {
        float square_sum = 0.0F;
        for (uint32_t col = 0; col < kValidationTestWidth; ++col) {
            const float value = input[row * kValidationTestWidth + col];
            square_sum += value * value;
        }
        rms[row] = std::sqrt(square_sum / static_cast<float>(kValidationTestWidth) + kValidationTestEpsilon);
    }
    return rms;
}

void expect_validation_test_forward_matches(
    const ttnn::Tensor& output,
    const ttnn::Tensor& rms,
    const std::vector<float>& input,
    const std::vector<float>& gamma) {
    const auto actual_output = ttml::core::to_vector<float>(output);
    const auto actual_rms = ttml::core::to_vector<float>(rms);
    const auto expected_rms = validation_test_rms_reference(input);

    ASSERT_EQ(actual_output.size(), input.size());
    ASSERT_EQ(actual_rms.size(), expected_rms.size());
    for (uint32_t row = 0; row < expected_rms.size(); ++row) {
        EXPECT_NEAR(actual_rms[row], expected_rms[row], 3.0e-2F) << "row=" << row;
        for (uint32_t col = 0; col < kValidationTestWidth; ++col) {
            const size_t index = row * kValidationTestWidth + col;
            const float expected = input[index] * gamma[col] / expected_rms[row];
            EXPECT_NEAR(actual_output[index], expected, 4.0e-2F) << "row=" << row << ", col=" << col;
        }
    }
}

void expect_validation_test_backward_matches(
    const ttnn::Tensor& da,
    const ttnn::Tensor& dgamma_components,
    const std::vector<float>& input,
    const std::vector<float>& gamma,
    const std::vector<float>& rms,
    const std::vector<float>& upstream_grad) {
    const auto actual_da = ttml::core::to_vector<float>(da);
    const auto actual_dgamma = ttml::core::to_vector<float>(dgamma_components);

    ASSERT_EQ(actual_da.size(), input.size());
    ASSERT_EQ(actual_dgamma.size(), input.size());
    for (uint32_t row = 0; row < rms.size(); ++row) {
        float dot = 0.0F;
        for (uint32_t col = 0; col < kValidationTestWidth; ++col) {
            const size_t index = row * kValidationTestWidth + col;
            dot += upstream_grad[index] * gamma[col] * input[index];
        }
        dot /= static_cast<float>(kValidationTestWidth);

        for (uint32_t col = 0; col < kValidationTestWidth; ++col) {
            const size_t index = row * kValidationTestWidth + col;
            const float expected_da =
                upstream_grad[index] * gamma[col] / rms[row] - input[index] * dot / (rms[row] * rms[row] * rms[row]);
            const float expected_dgamma = upstream_grad[index] * input[index] / rms[row];
            EXPECT_NEAR(actual_da[index], expected_da, 5.0e-2F) << "row=" << row << ", col=" << col;
            EXPECT_NEAR(actual_dgamma[index], expected_dgamma, 5.0e-2F) << "row=" << row << ", col=" << col;
        }
    }
}

}  // namespace

class RMSNormOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
        ttml::autograd::ctx().set_seed(42);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(RMSNormOpTest, RawPrimitivesPreserveOverpaddedHeightWithAutomaticOutputs) {
    auto* device = &ttml::autograd::ctx().get_device();
    const ttnn::Shape input_shape({kValidationTestBatches, 1U, kValidationTestRows, kValidationTestWidth});
    const ttnn::Shape gamma_shape({1U, 1U, 1U, kValidationTestWidth});
    const tt::tt_metal::Alignment overpadded_alignment({1U, 1U, 64U, 32U});

    const auto input_data = make_validation_test_input(0.25F);
    const auto gamma_data = make_validation_test_gamma(0.75F);
    const auto upstream_grad_data = make_validation_test_upstream_grad(0.125F);
    auto input = make_validation_test_tensor(input_data, input_shape, device, overpadded_alignment);
    auto gamma = make_validation_test_tensor(gamma_data, gamma_shape, device);
    auto upstream_grad = make_validation_test_tensor(upstream_grad_data, input_shape, device, overpadded_alignment);

    const auto forward =
        ttnn::prim::ttml_rmsnorm_fw(input, gamma, /*return_intermediates=*/true, kValidationTestEpsilon);
    ASSERT_EQ(forward.size(), 2U);
    EXPECT_EQ(forward[0].tensor_spec(), input.tensor_spec());
    auto rms_shape = input_shape;
    rms_shape[-1] = 1U;
    const auto expected_rms_spec = tt::tt_metal::TensorSpec(rms_shape, input.tensor_spec().tensor_layout());
    EXPECT_EQ(forward[1].tensor_spec(), expected_rms_spec);
    expect_validation_test_forward_matches(forward[0], forward[1], input_data, gamma_data);

    const auto rms_data = validation_test_rms_reference(input_data);
    const auto backward = ttnn::prim::ttml_rmsnorm_bw(input, gamma, forward[1], upstream_grad, kValidationTestEpsilon);
    ASSERT_EQ(backward.size(), 2U);
    EXPECT_EQ(backward[0].tensor_spec(), input.tensor_spec());
    EXPECT_EQ(backward[1].tensor_spec(), input.tensor_spec());
    expect_validation_test_backward_matches(
        backward[0], backward[1], input_data, gamma_data, rms_data, upstream_grad_data);
}

TEST_F(RMSNormOpTest, ForwardRejectsMalformedContractsWithColdAndWarmCache) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    device->clear_program_cache();

    const ttnn::Shape input_shape({kValidationTestBatches, 1U, kValidationTestRows, kValidationTestWidth});
    const ttnn::Shape gamma_shape({1U, 1U, 1U, kValidationTestWidth});
    const ttnn::Shape rms_shape({kValidationTestBatches, 1U, kValidationTestRows, 1U});
    const tt::tt_metal::Alignment overpadded_alignment({1U, 1U, 64U, 32U});
    const auto input_data = make_validation_test_input(0.25F);
    const auto gamma_data = make_validation_test_gamma(0.75F);

    auto input = make_validation_test_tensor(input_data, input_shape, device, overpadded_alignment);
    auto gamma = make_validation_test_tensor(gamma_data, gamma_shape, device);
    auto rank3_gamma = make_validation_test_tensor(gamma_data, ttnn::Shape({1U, 1U, kValidationTestWidth}), device);
    auto undersized_output = make_validation_test_tensor(
        std::vector<float>(kValidationTestBatches * kValidationTestRows * 32U, -7.0F),
        ttnn::Shape({kValidationTestBatches, 1U, kValidationTestRows, 32U}),
        device,
        overpadded_alignment);
    auto wrong_stride_rms =
        make_validation_test_tensor(std::vector<float>(rms_shape.volume(), -7.0F), rms_shape, device);
    auto overwide_input =
        make_validation_test_tensor(input_data, input_shape, device, tt::tt_metal::Alignment({1U, 1U, 64U, 96U}));
    auto wide_aligned_input =
        make_validation_test_tensor(input_data, input_shape, device, tt::tt_metal::Alignment({1U, 1U, 64U, 64U}));
    auto narrow_tile_input = make_validation_test_tensor(
        input_data, input_shape, device, overpadded_alignment, tt::tt_metal::Tile({16U, 16U}));

    const auto expect_malformed_contracts_rejected = [&] {
        EXPECT_ANY_THROW((void)ttnn::prim::ttml_rmsnorm_fw(
            input, rank3_gamma, /*return_intermediates=*/true, kValidationTestEpsilon));
        EXPECT_ANY_THROW((void)ttnn::prim::ttml_rmsnorm_fw(
            input,
            gamma,
            /*return_intermediates=*/true,
            kValidationTestEpsilon,
            std::nullopt,
            undersized_output));
        EXPECT_ANY_THROW((void)ttnn::prim::ttml_rmsnorm_fw(
            input,
            gamma,
            /*return_intermediates=*/true,
            kValidationTestEpsilon,
            wrong_stride_rms));
        EXPECT_ANY_THROW((void)ttnn::prim::ttml_rmsnorm_fw(
            input,
            gamma,
            /*return_intermediates=*/false,
            kValidationTestEpsilon,
            wrong_stride_rms));
        EXPECT_ANY_THROW((void)ttnn::prim::ttml_rmsnorm_fw(
            overwide_input, gamma, /*return_intermediates=*/true, kValidationTestEpsilon));
        EXPECT_ANY_THROW((void)ttnn::prim::ttml_rmsnorm_fw(
            wide_aligned_input, gamma, /*return_intermediates=*/true, kValidationTestEpsilon));
        EXPECT_ANY_THROW((void)ttnn::prim::ttml_rmsnorm_fw(
            narrow_tile_input, gamma, /*return_intermediates=*/true, kValidationTestEpsilon));
    };

    expect_malformed_contracts_rejected();
    const auto valid = ttnn::prim::ttml_rmsnorm_fw(input, gamma, /*return_intermediates=*/true, kValidationTestEpsilon);
    ASSERT_EQ(valid.size(), 2U);
    expect_malformed_contracts_rejected();
}

TEST_F(RMSNormOpTest, BackwardRejectsMalformedContractsWithColdAndWarmCache) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();
    device->clear_program_cache();

    const ttnn::Shape input_shape({kValidationTestBatches, 1U, kValidationTestRows, kValidationTestWidth});
    const ttnn::Shape gamma_shape({1U, 1U, 1U, kValidationTestWidth});
    const ttnn::Shape rms_shape({kValidationTestBatches, 1U, kValidationTestRows, 1U});
    const tt::tt_metal::Alignment overpadded_alignment({1U, 1U, 64U, 32U});
    const auto input_data = make_validation_test_input(0.25F);
    const auto gamma_data = make_validation_test_gamma(0.75F);
    const auto rms_data = validation_test_rms_reference(input_data);
    const auto upstream_grad_data = make_validation_test_upstream_grad(0.125F);

    auto input = make_validation_test_tensor(input_data, input_shape, device, overpadded_alignment);
    auto gamma = make_validation_test_tensor(gamma_data, gamma_shape, device);
    auto rms = make_validation_test_tensor(rms_data, rms_shape, device, overpadded_alignment);
    auto upstream_grad = make_validation_test_tensor(upstream_grad_data, input_shape, device, overpadded_alignment);
    auto rank3_gamma = make_validation_test_tensor(gamma_data, ttnn::Shape({1U, 1U, kValidationTestWidth}), device);
    auto wrong_stride_rms = make_validation_test_tensor(rms_data, rms_shape, device);
    auto undersized_grad = make_validation_test_tensor(
        std::vector<float>(kValidationTestBatches * kValidationTestRows * 32U, 0.0F),
        ttnn::Shape({kValidationTestBatches, 1U, kValidationTestRows, 32U}),
        device,
        overpadded_alignment);
    auto undersized_da = make_validation_test_tensor(
        std::vector<float>(kValidationTestBatches * kValidationTestRows * 32U, -7.0F),
        ttnn::Shape({kValidationTestBatches, 1U, kValidationTestRows, 32U}),
        device,
        overpadded_alignment);
    auto overwide_input =
        make_validation_test_tensor(input_data, input_shape, device, tt::tt_metal::Alignment({1U, 1U, 64U, 96U}));
    auto narrow_tile_input = make_validation_test_tensor(
        input_data, input_shape, device, overpadded_alignment, tt::tt_metal::Tile({16U, 16U}));

    const auto expect_malformed_contracts_rejected = [&] {
        EXPECT_ANY_THROW(
            (void)ttnn::prim::ttml_rmsnorm_bw(input, rank3_gamma, rms, upstream_grad, kValidationTestEpsilon));
        EXPECT_ANY_THROW(
            (void)ttnn::prim::ttml_rmsnorm_bw(input, gamma, wrong_stride_rms, upstream_grad, kValidationTestEpsilon));
        EXPECT_ANY_THROW((void)ttnn::prim::ttml_rmsnorm_bw(input, gamma, rms, undersized_grad, kValidationTestEpsilon));
        EXPECT_ANY_THROW(
            (void)ttnn::prim::ttml_rmsnorm_bw(input, gamma, rms, upstream_grad, kValidationTestEpsilon, undersized_da));
        EXPECT_ANY_THROW(
            (void)ttnn::prim::ttml_rmsnorm_bw(overwide_input, gamma, rms, upstream_grad, kValidationTestEpsilon));
        EXPECT_ANY_THROW(
            (void)ttnn::prim::ttml_rmsnorm_bw(narrow_tile_input, gamma, rms, upstream_grad, kValidationTestEpsilon));
    };

    expect_malformed_contracts_rejected();
    const auto valid = ttnn::prim::ttml_rmsnorm_bw(input, gamma, rms, upstream_grad, kValidationTestEpsilon);
    ASSERT_EQ(valid.size(), 2U);
    expect_malformed_contracts_rejected();
}

// ============================================================================
// Section 1: RMSNorm Kernel vs PyTorch Reference Implementation
// ============================================================================
// These tests validate the optimized RMSNorm kernel implementation against
// PyTorch's reference implementation to ensure numerical correctness.
//
// Test methodology:
// 1. Create test tensor `x` of shape [N,C,H,W] with x.requires_grad = True
// 2. Compute PyTorch RMSNorm: `x_norm_sum = torch.nn.functional.rms_norm(x).sum()`
// 3. Compute PyTorch gradient: `x_grad = torch.autograd.grad(x_norm_sum, x)[0]`
// 4. Compare TTML kernel results with PyTorch reference results
// ============================================================================
TEST_F(RMSNormOpTest, RMSNorm_Small_Forward) {
    using namespace ttml;

    [[maybe_unused]] uint32_t N = 1, C = 1, H = 1, W = 8;

    xt::xarray<float> example_xtensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto example_tensor = autograd::create_tensor(core::from_xtensor(example_xtensor, &autograd::ctx().get_device()));
    auto gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, W}), &autograd::ctx().get_device()));

    auto result = ops::rmsnorm(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());
    xt::xarray<float> expected_result = {{0.3652F, 0.7305F, 1.0938F, 1.4609F, 0.3652F, 0.7305F, 1.0938F, 1.4609F}};
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F));
}

TEST_F(RMSNormOpTest, RMSNorm_Small_Backward) {
    using namespace ttml;

    [[maybe_unused]] uint32_t N = 1, C = 1, H = 1, W = 8;

    xt::xarray<float> example_xtensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto example_tensor = autograd::create_tensor(
        core::from_xtensor(example_xtensor, &autograd::ctx().get_device()), /* requires_grad */ true);
    auto gamma = autograd::create_tensor(
        core::ones(ttnn::Shape({1, 1, 1, W}), &autograd::ctx().get_device()), /* requires_grad */ true);

    auto result = ops::rmsnorm(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());

    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto mse_result = ops::mse_loss(result, target);
    mse_result->backward();
    auto example_tensor_grad = core::to_xtensor(example_tensor->get_grad());
    auto expected_example_tensor_grad = xt::xarray<float>(
        {{{{5.2452e-05F,
            1.0490e-04F,
            -2.0742e-05F,
            2.0981e-04F,
            5.2452e-05F,
            1.0490e-04F,
            -2.0742e-05F,
            2.0981e-04F}}}});
    EXPECT_TRUE(xt::allclose(example_tensor_grad, expected_example_tensor_grad, 1.0e-3F, 1e-2F));

    auto gamma_grad = core::to_xtensor(gamma->get_grad());
    auto expected_gamma_grad =
        xt::xarray<float>({{{{0.0334F, 0.1338F, 0.2988F, 0.5352F, 0.0334F, 0.1338F, 0.2988F, 0.5352F}}}});
    EXPECT_TRUE(xt::allclose(gamma_grad, expected_gamma_grad, 1.0e-3F, 1e-2F));
}

TEST_F(RMSNormOpTest, NIGHTLY_RMSNorm_Forward_Batch) {
    using namespace ttml;

    // 2 batches, 1 sequence, 20 tokens, 5-dim'l embedding space.
    std::array<uint32_t, 4> a_shape = {2, 1, 20, 5};
    xt::xarray<float> a_xarray = xt::xarray<float>::from_shape(a_shape);
    std::generate(a_xarray.begin(), a_xarray.end(), [cur = 0.0F]() mutable { return (cur++); });

    auto example_tensor = autograd::create_tensor(core::from_xtensor(a_xarray, &autograd::ctx().get_device()));
    auto gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, 5}), &autograd::ctx().get_device()));

    auto result = ops::rmsnorm(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());
    xt::xarray<float> expected_result = {
        {{{0.00000F, 0.40820F, 0.81641F, 1.22656F, 1.63281F}, {0.69922F, 0.83984F, 0.98047F, 1.11719F, 1.25781F},
          {0.82812F, 0.91016F, 0.99219F, 1.07812F, 1.15625F}, {0.87891F, 0.93750F, 0.99609F, 1.05469F, 1.11719F},
          {0.90625F, 0.95312F, 0.99609F, 1.04688F, 1.08594F}, {0.92578F, 0.96094F, 1.00000F, 1.03906F, 1.07031F},
          {0.93750F, 0.96875F, 1.00000F, 1.03125F, 1.06250F}, {0.94531F, 0.97266F, 1.00000F, 1.02344F, 1.05469F},
          {0.95312F, 0.97656F, 1.00000F, 1.02344F, 1.04688F}, {0.95703F, 0.97656F, 1.00000F, 1.02344F, 1.03906F},
          {0.96094F, 0.98047F, 1.00000F, 1.01562F, 1.03906F}, {0.96484F, 0.98047F, 1.00000F, 1.01562F, 1.03125F},
          {0.96875F, 0.98438F, 1.00000F, 1.01562F, 1.03125F}, {0.96875F, 0.98438F, 1.00000F, 1.01562F, 1.03125F},
          {0.97266F, 0.98438F, 1.00000F, 1.01562F, 1.03125F}, {0.97266F, 0.98828F, 1.00000F, 1.01562F, 1.02344F},
          {0.97656F, 0.98828F, 1.00000F, 1.01562F, 1.02344F}, {0.97656F, 0.98828F, 1.00000F, 1.00781F, 1.02344F},
          {0.97656F, 0.98828F, 1.00000F, 1.00781F, 1.02344F}, {0.98047F, 0.98828F, 1.00000F, 1.00781F, 1.02344F}}},
        {{{0.98047F, 0.98828F, 1.00000F, 1.00781F, 1.01562F}, {0.98047F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98047F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}, {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F},
          {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}, {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F},
          {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}, {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}}}};
    assert((expected_result.shape() == result_xtensor.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 6e-2F, 1e-8F));
}

TEST_F(RMSNormOpTest, NIGHTLY_RMSNorm_Backward_Batch) {
    using namespace ttml;

    // 2 batches, 1 sequence, 20 tokens, 5-dim'l embedding space.
    std::array<uint32_t, 4> a_shape = {2, 1, 20, 5};
    xt::xarray<float> a_xarray = xt::xarray<float>::from_shape(a_shape);
    std::generate(a_xarray.begin(), a_xarray.end(), [cur = 0.0F]() mutable { return (cur++); });

    auto example_tensor =
        autograd::create_tensor(core::from_xtensor(a_xarray, &autograd::ctx().get_device()), /* requires_grad */ true);
    auto gamma = autograd::create_tensor(
        core::ones(ttnn::Shape({1, 1, 1, 5}), &autograd::ctx().get_device()), /* requires_grad */ true);

    auto result = ops::rmsnorm(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());

    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto mse_result = ops::mse_loss(result, target);
    mse_result->backward();

    auto example_tensor_grad = core::to_xtensor(example_tensor->get_grad());
    xt::xarray<float> expected_example_tensor_grad = xt::zeros_like(a_xarray);
    EXPECT_TRUE(xt::allclose(example_tensor_grad, expected_example_tensor_grad, 5e-2F, 1e-3F));

    auto gamma_grad = core::to_xtensor(gamma->get_grad());
    xt::xarray<float> expected_gamma_grad = {{{{0.36111F, 0.37644F, 0.39589F, 0.41945F, 0.44712F}}}};
    EXPECT_TRUE(xt::allclose(gamma_grad, expected_gamma_grad, 5e-2F));
}

// ============================================================================
// Section 2: RMSNorm Composite vs PyTorch Reference Implementation
// ============================================================================
// These tests validate the composite RMSNorm implementation (built from basic ops)
// against PyTorch's reference implementation to ensure numerical correctness.
//
// The composite implementation serves as a reference for the optimized kernel
// and uses standard operations like power, mean, sqrt, and multiply.
// Same test methodology as Section 1, but using rmsnorm_composite() instead.
// ============================================================================
TEST_F(RMSNormOpTest, NIGHTLY_CompositeRMSNorm_Small_Forward) {
    using namespace ttml;

    [[maybe_unused]] uint32_t N = 1, C = 1, H = 1, W = 8;

    xt::xarray<float> example_xtensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto example_tensor = autograd::create_tensor(core::from_xtensor(example_xtensor, &autograd::ctx().get_device()));
    auto gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, W}), &autograd::ctx().get_device()));

    auto result = ops::rmsnorm_composite(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());
    xt::xarray<float> expected_result = {{0.3652F, 0.7305F, 1.0938F, 1.4609F, 0.3652F, 0.7305F, 1.0938F, 1.4609F}};
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 1e-2F));
}

TEST_F(RMSNormOpTest, NIGHTLY_CompositeRMSNorm_Small_Backward) {
    using namespace ttml;

    [[maybe_unused]] uint32_t N = 1, C = 1, H = 1, W = 8;

    xt::xarray<float> example_xtensor = {{{{1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F}}}};
    auto example_tensor = autograd::create_tensor(
        core::from_xtensor(example_xtensor, &autograd::ctx().get_device()), /* requires_grad */ true);
    auto gamma = autograd::create_tensor(
        core::ones(ttnn::Shape({1, 1, 1, W}), &autograd::ctx().get_device()), /* requires_grad */ true);

    auto result = ops::rmsnorm_composite(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());

    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto mse_result = ops::mse_loss(result, target);
    mse_result->backward();
    auto example_tensor_grad = core::to_xtensor(example_tensor->get_grad());
    auto expected_example_tensor_grad = xt::xarray<float>(
        {{{{5.2452e-05F,
            1.0490e-04F,
            -2.0742e-05F,
            2.0981e-04F,
            5.2452e-05F,
            1.0490e-04F,
            -2.0742e-05F,
            2.0981e-04F}}}});
    EXPECT_TRUE(xt::allclose(example_tensor_grad, expected_example_tensor_grad, 1.0e-3F, 1e-2F));

    auto gamma_grad = core::to_xtensor(gamma->get_grad());
    auto expected_gamma_grad =
        xt::xarray<float>({{{{0.0334F, 0.1338F, 0.2988F, 0.5352F, 0.0334F, 0.1338F, 0.2988F, 0.5352F}}}});
    EXPECT_TRUE(xt::allclose(gamma_grad, expected_gamma_grad, 1.0e-3F, 1e-2F));
}

TEST_F(RMSNormOpTest, NIGHTLY_CompositeRMSNorm_Forward_Batch) {
    using namespace ttml;

    // 2 batches, 1 sequence, 20 tokens, 5-dim'l embedding space.
    std::array<uint32_t, 4> a_shape = {2, 1, 20, 5};
    xt::xarray<float> a_xarray = xt::xarray<float>::from_shape(a_shape);
    std::generate(a_xarray.begin(), a_xarray.end(), [cur = 0.0F]() mutable { return (cur++); });

    auto example_tensor = autograd::create_tensor(core::from_xtensor(a_xarray, &autograd::ctx().get_device()));
    auto gamma = autograd::create_tensor(core::ones(ttnn::Shape({1, 1, 1, 5}), &autograd::ctx().get_device()));

    auto result = ops::rmsnorm_composite(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());
    xt::xarray<float> expected_result = {
        {{{0.00000F, 0.40820F, 0.81641F, 1.22656F, 1.63281F}, {0.69922F, 0.83984F, 0.98047F, 1.11719F, 1.25781F},
          {0.82812F, 0.91016F, 0.99219F, 1.07812F, 1.15625F}, {0.87891F, 0.93750F, 0.99609F, 1.05469F, 1.11719F},
          {0.90625F, 0.95312F, 0.99609F, 1.04688F, 1.08594F}, {0.92578F, 0.96094F, 1.00000F, 1.03906F, 1.07031F},
          {0.93750F, 0.96875F, 1.00000F, 1.03125F, 1.06250F}, {0.94531F, 0.97266F, 1.00000F, 1.02344F, 1.05469F},
          {0.95312F, 0.97656F, 1.00000F, 1.02344F, 1.04688F}, {0.95703F, 0.97656F, 1.00000F, 1.02344F, 1.03906F},
          {0.96094F, 0.98047F, 1.00000F, 1.01562F, 1.03906F}, {0.96484F, 0.98047F, 1.00000F, 1.01562F, 1.03125F},
          {0.96875F, 0.98438F, 1.00000F, 1.01562F, 1.03125F}, {0.96875F, 0.98438F, 1.00000F, 1.01562F, 1.03125F},
          {0.97266F, 0.98438F, 1.00000F, 1.01562F, 1.03125F}, {0.97266F, 0.98828F, 1.00000F, 1.01562F, 1.02344F},
          {0.97656F, 0.98828F, 1.00000F, 1.01562F, 1.02344F}, {0.97656F, 0.98828F, 1.00000F, 1.00781F, 1.02344F},
          {0.97656F, 0.98828F, 1.00000F, 1.00781F, 1.02344F}, {0.98047F, 0.98828F, 1.00000F, 1.00781F, 1.02344F}}},
        {{{0.98047F, 0.98828F, 1.00000F, 1.00781F, 1.01562F}, {0.98047F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98047F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98438F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F}, {0.98828F, 0.99219F, 1.00000F, 1.00781F, 1.01562F},
          {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}, {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F},
          {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}, {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F},
          {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}, {0.98828F, 0.99609F, 1.00000F, 1.00781F, 1.00781F}}}};
    assert((expected_result.shape() == result_xtensor.shape()));
    EXPECT_TRUE(xt::allclose(result_xtensor, expected_result, 6e-2F, 1e-8F));
}

TEST_F(RMSNormOpTest, NIGHTLY_CompositeRMSNorm_Backward_Batch) {
    using namespace ttml;

    // 2 batches, 1 sequence, 20 tokens, 5-dim'l embedding space.
    std::array<uint32_t, 4> a_shape = {2, 1, 20, 5};
    xt::xarray<float> a_xarray = xt::xarray<float>::from_shape(a_shape);
    std::generate(a_xarray.begin(), a_xarray.end(), [cur = 0.0F]() mutable { return (cur++); });

    auto example_tensor =
        autograd::create_tensor(core::from_xtensor(a_xarray, &autograd::ctx().get_device()), /* requires_grad */ true);
    auto gamma = autograd::create_tensor(
        core::ones(ttnn::Shape({1, 1, 1, 5}), &autograd::ctx().get_device()), /* requires_grad */ true);

    auto result = ops::rmsnorm_composite(example_tensor, gamma, 0.0078125F);
    auto result_xtensor = core::to_xtensor(result->get_value());

    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto mse_result = ops::mse_loss(result, target);
    mse_result->backward();

    auto example_tensor_grad = core::to_xtensor(example_tensor->get_grad());
    xt::xarray<float> expected_example_tensor_grad = xt::zeros_like(a_xarray);
    EXPECT_TRUE(xt::allclose(example_tensor_grad, expected_example_tensor_grad, 5e-2F, 1e-3F));

    auto gamma_grad = core::to_xtensor(gamma->get_grad());
    xt::xarray<float> expected_gamma_grad = {{{{0.36111F, 0.37644F, 0.39589F, 0.41945F, 0.44712F}}}};
    EXPECT_TRUE(xt::allclose(gamma_grad, expected_gamma_grad, 5e-2F));
}

// ============================================================================
// Section 3: RMSNorm Kernel vs Composite Implementation Comparison
// ============================================================================
// These tests compare the optimized RMSNorm kernel implementation against the
// composite implementation to ensure they produce identical results across
// different tensor shapes, sizes, and edge cases.
//
// This validation ensures that the kernel optimization maintains correctness
// while providing performance benefits. Both implementations should produce
// identical forward and backward pass results.
// ============================================================================

/**
 * Helper function to compare kernel vs composite implementations of RMSNorm
 *
 * This function tests both forward and backward passes to ensure:
 * 1. Forward pass: kernel and composite produce identical results
 * 2. Backward pass: gradients computed by both implementations match
 * 3. All outputs and gradients are finite (no NaN/Inf values)
 *
 * @param shape Input tensor shape [N, C, H, W] where:
 *   - N: batch size
 *   - C: number of channels (normalized dimension)
 *   - H: height
 *   - W: width
 *
 * Test cases cover various scenarios:
 * - Alignment: C % 32 == 0 (aligned) vs C % 32 != 0 (unaligned/masking)
 * - Memory: fits in L1 cache vs exceeds L1 cache capacity
 * - Block size: odd C (block_size=1) vs even C (block_size=2)
 * - Scale: small to very large C dimensions
 */
static void CompareKernelVsComposite(const std::vector<uint32_t>& shape) {
    using namespace ttml;
    auto* device = &autograd::ctx().get_device();
    float eps = 0.0078125F;

    // Generate random input data
    std::array<uint32_t, 4> gamma_shape = {1, 1, 1, shape[3]};
    auto rng = autograd::ctx().get_generator();
    uint32_t seed1 = rng();
    xt::xarray<float> x_data = ttml::test_utils::make_uniform_xarray<float>(shape, -1.0F, 1.0F, seed1);
    uint32_t seed2 = rng();
    xt::xarray<float> gamma_data = ttml::test_utils::make_uniform_xarray<float>(gamma_shape, 0.0F, 1.0F, seed2);

    // Test forward pass - kernel vs composite
    auto x_kernel = autograd::create_tensor(core::from_xtensor(x_data, device), /* requires_grad */ true);
    auto gamma_kernel = autograd::create_tensor(core::from_xtensor(gamma_data, device), /* requires_grad */ true);
    auto result_kernel = ops::rmsnorm(x_kernel, gamma_kernel, eps);
    auto result_kernel_xtensor = core::to_xtensor(result_kernel->get_value());

    auto x_composite = autograd::create_tensor(core::from_xtensor(x_data, device), /* requires_grad */ true);
    auto gamma_composite = autograd::create_tensor(core::from_xtensor(gamma_data, device), /* requires_grad */ true);
    auto result_composite = ops::rmsnorm_composite(x_composite, gamma_composite, eps);
    auto result_composite_xtensor = core::to_xtensor(result_composite->get_value());

    // Verify output shape matches input shape
    EXPECT_EQ(result_kernel_xtensor.shape(), x_data.shape());
    EXPECT_EQ(result_composite_xtensor.shape(), x_data.shape());

    // Compare forward results
    EXPECT_TRUE(xt::allclose(result_kernel_xtensor, result_composite_xtensor, 4e-2F, 3e-2F));
    EXPECT_TRUE(xt::all(xt::isfinite(result_kernel_xtensor)));
    EXPECT_TRUE(xt::all(xt::isfinite(result_composite_xtensor)));

    // Test backward pass - kernel vs composite
    auto target_composite = autograd::create_tensor(core::zeros_like(result_composite->get_value()));
    auto mse_composite = ops::mse_loss(result_composite, target_composite);
    mse_composite->backward();
    auto target_kernel = autograd::create_tensor(core::zeros_like(result_kernel->get_value()));
    auto mse_kernel = ops::mse_loss(result_kernel, target_kernel);
    mse_kernel->backward();

    // Since composite is finite, the kernel should also be finite
    auto x_grad_composite = core::to_xtensor(x_composite->get_grad());
    auto gamma_grad_composite = core::to_xtensor(gamma_composite->get_grad());
    auto x_grad_kernel = core::to_xtensor(x_kernel->get_grad());
    auto gamma_grad_kernel = core::to_xtensor(gamma_kernel->get_grad());
    // Should be composite here

    // Verify gradients have correct shapes and are finite
    EXPECT_EQ(x_grad_kernel.shape(), x_data.shape());
    EXPECT_EQ(gamma_grad_kernel.shape()[3], shape[3]);
    EXPECT_TRUE(xt::all(xt::isfinite(x_grad_kernel)));
    EXPECT_TRUE(xt::all(xt::isfinite(gamma_grad_kernel)));

    EXPECT_EQ(x_grad_composite.shape(), x_data.shape());
    EXPECT_EQ(gamma_grad_composite.shape()[3], shape[3]);
    EXPECT_TRUE(xt::all(xt::isfinite(x_grad_composite)));
    EXPECT_TRUE(xt::all(xt::isfinite(gamma_grad_composite)));

    // Compare backward results
    EXPECT_TRUE(xt::allclose(x_grad_kernel, x_grad_composite, 1.0e-3F, 2e-3F));
    EXPECT_TRUE(xt::allclose(gamma_grad_kernel, gamma_grad_composite, 1.0e-3F, 2e-3F));

    autograd::ctx().reset_graph();
}

// ============================================================================
// Section 3: Test Cases - RMSNorm Kernel vs Composite Comparison
// ============================================================================
// These tests systematically compare the optimized kernel implementation
// against the composite implementation across different scenarios:
//
// - Memory usage patterns: L1 cache fit vs overflow
// - Tensor alignment: 32-byte aligned vs unaligned (masking required)
// - Block sizes: odd vs even C dimensions
// - Scale testing: small to very large tensor dimensions
// - Training scenarios: realistic model shapes (NanoLlama, etc.)
// ============================================================================

TEST_F(RMSNormOpTest, RMSNorm_Compare_Basic_Small) {
    CompareKernelVsComposite({1U, 1U, 2U, 32U});
}

// Test aligned dimensions (C % 32 == 0) that fit in L1 cache
TEST_F(RMSNormOpTest, RMSNorm_Compare_Aligned_FitsInL1) {
    // C = 1024 (32 * 32), fits in L1 cache
    CompareKernelVsComposite({1U, 1U, 1U, 1024U});

    // C = 4096 (32 * 128), largest size that fits in L1 cache (1 << 12)
    CompareKernelVsComposite({1U, 1U, 1U, 4096U});
}

// Regression: Qwen3-32B hidden size. C = 5120 (Wt = 160) previously tripped the backward's
// under-counted L1 fit-check into the "everything fits in L1" path and then OOM'd on CB allocation.
TEST_F(RMSNormOpTest, RMSNorm_Compare_Aligned_Qwen3_32B_Hidden) {
    CompareKernelVsComposite({1U, 1U, 32U, 5120U});
}

// Test aligned dimensions (C % 32 == 0) that fit in L1 except for gamma
TEST_F(RMSNormOpTest, RMSNorm_Compare_Aligned_L1ExceptGamma) {
    // C = 8192 (1 << 13), fits in L1 except gamma parameter
    CompareKernelVsComposite({1U, 1U, 1U, 8192U});
}

// Test aligned dimensions (C % 32 == 0) that don't fit in L1 cache
TEST_F(RMSNormOpTest, RMSNorm_Compare_Aligned_DoesNotFitInL1) {
    // C = 16384 (1 << 14), does not fit in L1 cache
    CompareKernelVsComposite({1U, 1U, 1U, 16384U});
}

// Test aligned dimensions (C % 32 == 0) with very large C
TEST_F(RMSNormOpTest, RMSNorm_Compare_Aligned_VeryLargeC) {
    // C = 1048576 (1 << 20), very large C dimension (1M elements)
    CompareKernelVsComposite({1U, 1U, 1U, 1048576U});
}

// Test unaligned dimensions (C % 32 != 0) that fit in L1 cache
TEST_F(RMSNormOpTest, RMSNorm_Compare_Unaligned_FitsInL1) {
    // C = 1023 (32 * 31 + 31), requires masking, fits in L1
    CompareKernelVsComposite({1U, 1U, 1U, 1023U});

    // C = 4095 (32 * 127 + 31), requires masking, fits in L1
    CompareKernelVsComposite({1U, 1U, 1U, 4095U});
}

// Test unaligned dimensions (C % 32 != 0) that don't fit in L1 cache
TEST_F(RMSNormOpTest, RMSNorm_Compare_Unaligned_DoesNotFitInL1) {
    // C = 16383 (1 << 14 - 1), requires masking, does not fit in L1
    CompareKernelVsComposite({1U, 1U, 1U, 16383U});
}

// Test unaligned dimensions (C % 32 != 0) with very large C
TEST_F(RMSNormOpTest, RMSNorm_Compare_Unaligned_VeryLargeC) {
    // C = 1048575 (1 << 20 - 1), very large C with masking
    CompareKernelVsComposite({1U, 1U, 1U, 1048575U});

    // C = 1048558 (1 << 20 - 18), very large C with different masking pattern
    CompareKernelVsComposite({1U, 1U, 1U, 1048558U});
}

// Test block_size = 1 (C is odd)
TEST_F(RMSNormOpTest, RMSNorm_Compare_BlockSize1_OddC) {
    CompareKernelVsComposite({1U, 1U, 1U, 33U});   // C = 33 (odd)
    CompareKernelVsComposite({1U, 1U, 1U, 127U});  // C = 127 (odd)
}

// Test block_size = 2 (C is even)
TEST_F(RMSNormOpTest, RMSNorm_Compare_BlockSize2_EvenC) {
    CompareKernelVsComposite({1U, 1U, 1U, 34U});   // C = 34 (even)
    CompareKernelVsComposite({1U, 1U, 1U, 126U});  // C = 126 (even)
}

// Test training-like shapes with NanoLlama dimensions
TEST_F(RMSNormOpTest, RMSNorm_Compare_TrainingShapes_NanoLlama) {
    // NanoLlama training shape: batch=64, seq_len=256, hidden_dim=384
    CompareKernelVsComposite({64U, 1U, 256U, 384U});
}

// Test training-like shapes with LLaMA 7B dimensions
TEST_F(RMSNormOpTest, RMSNorm_Compare_TrainingShapes_NanoGPT) {
    CompareKernelVsComposite({1U, 1U, 512U, 4096U});
}

// Test small batch and sequence dimensions (non-1 values)
TEST_F(RMSNormOpTest, NIGHTLY_RMSNorm_Compare_SmallBatch_NonUnit) {
    CompareKernelVsComposite({2U, 1U, 4U, 64U});
    CompareKernelVsComposite({32U, 1U, 64U, 128U});
}

// Test different masking patterns with larger batches
TEST_F(RMSNormOpTest, NIGHTLY_RMSNorm_Compare_Masking_Patterns) {
    CompareKernelVsComposite({32U, 1U, 1024U, 4091U});  // C % 32 = 11
    CompareKernelVsComposite({32U, 1U, 1024U, 4079U});  // C % 32 = 31
    CompareKernelVsComposite({32U, 1U, 1024U, 4097U});  // C % 32 = 1
}
