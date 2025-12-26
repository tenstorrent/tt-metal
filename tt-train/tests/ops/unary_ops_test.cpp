// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/unary_ops.hpp"

#include <gtest/gtest.h>
#include <sys/random.h>

#include <algorithm>
#include <core/ttnn_all_includes.hpp>
#include <cstdint>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/random.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

namespace ttml::ops::tests {

namespace {

void load_random_data_from_os(std::span<float> data) {
    constexpr auto max_uint32 = std::numeric_limits<std::uint32_t>::max();

    // Get writable bytes from the float span
    auto float_bytes = std::as_writable_bytes(data);

    // Use getrandom to fill with random bytes from OS
    std::size_t total_read = 0;

    while (total_read < float_bytes.size()) {
        const auto remaining_bytes = float_bytes.subspan(total_read);
        // getrandom expects void* and size_t - std::byte* can be safely cast to void*
        const auto bytes_read = getrandom(static_cast<void*>(remaining_bytes.data()), remaining_bytes.size(), 0);

        if (bytes_read < 0) {
            // Fallback to std::random_device if getrandom fails
            std::random_device rd;
            std::uniform_int_distribution<std::uint8_t> dist;
            std::ranges::generate(remaining_bytes, [&]() { return static_cast<std::byte>(dist(rd)); });
            break;
        }
        total_read += static_cast<std::size_t>(bytes_read);
    }

    // Convert random bytes to floats in range [-1.0, 1.0]
    // Use std::as_bytes to safely reinterpret as uint32_t values
    const auto uint32_bytes = std::as_bytes(data);
    const auto uint32_span = std::span{reinterpret_cast<const std::uint32_t*>(uint32_bytes.data()), data.size()};

    std::ranges::transform(uint32_span, data.begin(), [](const std::uint32_t random_uint32) {
        // Convert uint32 to float in [0, 1) range, then scale to [-1.0, 1.0]
        const auto normalized = static_cast<float>(random_uint32) / static_cast<float>(max_uint32);
        return normalized * 2.0F - 1.0F;
    });
}

}  // namespace

class UnaryOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

TEST_F(UnaryOpsTest, GlobalMean) {
    std::vector<float> test_data = {1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F};

    auto shape = ttnn::Shape({2, 1, 1, 4});
    auto tensor = core::from_vector(test_data, shape, &autograd::ctx().get_device());

    auto tensor_ptr = autograd::create_tensor(tensor);

    auto result = mean(tensor_ptr);
    auto result_data = core::to_vector(result->get_value());

    ASSERT_EQ(result_data.size(), 1);
    EXPECT_FLOAT_EQ(result_data[0], 2.5F);

    result->backward();
    auto tensor_grad = core::to_vector(tensor_ptr->get_grad());
    ASSERT_EQ(tensor_grad.size(), test_data.size());
    for (float it : tensor_grad) {
        EXPECT_FLOAT_EQ(it, 0.125F);
    }
}

TEST_F(UnaryOpsTest, LogSoftmax) {
    auto* device = &autograd::ctx().get_device();
    std::vector<float> test_data = {-0.1F, -0.2F, -0.3F, -0.4F, 0.F, -0.2F, -0.3F, -0.4F};
    auto tensor = core::from_vector(test_data, ttnn::Shape({2, 1, 1, 4}), device);
    auto tensor_ptr = autograd::create_tensor(tensor);
    auto result = log_softmax_moreh(tensor_ptr, 3);
    auto result_data = core::to_vector(result->get_value());
    std::vector<float> expected_data = {
        -1.24253553F, -1.34253553F, -1.44253553F, -1.54253553F, -1.17244159F, -1.37244159F, -1.47244159F, -1.57244159F};
    EXPECT_EQ(result_data.size(), expected_data.size());
    for (uint32_t idx = 0; idx < result_data.size(); ++idx) {
        EXPECT_NEAR(result_data[idx], expected_data[idx], 2e-2F);
    }

    result->backward();
    auto tensor_grad = core::to_vector(tensor_ptr->get_grad());
    std::vector<float> expected_grad = {-0.156F, -0.03906F, 0.05078F, 0.1406F, -0.25F, -0.0156F, 0.07421F, 0.16406F};
    EXPECT_EQ(tensor_grad.size(), expected_grad.size());
    for (uint32_t idx = 0; idx < tensor_grad.size(); ++idx) {
        EXPECT_NEAR(tensor_grad[idx], expected_grad[idx], 2e-2F);
    }
}

TEST_F(UnaryOpsTest, Tanh) {
    // Test basic tanh functionality
    std::vector<float> test_data = {-2.0F, -1.0F, -0.5F, 0.0F, 0.5F, 1.0F, 2.0F, 3.0F};
    auto shape = ttnn::Shape({2, 1, 1, 4});
    auto tensor = core::from_vector(test_data, shape, &autograd::ctx().get_device());
    auto tensor_ptr = autograd::create_tensor(tensor);

    auto result = tanh(tensor_ptr);
    auto result_data = core::to_vector(result->get_value());

    // Expected tanh values (approximate)
    std::vector<float> expected_data = {
        -0.96402758F,  // tanh(-2.0)
        -0.76159416F,  // tanh(-1.0)
        -0.46211716F,  // tanh(-0.5)
         0.0F,         // tanh(0.0)
         0.46211716F,  // tanh(0.5)
         0.76159416F,  // tanh(1.0)
         0.96402758F,  // tanh(2.0)
         0.99505475F   // tanh(3.0)
    };

    ASSERT_EQ(result_data.size(), expected_data.size());
    for (uint32_t idx = 0; idx < result_data.size(); ++idx) {
        EXPECT_NEAR(result_data[idx], expected_data[idx], 1e-2F);
    }
}

TEST_F(UnaryOpsTest, TanhBackward) {
    // Test tanh backward pass
    std::vector<float> test_data = {-1.5F, -0.5F, 0.0F, 0.5F, 1.5F, 2.0F, -2.0F, -0.25F};
    auto shape = ttnn::Shape({2, 1, 1, 4});
    auto tensor = core::from_vector(test_data, shape, &autograd::ctx().get_device());
    auto tensor_ptr = autograd::create_tensor(tensor);

    // Apply tanh
    auto result = tanh(tensor_ptr);

    // Create a target of zeros and compute MSE loss
    auto target = autograd::create_tensor(core::zeros_like(result->get_value()));
    auto loss = mse_loss(result, target);

    // Backward pass
    loss->backward();

    // Get gradients
    auto tensor_grad = core::to_vector(tensor_ptr->get_grad());

    // For MSE loss with target=0, gradient should be: 2/N * tanh(x) * (1 - tanh(x)^2)
    // where N is the number of elements
    ASSERT_EQ(tensor_grad.size(), test_data.size());

    // Verify gradients are reasonable (non-zero for non-saturated values)
    for (uint32_t idx = 0; idx < tensor_grad.size(); ++idx) {
        float x = test_data[idx];
        float tanh_x = std::tanh(x);
        float expected_grad = (2.0F / test_data.size()) * tanh_x * (1.0F - tanh_x * tanh_x);
        EXPECT_NEAR(tensor_grad[idx], expected_grad, 1e-2F);
    }
}

TEST_F(UnaryOpsTest, TanhSaturation) {
    // Test tanh behavior at saturation regions
    std::vector<float> test_data = {
        -10.0F, -5.0F, -3.0F, -1.0F,  // Negative saturation region
         10.0F,  5.0F,  3.0F,  1.0F   // Positive saturation region
    };
    auto shape = ttnn::Shape({2, 1, 1, 4});
    auto tensor = core::from_vector(test_data, shape, &autograd::ctx().get_device());
    auto tensor_ptr = autograd::create_tensor(tensor);

    auto result = tanh(tensor_ptr);
    auto result_data = core::to_vector(result->get_value());

    // At extreme values, tanh should be close to ±1
    ASSERT_EQ(result_data.size(), 8);

    // Check negative saturation
    EXPECT_NEAR(result_data[0], -1.0F, 1e-4F);  // tanh(-10)
    EXPECT_NEAR(result_data[1], -0.99991F, 1e-3F);  // tanh(-5)

    // Check positive saturation
    EXPECT_NEAR(result_data[4], 1.0F, 1e-4F);  // tanh(10)
    EXPECT_NEAR(result_data[5], 0.99991F, 1e-3F);  // tanh(5)

    // Test gradients in saturation
    result->backward();
    auto tensor_grad = core::to_vector(tensor_ptr->get_grad());

    // Gradients should be very small in saturation regions
    EXPECT_NEAR(tensor_grad[0], 0.0F, 1e-3F);  // gradient at x=-10
    EXPECT_NEAR(tensor_grad[4], 0.0F, 1e-3F);  // gradient at x=10
}

TEST_F(UnaryOpsTest, Silu) {
    auto N = 4;
    auto C = 1;
    auto H = 20;
    auto W = 5;

    // Load random data from OS using getrandom and copy into tensor
    xt::xarray<float> a = xt::empty<float>({N, C, H, W});
    load_random_data_from_os(std::span{a.data(), a.size()});

    // Create two input tensors - one for kernel implementation, one for composite
    auto a_kernel = autograd::create_tensor(core::from_xtensor(a, &autograd::ctx().get_device()));
    auto a_composite = autograd::create_tensor(core::from_xtensor(a, &autograd::ctx().get_device()));

    // Forward pass - both use same forward implementation (ttnn::silu)
    // but will use different backward implementations
    auto result_kernel = silu(a_kernel);                                   // Default: uses metal kernel backward
    auto result_composite = silu(a_composite, /*use_composite_bw=*/true);  // Uses composite backward

    // Compare forward results - should be identical since forward is the same
    auto kernel_xtensor = core::to_xtensor(result_kernel->get_value());
    auto composite_xtensor = core::to_xtensor(result_composite->get_value());
    EXPECT_TRUE(xt::allclose(kernel_xtensor, composite_xtensor, 8e-3F, 4e-2F));

    // Backward pass - create zero targets for MSE loss
    auto target_kernel = autograd::create_tensor(core::zeros_like(result_kernel->get_value()));
    auto target_composite = autograd::create_tensor(core::zeros_like(result_composite->get_value()));

    // Compute MSE loss: mean((output - 0)^2) = mean(output^2)
    auto loss_kernel = mse_loss(result_kernel, target_kernel);
    auto loss_composite = mse_loss(result_composite, target_composite);

    // Execute backward pass - this triggers different backward implementations
    loss_kernel->backward();     // Uses metal::silu_bw()
    loss_composite->backward();  // Uses ttnn::silu_bw()

    // Compare backward gradients - both implementations should produce same gradients
    auto grad_kernel = core::to_xtensor(a_kernel->get_grad());
    auto grad_composite = core::to_xtensor(a_composite->get_grad());
    EXPECT_TRUE(xt::allclose(grad_kernel, grad_composite, 8e-3F, 4e-2F));
}

}  // namespace ttml::ops::tests
