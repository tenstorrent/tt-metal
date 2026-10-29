// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/unary_ops.hpp"

#include <gtest/gtest.h>
#include <sys/random.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/random.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"
#include "xtensor/core/xmath.hpp"

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

// GELU-related constants (compare to the ones in sfpu kernels)
constexpr float kSqrt2 = 1.41421356237309504880F;
constexpr float kInvSqrt2Pi = 0.3989422804014327F;   // 1 / sqrt(2 * pi)
constexpr float kSqrt2OverPi = 0.7978845608028654F;  // sqrt(2 / pi)
constexpr float kGeluTanhK = 0.044715F;

// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
xt::xarray<float> gelu_exact_reference(const xt::xarray<float>& x) {
    return 0.5F * x * (1.0F + xt::erf(x / kSqrt2));
}

// d/dx of exact GELU: Phi(x) + x * phi(x)
xt::xarray<float> gelu_exact_grad_reference(const xt::xarray<float>& x) {
    xt::xarray<float> phi_cdf = 0.5F * (1.0F + xt::erf(x / kSqrt2));
    xt::xarray<float> phi_pdf = kInvSqrt2Pi * xt::exp(-0.5F * x * x);
    return phi_cdf + x * phi_pdf;
}

// Hendrycks tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
xt::xarray<float> gelu_tanh_reference(const xt::xarray<float>& x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654F;  // sqrt(2 / pi)
    xt::xarray<float> t = xt::tanh(kSqrt2OverPi * (x + 0.044715F * x * x * x));
    return 0.5F * x * (1.0F + t);
}

// d/dx of the tanh approximation: 0.5*(1 + t) + 0.5*x*(1 - t^2)*sqrt(2/pi)*(1 + 3*0.044715*x^2)
xt::xarray<float> gelu_tanh_grad_reference(const xt::xarray<float>& x) {
    xt::xarray<float> t = xt::tanh(kSqrt2OverPi * (x + kGeluTanhK * x * x * x));
    return 0.5F * (1.0F + t) + 0.5F * x * (1.0F - t * t) * kSqrt2OverPi * (1.0F + 3.0F * kGeluTanhK * x * x);
}

}  // namespace

class UnaryOpsTest : public ::testing::Test {
public:
    static void SetUpTestSuite() {
        autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        autograd::ctx().close_device();
    }
};

TEST_F(UnaryOpsTest, GlobalMean) {
    std::vector<float> test_data = {1.F, 2.F, 3.F, 4.F, 1.F, 2.F, 3.F, 4.F};

    auto shape = ttnn::Shape({2, 1, 1, 4});
    auto tensor = core::from_vector(test_data, shape, &autograd::ctx().get_device());

    auto tensor_ptr = autograd::create_tensor(tensor, /* requires_grad */ true);

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
    auto tensor_ptr = autograd::create_tensor(tensor, /* requires_grad */ true);
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

class LogSoftmaxTest : public UnaryOpsTest, public ::testing::WithParamInterface<std::tuple<ttnn::DataType, int>> {
protected:
    void TearDown() override {
        autograd::ctx().reset_graph();
    }

    ttnn::Tensor from_data(const std::vector<float>& data, const ttnn::Shape& shape) {
        auto* device = &autograd::ctx().get_device();
        return std::get<0>(GetParam()) == ttnn::DataType::FLOAT32
                   ? core::from_vector<float, ttnn::DataType::FLOAT32>(data, shape, device)
                   : core::from_vector(data, shape, device);
    }

    void check_reference(
        const ttnn::Shape& shape,
        const std::vector<float>& logits,
        const std::vector<float>& upstream,
        bool expect_nonzero_max = false) {
        const int dim = std::get<1>(GetParam());
        const auto axis = dim < 0 ? dim + static_cast<int>(shape.rank()) : dim;
        size_t inner = 1;
        for (size_t d = axis + 1; d < shape.rank(); ++d) {
            inner *= shape[d];
        }
        const size_t width = shape[axis];
        auto input = autograd::create_tensor(from_data(logits, shape), /* requires_grad */ true);
        auto upstream_tensor = from_data(upstream, shape);
        // The oracle uses the actual quantized inputs, not the pre-conversion host data.
        auto x = core::to_vector(input->get_value(autograd::PreferredPrecision::NATIVE));
        auto dy = core::to_vector(upstream_tensor);
        auto output = log_softmax(input, dim);
        EXPECT_EQ(output->get_value(autograd::PreferredPrecision::NATIVE).dtype(), std::get<0>(GetParam()));
        auto y = core::to_vector(output->get_value(autograd::PreferredPrecision::NATIVE));
        output->set_grad(upstream_tensor);
        output->backward();
        auto dx = core::to_vector(input->get_grad());
        ASSERT_EQ(y.size(), x.size());
        ASSERT_EQ(dx.size(), x.size());
        EXPECT_EQ(input->get_grad().dtype(), std::get<0>(GetParam()));
        const double rtol = std::get<0>(GetParam()) == ttnn::DataType::FLOAT32 ? 3e-5 : 1e-2;
        const double atol = expect_nonzero_max ? 0.0 : 1e-7;
        for (size_t outer = 0; outer < x.size() / (width * inner); ++outer) {
            for (size_t in = 0; in < inner; ++in) {
                const auto index = [=](size_t k) { return (outer * width + k) * inner + in; };
                size_t pivot = 0;
                double sum_dy = 0;
                for (size_t k = 0; k < width; ++k) {
                    if (x[index(k)] > x[index(pivot)]) {
                        pivot = k;
                    }
                    sum_dy += dy[index(k)];
                }
                double tail = 0;
                for (size_t k = 0; k < width; ++k) {
                    if (k != pivot) {
                        tail += std::exp(static_cast<double>(x[index(k)]) - x[index(pivot)]);
                    }
                }
                const double normalizer = std::log1p(tail);
                double sum_others = 0;
                double sum_dx = 0;
                double norm_dx = 0;
                for (size_t k = 0; k < width; ++k) {
                    const double expected_y = static_cast<double>(x[index(k)]) - x[index(pivot)] - normalizer;
                    EXPECT_NEAR(y[index(k)], expected_y, std::abs(expected_y) * rtol);
                    if (k != pivot) {
                        const double expected_dx = dy[index(k)] - std::exp(expected_y) * sum_dy;
                        EXPECT_NEAR(dx[index(k)], expected_dx, std::abs(expected_dx) * rtol + atol);
                        sum_others += expected_dx;
                    }
                    sum_dx += dx[index(k)];
                    norm_dx += std::abs(dx[index(k)]);
                }
                EXPECT_NEAR(dx[index(pivot)], -sum_others, std::abs(sum_others) * rtol + atol);
                EXPECT_LE(std::abs(sum_dx), norm_dx * rtol + atol);
                if (expect_nonzero_max) {
                    EXPECT_LT(y[index(pivot)], 0.F);
                    EXPECT_LT(dx[index(pivot)], 0.F);
                }
            }
        }
    }
};

TEST_P(LogSoftmaxTest, PreservesConfidentLogProbabilitiesAndGradients) {
    // Cross both BF16 and FP32 precision cliffs, with tile padding on H and W.
    const ttnn::Shape shape({3, 3, 5, 35});
    const int dim = std::get<1>(GetParam());
    const auto axis = dim < 0 ? dim + 4 : dim;
    size_t inner = 1;
    for (int d = axis + 1; d < 4; ++d) {
        inner *= shape[d];
    }
    const size_t width = shape[axis];
    for (float gap : {8.F, 18.F, 24.F}) {
        SCOPED_TRACE(gap);
        std::vector<float> logits(shape.volume());
        std::vector<float> upstream(shape.volume(), 0.F);
        for (size_t outer = 0; outer < logits.size() / (width * inner); ++outer) {
            for (size_t in = 0; in < inner; ++in) {
                const size_t pivot = (outer + in) % width;
                // Negative maxima also catch accidental inclusion of zero tile padding.
                const float offset = (outer + in) % 2 == 0 ? -32.F : 32.F;
                for (size_t k = 0; k < width; ++k) {
                    logits[(outer * width + k) * inner + in] = offset - (k == pivot ? 0.F : gap);
                }
                upstream[(outer * width + pivot) * inner + in] = -1.F;
            }
        }
        check_reference(shape, logits, upstream, /* expect_nonzero_max */ true);
        autograd::ctx().reset_graph();
    }
}

TEST_P(LogSoftmaxTest, TiesAndGeneralUpstream) {
    const ttnn::Shape shape({3, 3, 5, 35});
    std::vector<float> logits(shape.volume());
    std::vector<float> upstream(shape.volume());
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = -static_cast<float>(i % 3);  // Includes tied maxima on several reduction axes.
        upstream[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 8.F;
    }
    check_reference(shape, logits, upstream);
}

TEST_P(LogSoftmaxTest, SingletonAxis) {
    std::vector<uint32_t> dimensions{2, 3, 5, 7};
    const int dim = std::get<1>(GetParam());
    dimensions[dim < 0 ? dim + 4 : dim] = 1;
    const ttnn::Shape shape(dimensions);
    check_reference(shape, std::vector<float>(shape.volume(), -3.F), std::vector<float>(shape.volume(), 0.25F));
}

TEST_F(UnaryOpsTest, LogSoftmaxLargeClassIndexAndGradientAccumulation) {
    const ttnn::Shape shape({1, 1, 1, 513});
    std::vector<float> logits(shape.volume(), -18.F);
    std::vector<float> upstream(shape.volume(), 0.F);
    logits[511] = 0.F;
    upstream[511] = -1.F;
    auto* device = &autograd::ctx().get_device();
    auto input = autograd::create_tensor(core::from_vector(logits, shape, device), /* requires_grad */ true);
    auto output = log_softmax(input, -1);
    auto y = core::to_vector(output->get_value());
    const double tail = 512.0 * std::exp(-18.0);
    EXPECT_NEAR(y[511], -std::log1p(tail), std::log1p(tail) * 1e-2);
    output->set_grad(core::from_vector(upstream, shape, device));
    output->backward(/* retain_graph */ true);
    auto first = core::to_vector(input->get_grad());
    EXPECT_NEAR(first[511], -tail / (1.0 + tail), tail * 1e-2);
    output->backward();
    auto accumulated = core::to_vector(input->get_grad());
    for (size_t i = 0; i < first.size(); ++i) {
        EXPECT_FLOAT_EQ(accumulated[i], 2.F * first[i]);
    }
    autograd::ctx().reset_graph();
}

INSTANTIATE_TEST_SUITE_P(
    PrecisionAndAxes,
    LogSoftmaxTest,
    ::testing::Combine(
        ::testing::Values(ttnn::DataType::BFLOAT16, ttnn::DataType::FLOAT32), ::testing::Values(0, 1, 2, 3, -1)));

TEST_F(UnaryOpsTest, Exp) {
    auto* device = &autograd::ctx().get_device();
    // e^0 = 1, e^1 ≈ 2.71828, e^-1 ≈ 0.36788
    xt::xarray<float> data = {{{{0.F, 1.F, -1.F, 0.5F}}}};
    auto tensor_ptr = autograd::create_tensor(core::from_xtensor(data, device), /* requires_grad */ true);

    auto result = exp(tensor_ptr);
    auto result_xt = core::to_xtensor(result->get_value());

    xt::xarray<float> expected = {{{{1.F, 2.71828F, 0.36788F, 1.64872F}}}};
    EXPECT_TRUE(xt::allclose(result_xt, expected, 1e-2F, 1e-2F));

    result->backward();
    auto grad = core::to_xtensor(tensor_ptr->get_grad());
    // d(e^x)/dx = e^x, upstream grad is 1
    EXPECT_TRUE(xt::allclose(grad, expected, 1e-2F, 1e-2F));
}

TEST_F(UnaryOpsTest, Clip) {
    auto* device = &autograd::ctx().get_device();
    //                 below lo   in range   at hi   above hi
    xt::xarray<float> data = {{{{-5.F, 2.F, 3.F, 10.F}}}};
    auto tensor_ptr = autograd::create_tensor(core::from_xtensor(data, device), /* requires_grad */ true);

    auto result = clip(tensor_ptr, 1.F, 3.F);
    auto result_xt = core::to_xtensor(result->get_value());

    xt::xarray<float> expected = {{{{1.F, 2.F, 3.F, 3.F}}}};
    EXPECT_TRUE(xt::allclose(result_xt, expected));

    result->backward();
    auto grad = core::to_xtensor(tensor_ptr->get_grad());
    // grad passes through where lo <= x <= hi, zero otherwise
    xt::xarray<float> expected_grad = {{{{0.F, 1.F, 1.F, 0.F}}}};
    EXPECT_TRUE(xt::allclose(grad, expected_grad));
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
    auto a_kernel =
        autograd::create_tensor(core::from_xtensor(a, &autograd::ctx().get_device()), /* requires_grad */ true);
    auto a_composite =
        autograd::create_tensor(core::from_xtensor(a, &autograd::ctx().get_device()), /* requires_grad */ true);

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

TEST_F(UnaryOpsTest, Gelu) {
    auto* device = &autograd::ctx().get_device();

    auto N = 4;
    auto C = 1;
    auto H = 20;
    auto W = 5;

    // Load random data from OS using getrandom and copy into tensor
    xt::xarray<float> data = xt::empty<float>({N, C, H, W});
    load_random_data_from_os(std::span{data.data(), data.size()});

    auto tensor_ptr = autograd::create_tensor(core::from_xtensor(data, device), /* requires_grad */ true);
    auto result = gelu(tensor_ptr);
    EXPECT_TRUE(xt::allclose(core::to_xtensor(result->get_value()), gelu_exact_reference(data), 8e-3F, 2e-2F));

    result->backward();
    EXPECT_TRUE(xt::allclose(core::to_xtensor(tensor_ptr->get_grad()), gelu_exact_grad_reference(data), 8e-3F, 2e-2F));
}

TEST_F(UnaryOpsTest, GeluTanh) {
    auto* device = &autograd::ctx().get_device();

    auto N = 4;
    auto C = 1;
    auto H = 20;
    auto W = 5;

    // Load random data from OS using getrandom and copy into tensor
    xt::xarray<float> data = xt::empty<float>({N, C, H, W});
    load_random_data_from_os(std::span{data.data(), data.size()});

    auto tensor_ptr = autograd::create_tensor(core::from_xtensor(data, device), /* requires_grad */ true);
    auto result = gelu(tensor_ptr, GeluVariant::TANH);
    EXPECT_TRUE(xt::allclose(core::to_xtensor(result->get_value()), gelu_tanh_reference(data), 8e-3F, 2e-2F));

    result->backward();
    EXPECT_TRUE(xt::allclose(core::to_xtensor(tensor_ptr->get_grad()), gelu_tanh_grad_reference(data), 8e-3F, 3e-2F));
}

}  // namespace ttml::ops::tests
