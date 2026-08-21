// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/unary_ops.hpp"

#include <gtest/gtest.h>
#include <sys/random.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
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

// Constants below match the ones the SFPU kernels use, see
// tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_gelu.h
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
    xt::xarray<float> t = xt::tanh(kSqrt2OverPi * (x + kGeluTanhK * x * x * x));
    return 0.5F * x * (1.0F + t);
}

// d/dx of the tanh approximation: 0.5*(1 + t) + 0.5*x*(1 - t^2)*sqrt(2/pi)*(1 + 3*0.044715*x^2)
xt::xarray<float> gelu_tanh_grad_reference(const xt::xarray<float>& x) {
    xt::xarray<float> t = xt::tanh(kSqrt2OverPi * (x + kGeluTanhK * x * x * x));
    return 0.5F * (1.0F + t) + 0.5F * x * (1.0F - t * t) * kSqrt2OverPi * (1.0F + 3.0F * kGeluTanhK * x * x);
}

// Same contract as xt::allclose, but reports the observed max absolute difference on failure so that
// tolerances can be tuned from measurements instead of guesses.
void expect_allclose(
    const xt::xarray<float>& got, const xt::xarray<float>& expected, float rtol, float atol, std::string_view label) {
    EXPECT_TRUE(xt::allclose(got, expected, rtol, atol))
        << label << ": rtol=" << rtol << " atol=" << atol << " max_abs_diff=" << xt::amax(xt::abs(got - expected))();
}

xt::xarray<float> random_input(const std::vector<std::size_t>& shape) {
    xt::xarray<float> data = xt::empty<float>(shape);
    load_random_data_from_os(std::span{data.data(), data.size()});
    return data;
}

// GELU tolerances. On uniform [-1, 1] inputs the measured max absolute error against the references
// above is stable run to run: ~3.7e-3 (ACCURATE fw), ~4.2e-3 (ACCURATE bw), ~3.7e-3 (TANH fw),
// ~1.2e-2 (TANH bw), ~2.4e-2 (FAST_LUT fw, a ~1% approximation by construction). Repeating the same
// measurement with FLOAT32 tensors gives the same numbers, so this is the SFPU approximation itself,
// not bf16/fp32 dest accumulation -- absorb it with atol rather than tightening rtol. GELU is
// ill-conditioned in relative terms on small negative inputs (the value goes to zero while the
// absolute error does not), so rtol alone cannot bound it. rtol matches the Silu test above.
constexpr float kGeluRtol = 8e-3F;
constexpr float kGeluAtol = 2e-2F;
constexpr float kGeluTanhBwAtol = 3e-2F;
constexpr float kGeluFastLutAtol = 5e-2F;

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

TEST_F(UnaryOpsTest, GeluVariantFromString) {
    EXPECT_EQ(gelu_variant_from_string("none"), GeluVariant::ACCURATE);
    EXPECT_EQ(gelu_variant_from_string("accurate"), GeluVariant::ACCURATE);
    EXPECT_EQ(gelu_variant_from_string("tanh"), GeluVariant::TANH);
    EXPECT_EQ(gelu_variant_from_string("fast_lut"), GeluVariant::FAST_LUT);

    EXPECT_THROW((void)gelu_variant_from_string("Tanh"), std::invalid_argument);
    EXPECT_THROW((void)gelu_variant_from_string(""), std::invalid_argument);
    EXPECT_THROW((void)gelu_variant_from_string("approximate"), std::invalid_argument);
}

// Default variant (== GeluVariant::ACCURATE) against the exact GELU and its derivative.
// backward() seeds dL/dout = 1, so the input gradient is GELU'(x) directly.
TEST_F(UnaryOpsTest, Gelu) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data = random_input({4U, 1U, 20U, 5U});
    auto tensor_ptr = autograd::create_tensor(core::from_xtensor(data, device), /* requires_grad */ true);

    auto result = gelu(tensor_ptr);
    expect_allclose(core::to_xtensor(result->get_value()), gelu_exact_reference(data), kGeluRtol, kGeluAtol, "gelu fw");

    result->backward();
    expect_allclose(
        core::to_xtensor(tensor_ptr->get_grad()), gelu_exact_grad_reference(data), kGeluRtol, kGeluAtol, "gelu bw");
}

// The default must stay ACCURATE: this is what keeps GPT numerics unchanged by this feature.
TEST_F(UnaryOpsTest, GeluDefaultVariantIsAccurate) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data = random_input({4U, 1U, 20U, 5U});
    auto default_ptr = autograd::create_tensor(core::from_xtensor(data, device), /* requires_grad */ true);
    auto explicit_ptr = autograd::create_tensor(core::from_xtensor(data, device), /* requires_grad */ true);

    auto default_result = gelu(default_ptr);
    auto explicit_result = gelu(explicit_ptr, GeluVariant::ACCURATE);
    EXPECT_EQ(
        xt::amax(
            xt::abs(core::to_xtensor(default_result->get_value()) - core::to_xtensor(explicit_result->get_value())))(),
        0.0F);

    default_result->backward();
    explicit_result->backward();
    EXPECT_EQ(
        xt::amax(xt::abs(core::to_xtensor(default_ptr->get_grad()) - core::to_xtensor(explicit_ptr->get_grad())))(),
        0.0F);
}

TEST_F(UnaryOpsTest, GeluTanh) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data = random_input({4U, 1U, 20U, 5U});
    auto tensor_ptr = autograd::create_tensor(core::from_xtensor(data, device), /* requires_grad */ true);

    auto result = gelu(tensor_ptr, GeluVariant::TANH);
    expect_allclose(
        core::to_xtensor(result->get_value()), gelu_tanh_reference(data), kGeluRtol, kGeluAtol, "gelu_tanh fw");

    result->backward();
    expect_allclose(
        core::to_xtensor(tensor_ptr->get_grad()),
        gelu_tanh_grad_reference(data),
        kGeluRtol,
        kGeluTanhBwAtol,
        "gelu_tanh bw");
}

// FAST_LUT is a forward-only approximation: ttnn has no LUT backward kernel, so its gradient is the
// exact GELU derivative, bit-identical to the ACCURATE path.
TEST_F(UnaryOpsTest, GeluFastLut) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data = random_input({4U, 1U, 20U, 5U});
    auto fast_ptr = autograd::create_tensor(core::from_xtensor(data, device), /* requires_grad */ true);
    auto accurate_ptr = autograd::create_tensor(core::from_xtensor(data, device), /* requires_grad */ true);

    auto fast_result = gelu(fast_ptr, GeluVariant::FAST_LUT);
    // ~1% absolute error against the exact GELU by construction, hence the looser bound.
    expect_allclose(
        core::to_xtensor(fast_result->get_value()),
        gelu_exact_reference(data),
        kGeluRtol,
        kGeluFastLutAtol,
        "fast_lut fw");

    auto accurate_result = gelu(accurate_ptr, GeluVariant::ACCURATE);
    fast_result->backward();
    accurate_result->backward();
    EXPECT_EQ(
        xt::amax(xt::abs(core::to_xtensor(fast_ptr->get_grad()) - core::to_xtensor(accurate_ptr->get_grad())))(), 0.0F);
}

// The two variants only separate above bfloat16 resolution in the negative tail: for |x| >= 1 the
// exact/tanh gap is below one bf16 ULP, so the [-1, 1] data used by the tests above cannot
// distinguish them. atol must stay 0 here or it swallows the signal.
TEST_F(UnaryOpsTest, GeluAccurateVsTanhDiffer) {
    auto* device = &autograd::ctx().get_device();
    xt::xarray<float> data = {{{{-4.F, -3.5F, -3.F, -2.5F}}}};
    auto accurate_ptr = autograd::create_tensor(core::from_xtensor(data, device), /* requires_grad */ true);
    auto tanh_ptr = autograd::create_tensor(core::from_xtensor(data, device), /* requires_grad */ true);

    auto accurate_xt = core::to_xtensor(gelu(accurate_ptr, GeluVariant::ACCURATE)->get_value());
    auto tanh_xt = core::to_xtensor(gelu(tanh_ptr, GeluVariant::TANH)->get_value());

    // Each variant tracks its own reference far more tightly than it tracks the other one.
    expect_allclose(accurate_xt, gelu_exact_reference(data), 5e-2F, 0.F, "accurate tail");
    expect_allclose(tanh_xt, gelu_tanh_reference(data), 5e-2F, 0.F, "tanh tail");

    EXPECT_GT(xt::amax(xt::abs(accurate_xt - tanh_xt))(), 1e-4F);
    EXPECT_FALSE(xt::allclose(accurate_xt, tanh_xt, 1e-2F, 0.F));
}

}  // namespace ttml::ops::tests
