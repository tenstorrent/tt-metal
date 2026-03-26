// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_op.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

namespace {

constexpr auto none = ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam>{};
enum class PolyNorm3ForwardVariant : uint8_t {
    Fused,
    CompositeComparisonOnly,
};
// Keep fused as the production default.
constexpr PolyNorm3ForwardVariant kPolyNorm3ForwardVariant = PolyNorm3ForwardVariant::Fused;

void validate_input_shapes(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& weight, const autograd::TensorPtr& bias) {
    const auto input_shape = tensor->get_value().logical_shape();
    if (input_shape.rank() != 4) {
        throw std::runtime_error("polynorm3 only supports rank-4 input tensors.");
    }

    const auto [batch, n, seq, channels] = input_shape.to_array_4D();
    (void)batch;
    (void)seq;
    (void)channels;
    if (n != 1U) {
        throw std::runtime_error("polynorm3 expects input shape [B, 1, S, C].");
    }

    const auto weight_shape = weight->get_value().logical_shape().to_array_4D();
    // PolyNorm3 is intentionally fixed to three polynomial terms (x, x^2, x^3).
    if (weight_shape != std::array<uint32_t, 4>{1, 1, 1, 3}) {
        throw std::runtime_error("polynorm3 expects weight shape [1, 1, 1, 3].");
    }

    const auto bias_shape = bias->get_value().logical_shape().to_array_4D();
    if (bias_shape != std::array<uint32_t, 4>{1, 1, 1, 1}) {
        throw std::runtime_error("polynorm3 expects bias shape [1, 1, 1, 1].");
    }
}

ttnn::Tensor extract_scalar_from_last_dim(const ttnn::Tensor& tensor, uint32_t index) {
    const ttsl::SmallVector<uint32_t> start = {0U, 0U, 0U, index};
    const ttsl::SmallVector<uint32_t> end = {1U, 1U, 1U, index + 1U};
    const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};
    return ttnn::slice(tensor, start, end, step);
}

std::array<ttnn::Tensor, 3> split_weight_scalars(const ttnn::Tensor& weight_tensor) {
    return {
        extract_scalar_from_last_dim(weight_tensor, 0U),
        extract_scalar_from_last_dim(weight_tensor, 1U),
        extract_scalar_from_last_dim(weight_tensor, 2U)};
}

std::pair<ttnn::Tensor, ttnn::Tensor> rms_normalize_last_dim(const ttnn::Tensor& x, float epsilon) {
    const auto x2 = ttnn::square(x);
    const auto mean_x2 = ttnn::mean(x2, /*dim_arg=*/-1, /*keepdim=*/true);
    const auto inv_rms = ttnn::rsqrt(ttnn::add(mean_x2, epsilon));
    return {ttnn::multiply(x, inv_rms, std::nullopt, std::nullopt, std::nullopt, none, none, none, false), inv_rms};
}

ttnn::Tensor grad_wrt_rmsnorm_input(
    const ttnn::Tensor& term,
    const ttnn::Tensor& grad_normed_term,
    const ttnn::Tensor& inv_rms,
    float inv_channel_count) {
    const auto inv_rms_cubed = ttnn::multiply(
        ttnn::multiply(inv_rms, inv_rms, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
        inv_rms,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        none,
        none,
        none,
        false);
    const auto scale = ttnn_fixed::sum_over_dim(
        ttnn::multiply(term, grad_normed_term, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
        3);  // [B,1,S,1]
    const auto rhs = ttnn::multiply(
        ttnn::multiply(term, scale, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
        ttnn::multiply(
            inv_rms_cubed, inv_channel_count, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        none,
        none,
        none,
        false);
    const auto lhs =
        ttnn::multiply(grad_normed_term, inv_rms, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
    return ttnn::subtract(lhs, rhs, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
}

ttnn::Tensor scalar_sum(const ttnn::Tensor& x) {
    return ttnn::sum(
        x,
        /* dim_arg */ ttsl::SmallVector<int>{0, 1, 2, 3},
        /* keep_dim */ true,
        /* output_mem_config */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::precise());
}

ttnn::Tensor polynorm3_composite_forward(
    const ttnn::Tensor& x, const ttnn::Tensor& weight_tensor, const ttnn::Tensor& bias_tensor, float epsilon) {
    const auto [w0, w1, w2] = split_weight_scalars(weight_tensor);
    const auto x2 = ttnn::square(x);
    const auto x3 = ttnn::multiply(x, x2, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

    const auto [x_norm, _x_inv_rms] = rms_normalize_last_dim(x, epsilon);
    const auto [x2_norm, _x2_inv_rms] = rms_normalize_last_dim(x2, epsilon);
    const auto [x3_norm, _x3_inv_rms] = rms_normalize_last_dim(x3, epsilon);
    (void)_x_inv_rms;
    (void)_x2_inv_rms;
    (void)_x3_inv_rms;

    const auto y3 = ttnn::multiply(x3_norm, w0, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
    const auto y2 = ttnn::multiply(x2_norm, w1, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
    const auto y1 = ttnn::multiply(x_norm, w2, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
    const auto sum = ttnn::add(
        ttnn::add(y3, y2, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
        y1,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        none,
        none,
        none,
        false);
    return ttnn::add(sum, bias_tensor, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
}

ttnn::Tensor polynorm3_forward_variant(
    const ttnn::Tensor& x,
    const ttnn::Tensor& weight_tensor,
    const ttnn::Tensor& bias_tensor,
    float epsilon,
    PolyNorm3ForwardVariant forward_variant) {
    if (forward_variant == PolyNorm3ForwardVariant::Fused) {
        if (x.logical_shape()[-1] % 32U != 0U) {
            throw std::runtime_error(
                "polynorm3 fused forward currently requires C to be divisible by 32 (no tail-channel masking yet).");
        }
        return metal::polynorm3_fw(x, weight_tensor, bias_tensor, epsilon);
    }
    // Comparison-only path for fused-vs-composite validation; do not use in production runs.
    return polynorm3_composite_forward(x, weight_tensor, bias_tensor, epsilon);
}

autograd::TensorPtr polynorm3_impl(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    float epsilon,
    PolyNorm3ForwardVariant forward_variant) {
    validate_input_shapes(tensor, weight, bias);

    const auto x = tensor->get_value();
    const auto weight_tensor = weight->get_value();
    const auto bias_tensor = bias->get_value();
    const auto out_value = polynorm3_forward_variant(x, weight_tensor, bias_tensor, epsilon, forward_variant);

    auto out = autograd::create_tensor(out_value);

    autograd::GradFunction grad = [tensor, weight, bias, out, epsilon]() {
        const auto x = tensor->get_value();
        const auto x2 = ttnn::square(x);
        const auto x3 = ttnn::multiply(x, x2, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

        const auto [x_norm, x_inv_rms] = rms_normalize_last_dim(x, epsilon);
        const auto [x2_norm, x2_inv_rms] = rms_normalize_last_dim(x2, epsilon);
        const auto [x3_norm, x3_inv_rms] = rms_normalize_last_dim(x3, epsilon);

        const auto dL_dout = out->get_grad();
        const auto [w0, w1, w2] = split_weight_scalars(weight->get_value());

        auto dL_dx = ttnn::multiply(x, 0.0F, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
        const float inv_channels = 1.0F / static_cast<float>(x.logical_shape()[-1]);

        const auto dL_dx_term1 = grad_wrt_rmsnorm_input(
            x,
            ttnn::multiply(dL_dout, w2, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
            x_inv_rms,
            inv_channels);
        dL_dx = ttnn::add(dL_dx, dL_dx_term1, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

        const auto dL_dx2 = grad_wrt_rmsnorm_input(
            x2,
            ttnn::multiply(dL_dout, w1, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
            x2_inv_rms,
            inv_channels);
        const auto dL_dx_term2 = ttnn::multiply(
            dL_dx2,
            ttnn::multiply(x, 2.0F, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none,
            false);
        dL_dx = ttnn::add(dL_dx, dL_dx_term2, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

        const auto dL_dx3 = grad_wrt_rmsnorm_input(
            x3,
            ttnn::multiply(dL_dout, w0, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
            x3_inv_rms,
            inv_channels);
        const auto dL_dx_term3 = ttnn::multiply(
            dL_dx3,
            ttnn::multiply(x2, 3.0F, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none,
            false);
        dL_dx = ttnn::add(dL_dx, dL_dx_term3, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
        tensor->add_grad(dL_dx);

        const auto dL_dw0 = scalar_sum(
            ttnn::multiply(dL_dout, x3_norm, std::nullopt, std::nullopt, std::nullopt, none, none, none, false));
        const auto dL_dw1 = scalar_sum(
            ttnn::multiply(dL_dout, x2_norm, std::nullopt, std::nullopt, std::nullopt, none, none, none, false));
        const auto dL_dw2 = scalar_sum(
            ttnn::multiply(dL_dout, x_norm, std::nullopt, std::nullopt, std::nullopt, none, none, none, false));

        auto dL_dw = ttnn::concat(std::vector<ttnn::Tensor>{dL_dw0, dL_dw1, dL_dw2}, /*dim=*/3);
        weight->add_grad(dL_dw);

        bias->add_grad(scalar_sum(dL_dout));
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor, weight, bias));
    return out;
}

}  // namespace

autograd::TensorPtr polynorm3(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    float epsilon) {
    return polynorm3_impl(tensor, weight, bias, epsilon, kPolyNorm3ForwardVariant);
}

autograd::TensorPtr polynorm3_composite(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    float epsilon) {
    return polynorm3_impl(tensor, weight, bias, epsilon, PolyNorm3ForwardVariant::CompositeComparisonOnly);
}

}  // namespace ttml::ops
