// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "polynorm_op.hpp"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

namespace {

using PolynomWeights = std::array<float, 3>;

constexpr auto none = ttsl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam>{};

void validate_input_shapes(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& weight, const autograd::TensorPtr& bias) {
    const auto input_shape = tensor->get_value().logical_shape();
    if (input_shape.rank() != 4) {
        throw std::runtime_error("polynorm only supports rank-4 input tensors.");
    }

    const auto [batch, n, seq, channels] = input_shape.to_array_4D();
    (void)batch;
    (void)seq;
    (void)channels;
    if (n != 1U) {
        throw std::runtime_error("polynorm expects input shape [B, 1, S, C].");
    }

    const auto weight_shape = weight->get_value().logical_shape().to_array_4D();
    if (weight_shape != std::array<uint32_t, 4>{1, 1, 1, 3}) {
        throw std::runtime_error("polynorm expects weight shape [1, 1, 1, 3].");
    }

    const auto bias_shape = bias->get_value().logical_shape().to_array_4D();
    if (bias_shape != std::array<uint32_t, 4>{1, 1, 1, 1}) {
        throw std::runtime_error("polynorm expects bias shape [1, 1, 1, 1].");
    }
}

PolynomWeights extract_weights(const autograd::TensorPtr& weight) {
    const auto weight_values = core::to_vector(weight->get_value());
    if (weight_values.size() != 3U) {
        throw std::runtime_error("polynorm weight tensor must have 3 elements.");
    }
    return {weight_values[0], weight_values[1], weight_values[2]};
}

float extract_bias(const autograd::TensorPtr& bias) {
    const auto bias_values = core::to_vector(bias->get_value());
    if (bias_values.size() != 1U) {
        throw std::runtime_error("polynorm bias tensor must have 1 element.");
    }
    return bias_values[0];
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

bool use_fused_forward_path() {
    const char* env = std::getenv("TTML_POLYNORM_USE_FUSED_FW");
    if (env == nullptr) {
        return false;
    }
    std::string value(env);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

bool use_fused_backward_path() {
    const char* env = std::getenv("TTML_POLYNORM_USE_FUSED_BW");
    if (env == nullptr) {
        return false;
    }
    std::string value(env);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

}  // namespace

autograd::TensorPtr polynorm(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    float epsilon) {
    validate_input_shapes(tensor, weight, bias);

    const auto w = extract_weights(weight);
    const auto b = extract_bias(bias);

    const auto x = tensor->get_value();
    const bool use_fused_path = use_fused_forward_path();
    auto out_value = ttnn::Tensor{};
    if (use_fused_path) {
        if (x.logical_shape()[-1] % 32U != 0U) {
            throw std::runtime_error(
                "polynorm fused forward currently requires C to be divisible by 32 (no tail-channel masking yet).");
        }
        out_value = metal::polynorm_fw(x, w[0], w[1], w[2], b, epsilon);
    } else {
        const auto x2 = ttnn::square(x);
        const auto x3 = ttnn::multiply(x, x2, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

        const auto [x_norm, _x_inv_rms] = rms_normalize_last_dim(x, epsilon);
        const auto [x2_norm, _x2_inv_rms] = rms_normalize_last_dim(x2, epsilon);
        const auto [x3_norm, _x3_inv_rms] = rms_normalize_last_dim(x3, epsilon);

        out_value = ttnn::add(
            ttnn::add(
                ttnn::multiply(x3_norm, w[0], std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
                ttnn::multiply(x2_norm, w[1], std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
                std::nullopt,
                std::nullopt,
                std::nullopt,
                none,
                none,
                none,
                false),
            ttnn::multiply(x_norm, w[2], std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none,
            false);
        out_value = ttnn::add(out_value, b, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
    }

    auto out = autograd::create_tensor(out_value);

    const bool use_fused_bw = use_fused_path && use_fused_backward_path();

    autograd::GradFunction grad = [tensor, weight, bias, out, epsilon, use_fused_bw]() {
        const auto x = tensor->get_value();
        const auto x2 = ttnn::square(x);
        const auto x3 = ttnn::multiply(x, x2, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);

        const auto [x_norm, x_inv_rms] = rms_normalize_last_dim(x, epsilon);
        const auto [x2_norm, x2_inv_rms] = rms_normalize_last_dim(x2, epsilon);
        const auto [x3_norm, x3_inv_rms] = rms_normalize_last_dim(x3, epsilon);
        (void)x_inv_rms;
        (void)x2_inv_rms;
        (void)x3_inv_rms;

        const auto dL_dout = out->get_grad();
        const auto [w0, w1, w2_value] = extract_weights(weight);

        ttnn::Tensor dL_dx;
        if (use_fused_bw) {
            dL_dx = metal::polynorm_bw(x, dL_dout, w0, w1, w2_value, epsilon);
        } else {
            dL_dx = ttnn::multiply(x, 0.0F, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
            const float inv_channels = 1.0F / static_cast<float>(x.logical_shape()[-1]);

            const auto dL_dx_term1 = grad_wrt_rmsnorm_input(
                x,
                ttnn::multiply(dL_dout, w2_value, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
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
        }
        tensor->add_grad(dL_dx);

        const auto dL_dw0 = scalar_sum(
            ttnn::multiply(dL_dout, x3_norm, std::nullopt, std::nullopt, std::nullopt, none, none, none, false));
        const auto dL_dw1 = scalar_sum(
            ttnn::multiply(dL_dout, x2_norm, std::nullopt, std::nullopt, std::nullopt, none, none, none, false));
        const auto dL_dw2 = scalar_sum(
            ttnn::multiply(dL_dout, x_norm, std::nullopt, std::nullopt, std::nullopt, none, none, none, false));

        auto dL_dw_values = std::vector<float>{
            core::to_vector(dL_dw0).front(), core::to_vector(dL_dw1).front(), core::to_vector(dL_dw2).front()};
        auto dL_dw = core::from_vector(dL_dw_values, ttnn::Shape({1, 1, 1, 3}), &autograd::ctx().get_device());
        weight->add_grad(dL_dw);

        bias->add_grad(scalar_sum(dL_dout));
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor, weight, bias));
    return out;
}

}  // namespace ttml::ops
