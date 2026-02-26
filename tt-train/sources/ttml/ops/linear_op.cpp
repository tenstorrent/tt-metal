// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "linear_op.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "ttnn_fixed/matmuls.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

using namespace tt::constants;

namespace ttml::ops {

void ttnn_linear_backward(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const autograd::TensorPtr& out) {
    const auto& tensor_value = tensor->get_value();
    auto volume_without_features =
        tensor_value.logical_volume() / static_cast<uint64_t>(tensor_value.logical_shape()[-1]);
    auto reshaped_tensor = ttnn::reshape(
        tensor_value, ttnn::Shape({static_cast<uint32_t>(volume_without_features), tensor_value.logical_shape()[-1]}));

    auto reshaped_grad = ttnn::reshape(
        out->get_grad(),
        ttnn::Shape({static_cast<uint32_t>(volume_without_features), out->get_grad().logical_shape()[-1]}));
    auto reshaped_weight_grad =
        ttnn_fixed::matmul(reshaped_grad, reshaped_tensor, /* transpose_a */ true, /* transpose_b */ false);
    auto reshaped_tensor_grad =
        ttnn_fixed::matmul(reshaped_grad, weight->get_value(), /* transpose_a */ false, /* transpose_b */ false);
    if (bias) {
        auto reshaped_bias_grad = ttnn_fixed::sum_over_dim(reshaped_grad, /* axis */ 0);
        auto bias_grad = ttnn::reshape(reshaped_bias_grad, bias->get_value().logical_shape());
        bias->add_grad(bias_grad);
    }
    auto weight_grad = ttnn::reshape(reshaped_weight_grad, weight->get_value().logical_shape());

    auto tensor_grad = ttnn::reshape(reshaped_tensor_grad, tensor_value.logical_shape());

    tensor->add_grad(tensor_grad);
    weight->add_grad(weight_grad);
}

void moreh_linear_backward(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& weight,
    const autograd::TensorPtr& bias,
    const autograd::TensorPtr& out) {
    auto tensor_grad = ttnn::empty_like(tensor->get_value());
    auto weight_grad = ttnn::empty_like(weight->get_value());

    auto res = ttnn::moreh_linear_backward(
        out->get_grad(),
        tensor->get_value(),
        weight->get_value(),
        /* are required outputs */ std::vector<bool>{true, true, bias != nullptr},
        bias != nullptr ? std::optional<tt::tt_metal::Tensor>(bias->get_value())
                        : std::optional<tt::tt_metal::Tensor>(std::nullopt),
        tensor_grad,
        weight_grad,
        bias ? std::optional<tt::tt_metal::Tensor>(ttnn::empty_like(bias->get_value()))
             : std::optional<tt::tt_metal::Tensor>(std::nullopt),
        /* input_grad_mem_config */ std::nullopt,
        /* weight_grad_mem_config */ std::nullopt,
        /* bias_grad_mem_config */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::matmul());

    if (!res[0].has_value()) {
        throw std::runtime_error("Tensor gradient is not available");
    }
    tensor->add_grad(res[0].value());

    if (!res[1].has_value()) {
        throw std::runtime_error("Weight gradient is not available");
    }
    weight->add_grad(res[1].value());

    if (res[2].has_value()) {
        bias->add_grad(res[2].value());
    }
}

autograd::TensorPtr linear_op(
    const autograd::TensorPtr& tensor, const autograd::TensorPtr& weight, const autograd::TensorPtr& bias) {
    auto out = autograd::create_tensor();

    const auto grid_size = tensor->get_value().device()->compute_with_storage_grid_size();
    auto core_grid = std::make_optional<ttnn::CoreGrid>(grid_size.x, grid_size.y);

    out->set_value(ttnn::linear(
        tensor->get_value(),
        weight->get_value(),
        bias != nullptr ? std::optional<tt::tt_metal::Tensor>(bias->get_value())
                        : std::optional<tt::tt_metal::Tensor>(std::nullopt),
        /* transpose_a */ false,
        /* tranpose_b */ true,
        /* memory_config */ std::nullopt,
        /* dtype */ std::nullopt,
        /* program_config */ std::nullopt,
        /* activation */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::matmul(),
        /* core_grid */ core_grid));

    autograd::GradFunction grad = [weight, bias, tensor, out]() {
        auto tensor_shape = tensor->get_value().logical_shape();
        auto grad_shape = out->get_grad().logical_shape();
        ttnn_linear_backward(tensor, weight, bias, out);
    };

    auto links = autograd::get_links(weight, tensor, bias);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops
