// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core/clip_grad_norm.hpp"

#include <fmt/format.h>

#include <cmath>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/distributed/distributed.hpp"
#include "serialization/serializable.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::core {

namespace {

std::vector<ttnn::Tensor> collect_grads(const serialization::NamedParameters& parameters) {
    std::vector<ttnn::Tensor> grads;
    grads.reserve(parameters.size());
    for (const auto& [_, tensor] : parameters) {
        if (tensor->get_requires_grad() && tensor->is_grad_initialized()) {
            grads.push_back(tensor->get_grad());
        }
    }
    return grads;
}

// step1/step3 take their accessor args from the first input only, so every gradient must match it.
void validate_grads(const std::vector<ttnn::Tensor>& grads) {
    const auto& reference = grads.front();
    for (const auto& grad : grads) {
        if (grad.logical_shape().rank() != 4U) {
            throw std::logic_error("clip_grad_norm: gradients must be 4D");
        }
        if (grad.dtype() != reference.dtype() || grad.memory_config() != reference.memory_config()) {
            throw std::logic_error("clip_grad_norm: all gradients must share dtype and memory config");
        }
    }
}

struct TpGrads {
    std::vector<ttnn::Tensor> sharded;
    std::vector<ttnn::Tensor> replicated;
};

// Splits the gradients by their parameter's placement on `cluster_axis`. Only `cluster_axis` is reduced, so a
// parameter sharded on any other mesh axis would leave each device with the norm of its own piece. With one device on
// `cluster_axis`, every gradient is counted once.
TpGrads split_by_placement(
    const serialization::NamedParameters& parameters,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t cluster_axis) {
    const auto tp_size = mesh_shape[static_cast<int32_t>(cluster_axis)];
    TpGrads grads;
    for (const auto& [name, tensor] : parameters) {
        if (!tensor->get_requires_grad() || !tensor->is_grad_initialized()) {
            continue;
        }
        const auto& value = tensor->get_value();
        for (uint32_t axis = 0; axis < mesh_shape.dims(); ++axis) {
            if (axis != cluster_axis && mesh_shape[static_cast<int32_t>(axis)] > 1U &&
                distributed::is_sharded_on_axis(value, axis)) {
                throw std::logic_error(fmt::format(
                    "clip_grad_norm: parameter '{}' is sharded on mesh axis {}, but only cluster_axis {} is "
                    "reduced (placements {})",
                    name,
                    axis,
                    cluster_axis,
                    value.tensor_topology().placements()));
            }
        }
        if (tp_size > 1U && distributed::is_sharded_on_axis(value, cluster_axis)) {
            grads.sharded.push_back(tensor->get_grad());
        } else {
            grads.replicated.push_back(tensor->get_grad());
        }
    }
    return grads;
}

autograd::TensorPtr clip_grad_norm_tp(
    const serialization::NamedParameters& parameters,
    float max_norm,
    float p_norm_type,
    bool error_if_nonfinite,
    uint32_t cluster_axis) {
    if (!std::isfinite(p_norm_type) || p_norm_type <= 0.0F) {
        throw std::logic_error("clip_grad_norm: p_norm_type must be finite and positive");
    }
    const auto mesh_shape = autograd::ctx().get_device().shape();
    if (cluster_axis >= mesh_shape.dims()) {
        throw std::logic_error(fmt::format(
            "clip_grad_norm: cluster_axis {} out of range for mesh with {} dims", cluster_axis, mesh_shape.dims()));
    }
    const auto [sharded, replicated] = split_by_placement(parameters, mesh_shape, cluster_axis);
    std::vector<ttnn::Tensor> grads = sharded;
    grads.insert(grads.end(), replicated.begin(), replicated.end());
    if (grads.empty()) {
        throw std::logic_error("clip_grad_norm: no gradients to clip");
    }
    validate_grads(grads);

    // step1/step3 currently ignore this (their factories build a default compute config) and accumulate
    // Sum[|g|^p] in bf16, so the norm is underestimated for gradients of more than ~1k tiles per device (#63505).
    const auto compute_config = ttml::core::ComputeKernelConfig::precise();

    // Sharded gradients are disjoint pieces of one tensor, so their partial sums add across the TP axis.
    // Replicated gradients are identical on every TP device, so they are counted once.
    std::optional<ttnn::Tensor> pow_sum;
    if (!sharded.empty()) {
        pow_sum = ttnn_fixed::distributed::all_reduce(
            ttnn::moreh_clip_grad_norm_pow_sum(sharded, p_norm_type, /* memory_config */ std::nullopt, compute_config),
            cluster_axis);
    }
    if (!replicated.empty()) {
        auto replicated_pow_sum = ttnn::moreh_clip_grad_norm_pow_sum(
            replicated, p_norm_type, /* memory_config */ std::nullopt, compute_config);
        pow_sum = pow_sum.has_value() ? ttnn::add(*pow_sum, replicated_pow_sum) : replicated_pow_sum;
    }

    // The p-th root, as in moreh_clip_grad_norm, on a [1, 1] tensor so the norm has the shape that op returns.
    const auto total_pow_sum = ttnn::reshape(*pow_sum, ttnn::Shape({1, 1}));
    auto total_norm = p_norm_type == 2.0F ? ttnn::sqrt(total_pow_sum) : ttnn::pow(total_pow_sum, 1.0F / p_norm_type);

    ttnn::moreh_clip_grad_norm_scale(
        grads, total_norm, max_norm, error_if_nonfinite, /* memory_config */ std::nullopt, compute_config);
    return autograd::create_tensor(total_norm);
}

}  // namespace

autograd::TensorPtr clip_grad_norm(
    const serialization::NamedParameters& parameters,
    float max_norm,
    float p_norm_type,
    bool error_if_nonfinite,
    std::optional<uint32_t> cluster_axis) {
    if (cluster_axis.has_value()) {
        return clip_grad_norm_tp(parameters, max_norm, p_norm_type, error_if_nonfinite, *cluster_axis);
    }
    auto grads = collect_grads(parameters);
    auto tt_result = ttnn::moreh_clip_grad_norm(
        grads,
        max_norm,
        p_norm_type,
        error_if_nonfinite,
        /* total_norm*/ std::nullopt,
        /* memory_config*/ std::nullopt,
        ttml::core::ComputeKernelConfig::precise());
    return autograd::create_tensor(tt_result);
}

}  // namespace ttml::core
