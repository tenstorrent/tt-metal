// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core/clip_grad_norm.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>

#include "core/compute_kernel_config.hpp"
#include "core/distributed/distributed.hpp"
#include "core/tt_tensor_utils.hpp"
#include "serialization/serializable.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/moreh_clip_grad_norm_step1_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/moreh_clip_grad_norm_step3_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::core {

namespace {

// Gradients keyed by the mesh axes (of size > 1) their parameter is sharded on. Along those axes the devices hold
// disjoint pieces of the gradient; along the others they hold copies.
using GradsBySharding = std::map<std::vector<uint32_t>, std::vector<ttnn::Tensor>>;

GradsBySharding group_by_sharding(const serialization::NamedParameters& parameters) {
    GradsBySharding groups;
    for (const auto& [_, tensor] : parameters) {
        if (!tensor->get_requires_grad() || !tensor->is_grad_initialized()) {
            continue;
        }
        const auto& value = tensor->get_value();
        const auto mesh_shape = value.device()->shape();
        std::vector<uint32_t> sharded_axes;
        for (uint32_t axis = 0; axis < mesh_shape.dims(); ++axis) {
            if (mesh_shape[static_cast<int32_t>(axis)] > 1U && distributed::is_sharded_on_axis(value, axis)) {
                sharded_axes.push_back(axis);
            }
        }
        groups[sharded_axes].push_back(tensor->get_grad());
    }
    return groups;
}

// step1/step3 take their accessor args from the first input only, so every gradient must match it.
void validate_grads(const GradsBySharding& groups) {
    const auto& reference = groups.begin()->second.front();
    for (const auto& [_, grads] : groups) {
        for (const auto& grad : grads) {
            if (grad.logical_shape().rank() != 4U) {
                throw std::logic_error("clip_grad_norm: gradients must be 4D");
            }
            if (grad.dtype() != reference.dtype() || grad.memory_config() != reference.memory_config()) {
                throw std::logic_error("clip_grad_norm: all gradients must share dtype and memory config");
            }
        }
    }
}

// step1/step3 assign one gradient per core, so each launch takes at most one gradient per core.
template <typename Launch>
void for_each_launch(const std::vector<ttnn::Tensor>& grads, Launch&& launch) {
    const auto grid = grads.front().device()->compute_with_storage_grid_size();
    const auto max_per_launch = static_cast<uint32_t>(grid.x * grid.y);
    const auto num_grads = static_cast<uint32_t>(grads.size());
    for (uint32_t offset = 0; offset < num_grads; offset += max_per_launch) {
        const auto end = std::min(offset + max_per_launch, num_grads);
        launch(std::vector<ttnn::Tensor>(grads.begin() + offset, grads.begin() + end), offset);
    }
}

// Sum[|g|^p] over every element of every gradient on each device, as a [1, 1, 1, 1] tensor.
ttnn::Tensor pow_sum(
    const std::vector<ttnn::Tensor>& grads, float p, const ttnn::DeviceComputeKernelConfig& compute_config) {
    const auto& reference = grads.front();
    // One tile per gradient, each holding that gradient's sum.
    auto per_grad = ttnn::empty(
        ttnn::Shape({static_cast<uint32_t>(grads.size()), 1U, 1U, 1U}),
        reference.dtype(),
        ttnn::Layout::TILE,
        reference.device(),
        reference.memory_config());
    for_each_launch(grads, [&](const std::vector<ttnn::Tensor>& launch_grads, uint32_t offset) {
        ttnn::prim::moreh_clip_grad_norm_step1(
            launch_grads, p, /* tile_offset_of_tmp_pow_sum */ offset, per_grad, std::nullopt, compute_config);
    });
    return ttnn_fixed::sum_over_dim(per_grad, /* dim */ 0);
}

}  // namespace

autograd::TensorPtr clip_grad_norm(
    const serialization::NamedParameters& parameters, float max_norm, float p_norm_type, bool error_if_nonfinite) {
    if (!std::isfinite(p_norm_type) || p_norm_type <= 0.0F) {
        throw std::logic_error("clip_grad_norm: p_norm_type must be finite and positive");
    }
    const auto groups = group_by_sharding(parameters);
    if (groups.empty()) {
        throw std::logic_error("clip_grad_norm: no gradients to clip");
    }
    validate_grads(groups);

    // step1/step3 currently ignore this (their factories build a default compute config) and accumulate
    // Sum[|g|^p] in bf16, so the norm is underestimated for gradients of more than ~1k tiles per device (#63505).
    const auto compute_config = ComputeKernelConfig::precise();

    // A gradient's pieces along its sharded axes add up to its sum; along the other axes each device holds a copy that
    // is counted once. Every device adds the same values in the same order, so all of them get the same norm.
    std::optional<ttnn::Tensor> total_pow_sum;
    for (const auto& [sharded_axes, grads] : groups) {
        auto group_pow_sum = pow_sum(grads, p_norm_type, compute_config);
        for (const auto axis : sharded_axes) {
            group_pow_sum = ttnn_fixed::distributed::all_reduce(group_pow_sum, axis);
        }
        total_pow_sum = total_pow_sum.has_value() ? ttnn::add(*total_pow_sum, group_pow_sum) : group_pow_sum;
    }

    // The p-th root, as in moreh_clip_grad_norm, on a [1, 1] tensor so the norm has the shape that op returns.
    const auto pow_sum_2d = ttnn::reshape(*total_pow_sum, ttnn::Shape({1, 1}));
    const auto total_norm = p_norm_type == 2.0F ? ttnn::sqrt(pow_sum_2d) : ttnn::pow(pow_sum_2d, 1.0F / p_norm_type);

    if (error_if_nonfinite) {
        const auto norms = core::to_xtensor(total_norm, core::IdentityComposer{});
        for (size_t device = 0; device < norms.size(); ++device) {
            if (!std::isfinite(norms[device](0))) {
                throw std::runtime_error(fmt::format(
                    "clip_grad_norm: the total norm of order {} of the gradients is non-finite on device {}, so they "
                    "cannot be clipped. Pass error_if_nonfinite=false to skip this check and get the non-finite norm "
                    "back; the clipped gradients are then unspecified.",
                    p_norm_type,
                    device));
            }
        }
    }

    // clip_coef = min(max_norm / (total_norm + 1e-6), 1), as in moreh_clip_grad_norm.
    const auto& reference = groups.begin()->second.front();
    auto& device = *reference.device();
    const auto clip_coef = ttnn::div(
        ttnn::full(ttnn::Shape({1}), max_norm, reference.dtype(), ttnn::Layout::TILE, std::ref(device)),
        ttnn::add(total_norm, 1e-6F));
    const auto clip_coef_clamped = ttnn::minimum(
        clip_coef, ttnn::full(ttnn::Shape({1}), 1.0F, reference.dtype(), ttnn::Layout::TILE, std::ref(device)));

    // All gradients share one coefficient, so step3 launches over them together.
    std::vector<ttnn::Tensor> all_grads;
    for (const auto& [_, grads] : groups) {
        all_grads.insert(all_grads.end(), grads.begin(), grads.end());
    }
    for_each_launch(all_grads, [&](const std::vector<ttnn::Tensor>& launch_grads, uint32_t /* offset */) {
        ttnn::prim::moreh_clip_grad_norm_step3(launch_grads, clip_coef_clamped, std::nullopt, compute_config);
    });
    return autograd::create_tensor(total_norm);
}

}  // namespace ttml::core
