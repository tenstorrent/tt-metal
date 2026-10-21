// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core/clip_grad_norm.hpp"

#include <cmath>
#include <map>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/distributed/distributed.hpp"
#include "core/tt_tensor_utils.hpp"
#include "serialization/serializable.hpp"
#include "ttnn/distributed/api.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm.hpp"
#include "ttnn/operations/moreh/moreh_norm/moreh_norm.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::core {

namespace {

// The mesh axes a gradient is sharded on. Checked against its parameter's: clipping reads a gradient's layout as its
// parameter's, which only holds once the gradients are synchronized (a partial gradient differs across ranks too).
std::vector<uint32_t> sharded_axes(const std::string& name, const autograd::TensorPtr& parameter) {
    const auto rank = autograd::ctx().get_device().shape().dims();
    std::vector<uint32_t> axes;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        const bool sharded = distributed::is_sharded_on_axis(parameter->get_grad(), axis);
        TT_FATAL(
            sharded == distributed::is_sharded_on_axis(parameter->get_value(), axis),
            "clip_grad_norm: the gradient of {} is {} on mesh axis {} but the parameter is not; synchronize the "
            "gradients before clipping them",
            name,
            sharded ? "sharded" : "replicated",
            axis);
        if (sharded) {
            axes.push_back(axis);
        }
    }
    return axes;
}

// sum(|g|^p) over all gradients of the mesh, scaling every gradient by min(max_norm / (norm + 1e-6), 1). A gradient
// sharded on an axis holds different elements on each rank there, so its sum is added up across that axis; one
// replicated on an axis holds a copy, counted once.
ttnn::Tensor clip_sharded_grad_norm(
    const serialization::NamedParameters& parameters, float max_norm, float p_norm_type, bool error_if_nonfinite) {
    TT_FATAL(std::isfinite(p_norm_type), "clip_grad_norm: the infinity norm of sharded gradients is not supported");
    std::map<std::vector<uint32_t>, std::optional<ttnn::Tensor>> sums_by_axes;
    for (const auto& [name, parameter] : parameters) {
        if (!parameter->get_requires_grad() || !parameter->is_grad_initialized()) {
            continue;
        }
        const auto norm = ttnn::moreh_norm(
            parameter->get_grad(),
            p_norm_type,
            /* dim */ std::nullopt,
            /* keepdim */ true,
            /* output */ std::nullopt,
            /* memory_config */ std::nullopt,
            ComputeKernelConfig::precise());
        const auto power = ttnn::pow(norm, p_norm_type);
        auto& sum = sums_by_axes[sharded_axes(name, parameter)];
        sum = sum.has_value() ? ttnn::add(*sum, power) : power;
    }
    std::optional<ttnn::Tensor> total;
    for (auto& [axes, sum] : sums_by_axes) {
        for (const auto axis : axes) {
            sum = ttnn_fixed::distributed::all_reduce(*sum, axis);
        }
        total = total.has_value() ? ttnn::add(*total, *sum) : *sum;
    }
    TT_FATAL(total.has_value(), "clip_grad_norm: no parameter has a gradient");
    const auto total_norm = ttnn::pow(*total, 1.0F / p_norm_type);
    if (error_if_nonfinite) {
        // Every device holds the same total.
        const auto value = core::to_vector<float>(ttnn::distributed::get_device_tensors(total_norm).front()).front();
        TT_FATAL(std::isfinite(value), "clip_grad_norm: the total norm of the gradients is {}", value);
    }
    const auto coefficient = ttnn::clamp(
        ttnn::multiply(ttnn::reciprocal(ttnn::add(total_norm, 1e-6F)), max_norm),
        /* min */ std::nullopt,
        /* max */ 1.0F);
    for (const auto& [name, parameter] : parameters) {
        if (parameter->get_requires_grad() && parameter->is_grad_initialized()) {
            parameter->set_grad(ttnn::multiply(parameter->get_grad(), coefficient));
        }
    }
    return total_norm;
}

}  // namespace

autograd::TensorPtr clip_grad_norm(
    const serialization::NamedParameters& parameters, float max_norm, float p_norm_type, bool error_if_nonfinite) {
    std::vector<ttnn::Tensor> grads;
    grads.reserve(parameters.size());
    bool any_sharded = false;
    for (const auto& [name, tensor] : parameters) {
        if (tensor->get_requires_grad() && tensor->is_grad_initialized()) {
            grads.push_back(tensor->get_grad());
            any_sharded = any_sharded || !sharded_axes(name, tensor).empty();
        }
    }
    if (any_sharded) {
        return autograd::create_tensor(clip_sharded_grad_norm(parameters, max_norm, p_norm_type, error_if_nonfinite));
    }
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
