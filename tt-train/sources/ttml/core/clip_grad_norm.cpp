// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core/clip_grad_norm.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <variant>

#include "core/compute_kernel_config.hpp"
#include "serialization/serializable.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm.hpp"
#include "ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/device/moreh_clip_grad_norm_step1_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/device/moreh_clip_grad_norm_step3_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

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

// moreh step1/step3 assign one tensor per core, so a single launch handles at most grid-size tensors.
uint32_t max_inputs_per_launch(const ttnn::distributed::MeshDevice& device) {
    const auto grid = device.compute_with_storage_grid_size();
    return static_cast<uint32_t>(grid.x * grid.y);
}

// Sum[|g|^p] over every element of every gradient, as a [1,1,1,1] device tensor.
ttnn::Tensor pow_sum(
    const std::vector<ttnn::Tensor>& grads, float p, const ttnn::DeviceComputeKernelConfig& compute_config) {
    auto* device = grads.front().device();
    const auto num_grads = static_cast<uint32_t>(grads.size());
    auto tmp_pow_sum = ttnn::empty(
        ttnn::Shape({num_grads, 1, 1, 1}),
        grads.front().dtype(),
        ttnn::Layout::TILE,
        device,
        grads.front().memory_config());

    const auto chunk = max_inputs_per_launch(*device);
    for (uint32_t offset = 0; offset < num_grads; offset += chunk) {
        const auto end = std::min(offset + chunk, num_grads);
        ttnn::prim::moreh_clip_grad_norm_step1(
            std::vector<ttnn::Tensor>(grads.begin() + offset, grads.begin() + end),
            p,
            /* tile_offset_of_tmp_pow_sum */ offset,
            tmp_pow_sum,
            /* memory_config */ std::nullopt,
            compute_config);
    }
    return ttnn_fixed::sum_over_dim(tmp_pow_sum, /* dim */ 0);
}

// step3 writes into the grads and relabels them with the union of all its inputs' placements, so a dim-2 shard
// scaled alongside a dim-3 shard would come back labelled dim 3. Each grad's own label is restored afterwards.
void scale_in_place(
    const std::vector<ttnn::Tensor>& grads,
    const ttnn::Tensor& clip_coef_clamped,
    const ttnn::DeviceComputeKernelConfig& compute_config) {
    std::vector<tt::tt_metal::TensorTopology> topologies;
    topologies.reserve(grads.size());
    for (const auto& grad : grads) {
        topologies.push_back(grad.tensor_topology());
    }

    const auto num_grads = static_cast<uint32_t>(grads.size());
    const auto chunk = max_inputs_per_launch(*grads.front().device());
    for (uint32_t offset = 0; offset < num_grads; offset += chunk) {
        const auto end = std::min(offset + chunk, num_grads);
        ttnn::prim::moreh_clip_grad_norm_step3(
            std::vector<ttnn::Tensor>(grads.begin() + offset, grads.begin() + end),
            clip_coef_clamped,
            /* memory_config */ std::nullopt,
            compute_config);
    }

    for (size_t i = 0; i < grads.size(); ++i) {
        // A Tensor is a handle: this copy shares the stored grad's attributes, so the relabel lands on the grad.
        auto grad = grads[i];
        grad.update_tensor_topology(topologies[i]);
    }
}

// step1/step3 take their accessor args from the first input only, so every gradient must match it.
void validate_grads(const std::vector<ttnn::Tensor>& grads, const ttnn::Tensor& reference) {
    for (const auto& grad : grads) {
        if (grad.logical_shape().rank() != 4U) {
            throw std::logic_error("clip_grad_norm_tp: gradients must be 4D");
        }
        if (grad.dtype() != reference.dtype() || grad.memory_config() != reference.memory_config()) {
            throw std::logic_error("clip_grad_norm_tp: all gradients must share dtype and memory config");
        }
    }
}

// A parameter in both groups would be counted twice and scaled twice.
void reject_duplicates(
    const serialization::NamedParameters& sharded, const serialization::NamedParameters& replicated) {
    std::unordered_set<const autograd::Tensor*> sharded_tensors;
    for (const auto& [_, tensor] : sharded) {
        sharded_tensors.insert(tensor.get());
    }
    for (const auto& [name, tensor] : replicated) {
        if (sharded_tensors.contains(tensor.get())) {
            throw std::logic_error(
                fmt::format("clip_grad_norm_tp: parameter '{}' is in both sharded and replicated", name));
        }
    }
}

// The two groups are reduced differently, so a parameter in the wrong one silently skews the norm. A tensor
// distributed over a collapsed 1-D shape carries a single placement for the whole mesh.
void validate_placements(
    const serialization::NamedParameters& parameters, uint32_t cluster_axis, bool expect_sharded) {
    for (const auto& [name, tensor] : parameters) {
        if (!tensor->get_requires_grad() || !tensor->is_grad_initialized()) {
            continue;
        }
        const auto& placements = tensor->get_value().tensor_topology().placements();
        const size_t idx = placements.size() == 1U ? 0U : cluster_axis;
        const bool is_sharded =
            idx < placements.size() &&
            std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placements[idx]);
        if (is_sharded != expect_sharded) {
            throw std::logic_error(fmt::format(
                "clip_grad_norm_tp: parameter '{}' is passed as {} but is {} on cluster_axis {}",
                name,
                expect_sharded ? "sharded" : "replicated",
                is_sharded ? "sharded" : "replicated",
                cluster_axis));
        }
    }
}

}  // namespace

autograd::TensorPtr clip_grad_norm(
    const serialization::NamedParameters& parameters, float max_norm, float p_norm_type, bool error_if_nonfinite) {
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

autograd::TensorPtr clip_grad_norm_tp(
    const serialization::NamedParameters& sharded,
    const serialization::NamedParameters& replicated,
    float max_norm,
    uint32_t cluster_axis,
    float p_norm_type) {
    if (!std::isfinite(p_norm_type) || p_norm_type <= 0.0F) {
        throw std::logic_error("clip_grad_norm_tp: p_norm_type must be finite and positive");
    }
    reject_duplicates(sharded, replicated);
    auto sharded_grads = collect_grads(sharded);
    auto replicated_grads = collect_grads(replicated);
    if (sharded_grads.empty() && replicated_grads.empty()) {
        throw std::logic_error("clip_grad_norm_tp: no gradients to clip");
    }
    const auto& reference = sharded_grads.empty() ? replicated_grads.front() : sharded_grads.front();
    validate_grads(sharded_grads, reference);
    validate_grads(replicated_grads, reference);

    auto* device = reference.device();
    const auto mesh_shape = device->shape();
    if (cluster_axis >= mesh_shape.dims()) {
        throw std::logic_error(fmt::format(
            "clip_grad_norm_tp: cluster_axis {} out of range for mesh with {} dims", cluster_axis, mesh_shape.dims()));
    }
    const auto tp_size = mesh_shape[static_cast<int32_t>(cluster_axis)];
    if (tp_size > 1U) {
        validate_placements(sharded, cluster_axis, /* expect_sharded */ true);
        validate_placements(replicated, cluster_axis, /* expect_sharded */ false);
    }

    // step1/step3 currently ignore this (their factories build a default compute config) and accumulate
    // Sum[|g|^p] in bf16, so the norm is underestimated for gradients of more than ~1k tiles per device (#63505).
    const auto compute_config = ttml::core::ComputeKernelConfig::precise();

    // Sharded gradients are disjoint pieces of one tensor, so their partial sums add across the TP axis.
    // Replicated gradients are identical on every TP device, so they are counted once.
    std::optional<ttnn::Tensor> total_pow_sum;
    if (!sharded_grads.empty()) {
        total_pow_sum = pow_sum(sharded_grads, p_norm_type, compute_config);
        if (tp_size > 1U) {
            total_pow_sum = ttnn_fixed::distributed::all_reduce(*total_pow_sum, cluster_axis);
        }
    }
    if (!replicated_grads.empty()) {
        auto replicated_pow_sum = pow_sum(replicated_grads, p_norm_type, compute_config);
        total_pow_sum = total_pow_sum.has_value() ? ttnn::add(*total_pow_sum, replicated_pow_sum) : replicated_pow_sum;
    }

    auto total_norm = p_norm_type == 2.0F ? ttnn::sqrt(*total_pow_sum) : ttnn::pow(*total_pow_sum, 1.0F / p_norm_type);

    // clip_coef = min(max_norm / (total_norm + 1e-6), 1), same as moreh_clip_grad_norm.
    auto clip_coef = ttnn::div(
        ttnn::full(ttnn::Shape({1}), max_norm, reference.dtype(), ttnn::Layout::TILE, std::ref(*device)),
        ttnn::add(total_norm, 1e-6F));
    auto clip_coef_clamped = ttnn::minimum(
        clip_coef, ttnn::full(ttnn::Shape({1}), 1.0F, reference.dtype(), ttnn::Layout::TILE, std::ref(*device)));

    if (!sharded_grads.empty()) {
        scale_in_place(sharded_grads, clip_coef_clamped, compute_config);
    }
    if (!replicated_grads.empty()) {
        scale_in_place(replicated_grads, clip_coef_clamped, compute_config);
    }
    return autograd::create_tensor(total_norm);
}

}  // namespace ttml::core
