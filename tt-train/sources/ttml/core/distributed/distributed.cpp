// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core/distributed/distributed.hpp"

#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/ccl/all_reduce/all_reduce.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::core::distributed {

namespace {

// fp32 sums of at least this many elements per device (1 MiB of bf16) go through a reduce-scatter in fp32 and an
// all-gather in bf16; smaller ones through one fp32 all-reduce. Measured on a Blackhole LoudBox (8x p150b) in 2026-10:
// below about 1 MiB a call costs host dispatch, and the reduce-scatter route issues more ops; above it, the route moves
// a quarter fewer bytes because the all-gather carries the bf16 result. Both routes give bit-identical results.
constexpr uint64_t kFp32ReduceScatterMinElements = 1ULL << 19U;

// The innermost dim that a reduce-scatter can split into `num_devices` whole-tile slices, if any. For a tensor of whole
// tiles it is the dim all_reduce scatters on internally, so both routes add the same values in the same order.
std::optional<int> whole_tile_scatter_dim(const ttnn::Tensor& tensor, uint32_t num_devices) {
    const auto& shape = tensor.logical_shape();
    const int rank = static_cast<int>(shape.rank());
    if (tensor.layout() != ttnn::Layout::TILE || rank < 2) {
        return std::nullopt;
    }
    const auto& tile = tensor.tensor_spec().tile();
    const uint32_t tile_height = tile.get_height();
    const uint32_t tile_width = tile.get_width();
    // Padded tiles would end up in some devices' slices.
    if (shape[-1] % tile_width != 0 || shape[-2] % tile_height != 0) {
        return std::nullopt;
    }
    if (shape[-1] % (tile_width * num_devices) == 0) {
        return rank - 1;
    }
    if (shape[-2] % (tile_height * num_devices) == 0) {
        return rank - 2;
    }
    return std::nullopt;
}

ttnn::Tensor sum_in_fp32(const ttnn::Tensor& tensor, const ttsl::SmallVector<uint32_t>& cluster_axes, uint32_t scaler) {
    const auto dtype = tensor.dtype();
    const float inv_scaler = 1.0F / static_cast<float>(scaler);
    const auto to_fp32 = [](const ttnn::Tensor& t) {
        return t.dtype() == ttnn::DataType::FLOAT32 ? t : ttnn::typecast(t, ttnn::DataType::FLOAT32);
    };
    const auto to_dtype = [dtype](const ttnn::Tensor& t) { return t.dtype() == dtype ? t : ttnn::typecast(t, dtype); };

    const auto scatter_dim = cluster_axes.size() == 1U && tensor.logical_volume() >= kFp32ReduceScatterMinElements
                                 ? whole_tile_scatter_dim(tensor, scaler)
                                 : std::nullopt;
    if (scatter_dim.has_value()) {
        const auto slice = ttnn::multiply(
            ttml::ttnn_fixed::distributed::reduce_scatter(to_fp32(tensor), *scatter_dim, cluster_axes[0]), inv_scaler);
        return ttml::ttnn_fixed::distributed::all_gather(to_dtype(slice), *scatter_dim, cluster_axes[0]);
    }

    auto result = to_fp32(tensor);
    for (const auto& cluster_axis : cluster_axes) {
        result = ttml::ttnn_fixed::distributed::all_reduce(result, cluster_axis);
    }
    return to_dtype(ttnn::multiply(result, inv_scaler));
}

}  // namespace

ttnn::Tensor synchronize_tensor(
    const ttnn::Tensor& tensor,
    const ttsl::SmallVector<uint32_t>& cluster_axes,
    std::optional<ttnn::DataType> reduce_dtype) {
    TT_FATAL(
        !reduce_dtype.has_value() || *reduce_dtype == ttnn::DataType::BFLOAT16 ||
            *reduce_dtype == ttnn::DataType::FLOAT32,
        "reduce_dtype must be BFLOAT16 or FLOAT32, got {}",
        *reduce_dtype);
    auto* device = &autograd::ctx().get_device();
    if (cluster_axes.size() == 0) {
        return tensor;
    }
    uint32_t scaler = 1U;
    for (const auto& cluster_axis : cluster_axes) {
        TT_FATAL(cluster_axis < device->shape().dims(), "Cluster axis must be within mesh shape");
        scaler *= device->shape()[cluster_axis];
    }
    if (scaler == 1U) {
        return tensor;
    }
    if (reduce_dtype == ttnn::DataType::FLOAT32) {
        return sum_in_fp32(tensor, cluster_axes, scaler);
    }
    auto result = tensor;
    for (const auto& cluster_axis : cluster_axes) {
        result = ttml::ttnn_fixed::distributed::all_reduce(result, cluster_axis);
    }

    result = ttnn::multiply(result, 1.0F / static_cast<float>(scaler));
    return result;
}

namespace {

// Returns true if the parameter's current placement on the given mesh axis is a
// Shard{...} rather than Replicate.
bool is_sharded_on_axis(const ttnn::Tensor& value, uint32_t axis) {
    const auto& topology = value.tensor_topology();
    const auto& placements = topology.placements();
    if (axis >= placements.size()) {
        return false;
    }
    return std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placements[axis]);
}

// Enforce all_reduce's contract that the reduced mesh axis is Replicate on output.
// ttnn::all_reduce_async::compute_output_topologies drops this relabel for the collapsed 1-D
// distribution TTML produces on a multi-D mesh, so redo it here (idx 0 when collapsed, else axis).
ttnn::Tensor force_replicate_axes(const ttnn::Tensor& t, const ttsl::SmallVector<uint32_t>& reduced_mesh_axes) {
    const auto& topology = t.tensor_topology();
    auto placements = topology.placements();
    bool changed = false;
    for (uint32_t axis : reduced_mesh_axes) {
        const size_t idx = (placements.size() == 1) ? 0 : axis;
        TT_FATAL(
            idx < placements.size(),
            "force_replicate_axes: reduced axis {} out of range for placements (size={})",
            axis,
            placements.size());
        if (!std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(placements[idx])) {
            placements[idx] = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
            changed = true;
        }
    }
    if (!changed) {
        return t;
    }
    auto out = t;
    out.update_tensor_topology(
        tt::tt_metal::TensorTopology(topology.distribution_shape(), std::move(placements), topology.mesh_coords()));
    return out;
}

// The dtype a gradient sync sums in: the one the call passes, else the training config's default.
std::optional<ttnn::DataType> gradient_reduce_dtype(std::optional<ttnn::DataType> reduce_dtype) {
    return reduce_dtype.has_value() ? reduce_dtype : autograd::ctx().get_gradient_reduce_dtype();
}

}  // namespace

void synchronize_gradients(
    const serialization::NamedParameters& parameters, std::optional<ttnn::DataType> reduce_dtype) {
    if (!autograd::ctx().is_parallelism_context_initialized()) {
        return;
    }
    const auto& pctx = autograd::ctx().get_parallelism_context();
    ttsl::SmallVector<uint32_t> cluster_axes;
    if (pctx.is_cp_enabled()) {
        cluster_axes.push_back(pctx.get_cp_axis().value());
    }
    if (pctx.is_ddp_enabled()) {
        cluster_axes.push_back(pctx.get_ddp_axis().value());
    }
    const auto dtype = gradient_reduce_dtype(reduce_dtype);
    for (auto& [name, tensor] : parameters) {
        if (!tensor->is_grad_initialized()) {
            continue;
        }
        // Build per-param axes, dropping any axis on which this parameter is
        // sharded: FSDP already reduce-scattered the grad over the DDP axis in
        // its backward-post hook.
        ttsl::SmallVector<uint32_t> axes_for_param;
        for (uint32_t axis : cluster_axes) {
            if (!is_sharded_on_axis(tensor->get_value(), axis)) {
                axes_for_param.push_back(axis);
            }
        }
        if (axes_for_param.empty()) {
            continue;
        }
        tensor->set_grad(
            force_replicate_axes(synchronize_tensor(tensor->get_grad(), axes_for_param, dtype), axes_for_param));
    }
}

void synchronize_gradients(
    const serialization::NamedParameters& parameters,
    const std::vector<uint32_t>& cluster_axes,
    std::optional<ttnn::DataType> reduce_dtype) {
    ttsl::SmallVector<uint32_t> axes(cluster_axes.begin(), cluster_axes.end());
    const auto dtype = gradient_reduce_dtype(reduce_dtype);
    for (auto& [name, tensor] : parameters) {
        if (tensor->is_grad_initialized()) {
            tensor->set_grad(force_replicate_axes(synchronize_tensor(tensor->get_grad(), axes, dtype), axes));
        }
    }
}

}  // namespace ttml::core::distributed
