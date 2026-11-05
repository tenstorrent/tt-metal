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
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::core::distributed {

namespace {

// fp32 sums of at least this many elements per device (1 MiB of bf16) go through a reduce-scatter in fp32 and an
// all-gather in bf16; smaller ones through one fp32 all-reduce. Measured on a Blackhole LoudBox (8x p150b) in 2026-10:
// below about 1 MiB a call costs host dispatch, and the reduce-scatter route issues more ops; above it, the route moves
// a quarter fewer bytes because the all-gather carries the bf16 result.
constexpr uint64_t kFp32ReduceScatterMinElements = 1ULL << 19U;

// Whether one of `placements` (one per mesh axis) splits tensor dim `dim` across the devices of its axis (a
// tensor-parallel weight's gradient, say).
bool is_split_by_a_mesh_axis(
    const ttnn::operations::ccl::common::TopologyPlacements& placements,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    const ttnn::Shape& shape,
    uint32_t dim) {
    for (size_t axis = 0; axis < placements.size(); ++axis) {
        const auto* shard = std::get_if<tt::tt_metal::distributed::MeshMapperConfig::Shard>(&placements[axis]);
        if (shard != nullptr && mesh_shape[axis] > 1U && shape.get_normalized_index(shard->dim) == dim) {
            return true;
        }
    }
    return false;
}

// The innermost dim that a reduce-scatter can split into `num_devices` whole-tile slices, if any. A dim that a mesh
// axis already splits, the reduced one included, is skipped: scattering it again leaves no topology label that
// describes the result. So is every dim of a tensor whose label can't be read per mesh axis, which then takes the
// all-reduce. ttnn's all_reduce over a cluster axis scatters the innermost dim that splits evenly by that axis's size,
// the same dim, so both routes add the same values in the same order and give the same bits (checked on 8 devices).
// Only for a dim another mesh axis splits do they differ: all_reduce still scatters it, so the last bit can differ.
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
    const auto mesh_shape = tensor.device()->shape();
    const auto placements = ttnn::operations::ccl::common::uncollapse_placements(tensor.tensor_topology(), mesh_shape);
    if (!placements.has_value()) {
        return std::nullopt;
    }
    if (shape[-1] % (tile_width * num_devices) == 0 &&
        !is_split_by_a_mesh_axis(*placements, mesh_shape, shape, static_cast<uint32_t>(rank - 1))) {
        return rank - 1;
    }
    if (shape[-2] % (tile_height * num_devices) == 0 &&
        !is_split_by_a_mesh_axis(*placements, mesh_shape, shape, static_cast<uint32_t>(rank - 2))) {
        return rank - 2;
    }
    return std::nullopt;
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
    const auto dtype = tensor.dtype();
    const float inv_scaler = 1.0F / static_cast<float>(scaler);
    // FLOAT32 sums a narrower tensor in fp32. Dividing by the device count writes the tensor's dtype directly, which
    // gives the same bits as a multiply and then a typecast. Each reassignment of `result` frees the previous
    // intermediate.
    const bool sum_in_fp32 = reduce_dtype == ttnn::DataType::FLOAT32 && dtype != ttnn::DataType::FLOAT32;
    ttnn::Tensor result = sum_in_fp32 ? ttnn::typecast(tensor, ttnn::DataType::FLOAT32) : tensor;
    // The reduce-scatter route pays off only for a sum in fp32 of a narrower tensor, whose all-gather then carries the
    // narrower dtype. It is taken for a sum over one axis, the DDP case it was measured for; a sum over several axes
    // (DDP with CP) keeps one all-reduce per axis.
    const auto scatter_dim =
        sum_in_fp32 && cluster_axes.size() == 1U && tensor.logical_volume() >= kFp32ReduceScatterMinElements
            ? whole_tile_scatter_dim(tensor, scaler)
            : std::nullopt;
    if (scatter_dim.has_value()) {
        result = ttml::ttnn_fixed::distributed::reduce_scatter(result, *scatter_dim, cluster_axes[0]);
        result = ttnn::multiply(result, inv_scaler, dtype);
        return ttml::ttnn_fixed::distributed::all_gather(result, *scatter_dim, cluster_axes[0]);
    }
    for (const auto& cluster_axis : cluster_axes) {
        result = ttml::ttnn_fixed::distributed::all_reduce(result, cluster_axis);
    }
    return ttnn::multiply(result, inv_scaler, dtype);
}

bool is_sharded_on_axis(const ttnn::Tensor& value, uint32_t axis) {
    const auto* device = value.device();
    TT_FATAL(
        device != nullptr,
        "is_sharded_on_axis: the tensor is not allocated on a device mesh (a host tensor, or a deallocated one); its "
        "label is read against the mesh it is on");
    const auto mesh_shape = device->shape();
    TT_FATAL(
        axis < mesh_shape.dims(),
        "is_sharded_on_axis: mesh axis {} is out of range for the {} device mesh",
        axis,
        mesh_shape);
    std::string reason;
    const auto placements =
        ttnn::operations::ccl::common::uncollapse_placements(value.tensor_topology(), mesh_shape, &reason);
    TT_FATAL(
        placements.has_value(),
        "is_sharded_on_axis: cannot tell whether the tensor is sharded on mesh axis {}: {}",
        axis,
        reason);
    return std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>((*placements)[axis]);
}

namespace {

// Averages each gradient along the axes in `cluster_axes` that its parameter is not sharded on, in the reduce dtype
// the call passes, else the training config's default.
void synchronize_gradients_along(
    const serialization::NamedParameters& parameters,
    const ttsl::SmallVector<uint32_t>& cluster_axes,
    std::optional<ttnn::DataType> reduce_dtype) {
    if (!reduce_dtype.has_value()) {
        reduce_dtype = autograd::ctx().get_gradient_reduce_dtype();
    }
    for (auto& [name, tensor] : parameters) {
        if (!tensor->is_grad_initialized()) {
            continue;
        }
        ttsl::SmallVector<uint32_t> axes_for_param;
        for (uint32_t axis : cluster_axes) {
            if (!is_sharded_on_axis(tensor->get_value(), axis)) {
                axes_for_param.push_back(axis);
            }
        }
        if (axes_for_param.empty()) {
            continue;
        }
        // The collective that finishes the sum, all_reduce or the reduce-scatter route's all_gather, labels the result.
        tensor->set_grad(synchronize_tensor(tensor->get_grad(), axes_for_param, reduce_dtype));
    }
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
    synchronize_gradients_along(parameters, cluster_axes, reduce_dtype);
}

void synchronize_gradients(
    const serialization::NamedParameters& parameters,
    const std::vector<uint32_t>& cluster_axes,
    std::optional<ttnn::DataType> reduce_dtype) {
    synchronize_gradients_along(
        parameters, ttsl::SmallVector<uint32_t>(cluster_axes.begin(), cluster_axes.end()), reduce_dtype);
}

ttnn::Tensor average_gradients(
    uint32_t count,
    const std::function<ttnn::Tensor(uint32_t)>& gradient,
    std::optional<ttnn::DataType> reduce_dtype,
    bool sum_over_devices) {
    TT_FATAL(count > 0U, "average_gradients needs at least one gradient");
    TT_FATAL(
        !reduce_dtype.has_value() || *reduce_dtype == ttnn::DataType::BFLOAT16 ||
            *reduce_dtype == ttnn::DataType::FLOAT32,
        "reduce_dtype must be BFLOAT16 or FLOAT32, got {}",
        *reduce_dtype);
    const float inv_count = 1.0F / static_cast<float>(count);
    ttnn::Tensor sum = gradient(0U);
    const auto dtype = sum.dtype();
    // In fp32 when asked for, or when the first gradient already is fp32, every operand is cast to fp32 for the exact
    // fp32 sum (see core::as_fp32). Otherwise the sum runs in the first gradient's dtype.
    const bool in_fp32 = reduce_dtype == ttnn::DataType::FLOAT32 || dtype == ttnn::DataType::FLOAT32;
    if (in_fp32) {
        sum = core::as_fp32(sum);
    }
    for (uint32_t i = 1U; i < count; ++i) {
        sum = ttnn::add(sum, in_fp32 ? core::as_fp32(gradient(i)) : gradient(i));
    }
    sum = ttnn::multiply(sum, inv_count);
    if (sum_over_devices) {
        sum = ttml::ttnn_fixed::distributed::all_reduce(sum);
    }
    return sum.dtype() == dtype ? sum : ttnn::typecast(sum, dtype);
}

}  // namespace ttml::core::distributed
