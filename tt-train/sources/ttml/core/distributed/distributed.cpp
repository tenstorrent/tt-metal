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

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor, const ttsl::SmallVector<uint32_t>& cluster_axes) {
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
    auto result = tensor;
    for (const auto& cluster_axis : cluster_axes) {
        result = ttml::ttnn_fixed::distributed::all_reduce(result, cluster_axis);
    }

    result = ttnn::multiply(result, 1.0F / static_cast<float>(scaler));
    return result;
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

// Averages each gradient along the axes in `cluster_axes` that its parameter is not sharded on.
void synchronize_gradients_along(
    const serialization::NamedParameters& parameters, const ttsl::SmallVector<uint32_t>& cluster_axes) {
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
        // all_reduce labels its own result.
        tensor->set_grad(synchronize_tensor(tensor->get_grad(), axes_for_param));
    }
}

}  // namespace

void synchronize_gradients(const serialization::NamedParameters& parameters) {
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
    synchronize_gradients_along(parameters, cluster_axes);
}

void synchronize_gradients(
    const serialization::NamedParameters& parameters, const std::vector<uint32_t>& cluster_axes) {
    synchronize_gradients_along(parameters, ttsl::SmallVector<uint32_t>(cluster_axes.begin(), cluster_axes.end()));
}

}  // namespace ttml::core::distributed
