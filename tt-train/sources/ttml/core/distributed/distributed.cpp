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

using Placements = ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>;

// `label` read per axis of `mesh_shape`, by ttnn's rule for collapsed labels. `whose` names the label in the error.
Placements read_per_axis(
    const tt::tt_metal::TensorTopology& label,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    const char* whose) {
    std::string reason;
    auto placements = ttnn::operations::ccl::common::uncollapse_placements(label, mesh_shape, &reason);
    TT_FATAL(placements.has_value(), "cannot read {} placement per mesh axis: {}", whose, reason);
    return std::move(*placements);
}

}  // namespace

Placements placements_per_axis(const ttnn::Tensor& value) {
    const auto* device = value.device();
    TT_FATAL(
        device != nullptr,
        "cannot read the placement per mesh axis of a tensor that is not allocated on a device mesh (a host tensor, "
        "or a deallocated one): its label is read against the mesh it is on");
    return read_per_axis(value.tensor_topology(), device->shape(), "the tensor's");
}

Placements placements_per_axis(
    const tt::tt_metal::distributed::MeshMapperConfig& config, const tt::tt_metal::distributed::MeshShape& mesh_shape) {
    // The label a tensor mapped with `config` gets. Its coordinates only matter when it covers the whole mesh, which a
    // mapper then lists in row-major order.
    const ttnn::MeshCoordinateRange whole_mesh(mesh_shape);
    const tt::tt_metal::TensorTopology label(
        config.mesh_shape_override.value_or(mesh_shape),
        config.placements,
        std::vector(whole_mesh.begin(), whole_mesh.end()));
    return read_per_axis(label, mesh_shape, "the mapper's");
}

bool is_sharded_on_axis(const ttnn::Tensor& value, uint32_t axis) {
    const auto placements = placements_per_axis(value);
    TT_FATAL(
        axis < placements.size(),
        "is_sharded_on_axis: mesh axis {} is out of range for a {}-axis device mesh",
        axis,
        placements.size());
    return std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placements[axis]);
}

void set_placements_per_axis(
    ttnn::Tensor& value, const ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>& placements) {
    const auto mesh_shape = value.device()->shape();
    TT_FATAL(
        placements.size() == mesh_shape.dims(),
        "set_placements_per_axis: {} placements for the {} device mesh",
        placements.size(),
        mesh_shape);
    value.update_tensor_topology(
        tt::tt_metal::TensorTopology(mesh_shape, placements, value.tensor_topology().mesh_coords()));
}

void label_grad_like_parameter_shards(const ttnn::Tensor& parameter, ttnn::Tensor& grad) {
    if (parameter.device() == nullptr || parameter.device()->shape().mesh_size() == 1U) {
        return;
    }
    const auto parameter_placements = placements_per_axis(parameter);
    std::optional<Placements> placements;
    for (size_t axis = 0; axis < parameter_placements.size(); ++axis) {
        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(parameter_placements[axis])) {
            if (!placements) {
                placements = placements_per_axis(grad);
            }
            (*placements)[axis] = parameter_placements[axis];
        }
    }
    if (placements) {
        set_placements_per_axis(grad, *placements);
    }
}

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
