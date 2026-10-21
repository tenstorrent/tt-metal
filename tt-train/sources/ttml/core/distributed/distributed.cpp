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

ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement> placements_per_axis(
    const ttnn::Tensor& value) {
    // The tensor's own mesh when it is on a device; the context's device otherwise (a host tensor).
    const auto* device = value.device();
    const auto mesh_shape = device != nullptr ? device->shape() : autograd::ctx().get_device().shape();
    std::string reason;
    auto placements =
        ttnn::operations::ccl::common::uncollapse_placements(value.tensor_topology(), mesh_shape, &reason);
    TT_FATAL(placements.has_value(), "cannot read the tensor's placement per mesh axis: {}", reason);
    return std::move(*placements);
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

std::vector<GradientReduction> gradient_reductions(const autograd::TensorPtr& parameter, const AxisKinds& kinds) {
    const auto mesh_shape = autograd::ctx().get_device().shape();
    TT_FATAL(
        kinds.size() == mesh_shape.dims(),
        "gradient_reductions: {} axis kinds for the {} device mesh; give one per mesh axis",
        kinds.size(),
        mesh_shape);
    std::vector<GradientReduction> reductions;
    if (!parameter->is_grad_initialized()) {
        return reductions;
    }
    const auto& value = parameter->get_value(autograd::PreferredPrecision::NATIVE);
    const auto& grad = parameter->get_grad();
    for (uint32_t axis = 0; axis < mesh_shape.dims(); ++axis) {
        // A sharded parameter's gradient is its own shard's, complete: TP and FSDP reduced it in the backward pass.
        if (mesh_shape[axis] == 1U || is_sharded_on_axis(value, axis)) {
            continue;
        }
        const auto& kind = kinds[axis];
        if (kind == AxisKind::Data) {
            reductions.push_back({axis, AxisKind::Data});
            continue;
        }
        if (!is_sharded_on_axis(grad, axis)) {
            continue;
        }
        TT_FATAL(
            kind.has_value(),
            "gradient_reductions: the gradient of a parameter replicated on mesh axis {} differs across that axis, "
            "which has no declared kind; declare it a data axis (averaged) or a model axis (summed)",
            axis);
        reductions.push_back({axis, AxisKind::Model});
    }
    return reductions;
}

void synchronize_gradients(const serialization::NamedParameters& parameters, const AxisKinds& kinds) {
    const auto mesh_shape = autograd::ctx().get_device().shape();
    for (const auto& [name, parameter] : parameters) {
        const auto reductions = gradient_reductions(parameter, kinds);
        if (reductions.empty()) {
            continue;
        }
        auto grad = parameter->get_grad();
        uint32_t data_ranks = 1U;
        for (const auto& [axis, kind] : reductions) {
            grad = ttml::ttnn_fixed::distributed::all_reduce(grad, axis);
            if (kind == AxisKind::Data) {
                data_ranks *= mesh_shape[axis];
            }
        }
        if (data_ranks > 1U) {
            grad = ttnn::multiply(grad, 1.0F / static_cast<float>(data_ranks));
        }
        parameter->set_grad(grad);
    }
}

void synchronize_gradients(const serialization::NamedParameters& parameters) {
    if (!autograd::ctx().is_parallelism_context_initialized()) {
        return;
    }
    const auto& pctx = autograd::ctx().get_parallelism_context();
    AxisKinds kinds(autograd::ctx().get_device().shape().dims());
    for (const auto axis : {pctx.get_ddp_axis(), pctx.get_cp_axis()}) {
        if (axis.has_value()) {
            kinds[*axis] = AxisKind::Data;
        }
    }
    if (const auto axis = pctx.get_tp_axis()) {
        kinds[*axis] = AxisKind::Model;
    }
    synchronize_gradients(parameters, kinds);
}

}  // namespace ttml::core::distributed
