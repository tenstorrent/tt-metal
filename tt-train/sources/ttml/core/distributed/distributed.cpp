// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/distributed/distributed.hpp"

#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/ccl/all_reduce/all_reduce.hpp"
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
        result = ttnn::all_reduce(result, cluster_axis);
    }

    result = ttnn::multiply(result, 1.0F / static_cast<float>(scaler));
    return result;
}

void synchronize_gradients(
    const serialization::NamedParameters& parameters, const std::optional<ttsl::SmallVector<uint32_t>>& cluster_axes) {
    ttsl::SmallVector<uint32_t> cluster_axes_to_use;
    if (cluster_axes.has_value()) {
        cluster_axes_to_use = cluster_axes.value();
    } else {
        if (!autograd::ctx().is_parallelism_context_initialized()) {
            return;
        }
        const auto& pctx = autograd::ctx().get_parallelism_context();
        if (pctx.is_cp_enabled()) {
            cluster_axes_to_use.push_back(pctx.get_cp_axis().value());
        }
        if (pctx.is_ddp_enabled()) {
            cluster_axes_to_use.push_back(pctx.get_ddp_axis().value());
        }
    }
    for (auto& [name, tensor] : parameters) {
        if (tensor->is_grad_initialized()) {
            tensor->set_grad(synchronize_tensor(tensor->get_grad(), cluster_axes_to_use));
        }
    }
}

}  // namespace ttml::core::distributed
