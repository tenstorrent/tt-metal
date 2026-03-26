// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/distributed/distributed.hpp"

#include <core/ttnn_all_includes.hpp>
#include <tt-metalium/experimental/tensor/topology/distributed_tensor_configs.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
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

void synchronize_gradients(const serialization::NamedParameters& parameters) {
    auto* device = &autograd::ctx().get_device();
    for (auto& [name, tensor] : parameters) {
        if (!tensor->is_grad_initialized()) {
            continue;
        }
        const auto& placements = tensor->get_value().tensor_topology().placements();
        for (size_t axis = 0; axis < placements.size(); ++axis) {
            if (!std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(placements[axis])) {
                continue;
            }
            uint32_t num_devices = device->shape()[axis];
            if (num_devices <= 1) {
                continue;
            }
            auto grad = ttnn_fixed::distributed::all_reduce(tensor->get_grad(), static_cast<uint32_t>(axis));
            tensor->set_grad(ttnn::multiply(grad, 1.0F / static_cast<float>(num_devices)));
        }
    }
}

}  // namespace ttml::core::distributed
