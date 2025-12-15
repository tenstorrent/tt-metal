// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/distributed/distributed.hpp"

#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::core::distributed {

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor) {
    auto* device = &autograd::ctx().get_device();
    auto devices_count = device->get_devices().size();
    assert(devices_count >= 1U);
    // no need to synchronize if there is only one device
    if (devices_count == 1U) {
        return tensor;
    }

    auto mesh_shape = device->shape();
    
    // For TP+DP (2D mesh): only synchronize across DP dimension (mesh dim 0), not TP dimension (mesh dim 1)
    // For 1D mesh: synchronize across all devices
    if (mesh_shape.dims() == 2 && mesh_shape[0] > 1) {
        // 2D mesh with multiple DP groups: only reduce along DP dimension (cluster_axis=0)
        // This averages gradients across DP groups while preserving TP sharding
        auto num_dp_groups = mesh_shape[0];
        auto result = ttnn::all_reduce(tensor, /* cluster_axis */ 0U);
        // Average by number of DP groups
        result = ttnn::multiply(result, 1.0F / static_cast<float>(num_dp_groups));
        return result;
    } else {
        // 1D mesh or single DP group: reduce across all devices
        auto result = ttnn::all_reduce(tensor, /* cluster_axis */ std::nullopt);
        // Average by total number of devices
        result = ttnn::multiply(result, 1.0F / static_cast<float>(devices_count));
        return result;
    }
}

void synchronize_gradients(const serialization::NamedParameters& parameters) {
    for (auto& [name, tensor] : parameters) {
        if (tensor->is_grad_initialized()) {
            tensor->set_grad(synchronize_tensor(tensor->get_grad()));
        }
    }
}

}  // namespace ttml::core::distributed
