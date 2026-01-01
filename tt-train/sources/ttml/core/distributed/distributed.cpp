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

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor, std::optional<uint32_t> dp_dim) {
    auto* device = &autograd::ctx().get_device();
    auto devices_count = device->get_devices().size();
    assert(devices_count >= 1U);
    // no need to synchronize if there is only one device
    if (devices_count == 1U) {
        return tensor;
    }

    auto mesh_shape = device->shape();
    if (dp_dim.has_value()) {
        TT_FATAL(dp_dim.value() >= 0 && dp_dim.value() < mesh_shape.dims(), "Cluster axis must be within mesh shape");
        auto result = ttnn::all_reduce(tensor, dp_dim.value());
        const auto dp_size = mesh_shape[dp_dim.value()];
        result = ttnn::multiply(result, 1.0F / static_cast<float>(dp_size));
        return result;
    } else {
        auto result = ttnn::all_reduce(tensor);
        result = ttnn::multiply(result, 1.0F / static_cast<float>(devices_count));
        return result;
    }
}

void synchronize_gradients(const serialization::NamedParameters& parameters) {
    for (auto& [name, tensor] : parameters) {
        if (tensor->is_grad_initialized()) {
            tensor->set_grad(synchronize_tensor(tensor->get_grad(), dp_dim));
        }
    }
}

}  // namespace ttml::core::distributed
