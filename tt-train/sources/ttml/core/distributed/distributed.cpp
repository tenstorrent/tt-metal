// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/distributed/distributed.hpp"

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::core::distributed {

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor) {
    auto* device = &autograd::ctx().get_device();
    auto devices_count = device->get_devices().size();
    assert(devices_count >= 1U);
    // no need to synchronize if there is only one device
    if (devices_count == 1U) {
        return tensor;
    }

    // all_reduce Mean is not supported, use sum and divide by #devices
    auto result = ttnn::experimental::all_reduce(
        tensor, ttnn::operations::reduction::ReduceType::Sum, 1, std::nullopt, ttnn::ccl::Topology::Ring);
    result = ttnn::multiply(result, 1.0F / static_cast<float>(devices_count));
    return result;
}

void synchronize_parameters(const serialization::NamedParameters& parameters) {
    for (auto& [name, tensor] : parameters) {
        if (tensor->is_grad_initialized()) {
            tensor->set_grad(synchronize_tensor(tensor->get_grad()));
        }
    }
}

}  // namespace ttml::core::distributed
