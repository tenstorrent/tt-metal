// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/distributed/distributed.hpp"

#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/creation/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

namespace ttml::core::distributed {

ttnn::Tensor synchronize_tensor(const ttnn::Tensor& tensor, const std::optional<uint32_t> dp_dim) {
    auto* device = &autograd::ctx().get_device();
    TT_FATAL(!dp_dim.has_value() || dp_dim.value() < device->shape().dims(), "Cluster axis must be within mesh shape");
    const auto dp_size = autograd::ctx().get_parallelism_context().get_ddp_size();
    assert(dp_size >= 1U);
    // no need to synchronize if there is only one device
    if (dp_size == 1U) {
        return tensor;
    }

    auto result = ttnn::all_reduce(tensor, dp_dim);
    result = ttnn::multiply(result, 1.0F / static_cast<float>(dp_size));
    return result;
}

void synchronize_gradients(const serialization::NamedParameters& parameters) {
    const auto& pctx = autograd::ctx().get_parallelism_context();
    const auto dp_dim = pctx.get_ddp_axis();
    for (auto& [name, tensor] : parameters) {
        if (tensor->is_grad_initialized()) {
            tensor->set_grad(synchronize_tensor(tensor->get_grad(), dp_dim));
        }
    }
}

}  // namespace ttml::core::distributed
