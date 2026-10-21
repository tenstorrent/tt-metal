// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dropout_op.hpp"

#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/distributed/distributed.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/experimental/dropout/dropout.hpp"

namespace ttml::ops {

autograd::TensorPtr dropout(const autograd::TensorPtr& tensor, float probability, std::optional<bool> per_device_seed) {
    if (probability == 0.0F) {
        return tensor;
    }
    const bool use_per_device_seed =
        per_device_seed.value_or(core::distributed::differs_on_every_device(tensor->get_value()));

    auto dropout_seed = autograd::ctx().get_generator()();
    auto scaler = 1.0F / (1.0F - probability);
    auto masked_out = ttnn::experimental::dropout(
        tensor->get_value(), probability, scaler, static_cast<uint32_t>(dropout_seed), use_per_device_seed);
    auto out = autograd::create_tensor();
    out->set_value(masked_out);
    autograd::GradFunction grad = [tensor, probability, scaler, out, dropout_seed, use_per_device_seed]() {
        auto res = ttnn::experimental::dropout(
            out->get_grad(), probability, scaler, static_cast<uint32_t>(dropout_seed), use_per_device_seed);
        tensor->add_grad(res);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));

    return out;
}

}  // namespace ttml::ops
