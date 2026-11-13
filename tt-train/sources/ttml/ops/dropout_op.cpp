// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dropout_op.hpp"

#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/distributed/distributed.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ttnn/operations/experimental/dropout/dropout.hpp"

namespace ttml::ops {

bool dropout_seed_per_device(const ttnn::Tensor& value) {
    const auto* device = value.device();
    TT_FATAL(device != nullptr, "dropout needs a tensor allocated on a device");
    const auto mesh_shape = device->shape();
    std::vector<uint32_t> replicated;
    std::vector<uint32_t> sharded;
    for (uint32_t axis = 0; axis < mesh_shape.dims(); ++axis) {
        if (mesh_shape[axis] == 1) {
            continue;
        }
        (core::distributed::is_sharded_on_axis(value, axis) ? sharded : replicated).push_back(axis);
    }
    TT_FATAL(
        replicated.empty() || sharded.empty(),
        "dropout cannot mask a tensor replicated across mesh axis {} and sharded across mesh axis {}: its copies need "
        "one mask and its shards different ones, which needs a dropout kernel that seeds per mesh axis; set the "
        "dropout probability to 0",
        replicated.front(),
        sharded.front());
    return !sharded.empty();
}

autograd::TensorPtr dropout(const autograd::TensorPtr& tensor, float probability) {
    if (probability == 0.0F) {
        return tensor;
    }
    // Decided once from the input: the backward must draw the same masks, whatever its gradient's label says.
    const bool use_per_device_seed = dropout_seed_per_device(tensor->get_value());

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
