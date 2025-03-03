// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dropout_op.hpp"

#include <core/ttnn_all_includes.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/tt_tensor_utils.hpp"

namespace ttml::ops {

autograd::TensorPtr dropout(const autograd::TensorPtr& tensor, float probability, bool use_per_device_seed) {
    if (probability == 0.0F) {
        return tensor;
    }

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

    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));

    return out;
}

}  // namespace ttml::ops
