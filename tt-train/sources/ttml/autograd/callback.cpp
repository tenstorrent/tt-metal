// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "callback.hpp"

#include <utility>

#include "auto_context.hpp"
#include "graph.hpp"
#include "graph_utils.hpp"

namespace ttml::autograd {

TensorPtr autograd_callback(const TensorPtr& input, std::function<void()> callback) {
    auto out = create_tensor(input->get_value(), /* requires_grad */ false);

    GradFunction grad = [input, out, callback = std::move(callback)]() {
        callback();
        if (out->is_grad_initialized()) {
            input->add_grad(out->get_grad());
            out->set_grad(ttnn::Tensor{});  // release our pin; input owns it now
        }
    };

    out->set_node(add_backward_node_always(std::move(grad), out, input));
    return out;
}

}  // namespace ttml::autograd
