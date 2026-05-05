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
    // Identity forward: share the same underlying ttnn::Tensor handle as the input.
    // `add_backward_node_always` below will flip `out->requires_grad` to true and register
    // the node unconditionally. That's intentional: FSDP uses this op purely as a *timing
    // trigger* (to position a callback at a known point in the backward topological order).
    // The node must fire even when no gradient actually flows into `input` — for example
    // at the root model whose input is a token-index tensor that never needs a grad.
    // When `out` has no grad written to it, we simply skip the grad forwarding below.
    auto out = create_tensor(input->get_value(), /* requires_grad */ false);

    GradFunction grad = [input, out, callback = std::move(callback)]() {
        // Fire the user callback at the module boundary. We do not hand grad_output to the
        // callback because TTML autograd closures are side-effect-only (they mutate
        // captured tensors via add_grad). The callback's job is to prepare state so the
        // subsequent / preceding backward closures see the right shapes (FSDP re-gathers
        // weights before matmul_bwd runs, or reduce-scatters grads after it).
        callback();
        if (out->is_grad_initialized()) {
            input->add_grad(out->get_grad());
        }
    };

    out->set_node(add_backward_node_always(std::move(grad), out, input));
    return out;
}

}  // namespace ttml::autograd
