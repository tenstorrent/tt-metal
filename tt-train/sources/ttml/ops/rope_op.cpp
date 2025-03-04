// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/rope_op.hpp"

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"

namespace ttml::ops {
// trans_mat, sin_cache, cos_cache all precomputed and stored somewhere in the module hierarchy
autograd::TensorPtr rope(const autograd::TensorPtr& input, const RotaryEmbeddingParams& params) {
    if (input->get_value().logical_shape().rank() != 4) {
        throw std::runtime_error(
            "rope only supports rank-4 input tensors, but got rank " +
            std::to_string(input->get_value().logical_shape().rank()));
    }

    // FIXME: mostly use defaults for now, try tweaking.
    auto out_tensor = ttnn::experimental::rotary_embedding_llama(
        input->get_value(), params.cos_cache, params.sin_cache, params.trans_mat);
    auto out = autograd::create_tensor(out_tensor);

    // In the backward pass we rotate by -θ, so we need negated cos and sin
    // caches. Note: we can just reuse trans_mat here since the data movement
    // should be the same on the backward pass (we use the same trick to speed
    // up the matmul, and the matrix used is specified by the cos/sin caches.)
    autograd::GradFunction grad_fn = [input, params, out]() {
        auto dL_dout = out->get_grad();
        auto dL_dinput = ttnn::experimental::rotary_embedding_llama(
            dL_dout, params.neg_cos_cache, params.neg_sin_cache, params.trans_mat);
        input->add_grad(dL_dinput);
    };

    auto links = autograd::get_links(input);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad_fn), links));

    return out;
}

}  // namespace ttml::ops
