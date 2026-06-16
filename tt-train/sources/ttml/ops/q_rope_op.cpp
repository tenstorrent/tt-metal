// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/q_rope_op.hpp"

#include <stdexcept>

#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "core/compute_kernel_config.hpp"
#include "metal/ops/q_rope_fw/q_rope_fw.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"

namespace ttml::ops {

namespace {

ttnn::Tensor composite_q_rope_bw(
    const ttnn::Tensor& d_out, const RotaryEmbeddingParams& rope_params, uint32_t qk_nope_dim, uint32_t qk_rope_dim) {
    const auto shape = d_out.logical_shape();
    const uint32_t B = shape[0];
    const uint32_t H = shape[1];
    const uint32_t S = shape[2];
    const uint32_t qk_head = qk_nope_dim + qk_rope_dim;

    ttsl::SmallVector<uint32_t> step = {1, 1, 1, 1};
    auto dL_dq_nope = ttnn::slice(
        d_out, ttsl::SmallVector<uint32_t>{0, 0, 0, 0}, ttsl::SmallVector<uint32_t>{B, H, S, qk_nope_dim}, step);
    auto dL_dq_pe = ttnn::slice(
        d_out, ttsl::SmallVector<uint32_t>{0, 0, 0, qk_nope_dim}, ttsl::SmallVector<uint32_t>{B, H, S, qk_head}, step);

    auto dL_dq_pe_in = ttnn::experimental::rotary_embedding_llama(
        dL_dq_pe,
        rope_params.neg_cos_cache,
        rope_params.neg_sin_cache,
        rope_params.trans_mat,
        /*is_decode_mode=*/false,
        /*memory_config=*/std::nullopt,
        core::ComputeKernelConfig::precise());

    return ttnn::concat(std::vector<ttnn::Tensor>{dL_dq_nope, dL_dq_pe_in}, /*dim=*/3);
}

}  // namespace

autograd::TensorPtr q_rope(
    const autograd::TensorPtr& q_full,
    const RotaryEmbeddingParams& rope_params,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim) {
    auto q_out = ttml::metal::q_rope_fw(
        q_full->get_value(),
        rope_params.cos_cache,
        rope_params.sin_cache,
        rope_params.trans_mat,
        qk_nope_dim,
        qk_rope_dim);

    auto out = autograd::create_tensor(q_out);

    autograd::GradFunction grad_fn = [q_full, rope_params, out, qk_nope_dim, qk_rope_dim]() {
        const auto& dL_dout = out->get_grad();

        const auto dL_dq_full = composite_q_rope_bw(dL_dout, rope_params, qk_nope_dim, qk_rope_dim);

        q_full->add_grad(dL_dq_full);
    };

    out->set_node(autograd::add_backward_node(std::move(grad_fn), out, q_full));
    return out;
}

}  // namespace ttml::ops
