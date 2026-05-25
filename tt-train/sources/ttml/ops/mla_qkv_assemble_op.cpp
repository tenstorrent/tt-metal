// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_qkv_assemble_op.hpp"

#include <cstdint>
#include <tuple>

#include "autograd/tensor.hpp"
#include "metal/operations.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttml::ops {

std::tuple<autograd::TensorPtr, autograd::TensorPtr, autograd::TensorPtr> mla_qkv_assemble_fw(
    const autograd::TensorPtr& q_pre,
    const autograd::TensorPtr& kv_up,
    const autograd::TensorPtr& k_pe,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim) {
    auto [q_raw, k_raw, v_raw] = ttml::metal::mla_qkv_assemble_fw(
        q_pre->get_value(), kv_up->get_value(), k_pe->get_value(), n_heads, qk_nope_dim, qk_rope_dim, v_dim);

    auto out_q = autograd::create_tensor(q_raw);
    auto out_k = autograd::create_tensor(k_raw);
    auto out_v = autograd::create_tensor(v_raw);

    return {out_q, out_k, out_v};
}

}  // namespace ttml::ops
