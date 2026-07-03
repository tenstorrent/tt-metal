// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mla_q_rope.hpp"

#include "device/mla_q_rope_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor mla_q_rope(
    const ttnn::Tensor& q_in,
    const ttnn::Tensor& cos_cache,
    const ttnn::Tensor& sin_cache,
    const ttnn::Tensor& trans_mat,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim) {
    return ttnn::prim::ttml_mla_q_rope(q_in, cos_cache, sin_cache, trans_mat, qk_nope_dim, qk_rope_dim);
}

}  // namespace ttml::metal
