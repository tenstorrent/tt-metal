// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "q_rope_fw.hpp"

#include "device/q_rope_fw_device_operation.hpp"

namespace ttml::metal {

ttnn::Tensor q_rope_fw(
    const ttnn::Tensor& q_in,
    const ttnn::Tensor& cos_cache,
    const ttnn::Tensor& sin_cache,
    const ttnn::Tensor& trans_mat,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim) {
    return ttnn::prim::ttml_q_rope_fw(q_in, cos_cache, sin_cache, trans_mat, qk_nope_dim, qk_rope_dim);
}

}  // namespace ttml::metal
