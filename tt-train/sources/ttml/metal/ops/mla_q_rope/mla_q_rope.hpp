// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <ttnn/tensor/tensor.hpp>

namespace ttml::metal {

// Fused forward for MLA Q RoPE on the rope slice only:
//   q_out[..., 0:qk_nope] = q_in[..., 0:qk_nope]   (reader -> writer, no compute)
//   q_out[..., qk_nope:] = RoPE(q_in[..., qk_nope:])
//
// q_in / q_out: [B, n_heads, S, qk_nope_dim + qk_rope_dim]  (head-major, TILE layout)
// cos_cache / sin_cache: [1, 1, S, qk_rope_dim]
// trans_mat: [1, 1, 32, 32]
// Requires qk_rope_dim <= 128 (fp32 dest accumulation, same as rotary_embedding_llama precise).
ttnn::Tensor mla_q_rope(
    const ttnn::Tensor& q_in,
    const ttnn::Tensor& cos_cache,
    const ttnn::Tensor& sin_cache,
    const ttnn::Tensor& trans_mat,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim);

}  // namespace ttml::metal
