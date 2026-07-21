// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Fused KV assembly for DeepSeek MLA (Q head-split + RoPE lives in mla_q_rope).
//
// Demuxes the per-head [k_nope_h | v_h] stripes from a single packed projection
// output (the result of `wkv_b`), and broadcasts the post-RoPE shared `k_pe`
// into every head's rope-suffix. Pure data movement.
//
// Shapes (TILE layout, INTERLEAVED memory):
//   kv_up  : [B, 1, S, n_heads * (qk_nope_dim + v_dim)]
//   k_pe   : [B, 1, S, qk_rope_dim]                       (shared across heads)
//   k      : [B, n_heads, S, qk_nope_dim + qk_rope_dim]   (output, k_nope | broadcast k_pe)
//   v      : [B, n_heads, S, v_dim]                       (output)
//
// Per-head dims and S must be tile-aligned (multiples of TILE_W=TILE_H=32).
//
// Returns: {k, v}
std::tuple<ttnn::Tensor, ttnn::Tensor> mla_kv_assemble_fw(
    const ttnn::Tensor& kv_up,
    const ttnn::Tensor& k_pe,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim);

}  // namespace ttml::metal
