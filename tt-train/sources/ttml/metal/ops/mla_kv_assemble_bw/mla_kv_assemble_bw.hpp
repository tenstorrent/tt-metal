// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Backward kernel for ``mla_kv_assemble_fw``.
//
// Inverts the forward op's tile routing:
//   - dkv_up  ← head-major [dK_nope | dV] packed flat per head.
//   - dk_pe   ← Σ_h dK[..., qk_nope:]  (head-axis sum, runs in compute kernel).
//
// Shapes (TILE layout, INTERLEAVED memory):
//   dK      : [B, n_heads, S, qk_nope_dim + qk_rope_dim]
//   dV      : [B, n_heads, S, v_dim]
//   dkv_up  : [B, 1, S, n_heads * (qk_nope_dim + v_dim)]         (output)
//   dk_pe   : [B, 1, S, qk_rope_dim]                             (output)
//
// Per-head dims and S must be tile-aligned.
//
// Returns: {dkv_up, dk_pe}
std::tuple<ttnn::Tensor, ttnn::Tensor> mla_kv_assemble_bw(
    const ttnn::Tensor& dK,
    const ttnn::Tensor& dV,
    uint32_t n_heads,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim,
    uint32_t v_dim);

}  // namespace ttml::metal
