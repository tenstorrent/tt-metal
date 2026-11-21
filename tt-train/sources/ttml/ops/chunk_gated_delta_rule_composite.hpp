// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "core/ttnn_all_includes.hpp"

namespace ttml::ops {

// What the composite forward returns.  All fp32 TILE except `output`; BH = B*HV, NC = T/chunk_size, and the per-head
// tensors are head-major (row bh = b*HV + hv).
struct ChunkGatedDeltaRuleCompositeResult {
    ttnn::Tensor output;       // [B, 1, T, HV*V], or [BH, 1, T, V] with output_head_major
    ttnn::Tensor final_state;  // [B, HV, K, V]
    // What the backward reads; empty when save_for_backward is false.
    std::optional<ttnn::Tensor> h;      // [BH, NC, K, V]  state entering each chunk; h[:, 0] is the initial state
    std::optional<ttnn::Tensor> v_new;  // [BH, 1, T, V]   T_inv @ (v*beta - (k*beta*exp(cumsum g)) @ h)
    std::optional<ttnn::Tensor> a;  // [BH, 1, T, C]   row t of its chunk's UT inverse (I + N)^-1, 0 above the diagonal
};

// Forward of the chunked gated delta rule (Gated DeltaNet) written only with existing ttnn ops, keeping the
// intermediates the backward (chunk_gated_delta_rule_bw) consumes, like scaled_dot_product_attention_composite keeps
// its attention weights.  It is the reference implementation of the forward: simple and easy to review, but it runs
// a Python-style loop of small ops over the chunks (T / chunk_size iterations), so it is slow.
//
// Inputs are token-major with the heads folded into the last dim:
//   q, k [B, 1, T, HK*K] BFLOAT16, L2-normalized along K (the recurrence diverges otherwise)
//   v    [B, 1, T, HV*V] BFLOAT16
//   g, beta [B, 1, T, HV] BFLOAT16 or FLOAT32 (the same dtype): log-space decay gate and write strength
//   initial_state [B, HV, K, V] FLOAT32, optional (zeros)
// HK must divide HV (value head h reads key head h / (HV / HK)); key_head_dim is K, which cannot be read off the flat
// q width.  T must be a multiple of chunk_size (32 or 64) and K, V multiples of 32.  scale defaults to K^-0.5.
//
// All intermediates and the state are fp32.  output_dtype is the dtype of `output` (FLOAT32 or BFLOAT16).
// save_for_backward = false (inference, or no input requires a gradient) returns h, v_new and a empty.
//
// (I + N)^-1, N strictly lower, is computed by exact forward substitution (chunk_size - 1 rounds of
// X <- I - N X over all chunks and heads at once).  A Neumann product (I - N)(I + N^2)(I + N^4)... is not used: it
// overflows for strongly correlated keys.
ChunkGatedDeltaRuleCompositeResult chunk_gated_delta_rule_fw_composite(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    uint32_t key_head_dim,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    uint32_t chunk_size = 64U,
    std::optional<float> scale = std::nullopt,
    bool output_head_major = false,
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::FLOAT32,
    bool save_for_backward = true);

}  // namespace ttml::ops
