// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttml::metal {

// Forward of the chunked gated delta rule (Gated DeltaNet), also returning what the backward needs.
// All fp32 TILE in DRAM (the output is fp32 or bf16, see output_dtype); BH = B*HV, NC = T/chunk_size, and the
// per-head tensors are head-major (row bh = b*HV + hv).
struct ChunkGatedDeltaRuleForwardResult {
    ttnn::Tensor output;       // [B, 1, T, HV*V], or [BH, 1, T, V] with output_head_major
    ttnn::Tensor final_state;  // [B, HV, K, V]
    // The three below are what the backward reads; empty when save_for_backward is false.
    std::optional<ttnn::Tensor> h;      // [BH, NC, K, V]  state entering each chunk; h[:, 0] is the initial state
    std::optional<ttnn::Tensor> v_new;  // [BH, 1, T, V]   T_inv @ (v*beta - (k*beta*exp(cumsum g)) @ h)
    std::optional<ttnn::Tensor> a;      // [BH, 1, T, C]   row t of its chunk's UT inverse (I - A_strict)^-1
};

// Token-major inputs with the heads folded into the last dim, the layout of chunk_gated_delta_rule_bw:
//   q, k [B, 1, T, HK*K] BFLOAT16, L2-normalized along K (the recurrence diverges otherwise)
//   v    [B, 1, T, HV*V] BFLOAT16
//   g, beta [B, 1, T, HV] BFLOAT16 or FLOAT32: log-space decay gate and write strength
//   initial_state [B, HV, K, V] FLOAT32, optional (zeros)
// HK must divide HV; value head h reads key head h / (HV / HK).  K cannot be read off the flat q
// width, so key_head_dim is required.  T must be a multiple of chunk_size (32 or 64), K and V
// multiples of 32 and HV <= 32.  The scan runs one (value head, V-block) per core and, with B*HV above
// the core count, several per core back to back.
//
// output_dtype is the dtype of `output`: FLOAT32 (default) or BFLOAT16, which the scan packs directly
// instead of leaving a typecast of the fp32 output to the caller.  Everything else stays fp32.
//
// save_for_backward = false (inference, or no input requires a gradient) skips the writes of h and v_new
// and returns h, v_new and a empty.  At B*HV = 24, K = V = 128, T = 4096 the skipped h is 100 MB of
// DRAM writes.
ChunkGatedDeltaRuleForwardResult chunk_gated_delta_rule_fw(
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
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool save_for_backward = true);

}  // namespace ttml::metal
