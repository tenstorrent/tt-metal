// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "metal/ttnn_all_includes.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttml::metal {

// Which gradients the caller will read.  The fused kernel computes and writes every one of them (they
// share the dh recurrence); an entry that is false only skips the host work that follows it: the
// reduction of dq / dk over each key head's value heads (G > 1 costs a `ttnn::sum` dispatch each), and
// the returned optional is left empty.
struct ChunkGatedDeltaRuleBwNeeds {
    bool dq = true;
    bool dk = true;
    bool dv = true;
    bool dg = true;
    bool dbeta = true;
    bool dh0 = true;
};

// Vector-Jacobian product of the chunked gated delta rule (Gated DeltaNet), in one dispatch.
//
// Token-major layout with the heads folded into the last dim, rank 3 or as [B, 1, T, X]:
//   q, k   [B, T, HK*K]   L2-normalized along K (the forward diverges otherwise)
//   v      [B, T, HV*V]
//   g, beta [B, T, HV]    log-space decay gate and write strength
//   d_out  [B, T, HV*V], or head-major [B*HV, 1, T, V]
//   initial_state [B, HV, K, V], optional, BFLOAT16 or FLOAT32 (the forward takes it in FLOAT32)
//   d_final_state [B, HV, K, V], optional, in q's dtype
// HK must divide HV; value head h reads key head h / (HV / HK).  K cannot be read off the flat q
// width, so key_head_dim is required.  T must be a multiple of chunk_size (32 or 64), K and V
// multiples of 32 and HV <= 32.  q, k, v, d_out and d_final_state share one dtype (BFLOAT16 or FLOAT32); g
// and beta share another, and initial_state a third, each BFLOAT16 or FLOAT32 whatever q's is, as in the
// forward.
//
// h [B*HV, NC, K, V], v_new [B*HV, 1, T, V], a [B*HV, 1, T, C]: the FLOAT32 intermediates returned by
// chunk_gated_delta_rule_fw (save_for_backward), all required.  The backward reads the forward's states, its
// corrected values and its UT inverse a = (I + N)^-1 and rebuilds none of them.  initial_state therefore
// only decides whether dh0 is produced: its values are already in h.
//
// DRAM scratch: the call allocates one transient FLOAT32 tensor of B*HV*T/chunk_size work items.  With
// Ct, Kt, Vt the chunk, K and V extents in tiles, an item is 2*Ct*Kt + 2*Ct*Vt + 2*Kt*Vt + 4*Ct tiles:
// at chunk_size 64, K = V = 128, 288 KB per item, i.e. 576 MiB for B*HV = 32 at T = 4096.
//
// Returns {dq, dk, dv, dg, dbeta, dh0} with the shapes of {q, k, v, g, beta, initial_state}: dq, dk and dv in
// q's dtype, dg and dbeta in the gates' dtype and dh0 in initial_state's.  The kernel accumulates all of them
// in fp32 and rounds once, on the way out.  dh0 is nullopt when initial_state is, and every entry that
// `needs` clears is nullopt.
std::vector<std::optional<ttnn::Tensor>> chunk_gated_delta_rule_bw(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& d_out,
    const ttnn::Tensor& h,
    const ttnn::Tensor& v_new,
    const ttnn::Tensor& a,
    uint32_t key_head_dim,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    const std::optional<ttnn::Tensor>& d_final_state = std::nullopt,
    uint32_t chunk_size = 64U,
    std::optional<float> scale = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const ChunkGatedDeltaRuleBwNeeds& needs = {});

}  // namespace ttml::metal
