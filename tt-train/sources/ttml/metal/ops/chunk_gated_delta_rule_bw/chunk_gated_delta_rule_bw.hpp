// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Vector-Jacobian product of the chunked gated delta rule (Gated DeltaNet), in one dispatch.
//
// Token-major layout with the heads folded into the last dim, rank 3 or as [B, 1, T, X]:
//   q, k   [B, T, HK*K]   L2-normalized along K (the forward diverges otherwise)
//   v      [B, T, HV*V]
//   g, beta [B, T, HV]    log-space decay gate and write strength
//   d_out  [B, T, HV*V], or head-major [B*HV, 1, T, V]
//   initial_state, d_final_state [B, HV, K, V], optional
// HK must divide HV; value head h reads key head h / (HV / HK).  K cannot be read off the flat q
// width, so key_head_dim is required.  T must be a multiple of chunk_size (32 or 64), K and V
// multiples of 32, HV <= 32, and every tensor must share one dtype (BFLOAT16 or FLOAT32).
//
// h, v_new, a: the FLOAT32 intermediates returned by chunk_gated_delta_rule_fw, given all together.
// With them the backward reads the forward's states, corrected values and UT inverse instead of
// recomputing them.
//
// DRAM scratch: the call allocates one transient FLOAT32 tensor of B*HV*T/chunk_size work items.  With
// Ct, Kt, Vt the chunk, K and V extents in tiles, an item is 2*Ct*Kt + 2*Ct*Vt + 2*Kt*Vt + 4*Ct tiles
// with the saved forward and Ct*Ct + 2*Ct*Vt + Kt*Vt more without it: at chunk_size 64, K = V = 128,
// 288 KB (432 KB) per item, i.e. 576 MiB for B*HV = 32 at T = 4096.
//
// Returns {dq, dk, dv, dg, dbeta, dh0} with the shapes of {q, k, v, g, beta, initial_state};
// dh0 is nullopt when initial_state is.
std::vector<std::optional<ttnn::Tensor>> chunk_gated_delta_rule_bw(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& v,
    const ttnn::Tensor& g,
    const ttnn::Tensor& beta,
    const ttnn::Tensor& d_out,
    uint32_t key_head_dim,
    const std::optional<ttnn::Tensor>& initial_state = std::nullopt,
    const std::optional<ttnn::Tensor>& d_final_state = std::nullopt,
    const std::optional<ttnn::Tensor>& h = std::nullopt,
    const std::optional<ttnn::Tensor>& v_new = std::nullopt,
    const std::optional<ttnn::Tensor>& a = std::nullopt,
    uint32_t chunk_size = 64U,
    std::optional<float> scale = std::nullopt,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttml::metal
