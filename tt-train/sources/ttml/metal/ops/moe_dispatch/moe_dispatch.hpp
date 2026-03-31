// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ttnn/tensor/tensor.hpp>

namespace ttml::metal {

// MoE streaming dispatch + fused expert matmul.
//
// Dispatches sorted token tile-rows to the device hosting each expert via
// fabric unicast, then computes matmul(tokens, W_up) on each device's local
// experts. Single program per device, all 32 devices run concurrently.
//
// Input:
//   sorted_hidden [1, 1, N_local, D]    — this device's tokens sorted by expert
//   w_up          [E_local, 1, D, ffn]  — local expert weights (per device)
//   cluster_axis  — mesh axis for dispatch (0 or 1)
//   expert_offsets_per_device[EP][E] — per-device start tile-row in sorted_hidden
//   expert_counts_per_device[EP][E]  — per-device padded tile-row count per expert
//   E_local       — experts per device
//
// Output:
//   [1, 1, N_local, ffn] — matmul results (local experts only)
ttnn::Tensor moe_dispatch(
    const ttnn::Tensor& sorted_hidden,
    const ttnn::Tensor& w_up,
    uint32_t cluster_axis,
    const std::vector<std::vector<uint32_t>>& expert_offsets_per_device,
    const std::vector<std::vector<uint32_t>>& expert_counts_per_device,
    uint32_t E_local);

}  // namespace ttml::metal
