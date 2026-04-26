// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttml::ops {

// Grouped (per-expert) matmul for tokens packed by the MoE group op.
//
// Shapes:
//   X       : [1, 1, T_cap, K]   bf16 TILE DRAM
//   W       : [E_local, K, N]    bf16 TILE DRAM
//   offsets : [E_local + 1]      uint32 L1 (or DRAM)
// Returns:
//   Y       : [1, 1, T_cap, N]   bf16 TILE DRAM
//
// Composite implementation: ttnn::slice per expert + ttnn::concat. The
// natural zero-copy alternative via ttnn::narrow is not viable on the
// packed [1,1,T_cap,H] layout because narrow requires
// `dim_size % length == 0`, and per-expert lengths derived from `offsets`
// do not satisfy that for arbitrary dispatch distributions. A custom
// kernel that reads `offsets` on-device is the path to a true zero-copy
// implementation.
ttnn::Tensor sparse_matmul(const ttnn::Tensor& X, const ttnn::Tensor& W, const ttnn::Tensor& offsets);

}  // namespace ttml::ops
