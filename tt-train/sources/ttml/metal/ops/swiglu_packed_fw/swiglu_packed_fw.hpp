// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Fused packed-SwiGLU gating forward. `packed` [.., R, 2*I] (TILE bf16) = [gate | up]; returns
// h = silu(gate) * up, [.., R, I], reading both halves in place (no slice).
// out_cols (>= I, multiple of 32; 0 = I): the output is [.., R, out_cols] with h in the first I columns (the
// wide LoRA input buffer, iteration 24); the columns past I are left as allocated.
ttnn::Tensor swiglu_packed_fw(
    const ttnn::Tensor& packed,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt,
    uint32_t out_cols = 0,
    // iteration 38: the output transposed ([.., out_cols, R], h^T in its first I rows, the rows past I zeroed)
    bool transposed_output = false,
    // iteration 39: packed is packed^T [1, 1, 2I, T] and the output h^T [1, 1, out_cols or I, T] (rows past I
    // zeroed); elementwise on the transposed layout, no transpose anywhere
    bool transposed_input = false);

}  // namespace ttml::metal
