// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Fused packed-SwiGLU gating backward. `packed` is the forward's [.., R, 2*I] = [gate | up];
// `dL_dh` is grad wrt h = silu(gate)*up, [.., R, I]. Returns dL/dpacked [.., R, 2*I] = [dgate | dup],
// written into the two halves of one tensor (no concat).
ttnn::Tensor swiglu_packed_bw(
    const ttnn::Tensor& packed,
    const ttnn::Tensor& dL_dh,
    const std::optional<ttnn::Tensor>& preallocated_dL_dpacked = std::nullopt,
    // iteration 38: dL_dh transposed ([.., >= I + 32, R], dh^T in rows [0, I), the scaled LoRA tail s_dh^T in
    // rows [I, I + 32)); dh += lowrank_scale * s_dh @ lora_a is folded in while the tiles are transposed in DEST
    const std::optional<ttnn::Tensor>& lora_a = std::nullopt,
    bool transposed_grad = false,
    float lowrank_scale = 1.0F,
    // iteration 39: every operand transposed (packed^T [1, 1, 2I, T], dL_dh = dh^T wide [1, 1, >= I + 32, T] with
    // the tail s_dh^T in rows [I, I + 32), dpacked^T out); dh^T += lowrank_scale * lora_a^T @ s_dh^T folded in
    bool transposed_io = false);

}  // namespace ttml::metal
