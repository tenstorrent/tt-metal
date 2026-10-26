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
    const std::optional<ttnn::Tensor>& preallocated_dL_dpacked = std::nullopt);

}  // namespace ttml::metal
