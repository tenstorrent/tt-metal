// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Fused packed-SwiGLU gating forward. `packed` [.., R, 2*I] (TILE bf16) = [gate | up]; returns
// h = silu(gate) * up, [.., R, I], reading both halves in place (no slice).
ttnn::Tensor swiglu_packed_fw(
    const ttnn::Tensor& packed, const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);

}  // namespace ttml::metal
