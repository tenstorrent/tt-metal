// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Grouped gated RMSNorm: `input`, `gate` [B, 1, T, H*V], `gamma` [1, 1, 1, V] ->
// rmsnorm over each V-wide head * gamma * silu(gate). Backward skips dL/dgamma when gamma is frozen.
autograd::TensorPtr gated_rmsnorm(
    const autograd::TensorPtr& input,
    const autograd::TensorPtr& gate,
    const autograd::TensorPtr& gamma,
    float epsilon = 1e-6F);

}  // namespace ttml::ops
