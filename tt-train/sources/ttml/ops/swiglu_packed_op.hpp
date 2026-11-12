// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Packed-SwiGLU gating: `packed` [.., R, 2*I] = [gate | up] -> h = silu(gate) * up, [.., R, I].
// Forward reads both halves in place; backward packs both grads into one tensor.
autograd::TensorPtr swiglu_packed(const autograd::TensorPtr& packed);

}  // namespace ttml::ops
