// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/tensor.hpp"

namespace ttml::ops {

// SwiGLU forward path. We keep all three for further tests and development:
//   GateUp (2-step): Production default. Fused gate-up kernel + matmul(M, W2).
//   Composite: Unfused matmul/silu/mul/matmul; same as LlamaMLP composite. Kept for tests and development.
//   FullFusion: Single fused kernel. Saves memory (no materialized M) but runtime is heavily suboptimal;
//               kept for tests and development, must NOT be used in production.
enum class SwigluFwPath { Composite, GateUp, FullFusion };

autograd::TensorPtr swiglu(
    const autograd::TensorPtr& tensor,
    const autograd::TensorPtr& w1,
    const autograd::TensorPtr& w2,
    const autograd::TensorPtr& w3);

}  // namespace ttml::ops
