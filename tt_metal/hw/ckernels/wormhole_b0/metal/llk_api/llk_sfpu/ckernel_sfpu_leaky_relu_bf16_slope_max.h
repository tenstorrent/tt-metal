// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_leaky_relu_bf16_replay_window.h"
#include "sfpu/ckernel_sfpu_leaky_relu_bf16_slope_max.h"

namespace ckernel::sfpu::bf16::leaky_relu {
// Stock lrelu reads no init state, so the call loads the slope (L4) and the boundary word (L5) itself.
template <typename Config>
inline void init_slope_max() {}
}  // namespace ckernel::sfpu::bf16::leaky_relu
