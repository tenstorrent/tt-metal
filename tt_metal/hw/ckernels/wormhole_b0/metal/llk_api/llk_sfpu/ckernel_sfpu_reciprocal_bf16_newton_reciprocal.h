// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_reciprocal_bf16_replay_window.h"
#include "sfpu/ckernel_sfpu_reciprocal_bf16_newton_reciprocal.h"

namespace ckernel::sfpu::bf16::reciprocal {
template <typename Config>
inline void init_newton_reciprocal() {}
}  // namespace ckernel::sfpu::bf16::reciprocal
