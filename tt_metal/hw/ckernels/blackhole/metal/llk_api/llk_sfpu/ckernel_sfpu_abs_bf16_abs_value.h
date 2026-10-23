// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_abs_bf16_replay_window.h"
#include "sfpu/ckernel_sfpu_abs_bf16_abs_value.h"

namespace ckernel::sfpu::bf16::abs {
template <typename Config>
inline void init_abs_value() {}
}  // namespace ckernel::sfpu::bf16::abs
