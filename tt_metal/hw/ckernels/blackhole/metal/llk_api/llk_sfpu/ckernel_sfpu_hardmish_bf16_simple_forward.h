// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_hardmish_bf16_replay_window.h"
#include "sfpu/ckernel_sfpu_hardmish_bf16_simple_forward.h"
#include "sfpu/ckernel_sfpu_hardmish_bf16_simple_algebraic.h"
namespace ckernel::sfpu::bf16::hardmish {
template <typename Config>
inline void init_simple_forward() {}
}  // namespace ckernel::sfpu::bf16::hardmish
