// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_log1p_bf16_replay_window.h"
#include "sfpu/ckernel_sfpu_log1p_bf16_log1p.h"
namespace ckernel::sfpu::bf16::log1p {
// The exponent term reads ln2*2^-23 from vConstFloatPrgm0, which the stock log1p_init sets to
// that value for log1p and the inverse hyperbolics alike; the polynomial lives in LREGs.
template <typename Config>
inline void init_log1p() {}
}  // namespace ckernel::sfpu::bf16::log1p
