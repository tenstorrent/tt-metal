// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_bf16_log1p.h"
namespace ckernel::sfpu::bf16 {
// The exponent term reads ln2*2^-23 from vConstFloatPrgm0, which the stock log1p_init sets to
// that value for log1p and the inverse hyperbolics alike; the polynomial lives in LREGs.
template <typename Config>
inline void init_log1p() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}
}  // namespace ckernel::sfpu::bf16
