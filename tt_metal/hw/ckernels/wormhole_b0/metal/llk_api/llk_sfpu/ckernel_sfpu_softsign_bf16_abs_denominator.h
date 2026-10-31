// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_softsign_bf16_abs_denominator_replay.h"
#include "sfpu/ckernel_sfpu_softsign_bf16_abs_denominator.h"
namespace ckernel::sfpu::bf16::softsign {
template <typename Config>
inline void init_abs_denominator() {
    ttpoly::softsign::abs_denominator_lut_init<Config::kLutSlopes, Config::kLutIntercepts>();
}
}  // namespace ckernel::sfpu::bf16::softsign
