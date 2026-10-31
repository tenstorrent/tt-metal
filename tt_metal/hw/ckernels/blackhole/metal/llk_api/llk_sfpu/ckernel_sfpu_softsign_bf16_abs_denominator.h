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
    // The rows load the bound themselves: Prgm1 belongs to stock's init.
    sfpu_reciprocal_init<false>();
#if !defined(DISABLE_SFPLOADMACRO)
    ttpoly::softsign::abs_denominator_loadmacro_init();
#endif
}
}  // namespace ckernel::sfpu::bf16::softsign
