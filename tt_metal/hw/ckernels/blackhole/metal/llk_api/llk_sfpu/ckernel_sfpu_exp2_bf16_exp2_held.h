// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_exp2_bf16_exp2_held.h"
// The exponent-ALU total form one row at a time, with every constant in L3..L7.
// The paired body needs eleven registers and reads its multiplier and two
// coefficients from the programmable constants; one row needs three working
// registers, so the constants fit in LREGs and the setup sets nothing.
// Each lane runs the paired body's instructions in its order, so results match it.
namespace ckernel::sfpu::bf16::exp2 {
template <typename Config>
inline void init_exp2_held() {}

}  // namespace ckernel::sfpu::bf16::exp2
