// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "sfpi.h"
#include <cstdint>
#include <limits>
namespace ckernel::sfpu::bf16::mish {
template <class Config>
inline void init_exponent_alu_product() {
    // Both rows load the P2 head's coefficients in the call; the symmetric row's init sets its LOADMACRO sequence.
    sfpu_reciprocal_init<false>();
}
}  // namespace ckernel::sfpu::bf16::mish
