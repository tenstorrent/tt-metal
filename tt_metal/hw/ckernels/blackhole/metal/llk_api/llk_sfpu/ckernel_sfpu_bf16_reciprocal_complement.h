// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
namespace sfpi {
#include "sfpu/ckernel_sfpu_bf16_mirrored_terminals.h"
}
#include "sfpu/ckernel_sfpu_bf16_reciprocal_complement_core.h"
#include "sfpu/ckernel_sfpu_bf16_reciprocal_complement.h"
namespace ckernel::sfpu::bf16 {
template <class Config>
inline void init_reciprocal_complement() {
    sfpu_reciprocal_init<false>();
}
}  // namespace ckernel::sfpu::bf16
