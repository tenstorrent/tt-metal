// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_bf16_signed_abs_affine.h"
#include "sfpu/ckernel_sfpu_bf16_cascade_signed_abs_affine.h"

namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_signed_abs_affine() {
    sfpi::signed_abs_affine_init<Config>();
}
}  // namespace ckernel::sfpu::bf16
