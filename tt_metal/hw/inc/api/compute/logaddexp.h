// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif
namespace ckernel {
ALWI void logaddexp_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((SFPU_BINARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sfpu_binary,
        (APPROX, BinaryOp::LOGADDEXP, 8 /* ITERATIONS */),
        idst0, idst1, odst, VectorMode::RC)));
}
ALWI void logaddexp_binary_tile_init() {
    MATH((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, BinaryOp::LOGADDEXP))));
}
}  // namespace ckernel
