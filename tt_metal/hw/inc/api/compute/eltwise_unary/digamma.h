// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_digamma.h"
#endif

namespace ckernel {

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void digamma_tile(uint32_t idst) {
    MATH((sfpu::Digamma<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void digamma_tile_init() {
    MATH((sfpu::Digamma<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
