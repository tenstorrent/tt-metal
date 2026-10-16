// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_tanhshrink.h"
#endif

namespace ckernel {

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tanhshrink_tile(std::uint32_t idst) {
    MATH((sfpu::Tanhshrink<APPROX, is_fp32_dest_acc_en>::run(idst)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void tanhshrink_tile_init() {
    MATH((sfpu::Tanhshrink<APPROX, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
