// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATE = false, bool save_reg = true /* Unused. Enough registers available. */>
sfpi_inline vFloat sfpu_reciprocal(const vFloat in) {
    return _sfpu_reciprocal_<APPROXIMATE ? 0 : 2>(in);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8, bool legacy_compat = false>
inline void calculate_reciprocal() {
    _calculate_reciprocal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en, legacy_compat>(ITERATIONS);
}

template <bool APPROXIMATION_MODE, bool legacy_compat = false>
void recip_init() {
    _init_reciprocal_<APPROXIMATION_MODE, legacy_compat>();
}

}  // namespace sfpu
}  // namespace ckernel
