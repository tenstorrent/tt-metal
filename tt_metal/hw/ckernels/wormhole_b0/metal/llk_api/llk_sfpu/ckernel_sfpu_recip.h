// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_recip.h"
using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <int max_iter = 3, bool save_reg = true /* Unused. Enough registers available. */>
sfpi_inline vFloat sfpu_reciprocal(const vFloat in)
{
    return _sfpu_reciprocal_<max_iter>(in);
}


template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_reciprocal()
{
    _calculate_reciprocal_<APPROXIMATION_MODE, ITERATIONS, is_fp32_dest_acc_en>(ITERATIONS);
}

template <bool APPROXIMATION_MODE>
void recip_init() {
    _init_reciprocal_<APPROXIMATION_MODE>();
}

} // namespace sfpu
} // namespace ckernel
