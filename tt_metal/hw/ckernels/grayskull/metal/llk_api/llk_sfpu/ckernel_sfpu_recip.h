// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"
#include "sfpu/ckernel_sfpu_recip.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool save_reg, int max_iter = 3>
sfpi_inline vFloat sfpu_reciprocal(const vFloat in)
{
    return _sfpu_reciprocal_<save_reg, max_iter>(in);
}

template <bool APPROXIMATION_MODE, int ITERATIONS=4>
inline void calculate_reciprocal()
{
    _calculate_reciprocal_<APPROXIMATION_MODE, ITERATIONS>();
}

} // namespace sfpu
} // namespace ckernel
