// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_cast_fp32_to_fp16a.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void cast_fp32_to_fp16a()
{
    _cast_fp32_to_fp16a_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS);
}

}  // namespace sfpu
}  // namespace ckernel
