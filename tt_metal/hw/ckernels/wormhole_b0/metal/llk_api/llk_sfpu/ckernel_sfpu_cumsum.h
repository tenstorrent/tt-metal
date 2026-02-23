// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_cumsum.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ckernel::ApproximationMode APPROX_MODE /*unused*/, int ITERATIONS = 8 /*unused*/>
inline void calculate_cumsum(bool first) {
    _calculate_cumsum_<false, 1>(first);  // There is only non APPROX_MODE implementation and one iteration
}

template <ckernel::ApproximationMode APPROX_MODE /*unused*/>
inline void cumsum_init() {
    _cumsum_init_<false>();  // There is only non APPROX_MODE implementation
}

}  // namespace sfpu
}  // namespace ckernel
