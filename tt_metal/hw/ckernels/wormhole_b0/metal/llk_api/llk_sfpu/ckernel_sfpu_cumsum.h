// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_cumsum.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE /*unused*/, int ITERATIONS = 8 /*unused*/>
inline void calculate_cumsum(bool first) {
    _calculate_cumsum_<false, 1>(first);  // There is only non APPROXIMATE implementation and one iteration
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void cumsum_init() {
    _cumsum_init_<false>();  // There is only non APPROXIMATE implementation
}

}  // namespace sfpu
}  // namespace ckernel
