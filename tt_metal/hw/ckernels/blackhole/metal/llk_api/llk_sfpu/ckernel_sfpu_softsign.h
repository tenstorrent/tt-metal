// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

namespace ckernel::sfpu {

// Implementation removed — depends on recip primitive (Family 3)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softsign() {}

template <bool APPROXIMATION_MODE>
inline void softsign_init() {}

}  // namespace ckernel::sfpu
