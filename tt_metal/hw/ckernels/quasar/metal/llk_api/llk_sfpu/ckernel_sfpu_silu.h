// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_silu.h"

namespace ckernel {
namespace sfpu {

inline void calculate_silu(const int iterations = SFPU_ITERATIONS) { _calculate_silu_(iterations); }

}  // namespace sfpu
}  // namespace ckernel
