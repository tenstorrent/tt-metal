// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_sigmoid.h"

namespace ckernel {
namespace sfpu {

inline void calculate_sigmoid(const int iterations = SFPU_ITERATIONS) { _calculate_sigmoid_(iterations); }

}  // namespace sfpu
}  // namespace ckernel
