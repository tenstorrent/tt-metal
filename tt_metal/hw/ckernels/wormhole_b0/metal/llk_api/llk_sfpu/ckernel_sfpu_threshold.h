// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The threshold kernel lives in tt-llk on this arch. This forwarding header gives the Compute API one
// include name for it on every arch (Quasar's kernel lives in this directory), and holds the metal
// threshold_init that llk_math_eltwise_unary_sfpu_init.h declares.
#include "sfpu/ckernel_sfpu_threshold.h"

namespace ckernel::sfpu {

inline void threshold_init() { _init_threshold_(); }

}  // namespace ckernel::sfpu
