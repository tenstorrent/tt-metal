// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The isinf_isnan kernel lives in tt-llk on this arch. This header gives the Compute API one include
// name for it on every arch (Quasar's kernel lives in this directory), and holds the five ops' inits.
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_isinf_isnan.h"

namespace ckernel::sfpu {

// The kernel's predicated store advances Dest through ADDR_MOD_6, which _init_sfpu_isinf_isnan_ programs.
inline void isinf_isnan_init() {
    _init_sfpu_isinf_isnan_();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

inline void isfinite_init() { isinf_isnan_init(); }

inline void isinf_init() { isinf_isnan_init(); }

inline void isnan_init() { isinf_isnan_init(); }

inline void isneginf_init() { isinf_isnan_init(); }

inline void isposinf_init() { isinf_isnan_init(); }

}  // namespace ckernel::sfpu
