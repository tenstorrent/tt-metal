// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The root's coefficient pass with the round-end marker (PMK mhc rev 5, README §9e): 7a's mhc_coeff_planes, then
// MARKER (an fp32 bit pattern distinct per round and per cycle, mhc_common.hpp marker_bits / marker_final) stored in
// plane PLANE_MARKER as the pass's LAST SFPU store. The pack thread's verified pack (mhc_compute.cpp
// pack_coef_verified) packs DST tile 0 out-of-order into CB_COEF page 0 until that marker is in L1, then once more.
//
// A HEADER, not a block in mhc_compute.cpp (assembly port, 2026-10-22): the program-A composite kernels
// (tt/pmk_layer.py composite_source) #include every body inside its own phase namespace ph<pid>_<i> and hoist only the
// body's include-guarded #include lines to the top level, so a `namespace ckernel::sfpu { ... }` inside the body
// becomes ph<pid>_<i>::ckernel::sfpu, where neither mhc_coeff_planes nor Converter is visible (Stage 16's pair-mock
// JIT error on trisc1: "mhc_coeff_planes was not declared in this scope; did you mean ckernel::sfpu::mhc_coeff_planes",
// "Converter has not been declared"). Hoisted, this header lands in ::ckernel::sfpu on the standalone build and on
// every composite alike, and SFPU_UNARY_CALL's ckernel::sfpu::pmk_coeff_planes resolves on both.
#pragma once

#ifdef TRISC_MATH
#include "mhc_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_sfpu.hpp"

namespace ckernel::sfpu {

template <uint32_t ITERS_, uint32_t EPS_BITS_>
inline void pmk_coeff_planes(uint32_t marker) {
    mhc_coeff_planes<ITERS_, EPS_BITS_>();
    const sfpi::vFloat m = Converter::as_float(marker);
    sfpi::dst_reg[pmk_mhc::PLANE_MARKER] = m;
}

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH
