// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PER-PHASE MATH FIDELITY (Stage 14, the per-layer kernel framework, per_layer_kernel_design.md 2.3): one TRISC
// composite runs phases of different fidelities (A3's kda items HiFi4 + A4's o_proj HiFi2 in one program). The
// composite's MATH_FIDELITY is the highest of its phases (pmk_layer.merge_compute, part_regime); a part that asked a
// lower one gets PMK_PART_FID and calls these instead of matmul_block_init / matmul_block. Same UNPACK / MATH calls as
// api/compute/matmul.h (and the BH dynamic throttling), the fidelity a template argument (kernels/mk/reader_c.cpp's
// mm_init / mm_block, with idst and transpose).
#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/matmul.h"

namespace pmk {

// F_ = the MathFidelity value (LoFi 0, HiFi2 2, HiFi3 3, HiFi4 4): PMK_PART_FID is a plain integer define

template <int F_>
ALWI void mm_block_init_fid(uint32_t in0, uint32_t in1, uint32_t transpose, uint32_t ct, uint32_t rt, uint32_t kt) {
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, transpose, ct, rt, kt)));
    MATH((llk_math_matmul_init<static_cast<ckernel::MathFidelity>(F_), MM_THROTTLE>(in0, in1, transpose, ct, rt)));
#ifdef ARCH_BLACKHOLE
    MATH((ckernel::throttled_mop_status = 0));
#endif
}

template <int F_>
ALWI void mm_block_fid(
    uint32_t in0,
    uint32_t in1,
    uint32_t i0,
    uint32_t i1,
    uint32_t idst,
    uint32_t transpose,
    uint32_t ct,
    uint32_t rt,
    uint32_t kt) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, i0, i1, ct, rt, kt)));
#ifdef TRISC_MATH
#ifdef ARCH_BLACKHOLE
    if (*(ckernel::throttle_ptr) % 2) {
        if (ckernel::throttled_mop_status != 1) {
            llk_math_matmul_init<static_cast<ckernel::MathFidelity>(F_), MM_THROTTLE_MAX>(in0, in1, transpose, ct, rt);
            ckernel::throttled_mop_status = 1;
        }
        llk_math_matmul<static_cast<ckernel::MathFidelity>(F_), MM_THROTTLE_MAX>(idst, ct, rt);
    } else {
        if (ckernel::throttled_mop_status != 0) {
            llk_math_matmul_init<static_cast<ckernel::MathFidelity>(F_), MM_THROTTLE>(in0, in1, transpose, ct, rt);
            ckernel::throttled_mop_status = 0;
        }
        llk_math_matmul<static_cast<ckernel::MathFidelity>(F_), MM_THROTTLE>(idst, ct, rt);
    }
#else
    llk_math_matmul<static_cast<ckernel::MathFidelity>(F_), MM_THROTTLE>(idst, ct, rt);
#endif
#endif
}

// Stage 16 M-B (program B as a window of the fused layer program): the combine's broadcast multiplies (COL: row t x
// token t's weight; SCALAR: the bf16 path) at the part's fidelity -- the same MATH / UNPACK calls as
// api/compute/bcast.h mul_bcast_cols_init / mul_bcast_scalar_init / mul_tiles_bcast<B>, MATH_FIDELITY -> F_
template <int F_, BroadcastType B_>
ALWI void mul_bcast_init_fid(uint32_t icb0, uint32_t icb1) {
    state_configure(icb0, icb1, __builtin_LINE());
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, B_, static_cast<ckernel::MathFidelity>(F_)>(
        icb0, icb1)));
    UNPACK((llk_unpack_AB_init<B_>(icb0, icb1)));
}

template <int F_, BroadcastType B_>
ALWI void mul_tiles_bcast_fid(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          B_,
          DST_ACCUM_MODE,
          static_cast<ckernel::MathFidelity>(F_),
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
    UNPACK((llk_unpack_AB<B_>(icb0, icb1, itile0, itile1)));
}

}  // namespace pmk
