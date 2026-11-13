// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPU probe: copies one fp32 tile (CB 0) to DST, applies PROBE_OP elementwise (MATH thread SFPU), packs to CB 16.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
namespace ckernel::sfpu {
template <bool APPROX, int ITERATIONS = 8>
inline void probe_op() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = PROBE_OP;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    compute_kernel_hw_startup(0, 16);
    copy_init(0);
    MATH((::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::unused, DST_ACCUM_MODE>()));
    cb_wait_front(0, 1);
    tile_regs_acquire();
    copy_tile(0, 0, 0);
    MATH((SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, probe_op, (false, 8), 0, VectorMode::RC)));
    tile_regs_commit();
    cb_reserve_back(16, 1);
    tile_regs_wait();
    pack_tile(0, 16);
    tile_regs_release();
    cb_push_back(16, 1);
}
