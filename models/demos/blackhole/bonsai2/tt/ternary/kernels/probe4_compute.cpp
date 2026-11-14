// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Probe: which DEST (row, col) of tile 0 does SFPU row K address (fp32 DEST)? Copies a zero tile in, writes K+1 to
// dst_reg[K] for each listed K, packs the tile out.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
inline void mark_rows(uint32_t a, uint32_t, uint32_t, uint32_t) {
    sfpi::dst_reg[0] = 1.0f;
    sfpi::dst_reg[1] = 2.0f;
    sfpi::dst_reg[2] = 3.0f;
    sfpi::dst_reg[8] = 9.0f;
    sfpi::dst_reg[9] = 10.0f;
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0, cb_out = tt::CBIndex::c_16;
    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);
    cb_wait_front(cb_in, 1);
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    addcmul_tile_init();
    MATH((SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, mark_rows, 0, 0, 0, 0, VectorMode::None)));
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_init(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
