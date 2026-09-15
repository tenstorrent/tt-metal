// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Register-resident macro-exp calibration; no recurring L1 subtraction,
// score reload/pack, matmul, or online state. Not an SDPA benchmark.
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/exp.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#endif

void kernel_main() {
    constexpr uint32_t repetitions = get_compile_time_arg_val(1);
    constexpr uint32_t scale = 0x3db504f3;
    compute_kernel_hw_startup(0, 16);
    cb_wait_front(0, 1);
    cb_wait_front(1, 4);
    cb_reserve_back(16, 4);
    tile_regs_acquire();
    copy_tile_init(0);
    for (uint32_t j = 0; j < 4; ++j) {
        copy_tile(0, 0, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    exp_packthread_tile_init<true, scale, InputClamping::None>();
    for (uint32_t rep = 0; rep < repetitions; ++rep) {
        PACK((ckernel::sfpu::init_sdpa_exp_grid<scale>()));
        PACK((
            SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_grid_batch, (128), 0, VectorMode::None)));
        PACK((SFPU_UNARY_CALL(
            DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_exp_stream_effective, (128), 0, VectorMode::None)));
    }
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    for (uint32_t j = 0; j < 4; ++j) {
        pack_tile<true>(j, 16, j);
    }
    tile_regs_release();
    cb_push_back(16, 4);
    cb_pop_front(0, 1);
    cb_pop_front(1, 4);
}
