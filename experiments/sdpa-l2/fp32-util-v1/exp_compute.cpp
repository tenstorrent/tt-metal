// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Register-resident SFPU instruction-throughput calibration. Deliberately
// excludes recurring score reloads, packing, QK/PV, and online state updates.
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/exp.h"
#include "tools/profiler/kernel_profiler.hpp"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#endif

void kernel_main() {
    constexpr uint32_t repetitions = get_compile_time_arg_val(1);
    constexpr uint32_t scale = 0x3db504f3;  // float32(1/sqrt(128))
    compute_kernel_hw_startup(0, 16);
    cb_wait_front(0, 1);
    cb_wait_front(1, 4);
    cb_reserve_back(16, 4);
    tile_regs_acquire();
    copy_tile_init(0);
    for (uint32_t j = 0; j < 4; ++j) {
        copy_tile(0, 0, j);
    }
    copy_init(1);
    copy_tile(1, 0, 2);  // maximum=16: keep all repeated scaled logits negative.
    tile_regs_commit();
    tile_regs_wait();
    PACK((llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, DST_ACCUM_MODE>()));
    PACK((ckernel::sfpu::calculate_sdpa_fused_sub_exp<scale, true>()));
    {
        DeviceZoneScopedN("UTIL_REGISTER_EXP");
        for (uint32_t rep = 0; rep < repetitions; ++rep) {
            PACK((SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_sdpa_fused_sub_exp,
                (scale, false, true),
                0,
                VectorMode::None)));
        }
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
