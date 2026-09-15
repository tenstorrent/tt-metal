// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#define EXP_APPROX_MODE 1
#ifdef PROBE_REFERENCE
#include "../single-core-resident-v1/candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#else
#include "candidate/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#endif

void kernel_main() {
    constexpr int pairs = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 0, 16);
    cb_wait_front(0, 3 * pairs + 1);
    cb_reserve_back(16, 2 * pairs);
    tile_regs_acquire();
    copy_init(0);
    for (int i = 0; i < 3 * pairs + 1; ++i) {
        copy_tile(0, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
#ifndef PROBE_REFERENCE
    if constexpr (pairs == 2) {
        PACK((ckernel::sfpu::init_sdpa_compensated_state_macros()));
    }
#endif
    PACK((SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sdpa_compensated_state, (pairs), 0, VectorMode::None)));
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    for (int i = 0; i < pairs; ++i) {
        pack_tile(3 * i, 16);
        pack_tile(3 * i + 1, 16);
    }
    tile_regs_release();
    cb_push_back(16, 2 * pairs);
    cb_pop_front(0, 3 * pairs + 1);
}
