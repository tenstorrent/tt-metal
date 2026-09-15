// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "round.hpp"

void kernel_main() {
    constexpr uint32_t tiles_per_head = get_compile_time_arg_val(0);
    static_assert(!DST_ACCUM_MODE && tiles_per_head % 4 == 0);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(0, 16);
    copy_init(0);
    negative_tile_init();
    MATH((ckernel::math::_configure_src_zero_flag_(false)));
    for (uint32_t i = 0; i < count; i += 4) {
        if (i == 0 || (start + i) % tiles_per_head == 0) {
            cb_wait_front(1, 4);
        }
        cb_wait_front(0, 4);
        cb_reserve_back(16, 4);
        tile_regs_acquire();
        for (uint32_t j = 0; j < 4; ++j) {
            copy_tile(1, j, j + 4);
            copy_tile(0, j, j);
            MATH(SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE, q_center_face, j, VectorMode::RC));
        }
        tile_regs_commit();
        cb_pop_front(0, 4);
        if (i + 4 == count || (start + i + 4) % tiles_per_head == 0) {
            cb_pop_front(1, 4);
        }
        tile_regs_wait();
        for (uint32_t j = 0; j < 4; ++j) {
            pack_tile(j, 16);
        }
        tile_regs_release();
        cb_push_back(16, 4);
    }
}
