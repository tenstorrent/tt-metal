// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "round.hpp"

void kernel_main() {
    constexpr uint32_t components = get_compile_time_arg_val(0);
    constexpr uint32_t batch = get_compile_time_arg_val(1);
    constexpr bool second_b8 = get_compile_time_arg_val(2) != 0;
    static_assert(components == 2 || components == 3);
    static_assert(!second_b8 || components == 2);
    static_assert(batch * components <= (DST_ACCUM_MODE ? 4 : 8));
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 16);
    copy_init(0);
    negative_tile_init();
    MATH((ckernel::math::_configure_src_zero_flag_(false)));
    for (uint32_t i = 0; i < count; i += batch) {
        cb_wait_front(0, batch);
        cb_reserve_back(16, batch);
        cb_reserve_back(17, batch);
        if constexpr (components == 3) {
            cb_reserve_back(18, batch);
        }
        tile_regs_acquire();
        for (uint32_t j = 0; j < batch; ++j) {
            const uint32_t dst = j * components;
            copy_tile(0, j, dst);
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                residual_round_face,
                (DST_ACCUM_MODE, components, second_b8),
                dst,
                VectorMode::RC));
        }
        tile_regs_commit();
        cb_pop_front(0, batch);
        tile_regs_wait();
        for (uint32_t c = 0; c < components; ++c) {
            if constexpr (second_b8) {
                if (c == 1) {
                    pack_reconfig_data_format(16, 17);
                }
            }
            for (uint32_t j = 0; j < batch; ++j) {
                pack_tile(j * components + c, 16 + c);
            }
        }
        if constexpr (second_b8) {
            pack_reconfig_data_format(17, 16);
        }
        tile_regs_release();
        cb_push_back(16, batch);
        cb_push_back(17, batch);
        if constexpr (components == 3) {
            cb_push_back(18, batch);
        }
    }
}
