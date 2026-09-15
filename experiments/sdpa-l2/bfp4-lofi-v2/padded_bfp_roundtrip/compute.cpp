// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t tile_count = get_compile_time_arg_val(1);
    static_assert(batch == (DST_ACCUM_MODE ? 4 : 8));
    compute_kernel_hw_startup(0, 8);

    for (uint32_t tile = 0; tile < tile_count; tile += batch) {
        // Stage 1: ordinary BF16 source -> native BFP within a possibly padded
        // page. Reconfigure formats/strides explicitly; never use blocked BFP.
        if (tile != 0) {
            reconfig_data_format_srca(8, 0);
            pack_reconfig_data_format(16, 8);
        }
        copy_init(0);
        MATH((ckernel::math::_configure_src_zero_flag_(false)));
        pack_init(8);
        cb_wait_front(0, batch);
        cb_reserve_back(8, batch);
        tile_regs_acquire();
        for (uint32_t j = 0; j < batch; ++j) {
            copy_tile(0, j, j);
        }
        tile_regs_commit();
        cb_pop_front(0, batch);
        tile_regs_wait();
        for (uint32_t j = 0; j < batch; ++j) {
            pack_tile<true>(j, 8, j);
        }
        tile_regs_release();
        cb_push_back(8, batch);

        // Stage 2: native BFP payload at the declared page stride -> ordinary
        // BF16 output. wait_front orders UNPACK after the BFP producer PACK.
        cb_wait_front(8, batch);
        reconfig_data_format_srca(0, 8);
        copy_init(8);
        MATH((ckernel::math::_configure_src_zero_flag_(false)));
        pack_reconfig_data_format(8, 16);
        pack_init(16);
        cb_reserve_back(16, batch);
        tile_regs_acquire();
        for (uint32_t j = 0; j < batch; ++j) {
            copy_tile(8, j, j);
        }
        tile_regs_commit();
        cb_pop_front(8, batch);
        tile_regs_wait();
        for (uint32_t j = 0; j < batch; ++j) {
            pack_tile<true>(j, 16, j);
        }
        tile_regs_release();
        cb_push_back(16, batch);
    }
}
