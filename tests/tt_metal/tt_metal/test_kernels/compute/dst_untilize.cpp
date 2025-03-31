// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t num_faces = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows_per_face = get_compile_time_arg_val(3);

    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_to_dst_init_short(tt::CBIndex::c_0);
    pack_untilize_dst_init_short<per_core_block_tile_cnt>(tt::CBIndex::c_16, num_rows_per_face, num_faces);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_wait_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_tile_cnt);

        tile_regs_acquire();
        for (uint32_t i = 0; i < per_core_block_tile_cnt; ++i) {
            copy_tile(tt::CBIndex::c_0, i, i);
        }
        tile_regs_commit();

        tile_regs_wait();
        pack_untilize_dst<per_core_block_tile_cnt>(tt::CBIndex::c_16, 1, 0, num_rows_per_face, num_faces);
        tile_regs_release();

        cb_push_back(tt::CBIndex::c_16, per_core_block_tile_cnt);
        cb_pop_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
    }

    pack_untilize_uninit(tt::CBIndex::c_16);
}
}  // namespace NAMESPACE
