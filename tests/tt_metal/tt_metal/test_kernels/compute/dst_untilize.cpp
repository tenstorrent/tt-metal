// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#include "tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t num_faces = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows_per_face = get_compile_time_arg_val(3);

    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_to_dst_init_short(tt::CBIndex::c_0);
    pack_untilize_dst_init_short<per_core_block_tile_cnt>(tt::CBIndex::c_16, num_rows_per_face, num_faces);

// If LLK perf is measured on OP level put profiler zone around complete operation
#ifdef LLK_PERF_OP
    {
        DeviceZoneScopedN("UNTILIZE-OP")
#endif
            for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
// If LLK perf is measured disable sync with DM cores/kernels
#ifndef LLK_PERF_NO_DM
            cb_wait_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
            cb_reserve_back(tt::CBIndex::c_16, per_core_block_tile_cnt);
#endif
// If LLK perf is measured on block level put profiler zone around *_block operation
#ifdef LLK_PERF_BLOCK
            {
                DeviceZoneScopedN("UNTILIZE-BLOCK");
#endif
                tile_regs_acquire();
                for (uint32_t i = 0; i < per_core_block_tile_cnt; ++i) {
                    copy_tile(tt::CBIndex::c_0, i, i);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_untilize_dst<per_core_block_tile_cnt>(tt::CBIndex::c_16, 1, 0, num_rows_per_face, num_faces);
                tile_regs_release();
// If LLK perf is measured on block level put profiler zone around *_block operation
#ifdef LLK_PERF_BLOCK
            }
#endif
// If LLK perf is measured disable sync with DM cores/kernels
#ifndef LLK_PERF_NO_DM
            cb_push_back(tt::CBIndex::c_16, per_core_block_tile_cnt);
            cb_pop_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
#endif
        }
// If LLK perf is measured on OP level put profiler zone around complete operation
#ifdef LLK_PERF_OP
    }
#endif

    pack_untilize_uninit(tt::CBIndex::c_16);
}
}  // namespace NAMESPACE
