// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#include "tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

#ifdef SHORT_INIT
    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    pack_untilize_init_short<per_core_block_tile_cnt>(tt::CBIndex::c_0, tt::CBIndex::c_16);
#else
    pack_untilize_init<per_core_block_tile_cnt>(tt::CBIndex::c_0, tt::CBIndex::c_16);
#endif

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
                pack_untilize_block<per_core_block_tile_cnt>(tt::CBIndex::c_0, 1, tt::CBIndex::c_16);
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
