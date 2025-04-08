// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

#include "tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    DPRINT_PACK(DPRINT << "per_core_block_cnt: " << per_core_block_cnt << ENDL());
    DPRINT_PACK(DPRINT << "per_core_block_tile_cnt: " << per_core_block_tile_cnt << ENDL());
#ifndef SHORT_INIT
    tilize_init(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);
#else
    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    tilize_init_short(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);
#endif

// If LLK perf is measured on OP level put profiler zone around complete operation
#ifdef LLK_PERF_OP
    {
        DeviceZoneScopedN("TILIZE-OP")
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
                DeviceZoneScopedN("TILIZE-BLOCK");
#endif
                tilize_block(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);
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

    tilize_uninit(tt::CBIndex::c_0, tt::CBIndex::c_16);
}
}  // namespace NAMESPACE
