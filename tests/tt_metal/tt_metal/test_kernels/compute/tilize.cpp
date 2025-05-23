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
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t fast = 1;
    // constexpr uint32_t block_count = per_core_block_tile_cnt >= 8 ? 8 : per_core_block_tile_cnt;
    constexpr uint32_t block_count = per_core_block_tile_cnt % 8 == 0   ? 8
                                     : per_core_block_tile_cnt % 4 == 0 ? 4
                                     : per_core_block_tile_cnt % 2 == 0 ? 2
                                                                        : 1;
    // constexpr uint32_t block_count = 1;

    if constexpr (fast) {
#ifndef SHORT_INIT
        fast_tilize_init(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);
#else
        unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
        fast_tilize_init_short(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);
#endif
    } else {
#ifndef SHORT_INIT
        tilize_init(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);
#else
        unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
        tilize_init_short(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);
#endif
    }

    cb_wait_front(tt::CBIndex::c_0, per_core_block_cnt * per_core_block_tile_cnt);
    cb_reserve_back(tt::CBIndex::c_16, per_core_block_cnt * per_core_block_tile_cnt);

    volatile std::uint32_t* base_address = (std::uint32_t*)MEM_LLK_DEBUG_BASE;
    uint64_t start = 0;
    uint64_t end = 0;

    UNPACK((base_address[1] = 1));
    MATH((base_address[2] = 2));
    PACK((base_address[3] = 3));
    while (base_address[1] != 1) {
        asm("nop");
    }
    while (base_address[2] != 2) {
        asm("nop");
    }
    while (base_address[3] != 3) {
        asm("nop");
    }
    UNPACK((base_address[5] = 5));
    MATH((base_address[6] = 6));
    PACK((base_address[7] = 7));
    while (base_address[5] != 5) {
        asm("nop");
    }
    while (base_address[6] != 6) {
        asm("nop");
    }
    while (base_address[7] != 7) {
        asm("nop");
    }
    UNPACK((base_address[1] = 0));
    MATH((base_address[2] = 0));
    PACK((base_address[3] = 0));
    while (base_address[1] != 0) {
        asm("nop");
    }
    while (base_address[2] != 0) {
        asm("nop");
    }
    while (base_address[3] != 0) {
        asm("nop");
    }
    UNPACK((base_address[5] = 0));
    MATH((base_address[6] = 0));
    PACK((base_address[7] = 0));

    {
        DeviceZoneScopedN("tilize-sync");
        {
            DeviceZoneScopedN("tilize-loop");
            start = read_wall_clock();
            for (uint32_t i = 0; i < 1024; ++i) {
                for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
                    if constexpr (fast) {
                        fast_tilize_block(
                            tt::CBIndex::c_0,
                            block_count,
                            per_core_block_tile_cnt,
                            tt::CBIndex::c_16,
                            b * per_core_block_tile_cnt,
                            b * per_core_block_tile_cnt);
                    } else {
                        tilize_block(
                            tt::CBIndex::c_0,
                            per_core_block_tile_cnt,
                            tt::CBIndex::c_16,
                            b * per_core_block_tile_cnt,
                            b * per_core_block_tile_cnt);
                    }
                }
            }
        }
        tensix_sync();
        end = read_wall_clock();
    }

    DPRINT << "time: " << (end - start) / (1024 * per_core_block_cnt * per_core_block_tile_cnt) << " cycles" << ENDL();

    cb_pop_front(tt::CBIndex::c_0, per_core_block_cnt * per_core_block_tile_cnt);
    cb_push_back(tt::CBIndex::c_16, per_core_block_cnt * per_core_block_tile_cnt);

    if constexpr (fast) {
        fast_tilize_uninit();
    } else {
        tilize_uninit(tt::CBIndex::c_0, tt::CBIndex::c_16);
    }
}
}  // namespace NAMESPACE
