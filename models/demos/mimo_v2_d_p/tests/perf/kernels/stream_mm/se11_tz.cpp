// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end flat expert: x relay tilizer (TRISC). Each super-block (MT row tiles x 32 K tiles of one virtual expert)
// arrives as MT row-major chunks of 32 rows x 32 tiles (se11_xrd.cpp) and leaves as MT x 32 bfp8 tiles, row tile
// major, for the relay's multicaster (se11_xmc.cpp).
// CT: 0 RM_CB, 1 SB_CB, 2 MT; RT: 0 super-blocks in total
#include <cstdint>
#include "api/compute/tilize.h"
#include "api/compute/compute_kernel_hw_startup.h"
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#endif

void kernel_main() {
    constexpr uint32_t rm_cb = get_compile_time_arg_val(0);
    constexpr uint32_t sb_cb = get_compile_time_arg_val(1);
    constexpr uint32_t mt = get_compile_time_arg_val(2);
#ifdef SE_DYN
    cb_wait_front(tt::CBIndex::c_6, 1);  // this relay's super-block count (from se11_xrd.cpp)
    const uint32_t num_sb = read_tile_value(tt::CBIndex::c_6, 0, 0);
    cb_pop_front(tt::CBIndex::c_6, 1);
#else
    const uint32_t num_sb = get_arg_val<uint32_t>(0);
#endif
    compute_kernel_hw_startup(rm_cb, sb_cb);
    fast_tilize_init(rm_cb, 32, sb_cb);
    for (uint32_t b = 0; b < num_sb; ++b) {
        {
#ifdef SE_ZONES
            DeviceZoneScopedN("TZ_OUT");
#endif
            cb_reserve_back(sb_cb, mt * 32);
        }
        for (uint32_t m = 0; m < mt; ++m) {
            {
#ifdef SE_ZONES
                DeviceZoneScopedN("TZ_IN");
#endif
                cb_wait_front(rm_cb, 32);
            }
            fast_tilize_block(rm_cb, 32, sb_cb, 0, m * 32);
            cb_pop_front(rm_cb, 32);
        }
        cb_push_back(sb_cb, mt * 32);
    }
    fast_tilize_uninit(rm_cb, sb_cb, 32);
}
