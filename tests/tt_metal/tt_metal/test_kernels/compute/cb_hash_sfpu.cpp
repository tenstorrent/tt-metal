// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: drives hash_cb_sfpu() (SFPU lanewise FNV23 with MATH → L1 →
// UNPACK round trip) over an INT32 input CB. Used by
// tests/tt_metal/tt_metal/debug_tools/dprint/test_cb_hash.cpp.

#define DEBUG_CB_HASH 1

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/debug/cb_hash.h"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    uint32_t label = get_compile_time_arg_val(1);

    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);

    acquire_dst();
    cb_reserve_back(tt::CBIndex::c_16, per_core_tile_cnt);

    // The hash probe: SFPU FNV23 over the L1 bytes of CB0. Internally waits
    // on CB0 (cb_wait_front), pops it after consuming, and routes the result
    // through MATH → MEM_LLK_DEBUG L1 slot → UNPACK → DPRINT.
    hash_cb_sfpu(tt::CBIndex::c_0, per_core_tile_cnt, label);

    // CB0 was already popped by hash_cb_sfpu. We can't re-read its tiles, so
    // just push zero tiles to CB16 to keep the writer's cb_pop_front happy.
    // (The test ignores buffer_Res content — it only checks DPRINT.)
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        pack_tile(/*ifrom_dst=*/0, tt::CBIndex::c_16);
        cb_push_back(tt::CBIndex::c_16, 1);
    }

    release_dst();
}
