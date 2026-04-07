// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: copies tiles and uses debug_dump_cb_typed for format-aware output.
// On TRISC0 (Unpack), prints tile data interpreted by the CB's data format.
// On other TRISCs, the call is a no-op.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/dump.h"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);

    acquire_dst();
    cb_wait_front(tt::CBIndex::c_0, per_core_tile_cnt);
    cb_reserve_back(tt::CBIndex::c_16, per_core_tile_cnt);

    // Test debug_dump_cb_typed: format-aware tile print (active on TRISC0/Unpack)
    debug_dump_cb_typed(0, 0);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        copy_tile(tt::CBIndex::c_0, b, b);
    }

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        pack_tile(b, tt::CBIndex::c_16);
        cb_pop_front(tt::CBIndex::c_0, 1);
        cb_push_back(tt::CBIndex::c_16, 1);
    }

    release_dst();
}
