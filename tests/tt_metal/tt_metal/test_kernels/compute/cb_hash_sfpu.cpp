// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel: drives hash_cb_sfpu() (SFPU lanewise FNV23) over an INT32 input
// CB. hash_cb_sfpu leaves the result tile in DEST slot 0 (row 0 = the 32
// per-lane accumulators, rest zeroed); this kernel packs that single tile to
// CB16 so the host can XOR-fold it. Used by
// tests/tt_metal/tt_metal/debug_tools/device_print/test_cb_hash.cpp.

#define DEBUG_CB_HASH 1

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/debug/cb_hash.h"

void kernel_main() {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_init(tt::CBIndex::c_0);

    tile_regs_acquire();
    cb_reserve_back(tt::CBIndex::c_16, 1);

    // SFPU FNV23 over CB0. Internally waits on CB0 (cb_wait_front), pops it,
    // and leaves the result tile in DEST slot 0.
    hash_cb_sfpu(tt::CBIndex::c_0, per_core_tile_cnt);

    // Pack the single result tile to CB16 for the host to fold.
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(/*ifrom_dst=*/0, tt::CBIndex::c_16);
    cb_push_back(tt::CBIndex::c_16, 1);

    tile_regs_release();
}
