// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Bank-streamed decode linear, compute (tt/fused/bank_linear.py): per pass, the reference's fp32-Dest schedule of
// bmm_dram_sharded_multicore.cpp (MT = 1, no L1 accumulation): matmul_block_init(ct = ncq), then over the K blocks
// in walk order j = (p + rot) % nblk, matmul_block(x tile p * kb + r (cb_x holds x in walk order), weight row r of
// the block, ct = ncq), all K
// in Dest; pack the ncq output tiles (fp32) into one slot of DST_TILES tiles of cb_out.
// Runtime args: rot (unused here), number of passes, then ncq per pass.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_out = 16;
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");
constexpr uint32_t nblk = get_named_compile_time_arg_val("nblk");
constexpr uint32_t max_ncq = get_named_compile_time_arg_val("max_ncq");
constexpr uint32_t SLOT = kb * max_ncq;
constexpr uint32_t DST_TILES = 8;

void kernel_main() {
    uint32_t a = 0;
    a++;  // rot: the writer lays x out in walk order
    const uint32_t npass = get_arg_val<uint32_t>(a++);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    for (uint32_t q = 0; q < npass; ++q) {
        const uint32_t ncq = get_arg_val<uint32_t>(a++);
        matmul_block_init(cb_x, cb_w, false, ncq, 1, 1);
        tile_regs_acquire();
        for (uint32_t p = 0; p < nblk; ++p) {
            if (q == 0) {  // x arrives in walk order: block p (= K block (p + rot) % nblk) at position p
                cb_wait_front(cb_x, (p + 1) * kb);
            }
            cb_wait_front(cb_w, SLOT);
            for (uint32_t r = 0; r < kb; ++r) {
                matmul_block(cb_x, cb_w, p * kb + r, r * ncq, 0, false, ncq, 1, 1);
            }
            cb_pop_front(cb_w, SLOT);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, DST_TILES);
        tile_regs_wait();
        for (uint32_t t = 0; t < ncq; ++t) {
            pack_tile<true>(t, cb_out, t);
        }
        tile_regs_release();
        cb_push_back(cb_out, DST_TILES);
    }
    cb_pop_front(cb_x, kt);
}
