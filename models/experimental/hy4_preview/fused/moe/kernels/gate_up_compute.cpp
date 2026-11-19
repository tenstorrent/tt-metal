// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Gate|up partial sums of the Hy4 decode MoE program (TRISC) (fused/moe/program.py).
//
// One item = the K range of one (expert, pair of gate / up tile columns) this core's DRAM bank holds: item_blocks
// blocks of kb K tiles, K-major, [gate tile, up tile] per K tile. x is the 8 token rows as [ROWS, 32] BF16 tiles
// (resident for the whole program). Per item: DST 0 = x[K range] . gate, DST 1 = x[K range] . up (custom_mm, LoFi,
// FP32 accumulation), packed as two FP32 [ROWS, 32] tiles for the unit's finisher core.
//
// With PLAIN_TERMS = 2 the item of a plain slot is followed by its second BFP4 term (the same K range again),
// accumulated into the same partial sums.
//
// runtime args: [k0, residue, modulus] (the first K tile of this core's items; its share of the bank's items)
// compile-time args: [cb_x, cb_w, cb_out, x_tiles, kb, item_blocks, pb]

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"

#ifndef PLAIN_TERMS
#define PLAIN_TERMS 1
#endif

void kernel_main() {
    const uint32_t k0 = get_arg_val<uint32_t>(0);
#if PLAIN_TERMS > 1
    const uint32_t residue = get_arg_val<uint32_t>(1);
    const uint32_t modulus = get_arg_val<uint32_t>(2);
    constexpr uint32_t pb = get_compile_time_arg_val(6);
#endif

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t x_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t kb = get_compile_time_arg_val(4);
    constexpr uint32_t item_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t ct = 2;

    constexpr bool transpose = false;
    constexpr auto dest_layout = CustomMmDestLayout::DefaultSplitAcc;
    custom_mm_block_init<transpose, dest_layout>(cb_x, cb_w, cb_out, ct);

    uint32_t n_items = 0;
    UNPACK(n_items = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(n_items = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(n_items = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    [[maybe_unused]] uint32_t first_plain = 0;  // slots from here on have PLAIN_TERMS terms
    UNPACK(first_plain = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    MATH(first_plain = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));
    PACK(first_plain = ckernel::mailbox_read(ckernel::ThreadId::BriscThreadId));

    if (n_items > 0) {
        cb_wait_front(cb_x, x_tiles);
    }
    for (uint32_t it = 0; it < n_items; ++it) {
        tile_regs_acquire();
#if PLAIN_TERMS > 1
        const uint32_t n_blocks =
            (residue + it * modulus) / pb >= first_plain ? PLAIN_TERMS * item_blocks : item_blocks;
#else
        constexpr uint32_t n_blocks = item_blocks;
#endif
        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint32_t blk = b % item_blocks;  // a plain slot's second term covers the same K range again
            cb_wait_front(cb_w, kb * ct);
#ifndef HY4_NO_MATH
            if (b + 1 == n_blocks) {
                custom_mm_block<dest_layout, true>(cb_x, cb_w, k0 + blk * kb, 0, 0, kb, ct);
            } else {
                custom_mm_block<dest_layout, false>(cb_x, cb_w, k0 + blk * kb, 0, 0, kb, ct);
            }
#endif
            cb_pop_front(cb_w, kb * ct);
        }
        tile_regs_commit();
        cb_reserve_back(cb_out, ct);
        tile_regs_wait();
        pack_tile(0, cb_out);
        pack_tile(1, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, ct);
    }
    custom_mm_block_uninit<dest_layout>(cb_out);
}
