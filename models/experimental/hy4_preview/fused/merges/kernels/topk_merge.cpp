// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The cross-chip top-k merge as a program of its own (fused/merges/topk_merge.py): one data-movement kernel per
// user on that user's core. It reads the user's chip-local survivor from DRAM (the index-key scan's output:
// 2 raw uint32 tiles, pages `survivor_page` and `survivor_page + 1` of an interleaved buffer), runs
// hy4_topk::merge and leaves
//   * its result in the piece's state tile on this core (persistent L1, read by the host):
//       word 0  the cut word of fused/indexer_scan (bit 31 | keep)     word 1  cut key | flags << 16
//       word 2  entries above the cut key, all chips                    word 3  tied entries on earlier chips
//       words 4..9 the merge's timing stamps, 10 kernel start, 11 survivor loaded, 12 kernel end
//   * its words 0..3 in the mailbox of user 0's core (words 32 + 4 * user of its state tile; word 64 counts
//     arrivals and is consumed by subtraction), which writes the chip's 64-byte cut page to DRAM for the cut
//     program. A mailbox slot is 16 bytes at a 16-byte boundary, like its source: a NoC write takes its data from
//     the source at the destination's offset within the 16-byte word (a 4-byte write to word 32 + user left the
//     sender's word `user % 4` there on the board).
// In a layer program the survivor is already in the root scan core's L1 and the cut runs on the same core:
// hy4_topk::merge is called there and neither the DRAM read nor the mailbox exists.
//
// Compile-time args: users, topk, BITS0, BITS1, cb_survivor (2 x 4096 B), cb_payload (one page of the site's
// part), cb_page (64 B).
// Runtime args (per core): user, this chip's slot, survivor buffer address, survivor page, cut buffer address,
// user 0's core NoC x, y, state tile address, then two hy4_collective::Use (round 0, round 1).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "topk_merge.hpp"

namespace {

constexpr uint32_t users = get_compile_time_arg_val(0);
constexpr uint32_t topk = get_compile_time_arg_val(1);
constexpr uint32_t bits0 = get_compile_time_arg_val(2);
constexpr uint32_t bits1 = get_compile_time_arg_val(3);
constexpr uint32_t cb_survivor = get_compile_time_arg_val(4);
constexpr uint32_t cb_payload = get_compile_time_arg_val(5);
constexpr uint32_t cb_page = get_compile_time_arg_val(6);
constexpr uint32_t word_tile_bytes = 4096;

constexpr uint32_t ST_CUT = 0, ST_KEY = 1, ST_ABOVE = 2, ST_BEFORE = 3, ST_STAMPS = 4;
constexpr uint32_t ST_START = 10, ST_LOADED = 11, ST_END = 12, ST_BOX = 32, ST_BOX_WORDS = 4, ST_COUNT = 64;

FORCE_INLINE uint64_t interleaved_page(uint32_t base, uint32_t page, uint32_t page_bytes) {
    return get_noc_addr_from_bank_id<true>(page % NUM_DRAM_BANKS, base + (page / NUM_DRAM_BANKS) * page_bytes);
}

}  // namespace

void kernel_main() {
#ifdef HY4_MERGE_NOP
    return;
#else
    size_t arg = 0;
    const uint32_t user = get_arg_val<uint32_t>(arg++);
    const uint32_t chip = get_arg_val<uint32_t>(arg++);
    const uint32_t survivor_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t survivor_page = get_arg_val<uint32_t>(arg++);
    const uint32_t cuts_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t collector_x = get_arg_val<uint32_t>(arg++);
    const uint32_t collector_y = get_arg_val<uint32_t>(arg++);
    const uint32_t state = get_arg_val<uint32_t>(arg++);
    const hy4_collective::Use use0 = hy4_collective::read_use(arg);
    const hy4_collective::Use use1 = hy4_collective::read_use(arg);

    volatile tt_l1_ptr uint32_t* st = hy4_collective::word(state);
    st[ST_START] = hy4_collective::now();
    hy4_collective::stamp(use0.ctrl, hy4_collective::STAMP_START);

    WAYPOINT("TKRD");
    const uint32_t survivor = get_write_ptr(cb_survivor);
    for (uint32_t tile = 0; tile < 2; ++tile) {
        noc_async_read(
            interleaved_page(survivor_addr, survivor_page + tile, word_tile_bytes),
            survivor + tile * word_tile_bytes,
            word_tile_bytes);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    st[ST_LOADED] = hy4_collective::now();

    WAYPOINT("TKMG");
    const hy4_topk::Result result = hy4_topk::merge<bits0, bits1>(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(survivor), get_write_ptr(cb_payload), use0, use1, chip, topk);
    st[ST_CUT] = (1u << 31) | result.keep;
    st[ST_KEY] = result.key | (result.flags << 16);
    st[ST_ABOVE] = result.above;
    st[ST_BEFORE] = result.before;
    for (uint32_t i = 0; i < 6; ++i) {
        st[ST_STAMPS + i] = result.stamps[i];
    }

    // The chip's cut page, through user 0's core.
    WAYPOINT("TKBX");
    const uint64_t counter = get_noc_addr(collector_x, collector_y, state + 4 * ST_COUNT);
    if (user == 0) {
        st[ST_BOX] = st[ST_CUT];
    } else {
        noc_async_write(
            state,
            get_noc_addr(collector_x, collector_y, state + 4 * (ST_BOX + ST_BOX_WORDS * user)),
            4 * ST_BOX_WORDS);
        noc_async_write_barrier();
    }
    noc_semaphore_inc(counter, 1);
    noc_async_atomic_barrier();
    if (user == 0) {
        uint32_t arrived = 0;
        if (hy4_collective::wait_min(state + 4 * ST_COUNT, users, arrived)) {
            noc_semaphore_inc(counter, 0u - users);
            noc_async_atomic_barrier();
        } else {
            st[ST_KEY] = st[ST_KEY] | (hy4_topk::FLAG_TIMEOUT << 16);
        }
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* page = hy4_collective::word(get_write_ptr(cb_page));
        for (uint32_t u = 0; u < 16; ++u) {
            page[u] = u < users ? st[ST_BOX + ST_BOX_WORDS * u] : 0;
        }
        noc_async_write(get_write_ptr(cb_page), get_noc_addr_from_bank_id<true>(0, cuts_addr), 64);
        noc_async_write_barrier();
    }
    st[ST_END] = hy4_collective::now();
    WAYPOINT("TKDN");
#endif
}
