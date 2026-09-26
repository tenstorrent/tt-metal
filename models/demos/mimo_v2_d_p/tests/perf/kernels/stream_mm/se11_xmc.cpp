// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// End-to-end flat expert: x relay multicaster (NCRISC). Like se8_xmc.cpp, but its source is the tilizer's
// super-blocks (MT row tiles x 32 K tiles, row tile major): each of a super-block's 4 x blocks ([MT x KBLK] tiles, row
// tile major in the gate/up cores' x ring) goes out as MT linked multicasts of one row tile's KBLK tiles, once the
// block's ring slot is free on every core of the rectangle (minimum of the cores' freed words). XARR is then set to
// the number of blocks delivered, by a multicast that follows the linked data.
// CT: 0 SB_CB, 1 MT, 2 TILE_BYTES, 3 X_SLOTS, 4 XARR_SEM, 5 WORD_SEM, 6 KBLK
// RT: 0 x ring address, 1 rect start xy, 2 end xy, 3 dests, 4 freed words address, 5 cores whose words gate a slot,
//     6 super-blocks this relay sends, 7 XARR semaphore id, 8 STRIDE, 9 OFF (this relay sends super-blocks OFF,
//     OFF + STRIDE, ... in stream order; a block's ring slot follows its global index), 10 unused
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_DYN
#include "se_dyn.hpp"
#endif
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define ZW(name) DeviceZoneScopedN(name)
#else
#define ZW(name)
#endif

void kernel_main() {
    constexpr uint32_t sb_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mt = get_compile_time_arg_val(1);
    constexpr uint32_t tb = get_compile_time_arg_val(2);
    constexpr uint32_t x_slots = get_compile_time_arg_val(3);
    constexpr uint32_t word_sem = get_compile_time_arg_val(5);
    constexpr uint32_t kblk = get_compile_time_arg_val(6);
    constexpr uint32_t per_sb = 32 / kblk;
    constexpr uint32_t sb_tiles = mt * 32;
    constexpr uint32_t blk_bytes = mt * kblk * tb, piece = kblk * tb;
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t a0 = get_arg_val<uint32_t>(1), a1 = get_arg_val<uint32_t>(2), dests = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
    const uint32_t n_cores = get_arg_val<uint32_t>(5);
#ifdef SE_DYN
    // Dynamic counts: this relay's super-blocks follow from the active experts' sub-blocks (RT 14.. are the
    // se_dyn.hpp args, CT 7 NUM_E, CB 7's upper half this RISC's scratch)
    constexpr uint32_t num_e = get_compile_time_arg_val(7), nsb = get_compile_time_arg_val(8);
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, 14, get_write_ptr(tt::CBIndex::c_7) + 2048, mt * 32);
    const uint32_t tot_sb = dyn.num_v * nsb, st_ = get_arg_val<uint32_t>(8), of_ = get_arg_val<uint32_t>(9);
    const uint32_t num_sb = tot_sb > of_ ? (tot_sb - of_ + st_ - 1) / st_ : 0;
#else
    const uint32_t num_sb = get_arg_val<uint32_t>(6);
#endif
    // up to two rectangles: RT 1-3 and, when RT 11 (second rectangle's dests) is non-zero, RT 12-13 start / end xy
    const uint32_t dests2 = get_arg_val<uint32_t>(11);
    const uint32_t nrect = dests2 ? 2 : 1;
    const uint32_t b0 = get_arg_val<uint32_t>(12), b1 = get_arg_val<uint32_t>(13);
    const uint64_t rects[2] = {
        get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0),
        dests2 ? get_noc_multicast_addr(b0 >> 16, b0 & 0xFFFF, b1 >> 16, b1 & 0xFFFF, 0) : 0};
    const uint32_t ndest[2] = {dests, dests2};
    volatile tt_l1_ptr uint32_t* word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(word_sem));
    const uint32_t xarr_addr = get_semaphore(get_arg_val<uint32_t>(7));
    const uint32_t stride = get_arg_val<uint32_t>(8), soff = get_arg_val<uint32_t>(9);
    // global index of this relay's k-th block
    auto gidx = [&](uint32_t k) { return ((k / per_sb) * stride + soff) * per_sb + k % per_sb; };
    auto min_freed = [&]() {
        uint32_t lo = freed[0];
        for (uint32_t i = 1; i < n_cores; ++i) {
            lo = freed[i] < lo ? freed[i] : lo;
        }
        return lo;
    };
    uint32_t sent = 0;
    for (uint32_t b = 0; b < num_sb; ++b) {
        {
            ZW("XMC_SB");
            cb_wait_front(sb_cb, sb_tiles);
        }
        const uint32_t src = get_read_ptr(sb_cb);
        uint32_t i = 0;
        while (i < per_sb) {
            invalidate_l1_cache();
            const uint32_t lim = min_freed() + x_slots;
            uint32_t n = 0;
            while (i + n < per_sb && gidx(sent + n) < lim) {
                ++n;
            }
            if (!n) {
                ZW("XMC_CRED");
                while (gidx(sent) >= min_freed() + x_slots) {
                    invalidate_l1_cache();
                }
                continue;
            }
            while (!ncrisc_noc_nonposted_writes_sent(noc_index)) {
            }  // the previous counter write has left L1 before its source word is reused
            *word = sent + n;
            // One linked chain per rectangle (a linked chain must keep one destination set; its last piece is
            // unlinked), then that rectangle's XARR update, which follows the data on the same route and VC.
            for (uint32_t q = 0; q < nrect; ++q) {
                for (uint32_t k = 0; k < n; ++k) {
                    const uint32_t dst = ring + (gidx(sent + k) % x_slots) * blk_bytes;
                    for (uint32_t m = 0; m < mt; ++m) {
                        const bool last = k + 1 == n && m + 1 == mt;  // the chain ends before the next rectangle's
                        noc_async_write_multicast(
                            src + (m * 32 + (i + k) * kblk) * tb, rects[q] | (dst + m * piece), piece, ndest[q], !last);
                    }
                }
                noc_semaphore_set_multicast(reinterpret_cast<uint32_t>(word), rects[q] | xarr_addr, ndest[q]);
            }
            sent += n;
            i += n;
            noc_async_writes_flushed();
        }
        cb_pop_front(sb_cb, sb_tiles);
    }
    noc_async_write_barrier();
}
