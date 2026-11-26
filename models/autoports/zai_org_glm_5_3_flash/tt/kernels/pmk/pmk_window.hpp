// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// PMK PHASE WINDOWS (Stage 14, the per-layer kernel framework: per_layer_kernel_design.md 2.1 / 2.2). One program runs
// several phase families back to back on a role whose CB union does not fit L1 (A3's kda items + A4's o_proj on the
// members). tt/pmk_layer.py declares window 1's CBs IN PLACE (contiguous from CB PMK_WIN_BASE) and every later window's
// CBs ONE PAGE each (their formats); at the entry of window W the role's three composites
//   1. finish window W - 1: BRISC / NCRISC barrier their NoC traffic, the TRISC pack thread pushes after its last pack;
//   2. raise their own barrier CB (PMK_WIN_CB_D0 / D1 / T: counters only, reset by the firmware at every launch) and
//   wait
//      for the other two (>= W): no thread of the core touches window W's memory while another still uses window W -
//      1's;
//   3. re-lay out THEIR OWN LocalCBInterface copy of window W's CBs at (window-1 base + offset, size, page) from the
//      host table PMK_WIN<W>_TAB = {cb, offset, size, page, ..} (bytes; the CBs are EMPTY and untouched before W).
// Remote writers address window W's CBs through runtime publishes made inside W (PfStream's PARKADDR after its READY),
// never through host-computed addresses. Precondition (host): window W's CBs fit window 1's span; distinct indices.
#pragma once

#include <cstdint>

namespace pmk_win {

// this RISC's interface of `cb` -> [addr, addr + size), page bytes (TRISC addresses in 16-B words: cb_addr_shift)
template <bool RD, bool WR>
FORCE_INLINE void place(uint32_t cb, uint32_t addr, uint32_t size, uint32_t page) {
    LocalCBInterface& li = get_local_cb_interface(cb);
    const uint32_t a = addr >> cb_addr_shift, s = size >> cb_addr_shift;
    li.fifo_size = s;
    li.fifo_limit = a + s;
    li.fifo_page_size = page >> cb_addr_shift;
    if constexpr (RD) {
        li.fifo_rd_ptr = a;
    }
    if constexpr (WR) {
        li.fifo_wr_ptr = a;
        li.fifo_num_pages = size / page;
        li.fifo_wr_tile_ptr = 0;
    }
}

// the window-1 region's base in bytes (CB PMK_WIN_BASE as the firmware set it up at launch: never re-laid out)
FORCE_INLINE uint32_t base_bytes() {
    const LocalCBInterface& li = get_local_cb_interface(PMK_WIN_BASE);
    return (li.fifo_limit - li.fifo_size) << cb_addr_shift;
}

// b = the window-1 region's base, read by the composite ONCE at kernel start (pmk_win_b = base_bytes() before the first
// window): CB PMK_WIN_BASE itself may be re-laid out (A15: the X / XS group is placed per window) or killed by a later
// window, so reading it at a later entry gave a wrong base (A15 watched re-run 04:18Z: window 4 laid out from address
// 0, 'Local L1 overwrites mailboxes')
template <bool RD, bool WR>
FORCE_INLINE void relayout(const uint32_t* tab, uint32_t n, uint32_t b) {
    for (uint32_t i = 0; i < n; ++i) {
        place<RD, WR>(tab[4 * i], b + tab[4 * i + 1], tab[4 * i + 2], tab[4 * i + 3]);
    }
}

// this RISC's interfaces of the stale windowed CBs -> unused (fifo_size 0): the watcher's CB sanitizer (sanitize.h
// debug_valid_cb_addr) judges a transfer by the first interface containing its start; an earlier window's or a launch
// declaration's range overlapping this window's region would flag a correct transfer. Data-movement RISCs only.
FORCE_INLINE void kill(const uint32_t* tab, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        LocalCBInterface& li = get_local_cb_interface(tab[i]);
        li.fifo_size = 0;
        li.fifo_limit = 0;
    }
}

// raw counter of a barrier CB (no pops: received only grows within a launch)
FORCE_INLINE uint32_t count(uint32_t cb) { return static_cast<uint16_t>(*get_cb_tiles_received_ptr(cb)); }

// THE RECLAIM WINDOW (Stage 16 M-B, 2026-10-24: program B as the last window of the fused layer program): the window
// is laid out from the CB BASE (every earlier CB is dead on the core: b = window 1's base minus the shared bytes) and
// its CBs may REUSE indices an earlier window used -- possibly with another producer / consumer RISC. The L1 page
// counters of those indices are balanced first (acked := received, by the BRISC after the other two RISCs left the
// earlier windows), then every RISC re-reads them into its local copies (the TRISC packer keeps tiles_received, the
// unpacker tiles_acked: llk_io_pack.h / llk_io_unpack.h; the data-movement API reads L1 only)
FORCE_INLINE void balance(const uint32_t* tab, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t cb = tab[4 * i];
        *get_cb_tiles_acked_ptr(cb) = *get_cb_tiles_received_ptr(cb);
    }
}
template <bool RD, bool WR>
FORCE_INLINE void resync(const uint32_t* tab, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t cb = tab[4 * i];
        LocalCBInterface& li = get_local_cb_interface(cb);
        if constexpr (RD) {
            li.tiles_acked = static_cast<uint16_t>(*get_cb_tiles_acked_ptr(cb));
        }
        if constexpr (WR) {
            li.tiles_received = static_cast<uint16_t>(*get_cb_tiles_received_ptr(cb));
        }
    }
}

}  // namespace pmk_win

#if defined(COMPILE_FOR_TRISC)
namespace pmk_win {
// TRISC: the pack thread's push follows its last pack (llk_push_tiles stalls on the packer); unpack / pack wait for
// both data-movement RISCs; the unpacker owns the read side of the interfaces, the packer the write side (math has
// none)
template <uint32_t W>
ALWI void enter(const uint32_t* tab, uint32_t n, const uint32_t* /*kill_tab*/, uint32_t /*nk*/, uint32_t b) {
    cb_reserve_back(PMK_WIN_CB_T, 1);
    cb_push_back(PMK_WIN_CB_T, 1);
    UNPACK(({
        while (count(PMK_WIN_CB_D0) < W || count(PMK_WIN_CB_D1) < W) {
        }
        relayout<true, false>(tab, n, b);
        resync<true, false>(tab, n);  // the BRISC balanced them before raising D0 (every window: wt-fuse3 eabf3f5a186)
    }));
    PACK(({
        while (count(PMK_WIN_CB_D0) < W || count(PMK_WIN_CB_D1) < W) {
        }
        relayout<false, true>(tab, n, b);
        resync<false, true>(tab, n);
    }));
}
// the reclaim window (see balance): the BRISC raises D0 only after balancing, so both threads read balanced counters
template <uint32_t W>
ALWI void enter_reclaim(const uint32_t* tab, uint32_t n, const uint32_t* /*kill_tab*/, uint32_t /*nk*/, uint32_t b) {
    cb_reserve_back(PMK_WIN_CB_T, 1);
    cb_push_back(PMK_WIN_CB_T, 1);
    UNPACK(({
        while (count(PMK_WIN_CB_D0) < W || count(PMK_WIN_CB_D1) < W) {
        }
        relayout<true, false>(tab, n, b);
        resync<true, false>(tab, n);
    }));
    PACK(({
        while (count(PMK_WIN_CB_D0) < W || count(PMK_WIN_CB_D1) < W) {
        }
        relayout<false, true>(tab, n, b);
        resync<false, true>(tab, n);
    }));
}
}  // namespace pmk_win
#else
namespace pmk_win {
#if defined(COMPILE_FOR_BRISC)
constexpr uint32_t CB_MINE = PMK_WIN_CB_D0, CB_PEER = PMK_WIN_CB_D1;
#else
constexpr uint32_t CB_MINE = PMK_WIN_CB_D1, CB_PEER = PMK_WIN_CB_D0;
#endif
// BRISC / NCRISC: every read / write / atomic of the dying window has completed before the count goes up
template <uint32_t W>
FORCE_INLINE void enter(const uint32_t* tab, uint32_t n, const uint32_t* kill_tab, uint32_t nk, uint32_t b) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {  // a dynamic-NoC composite may have traffic on both NoCs
        noc_async_full_barrier(0);
        noc_async_full_barrier(1);
    } else {
        noc_async_full_barrier();
    }
#if defined(COMPILE_FOR_BRISC)
    // the BRISC enters LAST and balances the window's counters (an earlier window's unpopped pushes of a re-laid index:
    // the TP8 mhc's resident X hung A15's P10, wt-fuse3 eabf3f5a186)
    while (count(CB_PEER) < W || count(PMK_WIN_CB_T) < W) {
        invalidate_l1_cache();
    }
    balance(tab, n);
    cb_reserve_back(CB_MINE, 1);
    cb_push_back(CB_MINE, 1);
#else
    cb_reserve_back(CB_MINE, 1);
    cb_push_back(CB_MINE, 1);
    while (count(CB_PEER) < W || count(PMK_WIN_CB_T) < W) {
        invalidate_l1_cache();
    }
#endif
    kill(kill_tab, nk);
    relayout<true, true>(tab, n, b);
}
// the reclaim window: the BRISC waits for the NCRISC and the TRISC to leave the earlier windows, balances the counters
// of this window's indices, THEN raises D0 (the other two wait D0 before re-laying out: they see balanced counters)
template <uint32_t W>
FORCE_INLINE void enter_reclaim(const uint32_t* tab, uint32_t n, const uint32_t* kill_tab, uint32_t nk, uint32_t b) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        noc_async_full_barrier(0);
        noc_async_full_barrier(1);
    } else {
        noc_async_full_barrier();
    }
#if defined(COMPILE_FOR_BRISC)
    while (count(CB_PEER) < W || count(PMK_WIN_CB_T) < W) {
        invalidate_l1_cache();
    }
    balance(tab, n);
    cb_reserve_back(CB_MINE, 1);
    cb_push_back(CB_MINE, 1);
#else
    cb_reserve_back(CB_MINE, 1);
    cb_push_back(CB_MINE, 1);
    while (count(CB_PEER) < W || count(PMK_WIN_CB_T) < W) {
        invalidate_l1_cache();
    }
#endif
    kill(kill_tab, nk);
    relayout<true, true>(tab, n, b);
}
}  // namespace pmk_win
#endif
