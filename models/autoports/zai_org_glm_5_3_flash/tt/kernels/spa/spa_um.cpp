// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SP-row sparse attention v2, the device-side UNION (tt/spa_union.py), member NCRISC (M member cores, member i owns
// query rows [row0, row0 + nrows) of this chip):
//   1. its rows' selected token ids (uint32, SKIP tail) -> a partial window bitmap of POOLS (bit p = pool p = tokens 4
//   p ..
//      4 p + 3; word w = pools 32 w .. 32 w + 31 = the 128-token shard unit w);
//   2. the OR tree (pull): member i waits until its children i + 2^l (l < trailing zeros of i) signalled their bitmap
//      final, reads each child's bitmap and ORs it in, then signals its parent i - 2^tz(i); member 0 writes the chip's
//      union bitmap into the lane's bitmap slot of this chip and signals the lane (S_LOCAL);
//   3. after the lane's prefix table is ready (S_PFX): the chip's bitmap + exclusive word prefix counts -> each
//   selected id
//      t becomes its compact row 4 rank(t / 4) + t % 4 (rank = the pool's index in the union, pool order), written as
//      the ids_new rows (the SKIP tail kept). A rank >= cap maps to row t % 4 (the lane flags the overflow).
// Compile-time: CB_BM, CB_SCR, CB_PFX, CB_ROWS, WORDS, W (ids per row), CB_SLOTS, then the ids / ids_new
// TensorAccessorArgs. Runtime: ids, ids_new, row0, nrows, member index, M, lane x, lane y, this chip's slot index in
// CB_SLOTS, cap, the lane's DROP sync word address, then the members' NoC (x, y) in member order. Out-of-window input
// (an id past the WORDS x 32 pools: stale chunk inputs): never dereferenced -- no bitmap bit, its compact row is 0 --
// and counted: the member adds its count to the lane's DROP word (sync word 31, monotonic; SpaUnion.status()). 14b's
// realgate7 hang (2026-10-25 12:01Z) was such input indexing L1 out of bounds (bm[p >> 5] / pfx[w]).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB_BM = get_compile_time_arg_val(0);
constexpr uint32_t CB_SCR = get_compile_time_arg_val(1);
constexpr uint32_t CB_PFX = get_compile_time_arg_val(2);
constexpr uint32_t CB_ROWS = get_compile_time_arg_val(3);
constexpr uint32_t WORDS = get_compile_time_arg_val(4);
constexpr uint32_t W = get_compile_time_arg_val(5);
constexpr uint32_t CB_SLOTS =
    get_compile_time_arg_val(6);  // the lane's bitmap slots (one CB over members + lane: same address)
constexpr uint32_t SKIP = 0xFFFFFFFFu;
constexpr uint32_t NPOOLS = WORDS * 32;  // the window's pools (an id past it: dropped + counted, see the header)
constexpr uint32_t S_CHILD = 0, S_PFX = 1, S_LOCAL = 2;  // semaphore ids (tt/spa_union.py)

inline uint32_t popc(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}

void kernel_main() {
    constexpr auto ids_args = TensorAccessorArgs<7>();
    constexpr auto out_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    size_t a = 0;
    const uint32_t ids_addr = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t row0 = get_arg_val<uint32_t>(a++);
    const uint32_t nrows = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t m = get_arg_val<uint32_t>(a++);
    const uint32_t lane_x = get_arg_val<uint32_t>(a++);
    const uint32_t lane_y = get_arg_val<uint32_t>(a++);
    const uint32_t slot_idx = get_arg_val<uint32_t>(a++);  // this chip's slot in the lane's CB_SLOTS (member 0 only)
    const uint32_t cap = get_arg_val<uint32_t>(a++);
    const uint32_t drop_addr = get_arg_val<uint32_t>(a++);
    const uint32_t xy0 = a;  // members' NoC (x, y) from here
    const auto ids = TensorAccessor(ids_args, ids_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    constexpr uint32_t RB = W * 4;

    cb_reserve_back(CB_BM, 1);
    cb_reserve_back(CB_SCR, 1);
    cb_reserve_back(CB_PFX, 1);
    cb_reserve_back(CB_ROWS, 1);
    const uint32_t bm_l1 = get_write_ptr(CB_BM);
    const uint32_t scr_l1 = get_write_ptr(CB_SCR);
    const uint32_t pfx_l1 = get_write_ptr(CB_PFX);  // [WORDS] the chip's bitmap, then [WORDS] its exclusive prefix
    const uint32_t rows_l1 = get_write_ptr(CB_ROWS);
    volatile tt_l1_ptr uint32_t* bm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bm_l1);
    volatile tt_l1_ptr uint32_t* scr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_l1);
    for (uint32_t w = 0; w < WORDS; ++w) {
        bm[w] = 0;
    }
    // 1. the rows' ids (kept in L1 for step 3) -> the partial bitmap
    uint32_t drops = 0;
    for (uint32_t k = 0; k < nrows; ++k) {
        noc_async_read_page(row0 + k, ids, rows_l1 + k * RB);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();  // NoC-written L1 below: no stale data-cache lines (here and after every wait / barrier)
    for (uint32_t k = 0; k < nrows; ++k) {
        volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1 + k * RB);
        for (uint32_t j = 0; j < W; ++j) {
            const uint32_t v = row[j];
            if (v == SKIP) {
                break;
            }
            const uint32_t p = v >> 2;
            if (p < NPOOLS) {
                bm[p >> 5] |= 1u << (p & 31);
            } else {
                ++drops;
            }
        }
    }
    // 2. the OR tree (pull): children i + 2^l for l < tz(i) (member 0: every l with 2^l < m)
    uint32_t nch = 0;
    for (uint32_t l = 0; (1u << l) < m && ((me >> l) & 1u) == 0; ++l) {
        if (me + (1u << l) < m) {
            ++nch;
        }
    }
    const uint32_t s_child = get_semaphore(S_CHILD);
    volatile tt_l1_ptr uint32_t* s_child_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_child);
    if (nch) {
        noc_semaphore_wait_min(s_child_p, nch);
        for (uint32_t l = 0; (1u << l) < m && ((me >> l) & 1u) == 0; ++l) {
            const uint32_t c = me + (1u << l);
            if (c >= m) {
                continue;
            }
            const uint32_t cx = get_arg_val<uint32_t>(xy0 + 2 * c), cy = get_arg_val<uint32_t>(xy0 + 2 * c + 1);
            noc_async_read(get_noc_addr(cx, cy, bm_l1), scr_l1, WORDS * 4);
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t w = 0; w < WORDS; ++w) {
                bm[w] |= scr[w];
            }
        }
    }
    if (me != 0) {
        uint32_t tz = 0;
        while (((me >> tz) & 1u) == 0) {
            ++tz;
        }
        const uint32_t par = me - (1u << tz);
        const uint32_t px = get_arg_val<uint32_t>(xy0 + 2 * par), py = get_arg_val<uint32_t>(xy0 + 2 * par + 1);
        noc_semaphore_inc(get_noc_addr(px, py, s_child), 1);
    } else {  // member 0: CB_SLOTS is declared on {member 0, the lane} only (one descriptor: the same address there)
        cb_reserve_back(CB_SLOTS, 1);
        const uint32_t lane_slot = get_write_ptr(CB_SLOTS) + slot_idx * WORDS * 4;
        noc_async_write(bm_l1, get_noc_addr(lane_x, lane_y, lane_slot), WORDS * 4);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(lane_x, lane_y, get_semaphore(S_LOCAL)), 1);
    }
    // 3. the remap: the chip's bitmap + prefix from the lane (its CB_PFX holds both, same address on the lane)
    volatile tt_l1_ptr uint32_t* s_pfx_p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(S_PFX));
    noc_semaphore_wait_min(s_pfx_p, 1);
    noc_async_read(get_noc_addr(lane_x, lane_y, pfx_l1), pfx_l1, 2 * WORDS * 4);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* ubm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pfx_l1);
    volatile tt_l1_ptr uint32_t* pfx = ubm + WORDS;
    for (uint32_t k = 0; k < nrows; ++k) {
        volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1 + k * RB);
        for (uint32_t j = 0; j < W; ++j) {
            const uint32_t v = row[j];
            if (v == SKIP) {
                break;
            }
            const uint32_t p = v >> 2, w = p >> 5, b = p & 31;
            if (p >= NPOOLS) {
                row[j] = 0;
                continue;
            }
            uint32_t r = pfx[w] + popc(ubm[w] & ((1u << b) - 1u));
            if (r >= cap) {
                r = 0;
            }
            row[j] = 4 * r + (v & 3);
        }
        noc_async_write_page(row0 + k, out, rows_l1 + k * RB);
    }
    noc_async_write_barrier();
    if (drops) {
        noc_semaphore_inc(get_noc_addr(lane_x, lane_y, drop_addr), drops);
        noc_async_atomic_barrier();
    }
}
