// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// SP-row sparse attention v2, the device-side UNION + gather (tt/spa_union.py), lane NCRISC (the row lane on the 2x4):
//   1. waits for this chip's union bitmap (member 0 -> the bitmap slot of this chip's position, S_LOCAL);
//   2. its exclusive word prefix counts -> CB_PFX [bitmap | prefix] (the members' remap reads it), S_PFX on every
//   member;
//      the union size > cap is flagged in the sync word STATUS (the pools past cap are not gathered);
//   3. waits for every line peer's bitmap (the peers' lane BRISCs write them into the slots of their positions and bump
//      ARR_BM + q; generation GEN + 1);
//   TWO (the 2x4): 3b. the row's OR of the 4 bitmaps -> the column lane's OR slot 0 (+ S_ORMINE there: its BRISC sends
//   it
//      to the column partner, whose stage A pushes the pools THIS row needs from it into this chip's relay rows), and
//      the relay prefix table: relay index of pool p = its rank among the row-OR pools in the PARTNER's words;
//   4. walks each destination's bitmap (this chip first): word w = the 128-token unit w, owned by chip rank w % N; for
//      every pool this chip owns, its 4 latent rows (local row (w / N) 128 + 4 b .. + 3) -> CB_STAGE and (slot = its
//      rank in the destination's union, destination position) -> CB_META;
//   TWO: 4b. waits ARR_DATA_A (the partner's stage A landed), then walks again for the pools the partner owns: rows
//      4 ridx .. + 3 of the relay tensor;
//   then an end marker (slot SKIP).
// Compile-time: CB_STAGE, CB_META, CB_PFX, WORDS, TWO, CB_ROWOR, CB_RP, then the TensorAccessorArgs of the local store
// and of the relay tensor (the local store again when not TWO). Runtime: sync, bitmaps (n slots of WORDS words, same
// address on every chip), store, p, n, cap, members, N (the owner modulus: every chip), k (this chip's rank), kp (the
// column partner's rank), relay address, column lane x, y, its OR slot 0 address, relay cap (pools), then the members'
// NoC (x, y).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spa/spa_stage.hpp"

constexpr uint32_t CB_STAGE = get_compile_time_arg_val(0);
constexpr uint32_t CB_META = get_compile_time_arg_val(1);
constexpr uint32_t CB_PFX = get_compile_time_arg_val(2);
constexpr uint32_t WORDS = get_compile_time_arg_val(3);
constexpr uint32_t TWO = get_compile_time_arg_val(4);
constexpr uint32_t CB_ROWOR = get_compile_time_arg_val(5);
constexpr uint32_t CB_RP = get_compile_time_arg_val(6);
constexpr uint32_t CB_SLOTS = get_compile_time_arg_val(7);  // the bitmap slots (this lane's CB, n x WORDS words)
constexpr uint32_t CB_ORS = get_compile_time_arg_val(8);    // TWO: the column lane's OR slots (same address there)
constexpr uint32_t RB = 1024;                               // one latent row: 512 bf16
constexpr uint32_t UNIT = 128;                              // tokens per shard unit = one bitmap word
constexpr uint32_t SKIP = 0xFFFFFFFFu;
constexpr uint32_t S_PFX = 1, S_LOCAL = 2, S_ORMINE = 3;  // semaphore ids (tt/spa_union.py)
constexpr uint32_t GEN = 0,
                   ARR_BM = 1;  // sync words: GEN, ARR_BM + position, ARR_DATA (spa_us.cpp), STATUS, ARR_DATA_A

inline uint32_t popc(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}

void kernel_main() {
    constexpr auto lat_args = TensorAccessorArgs<9>();
    constexpr auto rel_args = TensorAccessorArgs<lat_args.next_compile_time_args_offset()>();
    size_t a = 0;
    const uint32_t sync = get_arg_val<uint32_t>(a++);
    a++;  // (the bitmap slots: CB_SLOTS)
    cb_reserve_back(CB_SLOTS, 1);
    const uint32_t bms = get_write_ptr(CB_SLOTS);
    const uint32_t lat_addr = get_arg_val<uint32_t>(a++);
    const uint32_t p = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t cap = get_arg_val<uint32_t>(a++);
    const uint32_t m = get_arg_val<uint32_t>(a++);
    const uint32_t nall = get_arg_val<uint32_t>(a++);
    const uint32_t k = get_arg_val<uint32_t>(a++);
    const uint32_t kp = get_arg_val<uint32_t>(a++);
    const uint32_t rel_addr = get_arg_val<uint32_t>(a++);
    const uint32_t col_x = get_arg_val<uint32_t>(a++);
    const uint32_t col_y = get_arg_val<uint32_t>(a++);
    a++;  // (the column lane's OR slot 0: CB_ORS, the same address on the column lane)
    uint32_t col_or0 = 0;
    if constexpr (TWO) {
        cb_reserve_back(CB_ORS, 1);
        col_or0 = get_write_ptr(CB_ORS);
    }
    const uint32_t rcap = get_arg_val<uint32_t>(a++);
    const uint32_t xy0 = a;
    const auto lat = TensorAccessor(lat_args, lat_addr);
    const auto rel = TensorAccessor(rel_args, rel_addr);
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync);
    invalidate_l1_cache();  // GEN / ARR / the bitmap slots are written by the BRISC, member 0 and the peers' fabric
                            // writes
    const uint32_t g = sw[GEN] + 1;  // read before anything (the BRISC bumps GEN at the very end of the launch)
    const uint32_t status = 1 + 2 * n, arr_a = 2 + 2 * n;

    // 1. this chip's union bitmap
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(S_LOCAL)), 1);
    invalidate_l1_cache();
    // 2. the prefix table for the members' remap
    cb_reserve_back(CB_PFX, 1);
    volatile tt_l1_ptr uint32_t* ubm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_PFX));
    volatile tt_l1_ptr uint32_t* pfx = ubm + WORDS;
    volatile tt_l1_ptr uint32_t* own = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bms + p * WORDS * 4);
    uint32_t run = 0;
    for (uint32_t w = 0; w < WORDS; ++w) {
        const uint32_t x = own[w];
        ubm[w] = x;
        pfx[w] = run;
        run += popc(x);
    }
    sw[status] = run > cap ? run : 0;
    const uint32_t s_pfx = get_semaphore(S_PFX);
    for (uint32_t i = 0; i < m; ++i) {
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(xy0 + 2 * i), get_arg_val<uint32_t>(xy0 + 2 * i + 1), s_pfx), 1);
    }
    // 3. every line peer's bitmap
    for (uint32_t q = 0; q < n; ++q) {
        if (q != p) {
            while (sw[ARR_BM + q] < g) {
                invalidate_l1_cache();
            }
        }
    }
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* rowor = nullptr;
    volatile tt_l1_ptr uint32_t* rp = nullptr;
    if constexpr (TWO) {  // 3b. the row's OR -> the column lane; the relay prefix table over the partner's words
        cb_reserve_back(CB_ROWOR, 1);
        cb_reserve_back(CB_RP, 1);
        const uint32_t ro_l1 = get_write_ptr(CB_ROWOR);
        rowor = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ro_l1);
        rp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_RP));
        uint32_t rr = 0;
        for (uint32_t w = 0; w < WORDS; ++w) {
            uint32_t x = 0;
            for (uint32_t q = 0; q < n; ++q) {
                x |= reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bms + q * WORDS * 4)[w];
            }
            rowor[w] = x;
            rp[w] = rr;
            if (w % nall == kp) {
                rr += popc(x);
            }
        }
        noc_async_write(ro_l1, get_noc_addr(col_x, col_y, col_or0), WORDS * 4);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(col_x, col_y, get_semaphore(S_ORMINE)), 1);
    }
    // 4. the owned pools of every destination's union (this chip first: local writes overlap the first remote reads);
    // TWO: then the partner's pools out of the relay rows, after its stage A landed
    spa::Stager<CB_STAGE, CB_META> st;
    for (uint32_t pass = 0; pass < (TWO ? 2u : 1u); ++pass) {
        const uint32_t owner = pass == 0 ? k : kp;
        if (pass == 1) {
            st.close();  // pass 0's last pools go out before the wait
            while (sw[arr_a] < g) {
                invalidate_l1_cache();
            }
            invalidate_l1_cache();
        }
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t d = (p + i) % n;
            volatile tt_l1_ptr uint32_t* bm = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bms + d * WORDS * 4);
            uint32_t rank = 0;
            for (uint32_t w = 0; w < WORDS; ++w) {
                const uint32_t x = bm[w];
                if (x != 0 && w % nall == owner) {
                    uint32_t r = rank;
                    for (uint32_t b = 0; b < 32; ++b) {
                        if (!((x >> b) & 1u)) {
                            continue;
                        }
                        if (r < cap) {
                            if (pass == 0) {
                                st.add(lat, (w / nall) * UNIT + 4 * b, r, d);
                            } else {
                                const uint32_t ridx = rp[w] + popc(rowor[w] & ((1u << b) - 1u));
                                if (ridx < rcap) {  // past the relay rows: not delivered (the column lane flags it)
                                    st.add(rel, 4 * ridx, r, d);
                                }
                            }
                        }
                        ++r;
                    }
                }
                rank += popc(x);
            }
        }
    }
    st.end();
}
