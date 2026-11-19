// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 XSL expert FFN (moe_xffn_common.hpp) -- reader (BRISC, NoC0): core i runs its share of the work units
// (expert, M group of <= MG tile rows); per unit: the control pages, then per K chunk the gate|up weight chunk (KC rows
// x 2W tiles, one 2-tile (gate_j, up_j) transaction per row and j) and -- unless the writer gathers them
// (XFFN_X_ON_NOC1) -- the group's x rows (through disp, KC tiles of columns per row; with XFFN_X_SPLIT the reader takes
// the EVEN K chunks into CB_XRM and the writer the odd ones into CB_XRM1: both NoCs gather), then per N chunk the down
// weight chunk (W rows x NC tiles, 2-tile transactions). Only a group's VALID rows are gathered (a padding row's output
// is never written: its CB rows may hold anything).
// Unit assignment: round robin over the units in expert order (unit u -> core u mod P), or with XFFN_SNAKE dealt by
// cost class -- full groups (MG tile rows) first, then each expert's last group by its tile rows MG-1 .. 1 -- snaking
// over the cores (u -> u mod P on even rounds, P - 1 - u mod P on odd ones): every core's tile-row total within one
// group of the others (round robin left whole groups' worth of imbalance: 2 vs 3 units per core at 2,048 rows).
// Compile-time: H_T, W_T, KC, NC, MG, E, EP, P, WTB (weight tile bytes), then the TensorAccessorArgs of gate_up and
// down (every expert's tensor shares that layout). Runtime: x_ext, disp, cnt, start, address-table addresses, core i,
// disp words (+ the probe tensor address: moe_xffn_probe.hpp).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "moe_xffn_common.hpp"
#include "moe_xffn_probe.hpp"

using namespace xffn;
using namespace xffn_probe_slot;

#if !defined(XFFN_X_ON_NOC1) || defined(XFFN_X_SPLIT)
#define XFFN_READER_GATHERS 1  // this kernel gathers x rows (all K chunks, or the even ones with XFFN_X_SPLIT)
#endif

void kernel_main() {
    constexpr uint32_t H_T = get_compile_time_arg_val(0);
    constexpr uint32_t W_T = get_compile_time_arg_val(1);
    constexpr uint32_t KC = get_compile_time_arg_val(2);
    constexpr uint32_t NC = get_compile_time_arg_val(3);
    constexpr uint32_t MG = get_compile_time_arg_val(4);
    constexpr uint32_t E = get_compile_time_arg_val(5);
    constexpr uint32_t EP = get_compile_time_arg_val(6);
    constexpr uint32_t P = get_compile_time_arg_val(7);
    constexpr uint32_t WTB = get_compile_time_arg_val(8);
    constexpr auto gu_args = TensorAccessorArgs<9>();
    constexpr auto dn_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    constexpr uint32_t GU_T = 2 * W_T;            // gate|up tiles per K row
    constexpr uint32_t ROW_BYTES = H_T * 32 * 2;  // one x row (bf16)
    constexpr uint32_t SEG = KC * 32 * 2;         // the K chunk of one x row
    constexpr uint32_t TBL_W = (2 * E + 15) / 16 * 16;
    static_assert(H_T % KC == 0 && H_T % NC == 0 && NC % 2 == 0 && EP % 16 == 0, "chunking");

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t disp_addr = get_arg_val<uint32_t>(1);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(2);
    const uint32_t start_addr = get_arg_val<uint32_t>(3);
    const uint32_t tbl_addr = get_arg_val<uint32_t>(4);
    const uint32_t i = get_arg_val<uint32_t>(5);
    const uint32_t cap = get_arg_val<uint32_t>(6);  // disp words (one page)
    XP_BEGIN();

    // tables: counts | starts | addresses | the group's disp rows
    const uint32_t tab = get_write_ptr(CB_TAB);
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab);
    volatile tt_l1_ptr uint32_t* start = cnt + EP;
    volatile tt_l1_ptr uint32_t* tbl = start + EP;
    volatile tt_l1_ptr uint32_t* dl = tbl + TBL_W;
#ifdef XFFN_SHARD_BASES
    // XFFN_SHARD_BASES (the host checked: every weight shard = the FULL height x 2 tiles, contiguous in one bank):
    // per unit, the NoC address of each shard's first page once; page (k, n) = base[n / 2] + (2 k + n % 2) WTB --
    // no TensorAccessor page math per read (that math was the reader's bottleneck: ~60 us per K chunk)
    uint64_t* gu_base = reinterpret_cast<uint64_t*>(reinterpret_cast<uint32_t>(dl + MG * 32 + 15) & ~63u);
    uint64_t* dn_base = gu_base + W_T;
#endif
    {
        XP_TIC();
        const InterleavedAddrGen<true> cnt_g{.bank_base_address = cnt_addr, .page_size = EP * 4};
        const InterleavedAddrGen<true> start_g{.bank_base_address = start_addr, .page_size = EP * 4};
        const InterleavedAddrGen<true> tbl_g{.bank_base_address = tbl_addr, .page_size = 2 * E * 4};
        noc_async_read(cnt_g.get_noc_addr(0), reinterpret_cast<uint32_t>(cnt), EP * 4);
        noc_async_read(start_g.get_noc_addr(0), reinterpret_cast<uint32_t>(start), EP * 4);
        noc_async_read(tbl_g.get_noc_addr(0), reinterpret_cast<uint32_t>(tbl), (2 * E * 4 + 63) / 64 * 64);
        noc_async_read_barrier();
        XP_TOC(R_TABLES);
    }
    const InterleavedAddrGen<true> x_g{.bank_base_address = x_addr, .page_size = ROW_BYTES};
    const InterleavedAddrGen<true> disp_g{.bank_base_address = disp_addr, .page_size = cap * 4};

    // one work unit: expert e's M group gi (tile rows g0 .. g0 + mt - 1 of its region)
    auto unit = [&](uint32_t e, uint32_t gi) {
        const uint32_t m = cnt[e];
        const uint32_t mt_all = (m + 31) / 32;
        const uint32_t g0 = gi * MG;
        const uint32_t mt = mt_all - g0 < MG ? mt_all - g0 : MG;
        const uint32_t rows = m - g0 * 32 < mt * 32 ? m - g0 * 32 : mt * 32;
        const uint32_t d0 = start[e] + g0 * 32;
        const auto gu = TensorAccessor(gu_args, tbl[2 * e], WTB);
        const auto dn = TensorAccessor(dn_args, tbl[2 * e + 1], WTB);
#ifdef XFFN_SHARD_BASES
        for (uint32_t j = 0; j < W_T; ++j) {
            gu_base[j] = gu.get_noc_addr(2 * j);
        }
        for (uint32_t sh = 0; sh < H_T / 2; ++sh) {
            dn_base[sh] = dn.get_noc_addr(2 * sh);
        }
#endif
        XP_ADD(R_UNITS, 1);
        cb_reserve_back(CB_CTL, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_CTL))[0] = mt;
        cb_push_back(CB_CTL, 1);
        cb_reserve_back(CB_WCTL, 1);
        {
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_WCTL));
            w[0] = mt;
            w[1] = rows;
            w[2] = d0;
        }
        cb_push_back(CB_WCTL, 1);
#ifdef XFFN_READER_GATHERS
        // the group's dispatch rows -> token rows (only the first `rows` are used)
        noc_async_read(disp_g.get_noc_addr(0, d0 * 4), reinterpret_cast<uint32_t>(dl), mt * 32 * 4);
        noc_async_read_barrier();
#endif

        for (uint32_t kc = 0; kc < H_T / KC; ++kc) {
            XP_TIC();
            cb_reserve_back(CB_WGU, KC * GU_T);
            XP_TOC(R_GU_WAIT);
            XP_TIC();
            const uint32_t wp = get_write_ptr(CB_WGU);
#ifdef XFFN_COALESCE  // diagnostics only: shard j's K chunk (KC rows x 2 tiles, contiguous in its bank) in ONE read
            for (uint32_t j = 0; j < W_T; ++j) {
                noc_async_read(gu_base[j] + kc * KC * 2 * WTB, wp + j * KC * 2 * WTB, KC * 2 * WTB);
            }
#else
            for (uint32_t k = 0; k < KC; ++k) {
#ifdef XFFN_SHARD_BASES
                const uint32_t off = (kc * KC + k) * 2 * WTB;
                for (uint32_t j = 0; j < W_T; ++j) {
                    noc_async_read(gu_base[j] + off, wp + (k * GU_T + 2 * j) * WTB, 2 * WTB);
                }
#else
                const uint32_t row = (kc * KC + k) * GU_T;
                for (uint32_t j = 0; j < W_T; ++j) {
                    noc_async_read(gu.get_noc_addr(row + 2 * j), wp + (k * GU_T + 2 * j) * WTB, 2 * WTB);
                }
#endif
            }
#endif
#ifdef XFFN_READER_GATHERS
#ifdef XFFN_X_SPLIT
            const bool mine = (kc & 1) == 0;  // the even K chunks (the writer gathers the odd ones over NoC1)
#else
            constexpr bool mine = true;
#endif
            if (mine) {
                // always MG tile rows' worth (a producer's pushes must land exactly on the CB end: MG KC pages per
                // push); tile row t = KC tile-sized pages holding its 32 rows of SEG bytes, rows >= `rows` unused
                cb_reserve_back(CB_XRM, MG * KC);
                const uint32_t xp = get_write_ptr(CB_XRM);
                for (uint32_t r = 0; r < rows; ++r) {
                    noc_async_read(x_g.get_noc_addr(dl[r], kc * SEG), xp + r * SEG, SEG);
                }
            }
#endif
            noc_async_read_barrier();
            XP_TOC(R_GU_READ);
            XP_ADD(R_KB, KC * GU_T * WTB / 1024);
            cb_push_back(CB_WGU, KC * GU_T);
#ifdef XFFN_READER_GATHERS
            if (mine) {
                cb_push_back(CB_XRM, MG * KC);
            }
#endif
        }
#ifndef XFFN_NO_DOWN
        for (uint32_t nc = 0; nc < H_T / NC; ++nc) {
            XP_TIC();
            cb_reserve_back(CB_WD, W_T * NC);
            XP_TOC(R_DN_WAIT);
            XP_TIC();
            const uint32_t wp = get_write_ptr(CB_WD);
#ifdef XFFN_COALESCE  // diagnostics only: each down shard (W rows x 2 tiles, contiguous) in ONE read
            for (uint32_t sl = 0; sl < NC / 2; ++sl) {
                noc_async_read(dn_base[nc * NC / 2 + sl], wp + sl * W_T * 2 * WTB, W_T * 2 * WTB);
            }
#else
            for (uint32_t k = 0; k < W_T; ++k) {
                for (uint32_t jj = 0; jj < NC; jj += 2) {
#ifdef XFFN_SHARD_BASES
                    noc_async_read(dn_base[(nc * NC + jj) / 2] + k * 2 * WTB, wp + (k * NC + jj) * WTB, 2 * WTB);
#else
                    noc_async_read(dn.get_noc_addr(k * H_T + nc * NC + jj), wp + (k * NC + jj) * WTB, 2 * WTB);
#endif
                }
            }
#endif
            noc_async_read_barrier();
            XP_TOC(R_DN_READ);
            XP_ADD(R_KB, W_T * NC * WTB / 1024);
            cb_push_back(CB_WD, W_T * NC);
        }
#endif
    };

#ifdef XFFN_SNAKE
    // cost classes: mt = MG (every full group), then MG - 1 .. 1 (an expert's last, partial group); unit v of the
    // dealing order goes to core v mod P on even rounds (v / P even), P - 1 - v mod P on odd ones
    uint32_t v = 0;
    for (uint32_t cls = MG; cls >= 1; --cls) {
        for (uint32_t e = 0; e < E; ++e) {
            const uint32_t m = cnt[e];
            if (m == 0) {
                continue;
            }
            const uint32_t mt_all = (m + 31) / 32;
            const uint32_t ng = (mt_all + MG - 1) / MG;
            const uint32_t last_mt = mt_all - (ng - 1) * MG;
            const uint32_t n_full = cls == MG ? ng - 1 : 0;  // the full groups (not counting the last)
            const uint32_t n_cls = n_full + (last_mt == cls ? 1 : 0);
            for (uint32_t j = 0; j < n_cls; ++j, ++v) {
                const uint32_t core = ((v / P) & 1) ? P - 1 - v % P : v % P;
                if (core == i) {
                    unit(e, j < n_full ? j : ng - 1);
                }
            }
        }
    }
#else
    // work units = (expert, M group), numbered in expert order; unit u runs on core u mod P -- a hot expert's groups
    // spread over consecutive cores (each re-reads its weights: ~1.8 MB, cheap next to the imbalance of one core
    // running every row of it; zipf-skewed routing doubled the FFN phase with whole experts per core)
    uint32_t u = 0;
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t m = cnt[e];
        if (m == 0) {
            continue;
        }
        const uint32_t ng = ((m + 31) / 32 + MG - 1) / MG;
        const uint32_t first = (i + P - u % P) % P;  // this core's first group of e: (u + first) mod P == i
        u += ng;
        for (uint32_t gi = first; gi < ng; gi += P) {
            unit(e, gi);
        }
    }
#endif
    cb_reserve_back(CB_CTL, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_CTL))[0] = 0;
    cb_push_back(CB_CTL, 1);
    cb_reserve_back(CB_WCTL, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_WCTL))[0] = 0;
    cb_push_back(CB_WCTL, 1);
    XP_FLUSH((tab + 63) & ~63u, get_arg_val<uint32_t>(7), i);  // (the tables are done: their L1 is the scratch)
}
