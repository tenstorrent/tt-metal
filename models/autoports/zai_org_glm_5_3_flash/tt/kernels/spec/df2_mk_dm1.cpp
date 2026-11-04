// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DFlash2 rung-3 owner core, NCRISC (NoC1): the RMS phases' all-gather of the 64 owners' partial sums of squares
// (kernels/head/head_dm1.cpp's pattern) and the HEADS phases' K / V cache writes (kernels/spec/df2_cache.cpp's
// read-modify-write of the destination tiles, here per owner: the owner of k / v head h writes the tiles of that
// head alone, so no two cores touch a tile). df2_mk_common.hpp documents the arguments.
// Runtime args: 0 k, 1 c0, 2 the owner table (DRAM, 64 x (x, y) uint32 = one 512-B page), phase blocks at 3.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spec/df2_bfp8.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spec/df2_mk_common.hpp"

using namespace df2mk;

namespace {

constexpr uint32_t NPH = get_compile_time_arg_val(CT_NPH);
constexpr uint32_t NQH = get_compile_time_arg_val(CT_NQH);
constexpr uint32_t NKVH = get_compile_time_arg_val(CT_NKVH);
constexpr uint32_t HT = get_compile_time_arg_val(CT_HT);
constexpr uint32_t NBLK = get_compile_time_arg_val(CT_NBLK);
constexpr uint32_t PB1 = 3;
constexpr uint32_t NH_ALL = NQH + 2 * NKVH;
constexpr uint32_t GQ = NQH / NKVH;
constexpr uint32_t POS_PAGE = 128;
constexpr uint32_t PT_ROOM = (NBLK * 4 + 63u) & ~63u;
constexpr uint32_t OWN_BYTES = 512;
#if defined(DF2_KV_BFP8)
constexpr uint32_t TBC = df2bfp8::TILE_BYTES;  // the cache tiles: bfloat8_b (GLM53_DF2_KV_DTYPE=bfp8; rows re-encoded)
#else
constexpr uint32_t TBC = TB;
#endif

FORCE_INLINE uint64_t dram(uint32_t addr, uint32_t page_size, uint32_t page) {
    const InterleavedAddrGen<true> g = {.bank_base_address = addr, .page_size = page_size};
    return g.get_noc_addr(page);
}

FORCE_INLINE volatile tt_l1_ptr uint32_t* u32(uint32_t l1) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
}
FORCE_INLINE volatile tt_l1_ptr int32_t* i32(uint32_t l1) { return reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1); }

// local L1 -> local L1, one 32-B face row (RISC copy)
FORCE_INLINE void copy_row32(uint32_t src, uint32_t dst) {
    volatile tt_l1_ptr uint32_t* s = u32(src);
    volatile tt_l1_ptr uint32_t* d = u32(dst);
    for (uint32_t w = 0; w < 8; ++w) {
        d[w] = s[w];
    }
}

FORCE_INLINE void zero_l1(uint32_t l1, uint32_t bytes) {
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < bytes; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, l1 + off, MEM_ZEROS_SIZE);
    }
}

// the RMS gather: this owner's 32 partials (column 0 of P) -> a 128-B record into slot k of EVERY owner's landing
// area (+ their SEM_G); wait for all 64; lay the records into the 2 G tiles (owner o: column o % 32 of tile o / 32)
FORCE_INLINE void gather(uint32_t k, uint32_t s_own) {
    const uint32_t land = get_write_ptr(CB_LAND);  // the same address on every owner (one CB, one core range)
    const uint32_t rec = get_write_ptr(CB_REC);
    cb_wait_front(CB_P, 1);
    invalidate_l1_cache();
    {
        volatile tt_l1_ptr uint32_t* p = u32(get_read_ptr(CB_P));
        volatile tt_l1_ptr uint32_t* r = u32(rec);
        for (uint32_t t = 0; t < REC_BYTES / 4; ++t) {
            r[t] = p[tile_off(t, 0)];
        }
    }
    cb_pop_front(CB_P, 1);
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* own = u32(s_own);
    for (uint32_t o = 0; o < N_OWNERS; ++o) {
        noc_async_write(rec, get_noc_addr(own[2 * o], own[2 * o + 1], land + k * REC_BYTES), REC_BYTES);
    }
    noc_async_write_barrier();  // every record landed before its owner's count is raised
    for (uint32_t o = 0; o < N_OWNERS; ++o) {
        noc_semaphore_inc(get_noc_addr(own[2 * o], own[2 * o + 1], get_semaphore(SEM_G)), 1);
    }
    noc_async_atomic_barrier();
    mk::wait_handoff(SEM_G, N_OWNERS);
    invalidate_l1_cache();
    cb_reserve_back(CB_G, 2);
    const uint32_t g = get_write_ptr(CB_G);
    zero_l1(g, 2 * TB32);
    noc_async_read_barrier();
    invalidate_l1_cache();
    {
        volatile tt_l1_ptr uint32_t* gt = u32(g);
        volatile tt_l1_ptr uint32_t* l = u32(land);
        for (uint32_t o = 0; o < N_OWNERS; ++o) {
            for (uint32_t t = 0; t < 32; ++t) {
                gt[(o >> 5) * 1024u + tile_off(t, o & 31u)] = l[o * (REC_BYTES / 4) + t];
            }
        }
    }
    invalidate_l1_cache();  // the RISC stores land before the unpacker reads the tiles
    cb_push_back(CB_G, 2);
}

// the HT tiles of one head (rows = the 32 block rows, at src_l1) -> the paged cache [NBLK, 1, 32, HD]'s rows pos[t]
// (tile (block, j) = page block * HT + j): rows grouped by destination tile, each tile read once, patched, written back
FORCE_INLINE void cache_rows(uint32_t cache, uint32_t src_l1, uint32_t s_pos, uint32_t s_pt, uint32_t s_tile) {
    volatile tt_l1_ptr int32_t* posv = i32(s_pos);
    volatile tt_l1_ptr uint32_t* ptv = u32(s_pt);
    for (uint32_t j = 0; j < HT; ++j) {
        uint32_t done = 0;  // bit t: row t written
        for (uint32_t t = 0; t < 32; ++t) {
            if ((done >> t) & 1u) {
                continue;
            }
            const int32_t p = posv[t];
            if (p < 0) {
                done |= 1u << t;
                continue;
            }
            const uint32_t tile0 = ptv[static_cast<uint32_t>(p) >> 5] * HT + j;
            noc_async_read(dram(cache, TBC, tile0), s_tile, TBC);
            noc_async_read_barrier();
            invalidate_l1_cache();
            for (uint32_t tt = t; tt < 32; ++tt) {
                if ((done >> tt) & 1u) {
                    continue;
                }
                const int32_t pp = posv[tt];
                if (pp < 0 || ptv[static_cast<uint32_t>(pp) >> 5] * HT + j != tile0) {
                    continue;
                }
                const uint32_t r = static_cast<uint32_t>(pp) & 31u;
                const uint32_t s = src_l1 + j * TB + row_off(tt);
#if defined(DF2_KV_BFP8)
                df2bfp8::patch_row(s, s_tile, r);
#else
                const uint32_t d = s_tile + row_off(r);
                copy_row32(s, d);
                copy_row32(s + 512, d + 512);
#endif
                done |= 1u << tt;
            }
            invalidate_l1_cache();  // the stores land before the NoC writes the tile back
            noc_async_write(s_tile, dram(cache, TBC, tile0), TBC);
            noc_async_write_barrier();
        }
    }
}

FORCE_INLINE void ph_heads(uint32_t rb, uint32_t k, uint32_t s_pos, uint32_t s_pt, uint32_t s_tile) {
    if (k >= NH_ALL || k < NQH) {
        return;  // not a k / v head (q heads: the BRISC writes them)
    }
    const uint32_t h = k;
    const bool isk = h < NQH + NKVH;
    const uint32_t g = isk ? h - NQH : h - NQH - NKVH;
    const uint32_t pos = get_arg_val<uint32_t>(rb + 5), pt = get_arg_val<uint32_t>(rb + 6);
    const uint32_t pt_bytes = get_arg_val<uint32_t>(rb + 7);
    const uint32_t cache = get_arg_val<uint32_t>(rb + PB_Q + (isk ? NKVH : 2 * NKVH) + g);
    noc_async_read(dram(pos, POS_PAGE, 0), s_pos, POS_PAGE);
    // the [NQH, blocks] page table's row of this kv head's first q head: block b of kv head g = g NBLK + b of the
    // layer's ONE [NKVH NBLK, 1, 32, hd] cache (the SDPA runs all q heads in one launch over it)
    noc_async_read(dram(pt, pt_bytes, g * GQ), s_pt, pt_bytes);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const uint32_t cb = isk ? CB_KOUT : CB_V;  // (one consumer RISC per CB: df2_mk_common.hpp)
    cb_wait_front(cb, HT);
    invalidate_l1_cache();
    {  // tap (GLM53_DF2_TAPS): the head's tiles as written (K after norm + RoPE / V) -> pages g HT + j
        const uint32_t tap = get_arg_val<uint32_t>(rb + (isk ? RT_TAP2 : RT_TAP3));
        if (tap != 0) {
            for (uint32_t j = 0; j < HT; ++j) {
                noc_async_write(get_read_ptr(cb) + j * TB, dram(tap, TB, g * HT + j), TB);
            }
            noc_async_write_barrier();
        }
    }
    cache_rows(cache, get_read_ptr(cb), s_pos, s_pt, s_tile);
    cb_pop_front(cb, HT);
}

template <uint32_t P>
FORCE_INLINE void run(uint32_t k, uint32_t s_pos, uint32_t s_pt, uint32_t s_tile, uint32_t s_own) {
    if constexpr (P < NPH) {
        constexpr uint32_t TYPE = get_compile_time_arg_val(CT_PH0 + 2 * P);
        constexpr uint32_t rb = PB1 + RT_W * P;
        if constexpr (TYPE == PH_RMS || TYPE == PH_RMS_AR) {
            gather(k, s_own);
        } else if constexpr (TYPE == PH_HEADS) {
            ph_heads(rb, k, s_pos, s_pt, s_tile);
        }
        mk::cb_rewind_range(
            0, CB_SCR);  // (the raw scratch CBs' pointers never move) the chunked CBs back to their base
        run<P + 1>(k, s_pos, s_pt, s_tile, s_own);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t k = get_arg_val<uint32_t>(0);
    const uint32_t tab = get_arg_val<uint32_t>(2);
    const uint32_t scr = (get_write_ptr(CB_SCR) + 63u) & ~63u;
    const uint32_t s_pos = scr, s_pt = scr + POS_PAGE, s_tile = s_pt + PT_ROOM, s_own = s_tile + TB;
    noc_async_read(dram(tab, OWN_BYTES, 0), s_own, OWN_BYTES);
    noc_async_read_barrier();
    invalidate_l1_cache();
    run<0>(k, s_pos, s_pt, s_tile, s_own);
}
