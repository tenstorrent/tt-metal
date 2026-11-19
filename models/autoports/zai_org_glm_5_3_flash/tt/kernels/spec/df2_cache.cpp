// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 17 DFlash2 in-trace drafter (tt/dflash2_kind.py TtnnOps.cache_write): the 32 K and V rows of one 32-row step
// into the drafter's paged K / V cache at positions pos[j], from ONE core, sequentially.
//
// Why not paged_update_cache: the drafter's 32 rows are 32 positions of ONE sequence (every "user" row's page table is
// the same block list). paged_update_cache runs one user per core and each core reads the whole cache tile(s) of its
// row, patches its row and writes the tile(s) back -- with 32 users on the same tile the read-modify-writes race and
// the last writer wins: 31 of 32 rows are lost (Stage 17 u1 on silicon: hidden PCC 0.3, drafts 0 / 16; the torch
// backend, which writes rows one by one, matches the reference). The op's share_cache mode (a serialising semaphore
// chain) is refused for paged caches. This kernel is that serial chain on one core (kernels/mla/lat_dm1.cpp's pattern).
//
// Inputs: K / V = the height-sharded decode-head tensors [1, 32 (row j), 32 (head rows, NKV valid), HD] bf16 TILE on
// the 32 batch cores (shard j = row j's [32, HD] = HT tiles at the same L1 address on core j; the cores' NoC (x, y)
// are runtime args); the cache [blocks, NKV, 32, HD] bf16 TILE interleaved DRAM (tile (block b, head h, col k) = page
// (b NKV + h) HT + k); the page table row 0 [blocks] int32 (row-major DRAM, one page); pos [32] int32 (one page).
// For every source (K, V) and head h: the rows are grouped by destination tile row (page(pos[j] / 32), pos[j] % 32)
// -- the tiles are read, every row of the group copied in (two 32-B face rows per tile column), written back, barrier;
// then the next group. Rows with pos < 0 are skipped (PREP clamps at 0, so none in practice).
// Compile-time args: NKV, HT (HD / 32), NSRC (32: the shard cores). Runtime args (fixed 7 + 2 NSRC): K cache, V cache,
// page table, pos addresses, K shard L1 address, V shard L1 address, page-table row bytes, then NSRC (x, y) pairs.
// CB 0: scratch = 128 (pos) + pt row (64-B rounded) + 2 x NSRC x HT tiles (the shards) + HT tiles (the RMW tiles).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/mla_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/spec/df2_bfp8.hpp"

namespace {
constexpr uint32_t TB = 2048;  // bf16 tile bytes (the source shards; the L1 slot stride of the RMW tiles)
#if defined(DF2_KV_BFP8)
constexpr uint32_t TBC = df2bfp8::TILE_BYTES;  // the cache tiles: bfloat8_b (GLM53_DF2_KV_DTYPE=bfp8; rows re-encoded)
#else
constexpr uint32_t TBC = TB;
#endif
constexpr uint32_t POS_PAGE = 128;

FORCE_INLINE uint64_t dram(uint32_t addr, uint32_t page_size, uint32_t page) {
    const InterleavedAddrGen<true> g = {.bank_base_address = addr, .page_size = page_size};
    return g.get_noc_addr(page);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t NKV = get_compile_time_arg_val(0);
    constexpr uint32_t HT = get_compile_time_arg_val(1);
    constexpr uint32_t NSRC = get_compile_time_arg_val(2);
    static_assert(NKV >= 1 && NKV <= 32 && HT >= 1 && NSRC == 32, "32 rows of up to 32 kv heads");
    const uint32_t kc = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t pt = get_arg_val<uint32_t>(2);
    const uint32_t pos = get_arg_val<uint32_t>(3);
    const uint32_t k_l1 = get_arg_val<uint32_t>(4);
    const uint32_t v_l1 = get_arg_val<uint32_t>(5);
    const uint32_t pt_bytes = get_arg_val<uint32_t>(6);
    const uint32_t pt_room = (pt_bytes + 63u) & ~63u;

    const uint32_t scr = (get_write_ptr(0) + 63u) & ~63u;
    const uint32_t s_pos = scr;
    const uint32_t s_pt = s_pos + POS_PAGE;
    const uint32_t s_k = s_pt + pt_room;           // NSRC x HT tiles
    const uint32_t s_v = s_k + NSRC * HT * TB;     // NSRC x HT tiles
    const uint32_t s_tile = s_v + NSRC * HT * TB;  // HT tiles (the RMW group)

    // the positions, the page-table row, the 32 K and V shards
    noc_async_read(dram(pos, POS_PAGE, 0), s_pos, POS_PAGE);
    noc_async_read(dram(pt, pt_bytes, 0), s_pt, pt_bytes);
    for (uint32_t j = 0; j < NSRC; ++j) {
        const uint32_t x = get_arg_val<uint32_t>(7 + 2 * j);
        const uint32_t y = get_arg_val<uint32_t>(8 + 2 * j);
        noc_async_read(get_noc_addr(x, y, k_l1), s_k + j * HT * TB, HT * TB);
        noc_async_read(get_noc_addr(x, y, v_l1), s_v + j * HT * TB, HT * TB);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr int32_t* posv = reinterpret_cast<volatile tt_l1_ptr int32_t*>(s_pos);
    volatile tt_l1_ptr uint32_t* ptv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s_pt);

    for (uint32_t src = 0; src < 2; ++src) {
        const uint32_t cache = src == 0 ? kc : vc;
        const uint32_t shards = src == 0 ? s_k : s_v;
        for (uint32_t h = 0; h < NKV; ++h) {
            uint32_t done = 0;  // bit j: row j written
            for (uint32_t j = 0; j < NSRC; ++j) {
                if ((done >> j) & 1u) {
                    continue;
                }
                const int32_t pj = posv[j];
                if (pj < 0) {
                    done |= 1u << j;
                    continue;
                }
                const uint32_t tile0 = (ptv[static_cast<uint32_t>(pj) / 32] * NKV + h) * HT;
                for (uint32_t k = 0; k < HT; ++k) {
                    noc_async_read(dram(cache, TBC, tile0 + k), s_tile + k * TB, TBC);
                }
                noc_async_read_barrier();
                invalidate_l1_cache();
                for (uint32_t jj = j; jj < NSRC; ++jj) {
                    if ((done >> jj) & 1u) {
                        continue;
                    }
                    const int32_t p = posv[jj];
                    if (p < 0 || (ptv[static_cast<uint32_t>(p) / 32] * NKV + h) * HT != tile0) {
                        continue;
                    }
                    const uint32_t r = static_cast<uint32_t>(p) % 32;
                    for (uint32_t k = 0; k < HT; ++k) {
                        const uint32_t s = shards + (jj * HT + k) * TB + mla::row_off(h);
#if defined(DF2_KV_BFP8)
                        df2bfp8::patch_row(s, s_tile + k * TB, r);
#else
                        const uint32_t d = s_tile + k * TB + mla::row_off(r);
                        mla::copy_row32(s, d);
                        mla::copy_row32(s + 512, d + 512);
#endif
                    }
                    done |= 1u << jj;
                }
                invalidate_l1_cache();  // the stores land before the NoC writes the tiles back
                for (uint32_t k = 0; k < HT; ++k) {
                    noc_async_write(s_tile + k * TB, dram(cache, TBC, tile0 + k), TBC);
                }
                noc_async_write_barrier();
            }
        }
    }
}
