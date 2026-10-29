// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident attention, paged KV (the attention worker reader and writer; named compile-time args kv_paged,
// kv_tpb, kv_max_n, kv_copy_pages, kv_pt_row_bytes).
//
// Contiguous mode (kv_paged 0): every user has its own KV cache per layer copy (attn_common.hpp).
// Paged mode: the K (and the V) caches of all users and layer copies are one pool of pages. A page is one
// position tile of one layer copy: Dt bf8 tiles, kKvRow bytes, contiguous in one bank. Page p lives in bank
// p % banks, at byte (p / banks) * kKvRow of the pool's per-bank shard. The pool is counted in blocks of
// kv_tpb position tiles (a vLLM block): tile j of physical block P of copy c is page
// c * kv_copy_pages + P * kv_tpb + j (kv_copy_pages = blocks * kv_tpb). Copy-major, so the pages of one layer
// spread over every bank.
//
// Page table (a DRAM tensor of the launch, written by the host apart from the token state): row-major int32,
// one row of kv_pt_row_bytes per row of the batch; entry b of row u is the physical block of user u's logical
// block b (positions b * 32 * kv_tpb ...). Verify: both rows are user 0 and read row 0.
//
// The workers partition the logical position tiles exactly as in contiguous mode (chunk_of), so both modes run
// the same arithmetic in the same order; only the DRAM addresses differ (a tile's bank is no longer the bank its
// worker was assigned for). At kernel start every worker (and the tail core) looks up the page of each of its
// tiles of every user once (cb_pages: batch x kv_max_n page numbers relative to copy 0, then a window of the page
// table); all of the step's layers use them. The tail core's writer reads its tail pages there too, after the
// reader's cb_ntiles push.
#pragma once

#include <stdint.h>

namespace resident_attn {

constexpr bool kv_paged = get_named_compile_time_arg_val("kv_paged") != 0;
constexpr uint32_t kv_tpb = get_named_compile_time_arg_val("kv_tpb");                    // position tiles per block
constexpr uint32_t kv_max_n = get_named_compile_time_arg_val("kv_max_n");                // page list entries per user
constexpr uint32_t kv_copy_pages = get_named_compile_time_arg_val("kv_copy_pages");      // pages per layer copy
constexpr uint32_t kv_pt_row_bytes = get_named_compile_time_arg_val("kv_pt_row_bytes");  // page table row
static_assert(!kv_paged || kv_pt_row_bytes % 64 == 0, "page table rows are whole 64 B DRAM reads");

constexpr uint32_t cb_pages = 30;     // worker / tail core: page lists, then the page-table window
constexpr uint32_t kPtWindow = 2048;  // bytes of a page-table row read at once (512 blocks)

// the first tile of chunk ch's k-th row (contiguous: shard row j + k J of bank `bank` = tile bank + banks * row)
FORCE_INLINE uint32_t chunk_tile(const Chunk& ch, uint32_t k) { return ch.bank + banks * (ch.j + k * ch.J); }

FORCE_INLINE uint32_t pages_base() {
    auto& iface = get_local_cb_interface(cb_pages);
    return iface.fifo_limit - iface.fifo_size;
}

// user u's page list: the page (relative to copy 0) of the k-th tile of this core's chunk
FORCE_INLINE volatile tt_l1_ptr uint32_t* page_list(uint32_t u) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pages_base()) + u * kv_max_n;
}

// NoC address of page `page` of the pool at `pool`
FORCE_INLINE uint64_t page_noc_addr(uint32_t pool, uint32_t page) {
    return get_noc_addr_from_bank_id<true>(page % banks, pool + (page / banks) * kKvRow);
}

// fills user u's page list from page-table row `row` for the n tiles of chunk ch (ascending, so the table is
// read in increasing windows)
FORCE_INLINE void translate_chunk(uint32_t pt_addr, uint32_t row, uint32_t u, const Chunk& ch, uint32_t n) {
    const InterleavedAddrGen<true> pt{.bank_base_address = pt_addr, .page_size = kv_pt_row_bytes};
    const uint32_t win = (pages_base() + 4 * batch * kv_max_n + 63) & ~63u;
    volatile tt_l1_ptr uint32_t* entries = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(win);
    volatile tt_l1_ptr uint32_t* out = page_list(u);
    uint32_t lo = 0, hi = 0;  // the window holds blocks [lo, hi)
    for (uint32_t k = 0; k < n; k++) {
        const uint32_t t = chunk_tile(ch, k);
        const uint32_t b = t / kv_tpb;
        if (b < lo || b >= hi) {
            lo = b & ~15u;  // 64 B aligned
            const uint32_t left = kv_pt_row_bytes - 4 * lo;
            const uint32_t bytes = left < kPtWindow ? left : kPtWindow;
            noc_async_read(pt.get_noc_addr(row, 4 * lo), win, bytes);
            noc_async_read_barrier();
            hi = lo + bytes / 4;
        }
        out[k] = entries[b - lo] * kv_tpb + t % kv_tpb;
    }
}

}  // namespace resident_attn
