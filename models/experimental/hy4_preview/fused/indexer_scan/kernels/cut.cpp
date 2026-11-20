// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The chip-local half of the cross-chip top-k merge: given the global cut of a user, the list of this chip's own
// winners in the form the sparse row reader takes (fused/sparse_mla: a uint32 table, row 0 = the users' counts,
// row 1 + u = user u's chip-local row ids).
//
// One core per user (data movement only). The user's survivor is the scan's result in DRAM: 2048 fused words
// [bf16 score | u16 stamp]; the entry of rank q (0 = best) is word ((q & 127) << 4) | (q >> 7) of its two tiles.
// The cut is one word per user:
//     bit 31 = 1: bits 15:0 = keep, the number of this chip's entries in the global top-k
//     bit 31 = 0: bits 15:0 = t (a BF16 score), bits 30:16 = ties: keep every entry with score > t and the first
//                 `ties` entries (in rank order) with score == t; ties = 0x7fff keeps all of them, which makes the
//                 count the answer to "how many of your scores are >= t"
// keep never exceeds the entries the query can see. The winners are the entries of rank < keep; their stamps are
// decoded to stripe rows and written in rank order. The counts travel to user 0's core through one program
// semaphore per user (count + 1, so that 0 means "not there yet"), and that core writes row 0.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "scan_common.hpp"

namespace {

using namespace hy4_scan;

constexpr uint32_t num_users = get_compile_time_arg_val(0);
constexpr uint32_t log2_chunks_per_bank = get_compile_time_arg_val(1);
constexpr uint32_t cb_survivor = 0;  // 2 x 4096 bytes
constexpr uint32_t cb_row = 1;       // 2048 x 4 bytes
constexpr uint32_t cb_cut = 2;       // 64 bytes
constexpr uint32_t row_bytes = topk * 4;
constexpr uint32_t neg_key = 0xFFFF - bf16_neg;  // order key of NEG

FORCE_INLINE uint64_t interleaved_page(uint32_t base, uint32_t page, uint32_t page_bytes) {
    return get_noc_addr_from_bank_id<true>(page % NUM_DRAM_BANKS, base + (page / NUM_DRAM_BANKS) * page_bytes);
}

// BF16 bits -> an unsigned key in the order of the values.
FORCE_INLINE uint32_t order_key(uint32_t bits) { return (bits & 0x8000) ? 0xFFFF - bits : bits | 0x8000; }

FORCE_INLINE uint32_t rank_word(uint32_t rank) { return ((rank & 127) << 4) | (rank >> 7); }

// Entries (in rank order: keys do not increase) whose key is above `key`, or not below it.
FORCE_INLINE uint32_t count_above(volatile tt_l1_ptr uint32_t* words, uint32_t key, bool or_equal) {
    uint32_t lo = 0, hi = topk;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        const uint32_t k = order_key(words[rank_word(mid)] >> 16);
        if (k > key || (or_equal && k == key)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

}  // namespace

void kernel_main() {
    const uint32_t user = get_arg_val<uint32_t>(0);
    const uint32_t survivor_addr = get_arg_val<uint32_t>(1);
    const uint32_t survivor_page = get_arg_val<uint32_t>(2);
    const uint32_t cut_addr = get_arg_val<uint32_t>(3);
    const uint32_t table_addr = get_arg_val<uint32_t>(4);
    const uint32_t collector_x = get_arg_val<uint32_t>(5);  // user 0's core
    const uint32_t collector_y = get_arg_val<uint32_t>(6);

    volatile tt_l1_ptr uint32_t* cut = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_cut));
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_survivor));
    volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_row));

    noc_async_read(get_noc_addr_from_bank_id<true>(0, cut_addr), get_write_ptr(cb_cut), 64);
    for (uint32_t tile = 0; tile < survivor_tiles; ++tile) {
        noc_async_read(
            interleaved_page(survivor_addr, survivor_page + tile, word_tile_bytes),
            get_write_ptr(cb_survivor) + tile * word_tile_bytes,
            word_tile_bytes);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();

    const uint32_t word = cut[user];
    const uint32_t visible = count_above(words, neg_key + 0x100, false);  // scores above NEG (and its -inf padding)
    uint32_t keep;
    if (word >> 31) {
        keep = word & 0xFFFF;
    } else {
        const uint32_t key = order_key(word & 0xFFFF);
        const uint32_t ties = (word >> 16) & 0x7FFF;
        const uint32_t above = count_above(words, key, false);
        const uint32_t tied = count_above(words, key, true) - above;
        keep = above + (ties < tied ? ties : tied);
    }
    if (keep > visible) {
        keep = visible;
    }

    for (uint32_t rank = 0; rank < keep; ++rank) {
        const uint32_t stamp = words[rank_word(rank)] & 0xFFFF;
        const uint32_t chunk = stamp >> 11;
        const uint32_t bank = chunk >> log2_chunks_per_bank;
        const uint32_t local = chunk & ((1u << log2_chunks_per_bank) - 1);
        const uint32_t bank_block = ((((local << 1) | ((stamp >> 5) & 1)) << 5) | ((stamp >> 6) & 31));
        row[rank] = (((bank_block * NUM_DRAM_BANKS) + bank) << 5) | (stamp & 31);
    }
    if (keep) {
        noc_async_write(
            get_write_ptr(cb_row), interleaved_page(table_addr, 1 + user, row_bytes), (keep * 4 + 63) & ~63u);
    }
    noc_semaphore_inc(get_noc_addr(collector_x, collector_y, get_semaphore(user)), keep + 1);

    if (user == 0) {
        // Row 0: the counts. Only its first 64 bytes are written.
        for (uint32_t u = 0; u < 16; ++u) {
            uint32_t count = 0;
            if (u < num_users) {
                volatile tt_l1_ptr uint32_t* box = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(u));
                noc_semaphore_wait_min(box, 1);
                count = *box - 1;
            }
            cut[u] = count;
        }
        noc_async_write(get_write_ptr(cb_cut), interleaved_page(table_addr, 0, row_bytes), 64);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
