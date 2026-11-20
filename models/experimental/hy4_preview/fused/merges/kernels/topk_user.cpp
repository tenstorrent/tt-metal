// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Cross-chip top-k merge, hub form (fused/merges/topk_merge.py TopkMergeHub): the kernel of a user core. It holds
// the user's chip-local survivor and makes the user's count vector of every round; the chip's hub core
// (topk_hub.cpp) puts the 8 users' vectors on the fabric as ONE payload per round and decides for all of them.
//
//   per round r   vector[i] = entries with key >= base + (i << shift), i = 0 .. n (topk_merge.hpp count_vector),
//                 vector[n + 1] = the bucket this core expects the cut in (its own entry of rank topk / chips);
//                 the vector is written into this user's part of the collective sender's staging buffer and
//                 counted on the hub. While the round is on the fabric the next round's vector of the expected
//                 bucket is counted ahead; the hub's record names the bucket the round found.
//   at the end    the hub writes the user's result (cut word, cut key | flags << 16, above, before) into words
//                 0..3 of this core's state tile.
// State tile words: 16 the bucket of the last round decided, 17 its flags (hub -> user, 16 bytes), 32 the count
// of hub records (consumed by subtraction); 8..15 timing stamps.
//
// Compile-time args: rounds, bits of round 0..3, topk, members, cb_survivor, cb_vectors (two vectors).
// Runtime args: user, survivor buffer address, survivor page, state address, hub NoC x | y << 16, sender NoC
// x | y << 16, this user's staging address on the sender core, vector bytes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "topk_merge.hpp"

namespace {

constexpr uint32_t rounds = get_compile_time_arg_val(0);
constexpr uint32_t bits[4] = {
    get_compile_time_arg_val(1), get_compile_time_arg_val(2), get_compile_time_arg_val(3), get_compile_time_arg_val(4)};
constexpr uint32_t topk = get_compile_time_arg_val(5);
constexpr uint32_t members = get_compile_time_arg_val(6);
constexpr uint32_t cb_survivor = get_compile_time_arg_val(7);
constexpr uint32_t cb_vectors = get_compile_time_arg_val(8);
constexpr uint32_t word_tile_bytes = 4096;
static_assert(rounds >= 1 && rounds <= 4 && bits[0] + bits[1] + bits[2] + bits[3] == 16);

constexpr uint32_t ST_RECORD = 16, ST_GO = 32, ST_STAMPS = 8;

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
    const uint32_t survivor_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t survivor_page = get_arg_val<uint32_t>(arg++);
    const uint32_t state = get_arg_val<uint32_t>(arg++);
    const uint32_t hub_xy = get_arg_val<uint32_t>(arg++);
    const uint32_t sender_xy = get_arg_val<uint32_t>(arg++);
    const uint32_t staging = get_arg_val<uint32_t>(arg++);
    const uint32_t vector_bytes = get_arg_val<uint32_t>(arg++);
    (void)user;

    volatile tt_l1_ptr uint32_t* st = hy4_collective::word(state);
    st[ST_STAMPS + 0] = hy4_collective::now();
    const uint64_t hub_counter = get_noc_addr(hub_xy & 0xFFFF, hub_xy >> 16, state + 4 * ST_GO);
    const uint64_t own_counter = get_noc_addr(state + 4 * ST_GO);
    const uint64_t staging_noc = get_noc_addr(sender_xy & 0xFFFF, sender_xy >> 16, staging);

    WAYPOINT("TURD");
    const uint32_t survivor = get_write_ptr(cb_survivor);
    for (uint32_t tile = 0; tile < 2; ++tile) {
        noc_async_read(
            interleaved_page(survivor_addr, survivor_page + tile, word_tile_bytes),
            survivor + tile * word_tile_bytes,
            word_tile_bytes);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(survivor);
    st[ST_STAMPS + 1] = hy4_collective::now();

    const uint32_t seen = hy4_topk::count_ge(words, hy4_topk::KEY_REAL_MIN, 0, hy4_topk::LIST);
    const uint32_t share = topk / members < seen ? topk / members : (seen ? seen - 1 : 0);
    const uint32_t share_key = seen ? hy4_topk::key_at(words, share) : 0;

    // Two vectors: the one of the round in flight (`sent`) and the one counted ahead (`ahead`).
    volatile tt_l1_ptr uint16_t* sent = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_vectors));
    volatile tt_l1_ptr uint16_t* ahead =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_vectors) + vector_bytes);
    uint32_t base = 0, shift = 16, guess = 0, recounts = 0;
    for (uint32_t r = 0; r < rounds; ++r) {
        const uint32_t n = 1u << bits[r];
        const uint32_t above_shift = shift;  // the shift of round r - 1
        shift -= bits[r];
        if (r == 0) {
            hy4_topk::count_vector(words, sent, 0, shift, n, 0, seen);
        } else {
            WAYPOINT("TUGW");
            uint32_t got = 0;
            if (hy4_collective::wait_min(state + 4 * ST_GO, 1, got)) {
                noc_semaphore_inc(own_counter, 0u - 1u);
                noc_async_atomic_barrier();
            }
            invalidate_l1_cache();
            const uint32_t bucket = st[ST_RECORD];
            base += bucket << above_shift;
            if (bucket == guess) {
                volatile tt_l1_ptr uint16_t* swap = sent;
                sent = ahead;
                ahead = swap;
            } else {
                const uint32_t lo = sent[bucket + 1];
                const uint32_t hi = sent[bucket];
                hy4_topk::count_vector(words, sent, base, shift, n, lo, hi);
                ++recounts;
            }
        }
        // The bucket of this round this core expects the cut in: a hint for the hub's search.
        const uint32_t top = base + (n << shift);
        guess = (seen == 0 || share_key < base) ? 0 : (share_key >= top ? n - 1 : (share_key - base) >> shift);
        sent[n + 1] = static_cast<uint16_t>(guess);
        st[ST_STAMPS + 2 + (r < 2 ? r : 2)] = hy4_collective::now();
        WAYPOINT("TUSW");
        noc_async_write(reinterpret_cast<uint32_t>(sent), staging_noc, vector_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(hub_counter, 1);
        noc_async_atomic_barrier();
        if (r + 1 < rounds) {
            hy4_topk::count_vector(
                words,
                ahead,
                base + (guess << shift),
                shift - bits[r + 1],
                1u << bits[r + 1],
                sent[guess + 1],
                sent[guess]);
        }
    }

    // The hub's result: words 0..3 of the state tile.
    WAYPOINT("TURW");
    uint32_t got = 0;
    if (hy4_collective::wait_min(state + 4 * ST_GO, 1, got)) {
        noc_semaphore_inc(own_counter, 0u - 1u);
        noc_async_atomic_barrier();
    } else {
        st[1] = hy4_topk::FLAG_TIMEOUT << 16;
    }
    st[ST_STAMPS + 5] = hy4_collective::now();
    st[ST_STAMPS + 6] = recounts;
    WAYPOINT("TUDN");
#endif
}
