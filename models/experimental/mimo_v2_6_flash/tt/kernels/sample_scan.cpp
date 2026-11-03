// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode sampling head, step 1, reader (tt/device_sampler.py): over a range of tile columns of this chip's vocab
// shard, the tiles for the compute kernel (sample_norm_compute.cpp), which sums exp((logit - the range's largest) / T)
// per row, and the first half of the user slots' K largest logits (sample_topk.hpp: users 0-1 of 4 slots, 0-3 of 8);
// the writer (sample_scan_writer.cpp) takes the other half.
//
// Input: logits [1, 1, 32, V] fp32 TILE, DRAM interleaved; users in rows 0..users-1 (row u: face 0 elements
// u*16 .. u*16+15 for columns 0-15, face 1 the same for columns 16-31). params: uint32 row-major [1, max_batch * 8],
// row r's word 4 = 1 / T as fp32 bits.
// Output: the first half of the user slots of this core's block of the top-K partials (raw 4096-byte pages: `slots`
// user slots (4, or 8 past 4 users) of K (key, local index) pairs, best first, at page core / (4096 / block), byte
// offset (core % (4096 / block)) * block, block = slots * K * 8); slots of users past `users` zero. To the compute
// kernel: the range's tiles (cb_x), and two tiles whose row u holds, in every column, user u's largest logit of the
// range (cb_m) and 1 / T (cb_a), rows past the users zero. To the writer: the tiles' address (cb_ready).
//
// Grammar: a user whose params word 5 is set reads its row of the mask (uint32 row-major [users, mask_stride] on this
// chip: word w, bit j allows local token 32 * w + j, so a tile column's word is its tile index), and every logit of the
// range it disallows becomes kMasked in L1 before anything reads the tiles.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "sample_topk.hpp"

namespace {

// -2^100: below every real logit, so a disallowed token ranks after every allowed one and weighs nothing next to it
// (exp((kMasked - m) / T) is 0, and the fixed-point merge saturates it to 0). Finite, so a range or a chip whose logits
// are all disallowed keeps a finite largest logit (its sum is relative to it and drops out of the merge); with -inf
// its exp((x - m) / T) would be NaN.
constexpr uint32_t kMasked = 0xF1800000u;

// Every column of row r of a 32x32 fp32 tile at `tile` set to `bits`.
FORCE_INLINE void fill_row(uint32_t* tile, uint32_t r, uint32_t bits) {
    uint32_t* face = tile + (r < 16 ? 0 : 512) + (r % 16) * 16;
    for (uint32_t c = 0; c < 16; ++c) {
        face[c] = bits;
        face[256 + c] = bits;
    }
}

}  // namespace

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t partials_addr = get_arg_val<uint32_t>(1);
    const uint32_t params_addr = get_arg_val<uint32_t>(2);
    const uint32_t tile_begin = get_arg_val<uint32_t>(3);
    const uint32_t tile_end = get_arg_val<uint32_t>(4);
    const uint32_t core = get_arg_val<uint32_t>(5);
    const uint32_t mask_addr = get_arg_val<uint32_t>(6);

    constexpr uint32_t users = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t max_batch = get_compile_time_arg_val(2);
    constexpr uint32_t cb_x = get_compile_time_arg_val(3);
    constexpr uint32_t cb_m = get_compile_time_arg_val(4);
    constexpr uint32_t cb_a = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(6);
    constexpr uint32_t cb_ready = get_compile_time_arg_val(7);
    constexpr uint32_t slots = get_compile_time_arg_val(8);        // user slots of a core's block: 4, or 8
    constexpr uint32_t mask_stride = get_compile_time_arg_val(9);  // words of a user's mask row (64-byte multiple)
    static_assert(users >= 1 && users <= slots && (slots == 4 || slots == 8), "users in rows 0-7");
    constexpr uint32_t half = slots / 2;  // the reader's slots
    constexpr uint32_t tile_bytes = 4096;
    constexpr uint32_t slot_bytes = K * 8;
    constexpr uint32_t block_bytes = slots * slot_bytes;
    static_assert(4096 % block_bytes == 0, "a page holds whole core blocks");
    constexpr uint32_t blocks_per_page = 4096 / block_bytes;
    constexpr uint32_t params_bytes = max_batch * 32;

    const uint32_t n = tile_end - tile_begin;
    cb_reserve_back(cb_x, n);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    cb_reserve_back(cb_scratch, 1);
    const uint32_t out_l1 = get_write_ptr(cb_scratch);  // the reader's slots, then the params page
    const uint32_t params_l1 = out_l1 + half * slot_bytes;
    const uint32_t mask_l1 = params_l1 + params_bytes;  // the users' mask rows

    const InterleavedAddrGen<true> logits_gen = {.bank_base_address = logits_addr, .page_size = tile_bytes};
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_read(get_noc_addr(tile_begin + i, logits_gen), x_l1 + i * tile_bytes, tile_bytes);
    }
    const InterleavedAddrGen<true> params_gen = {.bank_base_address = params_addr, .page_size = params_bytes};
    noc_async_read(get_noc_addr(0, params_gen), params_l1, params_bytes);
    noc_async_read_barrier();
    const uint32_t* params = reinterpret_cast<const uint32_t*>(params_l1);

    // Grammar: the flagged users' mask rows, then their disallowed logits of the range set to kMasked.
    constexpr uint32_t mask_row_bytes = mask_stride * 4;
    const InterleavedAddrGen<true> mask_gen = {.bank_base_address = mask_addr, .page_size = mask_row_bytes};
    bool masked = false;
    for (uint32_t u = 0; u < users; ++u) {
        if (params[u * 8 + 5] != 0) {
            noc_async_read(get_noc_addr(u, mask_gen), mask_l1 + u * mask_row_bytes, mask_row_bytes);
            masked = true;
        }
    }
    if (masked) {
        noc_async_read_barrier();
        uint32_t* tiles = reinterpret_cast<uint32_t*>(x_l1);
        for (uint32_t u = 0; u < users; ++u) {
            if (params[u * 8 + 5] == 0) {
                continue;
            }
            const uint32_t* words = reinterpret_cast<const uint32_t*>(mask_l1 + u * mask_row_bytes) + tile_begin;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t w = words[i];
                if (w == 0xFFFFFFFFu) {
                    continue;
                }
                uint32_t* row0 = tiles + i * 1024 + u * 16;  // columns 0-15 (face 0); 16-31 at + 256 (face 1)
                for (uint32_t c = 0; c < 16; ++c) {
                    if (!((w >> c) & 1u)) {
                        row0[c] = kMasked;
                    }
                    if (!((w >> (16 + c)) & 1u)) {
                        row0[256 + c] = kMasked;
                    }
                }
            }
        }
    }

    // Each user's largest logit of the range: the normalizer's shift, released to the compute kernel first.
    const uint32_t* x = reinterpret_cast<const uint32_t*>(x_l1);
    uint32_t largest[slots] = {};
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t u = 0; u < users; ++u) {
            const uint32_t* __restrict row0 = x + i * 1024 + u * 16;
            uint32_t best = largest[u];
            for (uint32_t c = 0; c < 16; ++c) {
                const uint32_t k0 = sample::sort_key(row0[c]);
                const uint32_t k1 = sample::sort_key(row0[256 + c]);
                best = k0 > best ? k0 : best;
                best = k1 > best ? k1 : best;
            }
            largest[u] = best;
        }
    }
    cb_reserve_back(cb_m, 1);
    cb_reserve_back(cb_a, 1);
    uint32_t* m_tile = reinterpret_cast<uint32_t*>(get_write_ptr(cb_m));
    uint32_t* a_tile = reinterpret_cast<uint32_t*>(get_write_ptr(cb_a));
    for (uint32_t r = 0; r < 32; ++r) {
        const bool user = r < users;
        const uint32_t key = user ? largest[r] : 0;
        fill_row(m_tile, r, user ? ((key & 0x80000000u) ? (key & 0x7FFFFFFFu) : ~key) : 0);
        fill_row(a_tile, r, user ? params[r * 8 + 4] : 0);
    }
    cb_push_back(cb_m, 1);
    cb_push_back(cb_a, 1);
    cb_push_back(cb_x, n);
    cb_reserve_back(cb_ready, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ready)) = x_l1;
    cb_push_back(cb_ready, 1);

    uint32_t* out = reinterpret_cast<uint32_t*>(out_l1);
    for (uint32_t u = 0; u < half; ++u) {
        if (u < users) {
            sample::range_topk<K>(x, n, tile_begin, u, out + u * 2 * K);
        } else {
            for (uint32_t j = 0; j < 2 * K; ++j) {
                out[u * 2 * K + j] = 0;
            }
        }
    }
    const InterleavedAddrGen<false> part_gen = {.bank_base_address = partials_addr, .page_size = 4096};
    noc_async_write(
        out_l1,
        get_noc_addr(core / blocks_per_page, part_gen, (core % blocks_per_page) * block_bytes),
        half * slot_bytes);
    noc_async_write_barrier();
}
