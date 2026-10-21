// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode sampling head, step 2 (tt/device_sampler.py): one user's top-K logits of this chip's vocab shard, from the
// scan cores' top-K lists. One core per user.
//
// Input: the scan partials (sample_scan.cpp): core c's K (key, local index) pairs of user u at page c / 4 (4096-byte
// pages), byte offset (c % 4) * 4 * K * 8 + u * K * 8.
// Output: user u's K pairs sorted by key descending, lower index first among equal keys, at byte offset u * K * 8 of
// the chip candidates page (uint32, 4096 bytes); and the user's largest logit (the first pair's value, fp32) at row u,
// column 0 of the chip-max tile ([1, 1, 32, 1] fp32 TILE: face 0, byte offset u * 64), the rest of that 16-byte row
// chunk zero.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

FORCE_INLINE bool better(uint32_t key, uint32_t idx, uint32_t other_key, uint32_t other_idx) {
    return key > other_key || (key == other_key && idx < other_idx);
}

// fp32 bits of a sort key (inverse of sample_scan.cpp's sort_key).
FORCE_INLINE uint32_t key_bits(uint32_t key) { return (key & 0x80000000u) ? (key & 0x7FFFFFFFu) : ~key; }

}  // namespace

void kernel_main() {
    const uint32_t partials_addr = get_arg_val<uint32_t>(0);
    const uint32_t cands_addr = get_arg_val<uint32_t>(1);
    const uint32_t max_addr = get_arg_val<uint32_t>(2);
    const uint32_t u = get_arg_val<uint32_t>(3);

    constexpr uint32_t num_cores = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t slot_bytes = K * 8;
    constexpr uint32_t block_bytes = 4 * slot_bytes;
    constexpr uint32_t blocks_per_page = 4096 / block_bytes;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t scratch = get_write_ptr(cb_scratch);  // num_cores slots, then the sorted list, then the max chunk
    const uint32_t list_l1 = scratch + num_cores * slot_bytes;
    const uint32_t max_l1 = list_l1 + slot_bytes;

    const InterleavedAddrGen<false> part_gen = {.bank_base_address = partials_addr, .page_size = 4096};
    for (uint32_t c = 0; c < num_cores; ++c) {
        const uint32_t offset = (c % blocks_per_page) * block_bytes + u * slot_bytes;
        noc_async_read(get_noc_addr(c / blocks_per_page, part_gen, offset), scratch + c * slot_bytes, slot_bytes);
    }
    noc_async_read_barrier();

    const volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    uint32_t keys[K];
    uint32_t ids[K];
    uint32_t n = 0;
    for (uint32_t e = 0; e < num_cores * K; ++e) {
        const uint32_t key = in[2 * e];
        const uint32_t idx = in[2 * e + 1];
        if (n == K && !better(key, idx, keys[K - 1], ids[K - 1])) {
            continue;
        }
        uint32_t j = n < K ? n++ : K - 1;
        while (j > 0 && better(key, idx, keys[j - 1], ids[j - 1])) {
            keys[j] = keys[j - 1];
            ids[j] = ids[j - 1];
            --j;
        }
        keys[j] = key;
        ids[j] = idx;
    }

    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(list_l1);
    for (uint32_t j = 0; j < K; ++j) {
        list[2 * j] = keys[j];
        list[2 * j + 1] = ids[j];
    }
    volatile tt_l1_ptr uint32_t* mx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(max_l1);
    mx[0] = key_bits(keys[0]);
    mx[1] = 0;
    mx[2] = 0;
    mx[3] = 0;

    const InterleavedAddrGen<false> cands_gen = {.bank_base_address = cands_addr, .page_size = 4096};
    noc_async_write(list_l1, get_noc_addr(0, cands_gen, u * slot_bytes), slot_bytes);
    const InterleavedAddrGen<false> max_gen = {.bank_base_address = max_addr, .page_size = 4096};
    noc_async_write(max_l1, get_noc_addr(0, max_gen, u * 64), 16);
    noc_async_write_barrier();
}
