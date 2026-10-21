// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode sampling head, step 1 (tt/device_sampler.py): per-core top-K logits of each decode user over a range of
// tile columns of this chip's vocab shard (the GreedyScan read pattern, with a top-K list instead of a maximum).
//
// Input: logits [1, 1, 32, V] bf16 or fp32 TILE, DRAM interleaved (one page per 32x32 tile column). Users sit in
// rows 0..users-1: their values are the first users*16 elements of face 0 (columns 0-15) and face 1 (columns 16-31).
// Output: this core's block of the partials tensor (uint32, raw pages of 4096 bytes): 4 user slots of K (key, local
// index) pairs, unsorted, at page core / 4, byte offset (core % 4) * 4 * K * 8. The key orders like the value
// (sign-adjusted bits). Among equal keys the lower indices are kept. Unused user slots hold zeros.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

FORCE_INLINE uint32_t sort_key(uint16_t bf16) {
    // bf16 widened to fp32 bits, so the keys compare across dtypes like GreedyScan's fp32 keys
    const uint32_t b = static_cast<uint32_t>(bf16) << 16;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
FORCE_INLINE uint32_t sort_key(uint32_t fp32) { return (fp32 & 0x80000000u) ? ~fp32 : (fp32 | 0x80000000u); }

// The K best (key, index) pairs seen, as a binary heap whose root is the worst kept pair (the smallest key, and among
// equal keys the highest index): a better pair replaces the root and sifts down.
template <uint32_t K>
struct TopK {
    uint32_t keys[K];
    uint32_t ids[K];
    uint32_t n = 0;

    static FORCE_INLINE bool worse(uint32_t ka, uint32_t ia, uint32_t kb, uint32_t ib) {
        return ka < kb || (ka == kb && ia > ib);
    }

    FORCE_INLINE uint32_t min_key() const { return keys[0]; }

    // Elements arrive in increasing index order, so an equal key never displaces a kept one.
    FORCE_INLINE void push(uint32_t key, uint32_t idx) {
        if (n < K) {
            uint32_t j = n++;
            while (j > 0) {
                const uint32_t parent = (j - 1) / 2;
                if (!worse(key, idx, keys[parent], ids[parent])) {
                    break;
                }
                keys[j] = keys[parent];
                ids[j] = ids[parent];
                j = parent;
            }
            keys[j] = key;
            ids[j] = idx;
            return;
        }
        uint32_t j = 0;
        while (true) {
            const uint32_t l = 2 * j + 1;
            if (l >= K) {
                break;
            }
            uint32_t c = l;
            if (l + 1 < K && worse(keys[l + 1], ids[l + 1], keys[l], ids[l])) {
                c = l + 1;
            }
            if (!worse(keys[c], ids[c], key, idx)) {
                break;
            }
            keys[j] = keys[c];
            ids[j] = ids[c];
            j = c;
        }
        keys[j] = key;
        ids[j] = idx;
    }
};

template <typename T, uint32_t K>
FORCE_INLINE void scan_row(uint32_t base, uint32_t col0, TopK<K>& top) {
    const T* v = reinterpret_cast<const T*>(base);  // read after the barrier
    for (uint32_t c = 0; c < 16; ++c) {
        const uint32_t k = sort_key(static_cast<T>(v[c]));
        if (top.n == K && k <= top.min_key()) {
            continue;
        }
        top.push(k, col0 + c);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t partials_addr = get_arg_val<uint32_t>(1);
    const uint32_t tile_begin = get_arg_val<uint32_t>(2);
    const uint32_t tile_end = get_arg_val<uint32_t>(3);
    const uint32_t core = get_arg_val<uint32_t>(4);

    constexpr uint32_t users = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t max_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(3);  // 2: bf16, 4: fp32
    constexpr uint32_t K = get_compile_time_arg_val(4);
    static_assert(users >= 1 && users <= 4, "users in rows 0-3");
    static_assert(elem_bytes == 2 || elem_bytes == 4, "bf16 or fp32 logits");
    constexpr uint32_t tile_bytes = 1024 * elem_bytes;
    constexpr uint32_t face_bytes = 256 * elem_bytes;
    constexpr uint32_t row_bytes = users * 16 * elem_bytes;  // `users` face rows of 16 values
    constexpr uint32_t slot_bytes = K * 8;
    constexpr uint32_t block_bytes = 4 * slot_bytes;
    static_assert(4096 % block_bytes == 0, "a page holds whole core blocks");
    constexpr uint32_t blocks_per_page = 4096 / block_bytes;

    cb_reserve_back(cb_scratch, 1);
    const uint32_t scratch = get_write_ptr(cb_scratch);  // per tile: face 0 rows | face 1 rows, then the out block
    const uint32_t out_l1 = scratch + max_tiles * 2 * row_bytes;

    const InterleavedAddrGen<true> logits_gen = {.bank_base_address = logits_addr, .page_size = tile_bytes};
    const uint32_t n = tile_end - tile_begin;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t tile = get_noc_addr(tile_begin + i, logits_gen);
        noc_async_read(tile, scratch + i * 2 * row_bytes, row_bytes);
        noc_async_read(tile + face_bytes, scratch + i * 2 * row_bytes + row_bytes, row_bytes);
    }
    noc_async_read_barrier();

    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    for (uint32_t u = 0; u < 4; ++u) {
        TopK<K> top;
        if (u < users) {
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t base = scratch + i * 2 * row_bytes + u * 16 * elem_bytes;
                const uint32_t col0 = (tile_begin + i) * 32;
                if constexpr (elem_bytes == 2) {
                    scan_row<uint16_t, K>(base, col0, top);
                    scan_row<uint16_t, K>(base + row_bytes, col0 + 16, top);
                } else {
                    scan_row<uint32_t, K>(base, col0, top);
                    scan_row<uint32_t, K>(base + row_bytes, col0 + 16, top);
                }
            }
        }
        for (uint32_t j = 0; j < K; ++j) {
            const bool real = j < top.n;
            out[u * 2 * K + 2 * j] = real ? top.keys[j] : 0;
            out[u * 2 * K + 2 * j + 1] = real ? top.ids[j] : 0;
        }
    }

    const InterleavedAddrGen<false> part_gen = {.bank_base_address = partials_addr, .page_size = 4096};
    const uint32_t offset = (core % blocks_per_page) * block_bytes;
    noc_async_write(out_l1, get_noc_addr(core / blocks_per_page, part_gen, offset), block_bytes);
    noc_async_write_barrier();
}
