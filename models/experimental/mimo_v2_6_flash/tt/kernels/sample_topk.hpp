// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Per-core top-K of the decode sampling head's scan (sample_scan.cpp, sample_scan_writer.cpp): a user's K largest
// logits over a range of tile columns, read from the range's fp32 tiles in L1.

#pragma once

#include <cstdint>

namespace sample {

// Larger key <=> larger fp32 value (sign-adjusted bits), branch-free.
inline uint32_t sort_key(uint32_t bits) {
    return bits ^ (static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) | 0x80000000u);
}

// The K best (key, index) pairs of a stream arriving in increasing index order, as a binary heap in local memory
// (L1 loads and stores cost the data movement cores far more) whose root is the worst kept pair (the smallest key,
// and among equal keys the highest index): a better pair replaces the root and sifts down. A pair equal to the root's
// key arrives after every kept pair of that key, so it never displaces one.
template <uint32_t K>
struct TopK {
    uint32_t keys[K];
    uint32_t ids[K];
    uint32_t n = 0;

    static bool worse(uint32_t ka, uint32_t ia, uint32_t kb, uint32_t ib) { return ka < kb || (ka == kb && ia > ib); }

    void push(uint32_t key, uint32_t idx) {
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

    // One face row (16 fp32 values) of a user, columns col0 .. col0 + 15. Once the heap is full only a key above its
    // root (kept in a register) costs more than a compare.
    void scan(const uint32_t* __restrict v, uint32_t col0) {
        uint32_t c = 0;
        for (; c < 16 && n < K; ++c) {
            push(sort_key(v[c]), col0 + c);
        }
        uint32_t floor = keys[0];
        for (; c < 16; ++c) {
            const uint32_t key = sort_key(v[c]);
            if (key > floor) {
                push(key, col0 + c);
                floor = keys[0];
            }
        }
    }

    // The list, best first (zero keys past the elements seen), at out[0..2K): repeatedly the best remaining pair.
    void write(uint32_t* out) {
        for (uint32_t j = 0; j < K; ++j) {
            uint32_t best = 0;
            bool found = false;
            for (uint32_t i = 0; i < n; ++i) {
                if (keys[i] != 0 && (!found || worse(keys[best], ids[best], keys[i], ids[i]))) {
                    best = i;
                    found = true;
                }
            }
            out[2 * j] = found ? keys[best] : 0;
            out[2 * j + 1] = found ? ids[best] : 0;
            if (found) {
                keys[best] = 0;
            }
        }
    }
};

// A user's K best over n tiles at x (fp32 tiles of 1024 words, users in rows 0-3), best first, into out[0..2K).
template <uint32_t K>
inline void range_topk(const uint32_t* x, uint32_t n, uint32_t tile_begin, uint32_t u, uint32_t* out) {
    TopK<K> top;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t* tile = x + i * 1024;
        const uint32_t col0 = (tile_begin + i) * 32;
        top.scan(tile + u * 16, col0);
        top.scan(tile + 256 + u * 16, col0 + 16);
    }
    top.write(out);
}

}  // namespace sample
