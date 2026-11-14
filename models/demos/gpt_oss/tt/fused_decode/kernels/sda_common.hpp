// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Split decode attention (fused_decode/split_attn.py), shared by its reader, writer and compute: one query row (the
// one-user decode) over the paged K / V cache, kv head g's 8 query heads on S slice cores. The attended positions
// [lo, p] (lo = p + 1 - window with a sliding window, else 0) are N = (p - c0) / 32 + 1 32-position chunks from
// c0 = lo & ~31; A = min(S, ceil(N / min_chunks)) slices are active: with A > 1 the last one holds only the newest
// chunk (it is read after QKV writes the position: that slice starts last) and slice s < A - 1 chunks
// [s (N - 1) / (A - 1), (s + 1) (N - 1) / (A - 1)).
// Each active slice runs an online softmax over its chunks (state m, O, L: FP32, rows 0..7 = the 8 heads), all-gathers
// its row max m with the kv head's other active slices (M = max over them), scales its state by exp(m - M) and writes
// it into one slot of the leader (slice 0), which sums the slots (a 0/1 fold matmul) and writes O / L.

#pragma once

#include <stdint.h>

namespace sda {

// circular buffers (the same on every core of the op)
constexpr uint32_t CB_Q = 0;       // BF16 [2]: the 8 heads' Q rows in rows 0..7 (rows 8..31 zero)
constexpr uint32_t CB_K = 1;       // K chunks, 2 tiles each, in groups of G
constexpr uint32_t CB_V = 2;       // V chunks
constexpr uint32_t CB_AUX = 3;     // BF16 [7]: scaler, masks first / last / single, sink-column mask, sink, zero
constexpr uint32_t CB_SCR = 4;     // pos (64 B) + page-table entries
constexpr uint32_t CB_S = 6;       // FP32 [G + 1]: scores of a group (+ the sink tile)
constexpr uint32_t CB_P = 7;       // BF16 [G + 1]: exp(S - m)
constexpr uint32_t CB_M = 8;       // FP32 [2]: running max
constexpr uint32_t CB_O = 9;       // FP32 [6]: running state [O_0, O_1, L] (L = rowsum P in every column)
constexpr uint32_t CB_AB = 10;     // FP32 [2]: correction factor exp(m - m')
constexpr uint32_t CB_ONES = 11;   // K / V format [1]: all ones (L = P . ones)
constexpr uint32_t CB_MOUT = 12;   // FP32 [1]: the final row max (to the writer's all-gather)
constexpr uint32_t CB_MX = 13;     // FP32 [1]: M (column 0 of rows 0..7), from the reader
constexpr uint32_t CB_W = 14;      // FP32 [1]: exp(m - M)
constexpr uint32_t CB_PART = 15;   // FP32 [3]: the scaled state [O_0, O_1, L] (to the writer)
constexpr uint32_t CB_POS = 16;    // the position, reader -> writer
constexpr uint32_t CB_MG = 17;     // the all-gathered row maxima: slot t = slice t's m[0..7] (32 bytes), never pushed
constexpr uint32_t CB_OPACK = 18;  // FP32 [3 K]: leader: pack tile k = O_0, O_1, L of slices 4 k + i in rows 8 i ..
constexpr uint32_t CB_FOLD = 19;   // FP32 [1]: leader: F[r, 8 i + r] = 1 (r < 8, i < 4)
constexpr uint32_t CB_ACC = 20;    // FP32 [3]: leader: sum O_0, O_1, 1 / sum L
constexpr uint32_t CB_OUT = 21;    // BF16 [2]: leader: out rows 0..7

constexpr uint32_t AUX_SCALER = 0, AUX_FIRST = 1, AUX_LAST = 2, AUX_SINGLE = 3, AUX_SINKCOL = 4, AUX_SINK = 5,
                   AUX_ZERO = 6, AUX_TILES = 7;

constexpr uint32_t G = 8;  // chunks per group (one running-softmax step)

struct Split {
    uint32_t lo;      // first attended position
    uint32_t c0;      // lo & ~31 (chunk 0's first position)
    uint32_t n;       // chunks of [c0, p]
    uint32_t active;  // active slices (A)
    uint32_t j0, j1;  // this slice's chunks [j0, j1) (j1 == j0: idle)
};

inline Split split(int32_t p, uint32_t window, uint32_t S, uint32_t min_chunks, uint32_t s) {
    Split r;
    const int32_t lo = window ? p + 1 - static_cast<int32_t>(window) : 0;
    r.lo = lo > 0 ? static_cast<uint32_t>(lo) : 0;
    r.c0 = r.lo & ~31u;
    r.n = (static_cast<uint32_t>(p) - r.c0) / 32 + 1;
    uint32_t a = (r.n + min_chunks - 1) / min_chunks;
    r.active = a < S ? a : S;
    if (s >= r.active) {
        r.j0 = r.j1 = 0;
    } else if (r.active == 1) {
        r.j0 = 0;
        r.j1 = r.n;
    } else if (s + 1 == r.active) {
        r.j0 = r.n - 1;
        r.j1 = r.n;
    } else {
        r.j0 = s * (r.n - 1) / (r.active - 1);
        r.j1 = (s + 1) * (r.n - 1) / (r.active - 1);
    }
    return r;
}

// mailbox word for compute: bits 0..12 chunks of this slice, 13 sink (slice 0), 14 leader, 15..20 active slices,
// 21 chunk 0 of the slice is cut at the start (lo % 32), 22 the slice's last chunk is cut at the end (p % 32 != 31)
inline uint32_t info_word(const Split& sp, int32_t p, uint32_t s) {
    const uint32_t nc = sp.j1 - sp.j0;
    if (nc == 0) {
        return 0;
    }
    const bool first_cut = sp.j0 == 0 && (sp.lo & 31u);
    const bool last_cut = sp.j1 == sp.n && ((static_cast<uint32_t>(p) & 31u) != 31u);
    return nc | (s == 0 ? (1u << 13) | (1u << 14) : 0u) | (sp.active << 15) | (first_cut ? 1u << 21 : 0u) |
           (last_cut ? 1u << 22 : 0u);
}

// byte offset of rows 8 i .. 8 i + 7, columns 16 f .. 16 f + 15 of a 32x32 tile: BF16 (256 bytes) / FP32 (512)
constexpr uint32_t rows8(uint32_t i, uint32_t f) { return ((i >> 1) * 2 + f) * 512 + (i & 1) * 256; }
constexpr uint32_t rows8_32(uint32_t i, uint32_t f) { return ((i >> 1) * 2 + f) * 1024 + (i & 1) * 512; }

}  // namespace sda
