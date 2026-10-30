// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// The geometry and the work-order arithmetic the six kernels of sdpa_decode_splitfree share (planner.py mirrors
// it). Every kernel of a phase gets the same compile-time prefix (indices below); the TensorAccessor args follow
// from CT_ACCESSORS on, per kernel. Nothing here reads cur_pos: the kernels pass the row's value in.
#pragma once

#include <cstdint>

namespace sf {

constexpr uint32_t CT_ACCESSORS = 24;  // first TensorAccessorArgs index

constexpr uint32_t B = get_compile_time_arg_val(0);         // padded batch rows (<= 32)
constexpr uint32_t G = get_compile_time_arg_val(1);         // local KV groups this call attends
constexpr uint32_t HPG = get_compile_time_arg_val(2);       // Q heads (tile rows) per group
constexpr uint32_t DHT = get_compile_time_arg_val(3);       // head dim in tiles
constexpr uint32_t SK = get_compile_time_arg_val(4);        // chunk in tiles
constexpr uint32_t NSUB = get_compile_time_arg_val(5);      // chunks per unit
constexpr uint32_t MAXU = get_compile_time_arg_val(6);      // unit indices the page table reaches
constexpr uint32_t NSLOTS = get_compile_time_arg_val(7);    // scratch slots per (row, group)
constexpr uint32_t M = get_compile_time_arg_val(8);         // unit modulus of the pair space
constexpr uint32_t NCORES = get_compile_time_arg_val(9);    // cores of the program
constexpr uint32_t WINDOW = get_compile_time_arg_val(10);   // sliding window (0: attend everything)
constexpr uint32_t CAP_T = get_compile_time_arg_val(11);    // bounded cache: wrap the virtual tile row mod this
constexpr uint32_t BLOCK_T = get_compile_time_arg_val(12);  // page block in tiles
constexpr uint32_t NKV = get_compile_time_arg_val(13);      // KV heads of the cache (the per-block stride)
constexpr uint32_t POS_STICK = get_compile_time_arg_val(14);
constexpr uint32_t PT_STICK = get_compile_time_arg_val(15);
constexpr uint32_t READS = get_compile_time_arg_val(16);  // K/V tile reads in flight before a barrier
constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(17);
constexpr uint32_t QK_IN1_SUB = get_compile_time_arg_val(18);
constexpr uint32_t QK_SUB_W = get_compile_time_arg_val(19);
constexpr uint32_t OUT_IN1_SUB = get_compile_time_arg_val(20);
constexpr uint32_t OUT_SUB_W = get_compile_time_arg_val(21);
constexpr uint32_t PAIRS = get_compile_time_arg_val(22);  // M * B * G
constexpr uint32_t CT_RESERVED = get_compile_time_arg_val(23);

constexpr uint32_t CHUNK = SK * 32;
constexpr uint32_t UNIT = CHUNK * NSUB;
constexpr uint32_t BG = B * G;
constexpr uint32_t TILES_PER_SLOT = 2 + DHT;
constexpr uint32_t NONE_POS = 0xFFFFFFFFu;  // cur_pos -1: a padding row

static_assert(B >= 1 && B <= 32, "padded rows fill one Q tile");
static_assert(G >= 1 && HPG >= 1 && G * HPG <= 32, "the local Q heads fill one Q tile");
static_assert(NSUB >= 1 && SK >= 1 && DHT >= 1 && NSLOTS >= 1 && M >= 1 && NCORES >= 1, "geometry");
static_assert(PAIRS == M * B * G, "the pair space is unit-major over (row, group)");
static_assert(CAP_T == 0 || CAP_T % BLOCK_T == 0, "a bounded cache wraps on block boundaries");

// Phase-1 (partials) circular buffers
constexpr uint32_t CB_Q = 0;
constexpr uint32_t CB_K = 1;
constexpr uint32_t CB_V = 2;
constexpr uint32_t CB_MASK = 3;      // the causal mask of the chunk holding cur_pos
constexpr uint32_t CB_WMASK = 4;     // the window mask of the first live chunk
constexpr uint32_t CB_SCALE = 5;     // the reduce scaler
constexpr uint32_t CB_ZERO = 6;      // a zero tile (matmul_blocks' unused mask path wants an id)
constexpr uint32_t CB_POS_C = 7;     // cur_pos for the compute
constexpr uint32_t CB_POS_D = 8;     // cur_pos for the dataflow kernel that reads it itself
constexpr uint32_t CB_PT = 9;        // one page-table row
constexpr uint32_t CB_OUT_O = 16;    // the unit's O (fp32)
constexpr uint32_t CB_OUT_M = 17;    // the unit's m (fp32)
constexpr uint32_t CB_OUT_L = 18;    // the unit's l (fp32)
constexpr uint32_t CB_QK = 24;       // scores (bf16)
constexpr uint32_t CB_OUT_IM = 25;   // PV of one chunk (fp32)
constexpr uint32_t CB_OUT_ACC = 26;  // the running O (fp32)
constexpr uint32_t CB_CUR_MAX = 27;
constexpr uint32_t CB_PREV_MAX = 28;
constexpr uint32_t CB_CUR_SUM = 29;
constexpr uint32_t CB_PREV_SUM = 30;
constexpr uint32_t CB_EMD = 31;  // exp(prev_max - cur_max)

// Phase-2 (combine) circular buffers
constexpr uint32_t CB_IN_M = 0;
constexpr uint32_t CB_IN_L = 1;
constexpr uint32_t CB_IN_O = 2;
constexpr uint32_t CB_OUT_FINAL = 16;  // bf16 output tiles
constexpr uint32_t CB_ACC = 20;        // the running O (fp32)
constexpr uint32_t CB_ACC_2 = 21;      // the unit's O being merged (fp32)
constexpr uint32_t CB_PREV_SUM_2 = 22;
constexpr uint32_t CB_EMD_2 = 23;
// CB_CUR_MAX .. CB_EMD as in phase 1

struct Row {
    uint32_t first_chunk;   // the first chunk the row attends (the window start's chunk, else 0)
    uint32_t tail_chunk;    // the chunk holding cur_pos
    uint32_t window_start;  // the first attended position (0 without a window)
    bool live;
};

inline Row row_of(uint32_t cur_pos) {
    Row r{0, 0, 0, false};
    if (cur_pos == NONE_POS) {
        return r;
    }
    r.live = true;
    r.tail_chunk = cur_pos / CHUNK;
    if constexpr (WINDOW > 0) {
        r.window_start = (cur_pos + 1 > WINDOW) ? (cur_pos + 1 - WINDOW) : 0;
        r.first_chunk = r.window_start / CHUNK;
    }
    return r;
}

inline uint32_t unit_lo(const Row& r) { return r.first_chunk / NSUB; }
inline uint32_t unit_hi(const Row& r) { return r.tail_chunk / NSUB; }
inline uint32_t chunk_lo(const Row& r, uint32_t u) {
    const uint32_t c = u * NSUB;
    return c > r.first_chunk ? c : r.first_chunk;
}
inline uint32_t chunk_hi(const Row& r, uint32_t u) {
    const uint32_t c = (u + 1) * NSUB - 1;
    return c < r.tail_chunk ? c : r.tail_chunk;
}
inline uint32_t slot_of(uint32_t bg, uint32_t u) { return bg * NSLOTS + (u % NSLOTS); }
inline bool chunk_has_window_mask(const Row& r, uint32_t j) {
    return WINDOW > 0 && r.window_start > 0 && j == r.first_chunk;
}
inline bool chunk_has_tail_mask(const Row& r, uint32_t j) { return j == r.tail_chunk; }

// Walks this core's units in the one order the three kernels of phase 1 share: ascending pair p (= u' * BG + bg,
// p == core mod NCORES), then the units u = u' + k * M of the row's live range, ascending. f(b, g, bg, u, row).
template <typename F>
inline void for_each_unit(uint32_t core, const uint32_t* pos, F&& f) {
    for (uint32_t p = core; p < PAIRS; p += NCORES) {
        const uint32_t up = p / BG;
        const uint32_t bg = p - up * BG;
        const uint32_t b = bg / G;
        const Row r = row_of(pos[b]);
        if (!r.live) {
            continue;
        }
        const uint32_t lo = unit_lo(r);
        const uint32_t hi = unit_hi(r);
        uint32_t u = up;
        if (u < lo) {
            u += ((lo - u + M - 1) / M) * M;
        }
        for (; u <= hi; u += M) {
            f(b, bg - b * G, bg, u, r);
        }
    }
}

}  // namespace sf
