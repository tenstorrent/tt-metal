// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// gather_payload_shrink MECHANISM PROBE — reader.
//
// Publishes the (already resident, zero-copy) input shard and hand-fills a bank
// of candidate reduce "scaler" tiles so the compute probe can discover, on real
// silicon, where a REDUCE_ROW result lands as a function of WHICH position of
// the scaler tile carries the 1.0.
//
// RAW L1 FILL — deliberate helper bypass. dataflow_kernel_lib's
// prepare_reduce_scaler / prepare_reduce_mask can only emit the canonical
// layouts (row 0 of every face, optionally prefix-masked along the reduce
// axis). This probe exists precisely to test NON-canonical positions, so no
// helper can express its input. Values are written as fp32 words at explicit
// (face, row, col) offsets.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"

namespace {
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_sc = 2;

// fp32 32x32 tile: 4 faces of 16x16, face f at f*256 words, row r at r*16.
constexpr uint32_t FACE_W = 16;
constexpr uint32_t FACE_WORDS = 256;

inline void zero_tile(uint32_t addr) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        p[i] = 0;
    }
}

// one word at (face, row-in-face, col-in-face)
inline void put(uint32_t addr, uint32_t face, uint32_t r, uint32_t c, float v) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    p[face * FACE_WORDS + r * FACE_W + c] = __builtin_bit_cast(uint32_t, v);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t NUM_IN_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_SC_TILES = get_compile_time_arg_val(1);

    // 1. publish the resident input shard (zero-copy CB, nothing to read).
    cb_reserve_back(cb_in, NUM_IN_TILES);
    cb_push_back(cb_in, NUM_IN_TILES);

    // 2. the scaler bank.
    const uint32_t tb = get_tile_size(cb_sc);
    cb_reserve_back(cb_sc, NUM_SC_TILES);
    uint32_t a = get_write_ptr(cb_sc);
    for (uint32_t t = 0; t < NUM_SC_TILES; ++t) {
        zero_tile(a + t * tb);
    }
    // t0 : CANONICAL SUM scaler — row 0 of every face, all 16 cols.
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t c = 0; c < FACE_W; ++c) {
            put(a + 0 * tb, f, 0, c, 1.0f);
        }
    }
    // t1 : same but FACE-ROW 1 of every face.
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t c = 0; c < FACE_W; ++c) {
            put(a + 1 * tb, f, 1, c, 1.0f);
        }
    }
    // t2 : same but FACE-ROW 3 of every face.
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t c = 0; c < FACE_W; ++c) {
            put(a + 2 * tb, f, 3, c, 1.0f);
        }
    }
    // t3 : one-hot at (face 0, row 0, col 0) only.
    put(a + 3 * tb, 0, 0, 0, 1.0f);
    // t4 : one-hot at (face 0, row 0, col 5) AND (face 2, row 0, col 5)
    //      -> the canonical layout restricted to reduce-axis position 5.
    put(a + 4 * tb, 0, 0, 5, 1.0f);
    put(a + 4 * tb, 2, 0, 5, 1.0f);
    // t5 : one-hot at (face 1, row 0, col 2) AND (face 3, row 0, col 2)
    //      -> reduce-axis position 18 (second face-column).
    put(a + 5 * tb, 1, 0, 2, 1.0f);
    put(a + 5 * tb, 3, 0, 2, 1.0f);
    // t6 : canonical row-0 but only face-row 0 (faces 0,1) — is the lower half of
    //      dest untouched?
    for (uint32_t c = 0; c < FACE_W; ++c) {
        put(a + 6 * tb, 0, 0, c, 1.0f);
        put(a + 6 * tb, 1, 0, c, 1.0f);
    }
    // t7 : COLUMN-0 layout (matmul-style): col 0 of faces 0 and 1, all rows.
    for (uint32_t r = 0; r < FACE_W; ++r) {
        put(a + 7 * tb, 0, r, 0, 1.0f);
        put(a + 7 * tb, 1, r, 0, 1.0f);
    }
    noc_async_write_barrier();  // no-op fence; keeps the pattern explicit
    cb_push_back(cb_sc, NUM_SC_TILES);
}
