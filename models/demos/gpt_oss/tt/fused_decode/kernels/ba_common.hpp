// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Block attention (dflash/block_attention.py), the work split shared by the reader and writer: the history of the
// block (positions 0 .. pos[M - 1], N 32-position chunks) is cut into A contiguous slices (A = min(S, ceil(N /
// min_chunks)): every slice core gets at least min_chunks chunks), slice s = chunks [s N / A, (s + 1) N / A). Slices
// merge their softmax partials up a binary tree: in round r, slice s with s % 2^(r + 1) == 2^r sends to s - 2^r.

#pragma once

#include <stdint.h>

struct BaSlice {
    uint32_t n_chunks;     // chunks of the block's history (all slices; from c_first)
    uint32_t active;       // slices with work (A)
    uint32_t c_lo, c_hi;   // this slice's chunks [c_lo, c_hi)
    uint32_t c_block;      // the first chunk holding a block row (pos[0] / 32)
    uint32_t recv_rounds;  // rounds in which this slice receives a child's partial (0 .. recv_rounds - 1)
    int32_t send_round;    // the round in which it sends to its parent (-1: the root)
};

// c_first: the history's first chunk (a window's start; 0 otherwise): the slices cut chunks [c_first, p_last / 32]
inline BaSlice ba_slice(
    uint32_t p_first, uint32_t p_last, uint32_t s, uint32_t S, uint32_t min_chunks, uint32_t c_first = 0) {
    BaSlice b;
    b.n_chunks = p_last / 32 + 1 - c_first;
    uint32_t a = (b.n_chunks + min_chunks - 1) / min_chunks;
    b.active = a < S ? a : S;
    if (b.active == 0) {
        b.active = 1;
    }
    const uint32_t q = s;
    b.c_lo = s < b.active ? c_first + q * b.n_chunks / b.active : 0;
    b.c_hi = s < b.active ? c_first + (q + 1) * b.n_chunks / b.active : 0;
    b.c_block = p_first / 32;
    b.recv_rounds = 0;
    b.send_round = -1;
    if (s < b.active) {
        // receives in rounds r while s % 2^(r + 1) == 0 and the child s + 2^r exists; sends in the first round r with
        // bit r of s set
        for (uint32_t r = 0; r < 8; ++r) {
            const uint32_t bit = 1u << r;
            if (s & bit) {
                b.send_round = static_cast<int32_t>(r);
                break;
            }
            if (s + bit < b.active) {
                b.recv_rounds = r + 1;
            } else if (bit >= b.active) {
                break;
            }
        }
    }
    return b;
}

// The rounds in which slice s receives (ba_slice's recv_rounds) among `active` slices
inline uint32_t ba_recv_rounds(uint32_t s, uint32_t active) {
    uint32_t n = 0;
    for (uint32_t r = 0; r < 8; ++r) {
        const uint32_t bit = 1u << r;
        if (s & bit) {
            break;
        }
        if (s + bit < active) {
            n = r + 1;
        } else if (bit >= active) {
            break;
        }
    }
    return n;
}

// This core's index in the op's core list (common runtime args from word `base`: one byte per grid core, row-major
// over a grid BA_GX wide, 0xFF = not in the op): every core's arguments are common, none per core (dispatch writes
// per-core runtime args core by core).
inline uint32_t ba_core_index(uint32_t base) {
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * BA_GX + get_absolute_logical_x();
    return (get_common_arg_val<uint32_t>(base + c / 4) >> (8 * (c % 4))) & 0xFF;
}
