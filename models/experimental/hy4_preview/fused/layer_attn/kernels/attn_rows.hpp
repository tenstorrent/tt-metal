// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// A pass of 16 token rows as two sub-passes of 8 with a row mask (fused/layer_attn/PLAN.md section 19): ONE header
// for everything two kernels of program P2 must agree on. Free of kernel-only and host-only dependencies:
// tests/host_rows.cpp includes it and runs the parties against each other for all 65,536 masks.
//
// Token row r = 8 p + i is row i of sub-pass p. The row mask of a pass is one host-written word, the same on every
// core of every chip (bit r: row r is live). A row that is not live leaves no partial, is in no payload, is counted
// nowhere and is waited for nowhere.
//   - a count word that had one count per row of 8 (C_GROUP[g], C_ROW, A_LOCAL) has one BYTE per sub-pass: byte p
//     counts the live rows of sub-pass p (one(p), field(word, p));
//   - a payload of X4 (a chip's partials of one head group for the rows of one sub-pass) holds the LIVE rows only,
//     in row order: row i at place(mask, p, i) x the partial's bytes, live_count(mask, p) partials in all. A
//     sub-pass without a live row is handed to the sender with 0 bytes (nothing is sent, no arrival is counted);
//   - an arrival counter (A_COLARR, A_ROWARR) therefore counts arrivals(mask) payloads, and the n-th one to arrive
//     from a source is sub-pass arrival_subpass(mask, n): a source's sender serves its hand-overs in the order they
//     were made, and packets of one route arrive in order;
//   - an arrival mask (A_COLMASK, A_ROWMASK) has bit 8 p + source, set only on the merge cores whose row is live in
//     sub-pass p (the others would never take it off).

#pragma once

#include <cstdint>

namespace hy4r {

constexpr uint32_t SUB = 8;                       // token rows of a sub-pass
constexpr uint32_t SUBPASSES = 2;                 // 16 token rows
constexpr uint32_t TOKEN_ROWS = SUB * SUBPASSES;  // (not "ROWS": the projection kernels have a macro of that name)

constexpr uint32_t popcount(uint32_t v) {
    uint32_t n = 0;
    for (; v != 0; v &= v - 1) {
        ++n;
    }
    return n;
}

constexpr uint32_t sub_mask(uint32_t mask, uint32_t p) { return (mask >> (SUB * p)) & ((1u << SUB) - 1u); }
constexpr bool live(uint32_t mask, uint32_t p, uint32_t i) { return ((mask >> (SUB * p + i)) & 1u) != 0; }
constexpr uint32_t live_count(uint32_t mask, uint32_t p) { return popcount(sub_mask(mask, p)); }
constexpr uint32_t live_rows(uint32_t mask) { return popcount(mask & ((1u << TOKEN_ROWS) - 1u)); }

// The place of row i of sub-pass p among the live rows of the sub-pass: its partial's index in an X4 payload and
// in the row staging.
constexpr uint32_t place(uint32_t mask, uint32_t p, uint32_t i) {
    return popcount(sub_mask(mask, p) & ((1u << i) - 1u));
}

// Sub-passes with a live row: the payloads one source lands on a landing in one pass.
constexpr uint32_t arrivals(uint32_t mask) {
    uint32_t n = 0;
    for (uint32_t p = 0; p < SUBPASSES; ++p) {
        n += live_count(mask, p) != 0 ? 1u : 0u;
    }
    return n;
}

// The sub-pass of the n-th payload (n = 0 first) a source lands; SUBPASSES when there is none.
constexpr uint32_t arrival_subpass(uint32_t mask, uint32_t n) {
    for (uint32_t p = 0; p < SUBPASSES; ++p) {
        if (live_count(mask, p) != 0) {
            if (n == 0) {
                return p;
            }
            --n;
        }
    }
    return SUBPASSES;
}

// Byte p of a count word.
constexpr uint32_t one(uint32_t p) { return 1u << (8 * p); }
constexpr uint32_t field(uint32_t word, uint32_t p) { return (word >> (8 * p)) & 0xFFu; }

// Bit of (sub-pass p, source index s) in an arrival mask.
constexpr uint32_t arrival_bit(uint32_t p, uint32_t s) { return 1u << (SUB * p + s); }

// f(i, k) for the live rows i of sub-pass p in row order, k = place(mask, p, i): how a payload is filled (the
// coordinator), and which merge cores hear of an arrival (the relay).
template <typename F>
inline __attribute__((always_inline)) void for_each_live(uint32_t mask, uint32_t p, F&& f) {
    uint32_t k = 0;
    for (uint32_t i = 0; i < SUB; ++i) {
        if (live(mask, p, i)) {
            f(i, k);
            ++k;
        }
    }
}

// How many times merge core i runs its chain in a pass: once per sub-pass in which its row is live. The
// data-movement kernel runs the chain for exactly these sub-passes in this order and hands this number to the merge
// kernel, which runs as many chains.
constexpr uint32_t chains(uint32_t mask, uint32_t i) {
    uint32_t n = 0;
    for (uint32_t p = 0; p < SUBPASSES; ++p) {
        n += live(mask, p, i) ? 1u : 0u;
    }
    return n;
}

}  // namespace hy4r
