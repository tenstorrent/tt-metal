// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// G[w] = window word at byte offset OFF[w], for the words [w0, w1) of one L1 page (the indexed read HPCG requires).
// Fastest loop form measured on Blackhole: 8 independent offset loads, then 8 value loads, then 8 stores (5.85
// cycles per word; 4-way 7.26, 12-way 5.62). Byte offsets avoid a shift per word. Not inlined: inlined copies ran up
// to 2x slower depending on where the compiler placed them.
#pragma once
// Launch options F_WSPLIT, F_ZEROX, F_RESTF, F_ADDB: one runtime flags word (reader arg 10, writer arg 12, compute arg
// 3) rather than defines, so launches that differ only in options share one kernel binary; shared with the host code
#define HPCG_KERNEL_SIDE
#include "../../../shared_constants.h"

__attribute__((noinline, aligned(64))) void gather_tile(
    uint32_t xbuf, const uint32_t* o, uint32_t* g, uint32_t w0, uint32_t w1) {
    const uint8_t* xbb = (const uint8_t*)xbuf;
    uint32_t w = w0;
    for (; w + 8 <= w1; w += 8) {
        const uint32_t a0 = o[w], a1 = o[w + 1], a2 = o[w + 2], a3 = o[w + 3];
        const uint32_t a4 = o[w + 4], a5 = o[w + 5], a6 = o[w + 6], a7 = o[w + 7];
        const uint32_t v0 = *(const uint32_t*)(xbb + a0), v1 = *(const uint32_t*)(xbb + a1);
        const uint32_t v2 = *(const uint32_t*)(xbb + a2), v3 = *(const uint32_t*)(xbb + a3);
        const uint32_t v4 = *(const uint32_t*)(xbb + a4), v5 = *(const uint32_t*)(xbb + a5);
        const uint32_t v6 = *(const uint32_t*)(xbb + a6), v7 = *(const uint32_t*)(xbb + a7);
        g[w] = v0;
        g[w + 1] = v1;
        g[w + 2] = v2;
        g[w + 3] = v3;
        g[w + 4] = v4;
        g[w + 5] = v5;
        g[w + 6] = v6;
        g[w + 7] = v7;
    }
    for (; w < w1; w += 4) {  // segments are multiples of 4 words
        const uint32_t a0 = o[w], a1 = o[w + 1], a2 = o[w + 2], a3 = o[w + 3];
        const uint32_t v0 = *(const uint32_t*)(xbb + a0), v1 = *(const uint32_t*)(xbb + a1);
        const uint32_t v2 = *(const uint32_t*)(xbb + a2), v3 = *(const uint32_t*)(xbb + a3);
        g[w] = v0;
        g[w + 1] = v1;
        g[w + 2] = v2;
        g[w + 3] = v3;
    }
}

// On-chip page geometry: TPW words per L1 page (1024 = a 32x32 tile; 256 = one 16x16 face on coarse grids, so the
// unpacker / packer move one face instead of four). DRAM-side tensors keep 4 KB pages.
#ifndef TPW
#define TPW 1024
#endif
#define OFFB 4             // bytes per column offset (32-bit, HPCG's local_int_t)
#define PGB (TPW * 4)      // bytes per value / gathered / output page in L1
#define OPGB (TPW * OFFB)  // bytes per offset page in L1
// unit lookahead: matrix / offset reads run LOOK work units ahead; unit u's reads carry transaction id TRID(u)
#ifndef LOOK
#define LOOK 1
#endif
#define TRID(u) (((u) % (LOOK + 1)) + 1)
#define USLOT(u) ((u) % (LOOK + 1))
// progress slot values: 1 = window loaded, phase q done = q + 1 + POFF
#define POFF 1
// Which processor gathers stencil slot j: the NG gatherers take slots round-robin. 0 = reader (BRISC), 1 = writer
// (NCRISC), 2 = the compute core's pack thread (TRISC2), 3 = its unpack thread (TRISC0).
#ifndef NSLOT
#define NSLOT 27
#endif
constexpr uint32_t owner(uint32_t j) { return j % NG; }
// first slot of gatherer o, NSLOT if none
constexpr uint32_t first_of(uint32_t o) {
    for (uint32_t j = 0; j < NSLOT; ++j) {
        if (owner(j) == o) {
            return j;
        }
    }
    return NSLOT;
}
// next slot after j of gatherer o, NSLOT if none
constexpr uint32_t next_of(uint32_t j, uint32_t o) {
    for (uint32_t k = j + 1; k < NSLOT; ++k) {
        if (owner(k) == o) {
            return k;
        }
    }
    return NSLOT;
}
// rank of slot j among gatherer o's slots in [lo, j), and the most slots o has in any group of gsz slots
constexpr uint32_t rank_in(uint32_t lo, uint32_t j, uint32_t o) {
    uint32_t c = 0;
    for (uint32_t k = lo; k < j; ++k) {
        c += owner(k) == o;
    }
    return c;
}
constexpr uint32_t max_in_group(uint32_t o, uint32_t gsz) {
    uint32_t m = 0;
    for (uint32_t lo = 0; lo < NSLOT; lo += gsz) {
        uint32_t c = 0;
        for (uint32_t k = lo; k < lo + gsz && k < NSLOT; ++k) {
            c += owner(k) == o;
        }
        m = c > m ? c : m;
    }
    return m;
}
constexpr uint32_t count_of(uint32_t o) {
    uint32_t c = 0;
    for (uint32_t j = 0; j < NSLOT; ++j) {
        c += owner(j) == o;
    }
    return c;
}
