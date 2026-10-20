// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Offset stores shared by the matrix-generation kernels: generate.cpp on the data-movement RISCs and
// generate_trisc.cpp on the compute RISCs, which share the offset slots of each chunk.
#pragma once
#include <cstdint>

// A run's description, written by the data-movement RISC that owns the chunk (uint32 words): [0] word of the run's
// first row in the staging page [1] x index i0 of that row [2] rows [3] window offset of the first row itself, then
// per slot j: [4 + 3 j] window offset at x index 0 (K), [5 + 3 j], [6 + 3 j]: rows [a, e) of the run that have the
// slot's neighbour (the others take their own row's offset)
constexpr uint32_t RUN_WORDS = 4 + 3 * 27;
constexpr uint32_t MAX_RUNS = 96;  // runs described per chunk; the owning RISC writes all slots of any more itself
// the work mailbox (CB_W, uint32 words): message (h, s) at 16 (2 h + s): [0] sequence number [1] runs [2] L1 address
// of the staging set's offset pages [3] L1 address of the run list; done (t, h, s) at 64 + 16 (4 t + 2 h + s): the
// sequence number TRISC t finished; end (h) at 256 + 16 h: the launch id, once RISC h has no more messages; run
// list (h, s) at 512 + (2 h + s) MAX_RUNS RUN_WORDS. Sequence numbers are (launch id << 16) | count, so words left
// by an earlier launch never look like a message.
constexpr uint32_t MB_DONE = 64, MB_END = 256, MB_RUNS = 512;
constexpr uint32_t MB_WORDS = MB_RUNS + 4 * MAX_RUNS * RUN_WORDS;

// o[0 .. n) = v, v + 4, v + 8, ...: a pointer walk, 8 stores per step with immediate offsets. Written as o[t] = v + 4 t
// the compiler rebuilt each address from the value (5 instructions per store, about 5.7 cycles on a data-movement RISC
// here); this takes about 1.4 instructions per store (a plain L1 store loop measured 4.0 cycles per store, unrolled
// by 8 2.3)
inline void ramp(uint32_t* o, int32_t n, uint32_t v) {
    uint32_t* p = o;
    for (uint32_t* pe = o + (n & ~7); p < pe; p += 8, v += 32) {
        p[0] = v;
        p[1] = v + 4;
        p[2] = v + 8;
        p[3] = v + 12;
        p[4] = v + 16;
        p[5] = v + 20;
        p[6] = v + 24;
        p[7] = v + 28;
    }
    for (uint32_t* pe = o + n; p < pe; ++p, v += 4) {
        *p = v;
    }
}

// one slot's offsets for a run: rows [a, e) the neighbour's (K + 4 x index), the others the row's own offset
inline void slot_offsets(uint32_t* o, uint32_t K, int32_t a, int32_t e, int32_t len, int32_t i0, uint32_t self0) {
    ramp(o + a, e - a, K + 4 * (uint32_t)(i0 + a));
    ramp(o, a, self0);
    ramp(o + e, len - e, self0 + 4 * (uint32_t)e);
}
