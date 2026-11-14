// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GDN decode step inputs prefetched into L1 (gdn_step.py prefetch): head j's small DRAM reads, done by an earlier op
// on the step's cores (step_prefetch_op.cpp, or a ternary linear's writer: tdec3_writer.cpp STEP_PREFETCH) into this
// core's persistent buffer PF, so the step itself reads only L1. Under the weight stream (bonsai2/tt/stream) a small
// DRAM read waited ~26-40 us behind the senders' long sequential ones.
//
// PF layout (bytes): a/b weight blocks 2 NBK tiles | ONES | E | taps rows 0-3 (face 0 / 1, 128 B each) x NCH |
// norm-weight row 0 (64 B per face) x HT | conv-state rows 0-3 (128 B per face) x NCH | recurrent state HT x HT tiles
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace step_pf {
constexpr uint32_t TB = 2048, NBK = 5, HT = 4, NCH = 3 * HT;
constexpr uint32_t WAB = 0, ONES = WAB + 2 * NBK * TB, E = ONES + TB, TAPS = E + TB, W = TAPS + NCH * 256;
constexpr uint32_t CONV = W + HT * 128, S = CONV + NCH * 256, BYTES = S + HT * HT * TB;

// head j's conv chunk c (q, k of its key head kh, then v of head j) -> its tile in the [q | k | v] conv layout
FORCE_INLINE uint32_t chunk_tile(uint32_t c, uint32_t j, uint32_t kh, uint32_t qt) {
    return c < HT ? kh * HT + c : (c < 2 * HT ? qt + kh * HT + (c - HT) : 2 * qt + j * HT + (c - 2 * HT));
}

// Issue head j's reads into PF (the caller waits: noc_async_read_barrier). nv: value heads; group: value heads per key
// head; qt: q (= k) width in tiles.
template <typename WA, typename CA, typename VA, typename SA>
FORCE_INLINE void prefetch(
    uint32_t pf,
    uint32_t j,
    uint32_t nv,
    uint32_t group,
    uint32_t qt,
    const WA& wab,
    const CA& cs,
    const VA& conv,
    const SA& st) {
    for (uint32_t t = 0; t < 2 * NBK; ++t) {
        noc_async_read(wab.get_noc_addr(j * 2 * NBK + t), pf + WAB + t * TB, TB);
    }
    // consts: [NV * NCH taps tiles | HT norm-weight tiles | E | ONES]
    noc_async_read(cs.get_noc_addr(nv * NCH + HT + 1), pf + ONES, TB);
    noc_async_read(cs.get_noc_addr(nv * NCH + HT), pf + E, TB);
    for (uint32_t c = 0; c < NCH; ++c) {
        const uint64_t s = cs.get_noc_addr(j * NCH + c);
        noc_async_read(s, pf + TAPS + c * 256, 128);
        noc_async_read(s + 512, pf + TAPS + c * 256 + 128, 128);
    }
    for (uint32_t i = 0; i < HT; ++i) {
        const uint64_t s = cs.get_noc_addr(nv * NCH + i);
        noc_async_read(s, pf + W + i * 128, 64);
        noc_async_read(s + 512, pf + W + i * 128 + 64, 64);
    }
    const uint32_t kh = j / group;
    for (uint32_t c = 0; c < NCH; ++c) {
        const uint64_t s = conv.get_noc_addr(chunk_tile(c, j, kh, qt));
        noc_async_read(s, pf + CONV + c * 256, 128);
        noc_async_read(s + 512, pf + CONV + c * 256 + 128, 128);
    }
    for (uint32_t t = 0; t < HT * HT; ++t) {
        noc_async_read(st.get_noc_addr(j * HT * HT + t), pf + S + t * TB, TB);
    }
}
}  // namespace step_pf
