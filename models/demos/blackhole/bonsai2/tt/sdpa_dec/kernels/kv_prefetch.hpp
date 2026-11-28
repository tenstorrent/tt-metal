// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// The decode SDPA's K / V prefetch (sdpa_dec.py, stream/decode.py kv_prefetch), run by an earlier op's writer on SDPA
// core j = g NG + c: from this core's record (position, page table: the step's embedding wrote it) the c-th share of
// kv head g's position tile rows, written back into the record ([1] rows, [2] last share, [3] valid columns of the
// last tile), and those rows' K / V tiles into the core's persistent buffer (K tile t Dt + d, then V at maxr Dt).
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace kv_pf {
constexpr uint32_t Dt = 8, KB = 1088;  // head_dim 256, bf8 tiles

template <typename KA, typename VA>
inline void prefetch(
    uint32_t j,
    uint32_t rec,
    uint32_t buf,
    const KA& ks,
    const VA& vs,
    uint32_t bs,
    uint32_t NG,
    uint32_t H,
    uint32_t maxr) {
    if (j >= H * NG) {
        return;
    }
    const uint32_t g = j / NG, c = j % NG;
    volatile tt_l1_ptr uint32_t* r = (volatile tt_l1_ptr uint32_t*)rec;
    const uint32_t n = r[0] + 1, nt = (n + 31) / 32;
    const uint32_t r0 = c * nt / NG, r1 = (c + 1) * nt / NG;
    r[1] = r1 - r0;
    r[2] = r1 > r0 && r1 == nt;
    r[3] = n - (nt - 1) * 32;
    for (uint32_t t = r0; t < r1; ++t) {
        const uint32_t p0 = 32 * t;
        const uint32_t base = ((r[16 + p0 / bs] * H + g) * (bs / 32) + (p0 % bs) / 32) * Dt;
        for (uint32_t d = 0; d < Dt; ++d) {
            noc_async_read(ks.get_noc_addr(base + d), buf + ((t - r0) * Dt + d) * KB, KB);
            noc_async_read(vs.get_noc_addr(base + d), buf + ((maxr + t - r0) * Dt + d) * KB, KB);
        }
    }
}
}  // namespace kv_pf
