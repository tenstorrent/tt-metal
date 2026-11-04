// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Ternary decode linear: stream every second output-tile record of this core into `cb` (NCRISC the even ones into
// c_1, BRISC the odd ones into c_6: two issuing RISCs read the 4-KB interleaved pages at 95% of the 512 GB/s peak,
// one at 82%, bonsai/mb/dram_bw.py). One 4-KB DRAM page per read (16 CB pages of 256 B), up to NB reads in flight,
// each with its own transaction id. The CB holds exactly two records and every record starts a new half, so a read
// never wraps around the CB's end.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

template <uint32_t ppr, uint32_t NB, typename Accessor>
FORCE_INLINE void stream_records(uint32_t cb, const Accessor& w, uint32_t first_nt, uint32_t num_nt, uint32_t parity) {
    constexpr uint32_t PAGE = 4096;
    constexpr uint32_t CBP = PAGE / 256;
    const uint32_t cb_base = get_write_ptr(cb);
    const uint32_t cb_end = cb_base + 2 * ppr * PAGE;
    uint32_t l1 = cb_base;
    uint32_t head = 0, count = 0, trid = 0;
    for (uint32_t i = parity; i < num_nt; i += 2) {
        const uint32_t page0 = (first_nt + i) * ppr;
        for (uint32_t j = 0; j < ppr; ++j) {
            if (count == NB) {
                noc_async_read_barrier_with_trid(head + 1);
                cb_push_back(cb, CBP);
                head = (head + 1) % NB;
                --count;
            }
            cb_reserve_back(cb, (count + 1) * CBP);
            noc_async_read_set_trid(trid + 1);
#ifndef FAKE_STREAM
            noc_async_read(w.get_noc_addr(page0 + j), l1, PAGE);
#endif
            trid = (trid + 1) % NB;
            ++count;
            l1 += PAGE;
            if (l1 >= cb_end) {
                l1 = cb_base;
            }
        }
    }
    while (count > 0) {
        noc_async_read_barrier_with_trid(head + 1);
        cb_push_back(cb, CBP);
        head = (head + 1) % NB;
        --count;
    }
    noc_async_read_set_trid(0);
}
