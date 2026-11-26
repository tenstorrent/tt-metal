// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Weight stream of the fused MoE expert step: blocks of SBK tiles into a CB ring of NBUF blocks, NBUF - 1 blocks
// in flight (one NOC transaction id per block, ids 1..NBUF on the calling RISC's NOC).

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

template <uint32_t sbk, uint32_t nbuf>
struct Stream {
    uint32_t cb, base, end, wr, tile_bytes;
    uint32_t in_flight = 0, trid_issue = 1, trid_wait = 1;

    FORCE_INLINE void open(uint32_t cb_id) {
        cb = cb_id;
        tile_bytes = get_tile_size(cb);
        base = get_write_ptr(cb);
        end = base + nbuf * sbk * tile_bytes;
        wr = base;
    }
    template <typename Acc>
    FORCE_INLINE void block(const Acc& acc, const uint32_t* pages) {
        // room for this block plus the ones still in flight
        cb_reserve_back(cb, sbk * (in_flight + 1));
        noc_async_read_set_trid(trid_issue);
        for (uint32_t i = 0; i < sbk; ++i) {
            noc_async_read(acc.get_noc_addr(pages[i]), wr, tile_bytes);
            wr += tile_bytes;
        }
        advance();
    }
    // one contiguous run of sbk tiles (stick layout: a column of one expert sits in one DRAM bank, in order)
    FORCE_INLINE void block_contig(uint64_t src) {
        cb_reserve_back(cb, sbk * (in_flight + 1));
        noc_async_read_set_trid(trid_issue);
        noc_async_read(src, wr, sbk * tile_bytes);
        wr += sbk * tile_bytes;
        advance();
    }
    FORCE_INLINE void advance() {
        if (wr >= end) {
            wr = base;
        }
        trid_issue = trid_issue == nbuf ? 1 : trid_issue + 1;
        if (++in_flight == nbuf - 1) {
            retire();
        }
    }
    FORCE_INLINE void retire() {
        noc_async_read_barrier_with_trid(trid_wait);
        cb_push_back(cb, sbk);
        trid_wait = trid_wait == nbuf ? 1 : trid_wait + 1;
        --in_flight;
    }
    // push one block without reading it (its tiles are stale; the consumer's result for it is dropped)
    FORCE_INLINE void skip() {
        drain();
        cb_reserve_back(cb, sbk);
        cb_push_back(cb, sbk);
        wr += sbk * tile_bytes;
        if (wr >= end) {
            wr = base;
        }
    }
    FORCE_INLINE void drain() {
        while (in_flight) {
            retire();
        }
    }
};
