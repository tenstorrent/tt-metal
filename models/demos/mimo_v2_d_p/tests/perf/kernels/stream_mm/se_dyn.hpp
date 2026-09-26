// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Dynamic token counts for the flat expert (data movement side). The routing produces, on device, a token count and a
// region row offset per global expert (uint32 [1, N] row-major tensors in DRAM). A program serves its chip's NUM_E
// local experts whose count lies in its band [LO, HI] (a hybrid runs two programs with disjoint bands); every other
// expert is skipped outright, weights included. Every kernel derives the same active-expert list from the same
// tensors, so the roles stay in step without any host round trip.
//
// Runtime args at DYN0 (identical on every core): counts address, regions address, row bytes (4 N), LO, HI, then the
// NUM_E local experts' global ids.
#pragma once
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

#ifndef SE_MAX_E
#define SE_MAX_E 16
#endif

struct SeDyn {
    uint32_t n_act = 0;       // active experts, in local order
    uint32_t num_v = 0;       // virtual experts (sub-blocks of MT row tiles) over all active experts
    uint32_t eid[SE_MAX_E];   // local expert index of active expert a
    uint32_t cnt[SE_MAX_E];   // its tokens
    uint32_t off[SE_MAX_E];   // its region's first row in the dispatch buffer (TILE_HEIGHT-aligned)
    uint32_t subs[SE_MAX_E];  // its sub-blocks
};

// Reads the counts / regions rows into SCRATCH (two 1 KB halves of L1 this RISC owns) and fills d.
template <uint32_t num_e>
inline void se_dyn_load(SeDyn& d, uint32_t dyn0, uint32_t scratch, uint32_t rows_per_sub) {
    static_assert(num_e <= SE_MAX_E);
    const uint32_t counts_addr = get_arg_val<uint32_t>(dyn0), regions_addr = get_arg_val<uint32_t>(dyn0 + 1);
    const uint32_t row_bytes = get_arg_val<uint32_t>(dyn0 + 2);
    const uint32_t lo = get_arg_val<uint32_t>(dyn0 + 3), hi = get_arg_val<uint32_t>(dyn0 + 4);
    const uint32_t rd = (row_bytes + 63) / 64 * 64;
    const InterleavedAddrGen<true> cg = {.bank_base_address = counts_addr, .page_size = row_bytes};
    const InterleavedAddrGen<true> rg = {.bank_base_address = regions_addr, .page_size = row_bytes};
    noc_async_read(get_noc_addr(0, cg), scratch, rd);
    noc_async_read(get_noc_addr(0, rg), scratch + 1024, rd);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    volatile tt_l1_ptr uint32_t* regions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 1024);
    d.n_act = 0;
    d.num_v = 0;
    for (uint32_t e = 0; e < num_e; ++e) {
        const uint32_t g = get_arg_val<uint32_t>(dyn0 + 5 + e);
        const uint32_t c = counts[g];
        if (c == 0 || c < lo || c > hi) {
            continue;
        }
        const uint32_t a = d.n_act++;
        d.eid[a] = e;
        d.cnt[a] = c;
        d.off[a] = regions[g];
        d.subs[a] = (c + rows_per_sub - 1) / rows_per_sub;
        d.num_v += d.subs[a];
    }
}

// Hands the active experts' sub-block counts to this core's compute kernel: one page of CB META_CB holding
// [n_act, num_v, subs[0], subs[1], ...] (read there with read_tile_value).
inline void se_dyn_publish(const SeDyn& d, uint32_t meta_cb) {
    cb_reserve_back(meta_cb, 1);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(meta_cb));
    p[0] = d.n_act;
    p[1] = d.num_v;
    for (uint32_t a = 0; a < d.n_act; ++a) {
        p[2 + a] = d.subs[a];
    }
    cb_push_back(meta_cb, 1);
}
