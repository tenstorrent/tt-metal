// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Data-movement helpers of the sparse-MLA kernels.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "sm_common.hpp"

namespace sm {

// Every semaphore wait of these kernels is bounded: on expiry the kernel goes on as if the event had happened, so
// a protocol error ends in wrong numbers (which the tests catch) and not in a core that waits for ever and takes
// the chip's command queue with it. The bounds are a fraction of a second to a few seconds of spinning: far above
// any real wait (the whole program runs for tens to hundreds of microseconds) and, also when several expire one
// after the other, below the host's operation timeout.
constexpr uint32_t WAIT_SPINS = 10u * 1000u * 1000u;  // a loop that only reads one word
constexpr uint32_t POLL_SPINS = 2u * 1000u * 1000u;   // a loop that polls several conditions

template <typename Pred>
inline __attribute__((always_inline)) bool wait_bounded(Pred&& done) {
    for (uint32_t i = 0; i < WAIT_SPINS; ++i) {
        invalidate_l1_cache();
        if (done()) {
            return true;
        }
    }
    return false;
}

inline __attribute__((always_inline)) bool wait_min_bounded(volatile tt_l1_ptr uint32_t* sem, uint32_t value) {
    return wait_bounded([&]() { return *sem >= value; });
}

// The per-user valid counts, row 0 of the selection table, through `scratch_cb` (one page, never pushed).
template <uint32_t USERS, typename Accessor>
inline __attribute__((always_inline)) void read_counts(const Accessor& sel, uint32_t scratch_cb, uint32_t* counts) {
    cb_reserve_back(scratch_cb, 1);
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(scratch_cb));
    noc_async_read(sel.get_noc_addr(0), reinterpret_cast<uint32_t>(hdr), USERS * sizeof(uint32_t));
    noc_async_read_barrier();
    for (uint32_t u = 0; u < USERS; ++u) {
        counts[u] = hdr[u];
    }
}

// One S-block's entry of the NoC coordinate table that starts at common runtime argument `tbl`.
struct Coords {
    uint32_t reader_x, reader_y, att_x, att_y;
};
inline __attribute__((always_inline)) Coords block_coords(uint32_t tbl, uint32_t b) {
    return Coords{
        get_common_arg_val<uint32_t>(tbl + COORD_WORDS * b),
        get_common_arg_val<uint32_t>(tbl + COORD_WORDS * b + 1),
        get_common_arg_val<uint32_t>(tbl + COORD_WORDS * b + 2),
        get_common_arg_val<uint32_t>(tbl + COORD_WORDS * b + 3)};
}

}  // namespace sm
