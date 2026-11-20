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

// A row of the selection table holds COUNT_CAP ids (the op's "topk"; sm_reader.cpp asserts that they agree). Every
// kernel that reads the count row clamps to it, so all of them (the row reader, the K sender and receiver, the
// attention core's io kernel) hold the same counts whatever the table holds, and the copy the io kernel hands to
// the compute kernel, two counts of 16 bits per word, is exact: no count can make the reader and the compute
// kernel disagree about the chunks.
constexpr uint32_t COUNT_CAP = 2048;
constexpr uint32_t clamp_count(uint32_t n) { return n > COUNT_CAP ? COUNT_CAP : n; }

#ifdef SM_MASK
// A layer program with a row mask (SM_MASK: the L1 address, the same on every core, of a host-written word; bit r =
// token row r is live this pass): a row that is not live has nothing to attend whatever the table's count row
// holds. Applied wherever the count row is read, so every kernel of the op sees the same counts.
FORCE_INLINE uint32_t live_rows() {
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SM_MASK);
}
template <uint32_t USERS>
inline __attribute__((always_inline)) void apply_row_mask(uint32_t* counts) {
    const uint32_t live = live_rows();
    for (uint32_t u = 0; u < USERS; ++u) {
        if (((live >> u) & 1u) == 0) {
            counts[u] = 0;
        }
    }
}

#endif
// The per-user valid counts, row 0 of the selection table, through `scratch_cb` (one page, never pushed).
template <uint32_t USERS, typename Accessor>
inline __attribute__((always_inline)) void read_counts(const Accessor& sel, uint32_t scratch_cb, uint32_t* counts) {
    cb_reserve_back(scratch_cb, 1);
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(scratch_cb));
    noc_async_read(sel.get_noc_addr(0), reinterpret_cast<uint32_t>(hdr), USERS * sizeof(uint32_t));
    noc_async_read_barrier();
    for (uint32_t u = 0; u < USERS; ++u) {
        counts[u] = clamp_count(hdr[u]);
    }
#ifdef SM_MASK
    apply_row_mask<USERS>(counts);
#endif
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
