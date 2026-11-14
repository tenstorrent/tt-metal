// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// R-core writer: writes each finished bf8 tile row back into the q buffer at the pages it was read
// from (what paged_fill_cache did). In place is safe: the paired N core read these pages before it
// produced the row this core consumed, and no other core touches them.
// Split output (runtime arg 10 = 1): local slice j's rows go to their own [1, H, sl, 128] buffer instead
// (j = 0: q, j = 1: q1), at the row's page in that slice.
//
// Compile-time args: 0+ accessor q, then accessor q1
// Runtime args: 0 q_addr, 1 num_rows, 2 h0, 3 j0, 4 tl0, 5 sl_t, 6 s0_t, 7 s1_t, 8 ht, 9 n_heads, 10 split,
//               11 q1_addr

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "qprep_common.hpp"

using namespace qprep;

void kernel_main() {
    constexpr auto q_args = TensorAccessorArgs<0>();
    constexpr auto q1_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    RowWalker w{
        get_arg_val<uint32_t>(2),
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(4),
        get_arg_val<uint32_t>(5),
        get_arg_val<uint32_t>(6),
        get_arg_val<uint32_t>(7),
        get_arg_val<uint32_t>(8),
        get_arg_val<uint32_t>(9)};

    const bool split = get_arg_val<uint32_t>(10) != 0;
    const auto q = TensorAccessor(q_args, q_addr);
    const auto q1 = TensorAccessor(q1_args, get_arg_val<uint32_t>(11));
    constexpr uint32_t bytes = get_tile_size(CB_QOUT);
    Noc noc;
    CircularBuffer cb(CB_QOUT);

    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t page = split ? w.slice_page() : w.q_page();
        cb.wait_front(WT);
        uint32_t off = 0;
        for (uint32_t i = 0; i < WT; ++i) {
            if (split && w.j == 1) {
                noc.async_write(cb, q1, bytes, {.offset_bytes = off}, {.page_id = page + i});
            } else {
                noc.async_write(cb, q, bytes, {.offset_bytes = off}, {.page_id = page + i});
            }
            off += bytes;
        }
        noc.async_writes_flushed();
        cb.pop_front(WT);
        w.next();
    }
    noc.async_write_barrier();
}
