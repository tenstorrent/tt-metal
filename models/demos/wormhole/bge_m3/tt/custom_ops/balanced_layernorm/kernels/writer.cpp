// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// LayerNorm writer: writes each row's output tiles, blk at a time.
//
// Compile-time args: 0 Wt, 1 blk, 2+ out accessor
// Runtime args: 0 out_addr, 1 row_start, 2 num_rows

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_OUT = 16;

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    const auto out = TensorAccessor(out_args, out_addr);
    constexpr uint32_t bytes = get_tile_size(CB_OUT);
    Noc noc;
    CircularBuffer cb(CB_OUT);

    for (uint32_t r = 0; r < num_rows; ++r) {
        const uint32_t base = (row_start + r) * Wt;
        for (uint32_t t = 0; t < Wt; t += blk) {
            cb.wait_front(blk);
            uint32_t off = 0;
            for (uint32_t i = 0; i < blk; ++i) {
                noc.async_write(cb, out, bytes, {.offset_bytes = off}, {.page_id = base + t + i});
                off += bytes;
            }
            // The data has left L1 once flushed; the final barrier waits for the acks.
            noc.async_writes_flushed();
            cb.pop_front(blk);
        }
    }
    noc.async_write_barrier();
}
