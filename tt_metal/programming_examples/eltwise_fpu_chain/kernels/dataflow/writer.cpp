// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the AXPY example. Pulls finished tiles out of cb_out and writes
// them back to DRAM at the corresponding page index.

#include <cstdint>

#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);

    Noc noc;
    CircularBuffer cb_out_buf(cb_out);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_out_buf.wait_front(1);
        noc.async_write(cb_out_buf, out, tile_size_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        cb_out_buf.pop_front(1);
    }
}
