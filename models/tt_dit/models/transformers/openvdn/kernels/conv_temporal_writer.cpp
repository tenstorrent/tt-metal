// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer for conv_temporal_compute.cpp: unit u's 4 column tiles to pages 4 u .. 4 u + 3 of the `[F, H, S, 128]` bf16
// features (the unit order is the tensor's tile-row order).
//
// Compile-time args: the tensor accessor args of the output. Runtime args: output address, first unit, units.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_out = 16;
constexpr uint32_t HEAD_TILES = 4;
}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t first = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);

    constexpr auto out_args = TensorAccessorArgs<0>();

    Noc noc;
    DataflowBuffer out_buf(cb_out);
    const uint32_t tile_bytes = out_buf.get_entry_size();
    const auto out = TensorAccessor(out_args, out_addr);

    for (uint32_t page = first * HEAD_TILES; page < (first + count) * HEAD_TILES; ++page) {
        out_buf.wait_front(1);
        noc.async_write(out_buf, out, tile_bytes, {}, {.page_id = page});
        noc.async_writes_flushed();
        out_buf.pop_front(1);
    }
    noc.async_write_barrier();
}
