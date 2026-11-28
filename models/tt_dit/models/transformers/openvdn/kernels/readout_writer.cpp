// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer for readout_compute.cpp: unit u = (f h + head) RT + rt goes to tile row `row_tile0 + f stride_tiles + rt`
// of the `[1, 1, R, h 128]` fp32 rows, column tiles 4 head .. 4 head + 3 (the token-major layout the all-to-all
// hands back to the rows' owners). Rows no unit covers are left as they are.
//
// Compile-time args: the tensor accessor args of the output.
// Runtime args: output address, first unit, units, heads, RT, row_tile0, stride_tiles.

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
    const uint32_t heads = get_arg_val<uint32_t>(3);
    const uint32_t rt_count = get_arg_val<uint32_t>(4);
    const uint32_t row_tile0 = get_arg_val<uint32_t>(5);
    const uint32_t stride_tiles = get_arg_val<uint32_t>(6);

    constexpr auto out_args = TensorAccessorArgs<0>();

    Noc noc;
    DataflowBuffer out_buf(cb_out);
    const uint32_t tile_bytes = out_buf.get_entry_size();
    const auto out = TensorAccessor(out_args, out_addr);
    const uint32_t col_tiles = heads * HEAD_TILES;

    for (uint32_t u = first; u < first + count; ++u) {
        const uint32_t group = u / rt_count;
        const uint32_t f = group / heads;
        const uint32_t head = group % heads;
        const uint32_t row_tile = row_tile0 + f * stride_tiles + u % rt_count;
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            out_buf.wait_front(1);
            noc.async_write(out_buf, out, tile_bytes, {}, {.page_id = row_tile * col_tiles + head * HEAD_TILES + c});
            noc.async_writes_flushed();
            out_buf.pop_front(1);
        }
    }
    noc.async_write_barrier();
}
