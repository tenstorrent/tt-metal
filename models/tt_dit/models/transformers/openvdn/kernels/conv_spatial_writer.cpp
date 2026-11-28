// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writer for conv_spatial_compute.cpp: unit u = f RT + rt of channel group g goes to the head layout
// `[F, H, S, 128]` fp32 tiled, head g / 4, column tile g % 4: page ((f H + g / 4) RT + rt) 4 + g % 4.
//
// Compile-time args: heads, RT, then the tensor accessor args of the output.
// Runtime args: output address, group, first unit, number of units.

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
    const uint32_t group = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);

    constexpr uint32_t heads = get_compile_time_arg_val(0);
    constexpr uint32_t RT = get_compile_time_arg_val(1);
    constexpr auto out_args = TensorAccessorArgs<2>();

    Noc noc;
    DataflowBuffer out_buf(cb_out);
    const uint32_t tile_bytes = out_buf.get_entry_size();
    const auto out = TensorAccessor(out_args, out_addr);
    const uint32_t head = group / HEAD_TILES;
    const uint32_t column = group % HEAD_TILES;

    for (uint32_t u = first; u < first + count; ++u) {
        const uint32_t f = u / RT;
        const uint32_t rt = u % RT;
        out_buf.wait_front(1);
        noc.async_write(out_buf, out, tile_bytes, {}, {.page_id = ((f * heads + head) * RT + rt) * HEAD_TILES + column});
        noc.async_writes_flushed();
        out_buf.pop_front(1);
    }
    noc.async_write_barrier();
}
