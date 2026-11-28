// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tile copy of src `[1, H, R, W]` into rows `row_tile0 * 32 ..` of dst `[1, H, Rd, W]` (same dtype, tiled,
// interleaved): src page ((h RT) + r) WT + c -> dst page ((h RTd) + row_tile0 + r) WT + c. One dataflow kernel per
// core reads a range of src pages into its buffer and writes them out.
//
// Compile-time args: the tensor accessor args of src, dst.
// Runtime args: src, dst addresses, first page, number of pages, RT (src row tiles), RTd (dst row tiles), WT,
// row_tile0.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_tiles = 0;
constexpr uint32_t BATCH = 8;
}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    const uint32_t rt = get_arg_val<uint32_t>(4);
    const uint32_t rt_dst = get_arg_val<uint32_t>(5);
    const uint32_t wt = get_arg_val<uint32_t>(6);
    const uint32_t row_tile0 = get_arg_val<uint32_t>(7);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    Noc noc;
    DataflowBuffer buf(cb_tiles);
    const uint32_t bytes = buf.get_entry_size();
    const auto src = TensorAccessor(src_args, src_addr);
    const auto dst = TensorAccessor(dst_args, dst_addr);

    buf.reserve_back(BATCH);  // scratch, never pushed
    for (uint32_t p = first; p < first + count; p += BATCH) {
        const uint32_t n = first + count - p < BATCH ? first + count - p : BATCH;
        for (uint32_t i = 0; i < n; ++i) {
            noc.async_read(src, buf, bytes, {.page_id = p + i}, {.offset_bytes = i * bytes});
        }
        noc.async_read_barrier();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t page = p + i;
            const uint32_t c = page % wt;
            const uint32_t row = page / wt;  // h RT + r
            const uint32_t out = ((row / rt) * rt_dst + row_tile0 + row % rt) * wt + c;
            noc.async_write(buf, dst, bytes, {.offset_bytes = i * bytes}, {.page_id = out});
        }
        noc.async_write_barrier();
    }
}
