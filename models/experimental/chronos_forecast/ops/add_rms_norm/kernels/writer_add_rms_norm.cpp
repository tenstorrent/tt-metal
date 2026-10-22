// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for [y = a + b,] n = rms_norm(y): each tile row of y (when fused), then of n, in blocks of blk tiles.
//
// Compile-time args: Wt, blk, fuse_add, TensorAccessorArgs(y), TensorAccessorArgs(n)
// Runtime args: y_addr, n_addr, num_rows, start_row

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

template <typename Accessor>
FORCE_INLINE void write_row(
    Noc& noc,
    CircularBuffer& cb,
    const Accessor& dst,
    uint32_t tile_bytes,
    uint32_t first_page,
    uint32_t Wt,
    uint32_t blk) {
    for (uint32_t j0 = 0; j0 < Wt; j0 += blk) {
        cb.wait_front(blk);
        for (uint32_t d = 0; d < blk; ++d) {
            noc.async_write(cb, dst, tile_bytes, {.offset_bytes = d * tile_bytes}, {.page_id = first_page + j0 + d});
        }
        noc.async_writes_flushed();
        cb.pop_front(blk);
    }
}

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    const uint32_t start_row = get_arg_val<uint32_t>(3);

    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t blk = get_compile_time_arg_val(1);
    constexpr bool fuse_add = get_compile_time_arg_val(2) != 0;
    constexpr auto y_args = TensorAccessorArgs<3>();
    constexpr auto n_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();

    constexpr uint32_t y_cb_id = tt::CBIndex::c_16;
    constexpr uint32_t n_cb_id = tt::CBIndex::c_17;

    Noc noc;
    CircularBuffer cb_y(y_cb_id);
    CircularBuffer cb_n(n_cb_id);
    const uint32_t y_tile_bytes = fuse_add ? get_tile_size(y_cb_id) : 0;
    const uint32_t n_tile_bytes = get_tile_size(n_cb_id);
    const auto y = TensorAccessor(y_args, y_addr);
    const auto n = TensorAccessor(n_args, n_addr);

    for (uint32_t row = start_row; row < start_row + num_rows; ++row) {
        if constexpr (fuse_add) {
            write_row(noc, cb_y, y, y_tile_bytes, row * Wt, Wt, blk);
        }
        write_row(noc, cb_n, n, n_tile_bytes, row * Wt, Wt, blk);
    }
    noc.async_write_barrier();
}
