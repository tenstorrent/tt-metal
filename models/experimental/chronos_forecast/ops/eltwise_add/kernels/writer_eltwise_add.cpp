// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Interleaved tile writer, adapted from binary_ng's kernels_ng/dataflow/writer_interleaved_no_bcast.cpp.
// A batch of tiles is written from the CB front, then flushed and popped together; one write barrier
// at the end. Pages start_tile_id, +stride, ... are written, matching the reader.
//
// Compile-time args: batch, stride, TensorAccessorArgs(out)
// Runtime args: out_addr, num_tiles, start_tile_id

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t stride = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    constexpr uint32_t cb_id_dst = tt::CBIndex::c_2;

    Noc noc;
    CircularBuffer cb_dst(cb_id_dst);
    const uint32_t dst_tile_bytes = get_tile_size(cb_id_dst);
    const auto dst = TensorAccessor(dst_args, dst_addr);

    uint32_t tile_id = start_tile_id;
    for (uint32_t done = 0; done < num_tiles; done += batch) {
        const uint32_t n = (num_tiles - done) < batch ? (num_tiles - done) : batch;
        cb_dst.wait_front(n);
        for (uint32_t i = 0; i < n; ++i, tile_id += stride) {
            noc.async_write(cb_dst, dst, dst_tile_bytes, {.offset_bytes = i * dst_tile_bytes}, {.page_id = tile_id});
        }
        noc.async_writes_flushed();
        cb_dst.pop_front(n);
    }
    noc.async_write_barrier();
}
