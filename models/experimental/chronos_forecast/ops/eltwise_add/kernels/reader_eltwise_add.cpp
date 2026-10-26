// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Same-shape reader for a and b on interleaved tiles, adapted from binary_ng's
// kernels_ng/dataflow/reader_interleaved_no_bcast.cpp. Tiles are read with offsets into the reserved
// CB region and pushed together after one barrier per batch. Pages start_tile_id, +stride, ... are read;
// with stride = number of L1 banks they all live in this core's bank.
//
// Compile-time args: batch, stride, TensorAccessorArgs(a), TensorAccessorArgs(b)
// Runtime args: a_addr, b_addr, num_tiles, start_tile_id

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t src_addr_b = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t stride = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto src_b_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_src = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_src_b = tt::CBIndex::c_1;

    Noc noc;
    CircularBuffer cb_src(cb_id_src);
    CircularBuffer cb_src_b(cb_id_src_b);
    const uint32_t src_tile_bytes = get_tile_size(cb_id_src);
    const uint32_t src_tile_bytes_b = get_tile_size(cb_id_src_b);
    const auto src = TensorAccessor(src_args, src_addr);
    const auto src_b = TensorAccessor(src_b_args, src_addr_b);

    uint32_t tile_id = start_tile_id;
    for (uint32_t done = 0; done < num_tiles; done += batch) {
        const uint32_t n = (num_tiles - done) < batch ? (num_tiles - done) : batch;
        cb_src.reserve_back(n);
        cb_src_b.reserve_back(n);
        for (uint32_t i = 0; i < n; ++i, tile_id += stride) {
            noc.async_read(src, cb_src, src_tile_bytes, {.page_id = tile_id}, {.offset_bytes = i * src_tile_bytes});
            noc.async_read(
                src_b, cb_src_b, src_tile_bytes_b, {.page_id = tile_id}, {.offset_bytes = i * src_tile_bytes_b});
        }
        noc.async_read_barrier();
        cb_src.push_back(n);
        cb_src_b.push_back(n);
    }
}
