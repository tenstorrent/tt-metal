// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One multicast phase of mcast_writer.cpp as a function: a producer core multicasts its `num_tiles` packed tiles from
// `src_cb` into the consumers' `dst_cb` at a tile offset, raises each consumer's semaphore `semaphore_id` by one,
// optionally writes the same tiles to a TILE tensor and one extra CB stream to another tensor.  Runtime args from
// `first_arg`: 0 dst tile offset, 1-4 NoC x0 y0 x1 y1, 5-7 tiles tensor (addr, first, stride), 8-12 extra stream (addr,
// count, first, stride, batch).  mcast_writer.cpp runs one phase; gr_recip_last's mcast_writer3.cpp three in sequence.

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

template <
    uint32_t src_cb,
    uint32_t dst_cb,
    uint32_t num_tiles,
    uint32_t write_tiles,
    uint32_t extra_cb,
    uint32_t semaphore_id,
    typename TilesArgs,
    typename ExtraArgs>
FORCE_INLINE void mcast_phase(const TilesArgs& tiles_args, const ExtraArgs& extra_args, uint32_t first_arg) {
    const uint32_t dst_tile_offset = get_arg_val<uint32_t>(first_arg + 0);
    const uint32_t x0 = get_arg_val<uint32_t>(first_arg + 1);
    const uint32_t y0 = get_arg_val<uint32_t>(first_arg + 2);
    const uint32_t x1 = get_arg_val<uint32_t>(first_arg + 3);
    const uint32_t y1 = get_arg_val<uint32_t>(first_arg + 4);
    const uint32_t num_dests = (x1 - x0 + 1) * (y1 - y0 + 1);
    const uint32_t tile_bytes = get_tile_size(src_cb);

    const Noc noc;
    DataflowBuffer src(src_cb);
    DataflowBuffer dst(dst_cb);
    Semaphore<> landed(semaphore_id);

    src.wait_front(num_tiles);
    const uint32_t src_addr = src.get_read_ptr();
    const uint32_t dst_addr =
        dst.get_write_ptr() +
        dst_tile_offset * tile_bytes;  // the CB sits at one address on every core that declares it
    MulticastEndpoint consumers;
    constexpr bool from_far_corner = noc_index != 0;  // NOC_1: start at the bottom-right corner
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(src_addr),
        consumers,
        num_tiles * tile_bytes,
        num_dests,
        {},
        {.noc_x_start = from_far_corner ? x1 : x0,
         .noc_y_start = from_far_corner ? y1 : y0,
         .noc_x_end = from_far_corner ? x0 : x1,
         .noc_y_end = from_far_corner ? y0 : y1,
         .addr = dst_addr});
    noc.async_write_barrier();
    for (uint32_t x = x0; x <= x1; ++x) {
        for (uint32_t y = y0; y <= y1; ++y) {
            landed.up(noc, x, y, 1);
        }
    }
    if constexpr (write_tiles) {
        const auto tiles = TensorAccessor(tiles_args, get_arg_val<uint32_t>(first_arg + 5));
        const uint32_t first = get_arg_val<uint32_t>(first_arg + 6);
        const uint32_t stride = get_arg_val<uint32_t>(first_arg + 7);
        for (uint32_t t = 0; t < num_tiles; ++t) {
            noc.async_write(src, tiles, tile_bytes, {.offset_bytes = t * tile_bytes}, {.page_id = first + t * stride});
        }
        noc.async_write_barrier();
    }
    src.pop_front(num_tiles);
    if constexpr (extra_cb != 0xFF) {
        const auto extra = TensorAccessor(extra_args, get_arg_val<uint32_t>(first_arg + 8));
        const uint32_t count = get_arg_val<uint32_t>(first_arg + 9);
        const uint32_t first = get_arg_val<uint32_t>(first_arg + 10);
        const uint32_t stride = get_arg_val<uint32_t>(first_arg + 11);
        const uint32_t batch = get_arg_val<uint32_t>(first_arg + 12);
        DataflowBuffer extra_stream(extra_cb);
        const uint32_t extra_bytes = get_tile_size(extra_cb);
        for (uint32_t done = 0; done < count; done += batch) {
            extra_stream.wait_front(batch);
            for (uint32_t t = 0; t < batch; ++t) {
                noc.async_write(
                    extra_stream,
                    extra,
                    extra_bytes,
                    {.offset_bytes = t * extra_bytes},
                    {.page_id = first + (done + t) * stride});
            }
            noc.async_write_barrier();
            extra_stream.pop_front(batch);
        }
    }
    noc.async_atomic_barrier();
}
