// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// v2 (QWEN38_GR_GATE_V2=1) of mcast_writer.cpp: the semaphore is raised by runtime arg 13 (a distinct bit per sender,
// so the receiver sees which sender's tile has landed) instead of by 1.  A producer core's writer: multicasts its
// `num_tiles` packed tiles into the consumers' CB (the same L1 address on every core that declares that CB) at a tile
// offset, then raises each consumer's semaphore by one; optionally also writes the same tiles to a TILE tensor, and one
// extra CB stream to another tensor.  Consumers are the NoC rectangle [x0..x1] x [y0..y1] (NoC-0 virtual coordinates
// measured by noc_probe.cpp, top-left first).  A writer kernel runs on NOC_1 (WriterDataMovementConfig = BRISC +
// preferred_noc_for_dram_write = NOC_1 on every arch), and NOC_1 routes a multicast from the opposite corner, so the
// start/end corners are swapped for it exactly as the DRAM-sharded matmul factory and
// tests/tt_metal/.../one_to_all/kernels/sender_multicast.cpp do. Compile-time args: 0 src cb, 1 dst cb, 2 num_tiles, 3
// write the tiles to DRAM (0/1), 4 extra stream cb (0xFF none),
//   5 semaphore id, 6.. two TensorAccessorArgs sets (tiles tensor, extra tensor; unused slots repeat).
// Runtime args: 0 dst tile offset, 1-4 NoC x0 y0 x1 y1, 5-7 tiles tensor (addr, first, stride), 8-12 extra stream
//   (addr, count, first, stride, batch).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

#include "../../kernels/zones.h"

template <
    uint32_t SRC_CB,
    uint32_t DST_CB,
    uint32_t NUM_TILES,
    uint32_t WRITE_TILES,
    uint32_t EXTRA_CB,
    uint32_t SEM_ID,
    typename TilesArgs,
    typename ExtraArgs>
FORCE_INLINE void mcast_phase_v2(const TilesArgs& tiles_args, const ExtraArgs& extra_args, uint32_t rt) {
    const uint32_t sem_bits = get_arg_val<uint32_t>(rt + 13);  // v2: this sender's own bit of the shared semaphore
    const uint32_t dst_tile_offset = get_arg_val<uint32_t>(rt + 0);
    const uint32_t x0 = get_arg_val<uint32_t>(rt + 1);
    const uint32_t y0 = get_arg_val<uint32_t>(rt + 2);
    const uint32_t x1 = get_arg_val<uint32_t>(rt + 3);
    const uint32_t y1 = get_arg_val<uint32_t>(rt + 4);
    const uint32_t num_dests = (x1 - x0 + 1) * (y1 - y0 + 1);
    const uint32_t tile_bytes = get_tile_size(SRC_CB);

    Noc noc;
    DataflowBuffer src(SRC_CB);
    DataflowBuffer dst(DST_CB);
    Semaphore<> sem(SEM_ID);

    src.wait_front(NUM_TILES);
    const uint32_t src_addr = src.get_read_ptr();
    const uint32_t dst_addr =
        dst.get_write_ptr() +
        dst_tile_offset * tile_bytes;  // the CB sits at one address on every core that declares it
    MulticastEndpoint mcast;
    constexpr bool from_far_corner = noc_index != 0;  // NOC_1: start at the bottom-right corner
    noc.async_write_multicast(
        CoreLocalMem<uint32_t>(src_addr),
        mcast,
        NUM_TILES * tile_bytes,
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
            sem.up(noc, x, y, sem_bits);
        }
    }
    if constexpr (WRITE_TILES) {
        const auto tiles = TensorAccessor(tiles_args, get_arg_val<uint32_t>(rt + 5));
        const uint32_t first = get_arg_val<uint32_t>(rt + 6);
        const uint32_t stride = get_arg_val<uint32_t>(rt + 7);
        for (uint32_t t = 0; t < NUM_TILES; ++t) {
            noc.async_write(src, tiles, tile_bytes, {.offset_bytes = t * tile_bytes}, {.page_id = first + t * stride});
        }
        noc.async_write_barrier();
    }
    src.pop_front(NUM_TILES);
    if constexpr (EXTRA_CB != 0xFF) {
        const auto extra = TensorAccessor(extra_args, get_arg_val<uint32_t>(rt + 8));
        const uint32_t count = get_arg_val<uint32_t>(rt + 9);
        const uint32_t first = get_arg_val<uint32_t>(rt + 10);
        const uint32_t stride = get_arg_val<uint32_t>(rt + 11);
        const uint32_t batch = get_arg_val<uint32_t>(rt + 12);
        DataflowBuffer extra_cb(EXTRA_CB);
        const uint32_t extra_bytes = get_tile_size(EXTRA_CB);
        for (uint32_t done = 0; done < count; done += batch) {
            extra_cb.wait_front(batch);
            for (uint32_t t = 0; t < batch; ++t) {
                noc.async_write(
                    extra_cb,
                    extra,
                    extra_bytes,
                    {.offset_bytes = t * extra_bytes},
                    {.page_id = first + (done + t) * stride});
            }
            noc.async_write_barrier();
            extra_cb.pop_front(batch);
        }
    }
    noc.async_atomic_barrier();
}

constexpr uint32_t SRC_CB = get_compile_time_arg_val(0);
constexpr uint32_t DST_CB = get_compile_time_arg_val(1);
constexpr uint32_t NUM_TILES = get_compile_time_arg_val(2);
constexpr uint32_t WRITE_TILES = get_compile_time_arg_val(3);
constexpr uint32_t EXTRA_CB = get_compile_time_arg_val(4);
constexpr uint32_t SEM_ID = get_compile_time_arg_val(5);
constexpr uint32_t ACCESSOR_BASE = 6;

void kernel_main() {
    constexpr auto tiles_args = TensorAccessorArgs<ACCESSOR_BASE>();
    constexpr auto extra_args = TensorAccessorArgs<tiles_args.next_compile_time_args_offset()>();
    {
        FUSED_ZONE("fz_gr_mcw_main");
        mcast_phase_v2<SRC_CB, DST_CB, NUM_TILES, WRITE_TILES, EXTRA_CB, SEM_ID>(tiles_args, extra_args, 0);
    }
}
