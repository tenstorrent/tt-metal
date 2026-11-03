// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "dataflow_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(2);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t packets_per_batch = get_compile_time_arg_val(5);
    constexpr uint32_t stripe_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t ring_size = get_compile_time_arg_val(7);
    constexpr bool transpose_b = get_compile_time_arg_val(8) == 1;
    constexpr uint32_t cb_packets = get_compile_time_arg_val(9);
    constexpr auto shard_args = TensorAccessorArgs<10>();
    constexpr auto pwb_args = TensorAccessorArgs<shard_args.next_compile_time_args_offset()>();
    constexpr uint32_t packet_bytes = tiles_per_packet * tile_size;

    uint32_t argidx = 0;
    const uint32_t shard_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t pwb_addr = get_arg_val<uint32_t>(argidx++);
    uint32_t stripe_sem_addrs[ring_size];
    for (uint32_t i = 0; i < ring_size; i++) {
        stripe_sem_addrs[i] = get_arg_val<uint32_t>(argidx++);
    }
    const uint32_t schedule_arg_idx = argidx;

    const auto shard = TensorAccessor(shard_args, shard_addr, tile_size);
    const auto pwb = TensorAccessor(pwb_args, pwb_addr, tile_size);
    Noc noc;
    CircularBuffer cb(cb_id);

    const uint32_t cb_base = cb.get_write_ptr();
    constexpr uint32_t cb_bytes = cb_packets * packet_bytes;
    uint32_t batched = 0;
    uint32_t batch_offset = 0;
    auto flush_batch = [&]() {
        if (batched > 0) {
            noc.async_read_barrier();
            cb.push_back(batched * tiles_per_packet);
            batched = 0;
        }
    };
    uint32_t relay_credits[ring_size] = {};

    for_each_send_packet(
        schedule_arg_idx,
        K_tiles,
        chunk_tiles,
        tiles_per_packet,
        [&](const SendSegment& seg, const SendPieceArgs&, uint32_t chunk_idx, uint32_t piece_idx) {
            if (seg.distance == 0) {
                return;
            }
            const uint32_t target = (chunk_idx * seg.num_pieces) + piece_idx + 1;
            auto* counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stripe_sem_addrs[seg.stripe]);
            invalidate_l1_cache();
            if (*counter < target) {
                flush_batch();
                noc_semaphore_wait_min(counter, target);
            }
            relay_credits[seg.stripe]++;
        },
        [&](const SendSegment& seg,
            const SendPieceArgs& piece,
            uint32_t t0,
            uint32_t count,
            uint32_t k_start,
            uint32_t chunk) {
            if (batched == 0) {
                cb.reserve_back(packets_per_batch * tiles_per_packet);
                batch_offset = cb.get_write_ptr() - cb_base;
            }
            uint32_t l1 = cb_base + ((batch_offset + (batched * packet_bytes)) % cb_bytes);
            for (uint32_t i = 0; i < count; ++i, l1 += tile_size) {
                const uint32_t t = t0 + i;
                const uint32_t row = piece.row_offset + (t / chunk);
                const uint32_t k = k_start + (t % chunk);
                if (seg.distance == 0) {
                    noc.async_read(
                        shard,
                        CoreLocalMem<uint32_t>(l1),
                        tile_size,
                        {.page_id = weight_page<transpose_b>(row, k, K_tiles, stripe_tiles)},
                        {});
                } else {
                    const uint32_t page = weight_page<transpose_b>(
                        (seg.stripe * stripe_tiles) + row, k, K_tiles, stripe_tiles * ring_size);
                    noc.async_read(pwb, CoreLocalMem<uint32_t>(l1), tile_size, {.page_id = page}, {});
                }
            }
            if (++batched == packets_per_batch) {
                flush_batch();
            }
        },
        [](const SendSegment&, const SendPieceArgs&, uint32_t, uint32_t) {});
    flush_batch();

    for (uint32_t s = 0; s < ring_size; s++) {
        if (relay_credits[s] > 0) {
            noc_semaphore_inc(get_noc_addr(stripe_sem_addrs[s]), -relay_credits[s]);
        }
    }
    noc.async_atomic_barrier();
}
