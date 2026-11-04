// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_common.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "dataflow_common.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"

void kernel_main() {
    Noc noc;
    constexpr uint32_t cb_id = tt::CBIndex::c_4;
    constexpr uint32_t tile_size = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(1);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t stripe_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t ring_size = get_compile_time_arg_val(5);
    constexpr bool transpose_b = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t slot_packets = get_compile_time_arg_val(7);
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t packets_per_batch = slot_packets / 2;
    constexpr ccl_routing_utils::line_unicast_route_info_t route =
        ccl_routing_utils::get_line_unicast_route_info_from_args<9>();
    constexpr auto shard_args = TensorAccessorArgs<9 + ccl_routing_utils::num_line_unicast_args>();
    constexpr auto pwb_args = TensorAccessorArgs<shard_args.next_compile_time_args_offset()>();
    constexpr uint32_t packet_bytes = tiles_per_packet * tile_size;

    size_t argidx = 0;
    const uint32_t shard_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t pwb_addr = get_arg_val<uint32_t>(argidx++);
    uint32_t stripe_sem_addrs[ring_size];
    for (uint32_t i = 0; i < ring_size; i++) {
        stripe_sem_addrs[i] = get_arg_val<uint32_t>(argidx++);
    }
    const uint32_t peer_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t peer_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    tt::tt_fabric::RoutingPlaneConnectionManager connection;
    tt::tt_fabric::common::experimental::open_connections(connection, 1, argidx);
    const uint32_t schedule_arg_idx = argidx;

    const auto shard = TensorAccessor(shard_args, shard_addr, tile_size);
    const auto pwb = TensorAccessor(pwb_args, pwb_addr, tile_size);
    const uint32_t slots_base = CircularBuffer(cb_id).get_write_ptr();

    FabricWriter<tile_size, packet_size_bytes, tt::tt_fabric::WorkerToFabricEdmSender> fabric(
        noc, &connection.get(0).sender, route.dst_chip_id, route.dst_mesh_id);

    auto send_inc = [&](uint32_t noc_x, uint32_t noc_y, uint32_t addr) {
        fabric.atomic_inc(safe_get_noc_addr(static_cast<uint8_t>(noc_x), static_cast<uint8_t>(noc_y), addr, 0), 1);
    };

    SendCursor reader(schedule_arg_idx, K_tiles, chunk_tiles, tiles_per_packet);
    SendCursor sender(schedule_arg_idx, K_tiles, chunk_tiles, tiles_per_packet);
    uint32_t read_slot = 0;
    uint32_t send_slot = 0;
    uint32_t unsent = 0;
    uint32_t relay_credits[ring_size] = {};

    auto relay_counter = [&](const SendCursor& c) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stripe_sem_addrs[c.segment().stripe]);
    };
    auto relay_target = [](const SendCursor& c) { return c.unit_index() + 1; };

    auto read_batch = [&]() -> uint32_t {
        uint32_t issued = 0;
        while (!reader.done() && issued < packets_per_batch && unsent < slot_packets) {
            const SendSegment& seg = reader.segment();
            if (reader.unit_begin() && seg.distance > 0) {
                invalidate_l1_cache();
                if (*relay_counter(reader) < relay_target(reader)) {
                    break;
                }
                relay_credits[seg.stripe]++;
            }
            const SendPieceArgs& piece = reader.piece();
            uint32_t l1 = slots_base + (read_slot * packet_bytes);
            for (uint32_t i = 0; i < reader.count(); ++i, l1 += tile_size) {
                const uint32_t t = reader.t0() + i;
                const uint32_t row = piece.row_offset + (t / reader.chunk());
                const uint32_t k = reader.k_start() + (t % reader.chunk());
                if (seg.distance == 0) {
                    const uint32_t page = weight_page<transpose_b>(row, k, K_tiles, stripe_tiles);
                    noc.async_read(shard, CoreLocalMem<uint32_t>(l1), tile_size, {.page_id = page}, {});
                } else {
                    const uint32_t page = weight_page<transpose_b>(
                        (seg.stripe * stripe_tiles) + row, k, K_tiles, stripe_tiles * ring_size);
                    noc.async_read(pwb, CoreLocalMem<uint32_t>(l1), tile_size, {.page_id = page}, {});
                }
            }
            read_slot = (read_slot + 1) % slot_packets;
            ++unsent;
            ++issued;
            reader.advance();
        }
        return issued;
    };

    auto send_steps = [&](uint32_t n) {
        while (n > 0) {
            const SendSegment& seg = sender.segment();
            const SendPieceArgs& piece = sender.piece();
            uint32_t l1 = slots_base + (send_slot * packet_bytes);
            for (uint32_t i = 0; i < sender.count(); ++i, l1 += tile_size) {
                const uint32_t t = sender.t0() + i;
                const uint32_t row = (seg.stripe * stripe_tiles) + piece.row_offset + (t / sender.chunk());
                const uint32_t page = weight_page<transpose_b>(
                    row, sender.k_start() + (t % sender.chunk()), K_tiles, stripe_tiles * ring_size);
                fabric.async_write(l1, tt::tt_fabric::linear::addrgen_detail::get_noc_address(pwb, page, 0));
            }
            fabric.async_writes_flushed();
            if (sender.unit_end()) {
                send_inc(piece.inj_noc_x, piece.inj_noc_y, stripe_sem_addrs[seg.stripe]);
                if (piece.notify) {
                    fabric.async_writes_flushed();
                    send_inc(peer_sender_noc_x, peer_sender_noc_y, stripe_sem_addrs[seg.stripe]);
                }
            }
            send_slot = (send_slot + 1) % slot_packets;
            --unsent;
            sender.advance();
            --n;
        }
    };

    uint32_t landed = read_batch();
    noc.async_read_barrier();
    while (!sender.done()) {
        const uint32_t in_flight = read_batch();
        send_steps(landed);
        noc.async_read_barrier();
        landed = in_flight;
        if (landed == 0 && !sender.done()) {
            noc_semaphore_wait_min(relay_counter(reader), relay_target(reader));
        }
    }

    for (uint32_t s = 0; s < ring_size; s++) {
        if (relay_credits[s] > 0) {
            noc_semaphore_inc(get_noc_addr(stripe_sem_addrs[s]), -relay_credits[s]);
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
    tt::tt_fabric::common::experimental::close_connections(connection);
    noc.async_write_barrier();
}
