// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <array>
#include <type_traits>

#include "api/alignment.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "dataflow_common.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

namespace {

namespace fabric_api = tt::tt_fabric::linear::experimental;

using Header = volatile tt_l1_ptr PACKET_HEADER_TYPE*;

// One router packet per L1 slot, with the slot's own header right in front of its payload: a packet leaves as a
// single NOC write, and a header is only rewritten once the packet that used it has left.
template <uint32_t tile_size, uint32_t tiles_per_packet, uint32_t header_bytes>
struct SendSlots {
    static constexpr uint32_t kSlotBytes = header_bytes + (tiles_per_packet * tile_size);
    static_assert(sizeof(PACKET_HEADER_TYPE) <= header_bytes);

    uint32_t base;

    uint32_t payload(uint32_t slot) const {
        return base + (slot * kSlotBytes) + header_bytes;
    }
    Header header(uint32_t slot) const {
        return reinterpret_cast<Header>(payload(slot) - sizeof(PACKET_HEADER_TYPE));
    }
    uint32_t packet_bytes(uint32_t slot) const {
        return sizeof(PACKET_HEADER_TYPE) + header(slot)->payload_size_bytes;
    }
};

// A unit's landed increment rides behind at most this many tiles as a flushed semaphore chunk: the neighbour's router
// bumps the counter only after the unit's writes have landed.
constexpr uint32_t kFusedIncMaxTiles = NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS;

// Points a slot's scatter header at `count` tiles whose addresses are already in it. A nonzero `landed` is a semaphore
// address bumped by a flushed chunk behind the writes; one tile + it is a 2-chunk packet, which the router accepts.
template <uint32_t tile_size>
void set_scatter(Header header, uint32_t count, uint64_t landed) {
    // The increment is chunk `count`; the router fuses it behind at most kFusedIncMaxTiles writes.
    ASSERT(landed == 0 || count <= kFusedIncMaxTiles);
    auto& scatter = header->command_fields.unicast_scatter_write;
    for (uint32_t i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS - 1; i++) {
        scatter.chunk_size[i] = tile_size;
    }
    uint8_t encoding = tt::tt_fabric::expand_encoding_to_all_chunks(
        tt::tt_fabric::NocScatterWriteChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE);
    if (landed != 0) {
        // The increment value travels in the chunk size.
        scatter.noc_address[count] = landed;
        scatter.chunk_size[count] = 1;
        encoding = static_cast<uint8_t>(
            encoding & ~(tt::tt_fabric::NOC_SCATTER_CHUNK_ENCODING_MASK
                         << (count * tt::tt_fabric::NOC_SCATTER_CHUNK_ENCODING_BITS)));
        tt::tt_fabric::set_chunk_encoding(
            encoding, tt::tt_fabric::NocScatterWriteChunkEncoding::CHUNK_ENCODING_SEMINC_FLUSH, count);
    }
    scatter.chunk_encoding = encoding;
    scatter.chunk_count = static_cast<uint8_t>(count + (landed != 0 ? 1 : 0));
    header->payload_size_bytes = static_cast<uint16_t>(count * tile_size);
}

}  // namespace

// Fabric sender on the sink row's free data mover. It reads a batch of packets into L1 slots (headers and
// destination addresses are filled in as it reads), then sends the batch that landed last round while the next one
// is read. Packets go out through the router connection's stateful command buffers, one NOC write per packet.
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
    constexpr uint32_t slot_header_bytes = get_compile_time_arg_val(8);
    static_assert(slot_packets >= 2 && slot_packets % 2 == 0, "the slots are split into a filling and a sending half");
    constexpr uint32_t packets_per_batch = slot_packets / 2;  // one half fills while the other half is sent
    constexpr ccl_routing_utils::line_unicast_route_info_t route =
        ccl_routing_utils::get_line_unicast_route_info_from_args<9>();
    constexpr auto shard_args = TensorAccessorArgs<9 + ccl_routing_utils::num_line_unicast_args>();
    constexpr auto pwb_args = TensorAccessorArgs<shard_args.next_compile_time_args_offset()>();
    // Every packet is a scatter write, which takes 2 to 4 chunks.
    static_assert(tiles_per_packet >= NOC_SCATTER_WRITE_MIN_CHUNKS && tiles_per_packet <= NOC_SCATTER_WRITE_MAX_CHUNKS);
    constexpr bool trailing_incs = tiles_per_packet > kFusedIncMaxTiles;
    auto next_slot = [](uint32_t slot) { return slot + 1 == slot_packets ? 0 : slot + 1; };

    size_t argidx = 0;
    const uint32_t shard_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t pwb_addr = get_arg_val<uint32_t>(argidx++);
    // One global semaphore per stripe, at the same L1 address on every core of every device.
    uint32_t stripe_sem_addrs[ring_size];
    for (uint32_t i = 0; i < ring_size; i++) {
        stripe_sem_addrs[i] = get_arg_val<uint32_t>(argidx++);
    }
    // The sender at our own core on the next device, which forwards what we send it.
    const uint32_t peer_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t peer_sender_noc_y = get_arg_val<uint32_t>(argidx++);

    tt::tt_fabric::RoutingPlaneConnectionManager connection;
    tt::tt_fabric::common::experimental::open_connections(connection, 1, argidx);
    const uint32_t schedule_arg_idx = argidx;

    const auto shard = TensorAccessor(shard_args, shard_addr, tile_size);
    const auto pwb = TensorAccessor(pwb_args, pwb_addr, tile_size);
    using Slots = SendSlots<tile_size, tiles_per_packet, slot_header_bytes>;
    const Slots slots{.base = align(CircularBuffer(cb_id).get_write_ptr(), NOC_DRAM_READ_ALIGNMENT_BYTES)};

    auto route_header = [&](Header header) {
        if constexpr (std::is_base_of_v<tt::tt_fabric::HybridMeshPacketHeader, PACKET_HEADER_TYPE>) {
            fabric_set_unicast_route(
                reinterpret_cast<volatile tt::tt_fabric::HybridMeshPacketHeader*>(header),
                route.dst_chip_id,
                route.dst_mesh_id);
        }
    };
    for (uint32_t s = 0; s < slot_packets; s++) {
        fabric_api::fabric_unicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
            slots.header(s), 1, nullptr, 0);
        route_header(slots.header(s));
    }
    // Header-only packet for increments that cannot ride behind a data packet: a unit's landed increment when its last
    // packet is too full and the notify to the next device's sender.
    Header inc_header = PacketHeaderPool::allocate_header(1);
    fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        inc_header, 1, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});
    route_header(inc_header);

    // Until the last packet is sent, this RISC writes only through the router connection's command buffers.
    auto& router = connection.get(0).sender;
    router.setup_stateful_send_cmd_bufs();
    uint32_t free_router_slots = 0;
    auto send_packet = [&](uint32_t header_addr, uint32_t bytes) {
        while (free_router_slots == 0) {
            free_router_slots = router.get_num_free_write_slots();
        }
        router.send_current_slot_stateful_non_blocking_from_address(header_addr, bytes);
        --free_router_slots;
    };
    auto send_inc = [&](uint64_t noc_address) {
        // The previous header-only packet must have left L1 before its header is rewritten.
        noc_async_writes_flushed();
        inc_header->command_fields.unicast_seminc.noc_address = noc_address;
        send_packet(reinterpret_cast<uint32_t>(inc_header), sizeof(PACKET_HEADER_TYPE));
    };

    SendCursor reader(schedule_arg_idx, K_tiles, chunk_tiles, tiles_per_packet);
    std::array<uint64_t, trailing_incs ? slot_packets : 1> slot_trailing_inc{};
    std::array<uint64_t, slot_packets> slot_notify_inc{};
    uint32_t read_slot = 0;
    uint32_t send_slot = 0;
    uint32_t relay_credits[ring_size] = {};

    // The previous device's sender bumps our semaphore for a stripe each time one of its units lands here.
    auto relay_counter = [&](const SendCursor& c) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stripe_sem_addrs[c.segment().stripe]);
    };
    auto relay_target = [](const SendCursor& c) { return c.unit_index() + 1; };  // bumped once per landed unit

    // Reads up to a batch of packets into the free slots and fills in their headers, stopping at a relayed unit that
    // has not landed yet.
    auto read_batch = [&]() -> uint32_t {
        uint32_t issued = 0;
        // At most packets_per_batch packets are unsent here, so a batch always fits in the slots.
        while (!reader.done() && issued < packets_per_batch) {
            const SendSegment& seg = reader.segment();
            // A relayed unit: has it landed yet? The counter is written remotely, hence the invalidate.
            if (reader.unit_begin() && seg.distance > 0) {
                invalidate_l1_cache();
                if (*relay_counter(reader) < relay_target(reader)) {
                    break;
                }
                relay_credits[seg.stripe]++;
            }
            const SendPieceArgs& piece = reader.piece();
            const Header header = slots.header(read_slot);
            auto& scatter = header->command_fields.unicast_scatter_write;
            const uint32_t count = reader.count();
            uint32_t l1 = slots.payload(read_slot);
            uint32_t row = piece.row_offset + reader.row();
            uint32_t k = reader.k();
            const uint32_t k_end = reader.k_start() + reader.chunk();
            const bool own_stripe = seg.distance == 0;
            for (uint32_t i = 0; i < count; ++i, l1 += tile_size) {
                // The tile's weight-buffer page: its place on the neighbour (same address there) and a relay's source.
                const uint32_t buffer_page =
                    weight_page<transpose_b>((seg.stripe * stripe_tiles) + row, k, K_tiles, stripe_tiles * ring_size);
                scatter.noc_address[i] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(pwb, buffer_page, 0);
                if (own_stripe) {
                    const uint32_t page = weight_page<transpose_b>(row, k, K_tiles, stripe_tiles);
                    noc.async_read<NocOptions::DEFAULT, tile_size>(
                        shard, CoreLocalMem<uint32_t>(l1), tile_size, {.page_id = page}, {});
                } else {
                    noc.async_read<NocOptions::DEFAULT, tile_size>(
                        pwb, CoreLocalMem<uint32_t>(l1), tile_size, {.page_id = buffer_page}, {});
                }
                if (++k == k_end) {
                    k = reader.k_start();
                    ++row;
                }
            }
            // The unit's last packet: tell its injector on the next device and that device's sender if it relays the
            // unit further.
            uint64_t landed_inc = 0;
            uint64_t notify_inc = 0;
            if (reader.unit_end()) {
                landed_inc = safe_get_noc_addr(
                    static_cast<uint8_t>(piece.inj_noc_x),
                    static_cast<uint8_t>(piece.inj_noc_y),
                    stripe_sem_addrs[seg.stripe],
                    edm_to_local_chip_noc);
                if (piece.notify) {
                    notify_inc = safe_get_noc_addr(
                        static_cast<uint8_t>(peer_sender_noc_x),
                        static_cast<uint8_t>(peer_sender_noc_y),
                        stripe_sem_addrs[seg.stripe],
                        edm_to_local_chip_noc);
                }
            }
            if constexpr (trailing_incs) {
                // Too many tiles for the fused increment: it follows as its own packet.
                const bool trailing = count > kFusedIncMaxTiles;
                slot_trailing_inc[read_slot] = trailing ? landed_inc : 0;
                if (trailing) {
                    landed_inc = 0;
                }
            }
            set_scatter<tile_size>(header, count, landed_inc);
            slot_notify_inc[read_slot] = notify_inc;
            reader.advance();
            read_slot = next_slot(read_slot);
            ++issued;
        }
        return issued;
    };

    // Each round sends the batch that landed last round while the next one is read into the slots sent before.
    uint32_t landed = 0;
    while (true) {
        // The slots read next were sent last round: their packets must have left L1.
        noc_async_writes_flushed();
        if (landed == 0) {
            if (reader.done()) {
                break;
            }
            if (reader.unit_begin() && reader.segment().distance > 0) {
                noc_semaphore_wait_min(relay_counter(reader), relay_target(reader));
            }
            landed = read_batch();
            noc_async_read_barrier();
        }
        const uint32_t in_flight = read_batch();
        for (uint32_t n = landed; n > 0; --n) {
            send_packet(reinterpret_cast<uint32_t>(slots.header(send_slot)), slots.packet_bytes(send_slot));
            if constexpr (trailing_incs) {
                if (slot_trailing_inc[send_slot] != 0) {
                    send_inc(slot_trailing_inc[send_slot]);
                }
            }
            if (slot_notify_inc[send_slot] != 0) {
                send_inc(slot_notify_inc[send_slot]);
            }
            send_slot = next_slot(send_slot);
        }
        noc_async_read_barrier();
        landed = in_flight;
    }

    // Subtract only this call's credits: a neighbour already in the next call may have bumped these too.
    noc_async_writes_flushed();
    for (uint32_t s = 0; s < ring_size; s++) {
        if (relay_credits[s] > 0) {
            noc_semaphore_inc(get_noc_addr(stripe_sem_addrs[s]), -relay_credits[s]);
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
    tt::tt_fabric::common::experimental::close_connections(connection);
    noc_async_write_barrier();
}
