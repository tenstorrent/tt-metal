// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <array>
#include <type_traits>

#include "api/alignment.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
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
// bumps the counter only after the unit's writes have landed. A packet holds no more tiles than this, so the increment
// always fits behind a unit's last packet.
constexpr uint32_t kFusedIncMaxTiles = NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS;

// Points a slot's scatter header at `count` tiles whose addresses are already in it. A nonzero `landed` is a semaphore
// address bumped by `value` in a flushed chunk behind the writes - one tile + it is a 2-chunk packet, which the router
// accepts.
template <uint32_t tile_size>
void set_scatter(Header header, uint32_t count, uint64_t landed, uint32_t value = 1) {
    // The increment is chunk `count` - the router fuses it behind at most kFusedIncMaxTiles writes.
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
        scatter.chunk_size[count] = static_cast<uint16_t>(value);
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

// Where the pages of an interleaved DRAM tensor are: they go round the banks, so page p lies p / kBanks pages into its
// bank and the bank starts where page p % kBanks does.
template <uint32_t page_bytes>
struct BankedPages {
    static constexpr uint32_t kBanks = NUM_DRAM_BANKS;
    uint64_t bank_start[kBanks] = {};

    template <typename AddressOf>
    void fill(const AddressOf& address_of) {
        for (uint32_t b = 0; b < kBanks; b++) {
            bank_start[b] = address_of(b);
        }
    }
    uint64_t operator()(uint32_t page) const {
        return bank_start[page % kBanks] + ((page / kBanks) * page_bytes);
    }
};

}  // namespace

// Fabric sender on the sink row's free data mover. It reads a batch of packets into L1 slots (headers and
// destination addresses are filled in as it reads), then sends the batch that landed last round while the next one
// is read. Packets go out through the router connection's stateful command buffers, one NOC write per packet.
void kernel_main() {
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
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(9);
    static_assert(chunk_tiles % K_block_tiles == 0, "a K chunk is a whole number of K blocks");
    static_assert(slot_packets >= 2 && slot_packets % 2 == 0, "the slots are split into a filling and a sending half");
    static_assert(slot_packets <= 32, "slots are flagged in 32-bit masks");
    constexpr uint32_t packets_per_batch = slot_packets / 2;  // one half fills while the other half is sent
    constexpr uint32_t N_tiles = stripe_tiles * ring_size;
    constexpr ccl_routing_utils::line_unicast_route_info_t route =
        ccl_routing_utils::get_line_unicast_route_info_from_args<kSendFixedCtArgs>();
    constexpr auto shard_args = TensorAccessorArgs<kSendFixedCtArgs + ccl_routing_utils::num_line_unicast_args>();
    constexpr auto pgb_args = TensorAccessorArgs<shard_args.next_compile_time_args_offset()>();
    // A packet carries exactly 2 tiles.
    static_assert(tiles_per_packet == NOC_SCATTER_WRITE_MIN_CHUNKS && tiles_per_packet == kFusedIncMaxTiles);
    auto next_slot = [](uint32_t slot) { return slot + 1 == slot_packets ? 0 : slot + 1; };

    size_t argidx = 0;
    const uint32_t shard_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t pgb_addr = get_arg_val<uint32_t>(argidx++);
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
    const auto pgb = TensorAccessor(pgb_args, pgb_addr, tile_size);
    // A tile's address by its page: read from the shard (our own stripe) or the gather buffer, written to
    // the same gather-buffer page on the neighbour.
    auto shard_address = [&](uint32_t page) { return shard.get_noc_addr(page, 0, noc_index); };
    auto buffer_address = [&](uint32_t page) { return pgb.get_noc_addr(page, 0, noc_index); };
    auto neighbour_address = [&](uint32_t page) {
        return tt::tt_fabric::linear::addrgen_detail::get_noc_address(pgb, page, 0);
    };
    constexpr bool banked = !shard_args.is_sharded && shard_args.is_dram && !pgb_args.is_sharded && pgb_args.is_dram;
    static_assert(
        !banked || (shard_args.get_aligned_page_size() == tile_size && pgb_args.get_aligned_page_size() == tile_size),
        "BankedPages steps through a bank one tile at a time");
    BankedPages<tile_size> shard_pages;
    BankedPages<tile_size> buffer_pages;
    BankedPages<tile_size> neighbour_pages;
    if constexpr (banked) {
        shard_pages.fill(shard_address);
        buffer_pages.fill(buffer_address);
        neighbour_pages.fill(neighbour_address);
    }

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
    // A slot's header stands as a full packet: only its tile addresses change from packet to
    // packet. The few packets that differ rewrite the rest, and the next packet in that slot puts it back.
    for (uint32_t s = 0; s < slot_packets; s++) {
        fabric_api::fabric_unicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::None>(
            slots.header(s), 1, nullptr, 0);
        route_header(slots.header(s));
        set_scatter<tile_size>(slots.header(s), tiles_per_packet, 0);
    }
    // Header-only packet for the notify to the next device's sender, which counts in the semaphore's high half.
    constexpr uint32_t kNotifyValue = 1U << kRelayCountShift;
    Header inc_header = PacketHeaderPool::allocate_header(1);
    fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        inc_header, 1, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});
    route_header(inc_header);

    auto& router = connection.get(0).sender;
    router.setup_stateful_send_cmd_bufs();
    uint32_t free_router_slots = 0;
    // The loop's helpers inlined so that its state stays in registers.
    auto send_packet = [&](uint32_t header_addr, uint32_t bytes) __attribute__((always_inline)) {
        while (free_router_slots == 0) {
            free_router_slots = router.get_num_free_write_slots();
        }
        router.send_current_slot_stateful_non_blocking_from_address(header_addr, bytes);
        --free_router_slots;
    };
    auto send_notify = [&](uint64_t noc_address) __attribute__((always_inline)) {
        noc_async_writes_flushed();
        inc_header->command_fields.unicast_seminc.noc_address = noc_address;
        inc_header->command_fields.unicast_seminc.val = kNotifyValue;
        send_packet(reinterpret_cast<uint32_t>(inc_header), sizeof(PACKET_HEADER_TYPE));
    };

    SendCursor cursor(schedule_arg_idx, K_tiles, chunk_tiles);
    std::array<uint64_t, slot_packets> slot_notify_inc{};
    uint32_t notify_slots = 0;
    uint32_t odd_slots = 0;  // slots whose header is not the standing one
    uint32_t read_slot = 0;
    uint32_t send_slot = 0;
    uint32_t relay_credits[ring_size] = {};

    // The unit being read. Its tiles go a K chunk at a time, row after row of the piece, and a tile's pages step with
    // it.
    uint32_t tiles_left = 0;  // 0: the cursor's unit has not been started
    uint32_t row_left = 0;    // tiles left in the row's K chunk
    uint32_t chunk = 0;
    uint32_t page = 0;
    uint32_t source_page = 0;
    uint32_t page_row_step = 0;  // from a row's last tile to the next row's first
    uint32_t source_row_step = 0;
    uint32_t source_k_step = 0;
    bool own_stripe = false;
    [[maybe_unused]] const BankedPages<tile_size>* source_pages = &shard_pages;
    // Page steps along K and from the end of a K chunk to the next row, for a tensor `n_cols` wide.
    auto k_step = [](uint32_t n_cols) { return gathered_page<transpose_b>(0, 1, K_tiles, n_cols); };
    auto row_step = [](uint32_t n_cols, uint32_t k_count) {
        return gathered_page<transpose_b>(1, 0, K_tiles, n_cols) -
               gathered_page<transpose_b>(0, k_count - 1, K_tiles, n_cols);
    };
    const uint32_t page_k_step = k_step(N_tiles);

    // The previous device's sender bumps the high half of our semaphore for a stripe each time a K chunk of the pieces
    // we relay has landed here.
    auto at_relayed_chunk = [&]() { return cursor.segment().distance > 0 && cursor.chunk_begin(); };
    auto relay_counter = [&]() {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stripe_sem_addrs[cursor.segment().stripe]);
    };
    auto relay_target = [&]() { return (cursor.chunk_index() + 1) << kRelayCountShift; };
    // Is the cursor at a relayed K chunk that has not landed yet? The counter is written remotely, hence the
    // invalidate.
    auto relay_pending = [&]() {
        if (!at_relayed_chunk()) {
            return false;
        }
        invalidate_l1_cache();
        return *relay_counter() < relay_target();
    };

    // Reads up to a batch of packets into the free slots and fills in their headers, stopping at a relayed K chunk
    // that has not landed yet.
    auto read_batch = [&]() __attribute__((always_inline)) -> uint32_t {
        uint32_t issued = 0;
        while (issued < packets_per_batch) {
            if (tiles_left == 0) {
                if (cursor.done()) {
                    break;
                }
                if (relay_pending()) {
                    break;
                }
                const SendSegment& seg = cursor.segment();
                if (at_relayed_chunk()) {
                    relay_credits[seg.stripe]++;
                }
                const SendPieceArgs& piece = cursor.piece();
                chunk = cursor.chunk();
                tiles_left = piece.width * chunk;
                row_left = chunk;
                page = gathered_page<transpose_b>(
                    (seg.stripe * stripe_tiles) + piece.row_offset, cursor.k_start(), K_tiles, N_tiles);
                page_row_step = row_step(N_tiles, chunk);
                own_stripe = seg.distance == 0;
                if (own_stripe) {
                    source_page = gathered_page<transpose_b>(piece.row_offset, cursor.k_start(), K_tiles, stripe_tiles);
                    source_k_step = k_step(stripe_tiles);
                    source_row_step = row_step(stripe_tiles, chunk);
                    source_pages = &shard_pages;
                } else {
                    source_page = page;
                    source_k_step = page_k_step;
                    source_row_step = page_row_step;
                    source_pages = &buffer_pages;
                }
            }
            const Header header = slots.header(read_slot);
            auto& scatter = header->command_fields.unicast_scatter_write;
            const uint32_t count = tiles_left < tiles_per_packet ? tiles_left : tiles_per_packet;
            uint32_t l1 = slots.payload(read_slot);
            for (uint32_t i = 0; i < count; ++i, l1 += tile_size) {
                uint64_t source = 0;
                if constexpr (banked) {
                    scatter.noc_address[i] = neighbour_pages(page);
                    source = (*source_pages)(source_page);
                } else {
                    scatter.noc_address[i] = neighbour_address(page);
                    source = own_stripe ? shard_address(source_page) : buffer_address(source_page);
                }
                noc_async_read_one_packet(source, l1, tile_size);
                if (--row_left == 0) {
                    row_left = chunk;
                    page += page_row_step;
                    source_page += source_row_step;
                } else {
                    page += page_k_step;
                    source_page += source_k_step;
                }
            }
            tiles_left -= count;
            const uint32_t slot_bit = 1U << read_slot;
            uint64_t landed_inc = 0;
            uint32_t landed_value = 0;
            if (tiles_left == 0) {
                // The unit's last packet. Every unit tells its chain's injector on the next device that it has landed.
                // Of the pieces the next device relays further, only the segment's last one has the notify flag, so
                // that device's sender is told once per K chunk.
                const SendSegment& seg = cursor.segment();
                const SendPieceArgs& piece = cursor.piece();
                landed_inc = safe_get_noc_addr(
                    static_cast<uint8_t>(piece.inj_noc_x),
                    static_cast<uint8_t>(piece.inj_noc_y),
                    stripe_sem_addrs[seg.stripe],
                    edm_to_local_chip_noc);
                if (piece.notify) {
                    slot_notify_inc[read_slot] = safe_get_noc_addr(
                        static_cast<uint8_t>(peer_sender_noc_x),
                        static_cast<uint8_t>(peer_sender_noc_y),
                        stripe_sem_addrs[seg.stripe],
                        edm_to_local_chip_noc);
                    notify_slots |= slot_bit;
                }
                landed_value = (chunk + K_block_tiles - 1) / K_block_tiles;
                cursor.next_unit();
            }
            if (count != tiles_per_packet || landed_inc != 0) {
                set_scatter<tile_size>(header, count, landed_inc, landed_value);
                odd_slots |= slot_bit;
            } else if ((odd_slots & slot_bit) != 0) {
                set_scatter<tile_size>(header, tiles_per_packet, 0);
                odd_slots &= ~slot_bit;
            }
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
            if (cursor.done()) {
                break;
            }
            if (at_relayed_chunk()) {
                noc_semaphore_wait_min(relay_counter(), relay_target());
            }
        }
        const uint32_t in_flight = read_batch();
        for (uint32_t n = landed; n > 0; --n) {
            send_packet(reinterpret_cast<uint32_t>(slots.header(send_slot)), slots.packet_bytes(send_slot));
            const uint32_t slot_bit = 1U << send_slot;
            if ((notify_slots & slot_bit) != 0) {
                notify_slots &= ~slot_bit;
                send_notify(slot_notify_inc[send_slot]);
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
            noc_semaphore_inc(get_noc_addr(stripe_sem_addrs[s]), -(relay_credits[s] << kRelayCountShift));
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
    tt::tt_fabric::common::experimental::close_connections(connection);
    noc_async_write_barrier();
}
