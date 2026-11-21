// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <array>
#include <type_traits>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "dataflow_common.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

namespace {
namespace fabric_api = tt::tt_fabric::linear::experimental;

constexpr uint32_t kReadyStrideBytes = 16;

volatile tt_l1_ptr uint32_t* ready_counter(uint32_t ready_base, uint32_t producer) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_base + (producer * kReadyStrideBytes));
}

struct ProducerArgs {
    uint32_t noc_x;
    uint32_t noc_y;
    CoreShare share;
};

template <
    uint32_t ring_size,
    uint32_t slice_tiles,
    uint32_t m_piece,
    uint32_t M_blocks,
    uint32_t N_blocks,
    uint32_t num_dirs,
    uint32_t dir,
    uint32_t tiles_per_packet>
class UnitCursor {
public:
    UnitCursor(const tt_l1_ptr ProducerArgs* producers, uint32_t num_producers, uint32_t ring_index, uint32_t N_tiles) :
        producers_(producers), num_producers_(num_producers), ring_index_(ring_index), N_tiles_(N_tiles) {
        if (num_producers_ == 0 || kUnitsPerProducer == 0) {
            done_ = true;
            return;
        }
        if (!load()) {
            next_unit();
        }
    }

    bool done() const {
        return done_;
    }
    uint32_t producer() const {
        return p_;
    }
    const tt_l1_ptr ProducerArgs& producer_args() const {
        return producers_[p_];
    }
    uint32_t ready_target() const {
        return j_ + 1;
    }
    bool unit_begin() const {
        return t0_ == 0;
    }
    bool unit_end() const {
        return t0_ + tiles_per_packet >= tiles_;
    }
    uint32_t count() const {
        return std::min(tiles_per_packet, tiles_ - t0_);
    }
    uint32_t page() const {
        return row_page_ + c_;
    }
    void next_tile() {
        if (++c_ == region_.cols) {
            c_ = 0;
            row_page_ += N_tiles_;
        }
    }
    void next_packet() {
        t0_ += tiles_per_packet;
        if (t0_ >= tiles_) {
            next_unit();
        }
    }

private:
    static constexpr uint32_t kBlocksPerStep = mmrs::blocks_in_direction(N_blocks, num_dirs, dir) * M_blocks;
    static constexpr uint32_t kUnitsPerProducer = (ring_size - 1) * kBlocksPerStep;

    void next_unit() {
        do {
            if (++p_ == num_producers_) {
                p_ = 0;
                if (++j_ == kUnitsPerProducer) {
                    done_ = true;
                    return;
                }
            }
        } while (!load());
    }

    bool load() {
        const uint32_t step = j_ / kBlocksPerStep;
        const uint32_t within = j_ % kBlocksPerStep;
        const uint32_t nb = mmrs::nth_block_in_direction(within / M_blocks, num_dirs, dir);
        const uint32_t mb = within % M_blocks;
        const uint32_t slice = mmrs::step_slice(step, ring_index_, ring_size, dir);
        const CoreShare share = producers_[p_].share;
        region_ = block_region<slice_tiles, m_piece, M_blocks, N_blocks>(slice, mb, nb, share);
        tiles_ = region_.rows * region_.cols;
        t0_ = 0;
        c_ = 0;
        row_page_ = (region_.row0 * N_tiles_) + region_.col0;
        return tiles_ > 0;
    }

    const tt_l1_ptr ProducerArgs* producers_;
    uint32_t num_producers_;
    uint32_t ring_index_;
    uint32_t N_tiles_;
    uint32_t p_ = 0;
    uint32_t j_ = 0;
    BlockRegion region_;
    uint32_t tiles_ = 0;
    uint32_t t0_ = 0;
    uint32_t c_ = 0;
    uint32_t row_page_ = 0;
    bool done_ = false;
};

using Header = volatile tt_l1_ptr PACKET_HEADER_TYPE*;

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
    static uint32_t packet_bytes(uint32_t tiles) {
        return sizeof(PACKET_HEADER_TYPE) + (tiles * tile_size);
    }
};

constexpr uint32_t kFusedIncMaxTiles = NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS;

template <uint32_t tile_size>
void set_scatter(Header header, uint32_t count, uint64_t landed) {
    using tt::tt_fabric::NocScatterWriteChunkEncoding;
    auto& scatter = header->command_fields.unicast_scatter_write;
    for (uint32_t i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS - 1; i++) {
        scatter.chunk_size[i] = tile_size;
    }
    uint32_t encoding =
        tt::tt_fabric::expand_encoding_to_all_chunks(NocScatterWriteChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE);
    if (landed != 0) {
        scatter.noc_address[count] = landed;
        scatter.chunk_size[count] = 1;
        encoding = (encoding & ~(0x3U << (2 * count))) |
                   (NocScatterWriteChunkEncoding::CHUNK_ENCODING_SEMINC_FLUSH << (2 * count));
    }
    scatter.chunk_encoding = static_cast<uint8_t>(encoding);
    scatter.chunk_count = static_cast<uint8_t>(count + (landed != 0 ? 1 : 0));
    header->payload_size_bytes = static_cast<uint16_t>(count * tile_size);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t slots_cb = tt::CBIndex::c_4;
    constexpr uint32_t ready_cb = tt::CBIndex::c_6;
    constexpr uint32_t tile_size = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(1);
    constexpr uint32_t slot_packets = get_compile_time_arg_val(2);
    constexpr uint32_t slot_header_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t slice_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t m_piece = get_compile_time_arg_val(6);
    constexpr uint32_t M_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t N_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t num_dirs = get_compile_time_arg_val(9);
    constexpr uint32_t dir = get_compile_time_arg_val(10);
    constexpr uint32_t ring_size = get_compile_time_arg_val(11);
    constexpr uint32_t go_semaphore_id = get_compile_time_arg_val(12);
    constexpr uint32_t packets_per_batch = slot_packets / 2;
    constexpr ccl_routing_utils::line_unicast_route_info_t route =
        ccl_routing_utils::get_line_unicast_route_info_from_args<13>();
    constexpr auto buffer_args = TensorAccessorArgs<13 + ccl_routing_utils::num_line_unicast_args>();
    static_assert(tiles_per_packet >= NOC_SCATTER_WRITE_MIN_CHUNKS && tiles_per_packet <= NOC_SCATTER_WRITE_MAX_CHUNKS);
    constexpr bool trailing_incs = tiles_per_packet > kFusedIncMaxTiles;

    size_t argidx = 0;
    const uint32_t buffer_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t landed_sem_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_index = get_arg_val<uint32_t>(argidx++);
    tt::tt_fabric::RoutingPlaneConnectionManager connection;
    tt::tt_fabric::common::experimental::open_connections(connection, 1, argidx);
    const uint32_t num_producers = get_arg_val<uint32_t>(argidx++);
    const auto* producers = reinterpret_cast<const tt_l1_ptr ProducerArgs*>(get_arg_addr(argidx));

    const auto buffer = TensorAccessor(buffer_args, buffer_addr, tile_size);
    const uint32_t ready_base = CircularBuffer(ready_cb).get_write_ptr();
    using Slots = SendSlots<tile_size, tiles_per_packet, slot_header_bytes>;
    constexpr uint32_t kAlign = NOC_DRAM_READ_ALIGNMENT_BYTES;
    const Slots slots{.base = (CircularBuffer(slots_cb).get_write_ptr() + kAlign - 1) & ~(kAlign - 1)};

    for (uint32_t p = 0; p < num_producers; p++) {
        *ready_counter(ready_base, p) = 0;
    }
    const uint32_t go_sem_addr = get_semaphore(go_semaphore_id);
    for (uint32_t p = 0; p < num_producers; p++) {
        noc_inline_dw_write(
            get_noc_addr(producers[p].noc_x, producers[p].noc_y, go_sem_addr), ready_base + (p * kReadyStrideBytes));
    }

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
    Header inc_header = nullptr;
    if constexpr (trailing_incs) {
        inc_header = PacketHeaderPool::allocate_header(1);
        fabric_api::fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            inc_header, 1, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0u, 1u});
        route_header(inc_header);
    }

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

    using Cursor = UnitCursor<ring_size, slice_tiles, m_piece, M_blocks, N_blocks, num_dirs, dir, tiles_per_packet>;
    Cursor reader(producers, num_producers, ring_index, N_tiles);
    std::array<uint32_t, slot_packets> slot_tiles{};
    std::array<uint64_t, trailing_incs ? slot_packets : 1> slot_trailing_inc{};
    uint32_t read_slot = 0;
    uint32_t send_slot = 0;

    auto read_batch = [&](uint32_t unsent) -> uint32_t {
        uint32_t issued = 0;
        while (!reader.done() && issued < packets_per_batch && unsent + issued < slot_packets) {
            if (reader.unit_begin()) {
                invalidate_l1_cache();
                if (*ready_counter(ready_base, reader.producer()) < reader.ready_target()) {
                    break;
                }
            }
            const Header header = slots.header(read_slot);
            auto& scatter = header->command_fields.unicast_scatter_write;
            const uint32_t count = reader.count();
            uint32_t l1 = slots.payload(read_slot);
            for (uint32_t i = 0; i < count; ++i, l1 += tile_size) {
                const uint32_t page = reader.page();
                noc_async_read_one_packet(buffer.get_noc_addr(page), l1, tile_size);
                scatter.noc_address[i] = buffer.get_noc_addr(page, 0, edm_to_local_chip_noc);
                reader.next_tile();
            }
            uint64_t landed_inc = 0;
            if (reader.unit_end()) {
                const auto& producer = reader.producer_args();
                landed_inc = safe_get_noc_addr(producer.noc_x, producer.noc_y, landed_sem_addr, edm_to_local_chip_noc);
            }
            if constexpr (trailing_incs) {
                slot_trailing_inc[read_slot] = count > kFusedIncMaxTiles ? landed_inc : 0;
                if (count > kFusedIncMaxTiles) {
                    landed_inc = 0;
                }
            }
            set_scatter<tile_size>(header, count, landed_inc);
            slot_tiles[read_slot] = count;
            reader.next_packet();
            read_slot = read_slot + 1 == slot_packets ? 0 : read_slot + 1;
            ++issued;
        }
        return issued;
    };

    uint32_t landed = 0;
    while (true) {
        if (landed == 0) {
            if (reader.done()) {
                break;
            }
            noc_semaphore_wait_min(ready_counter(ready_base, reader.producer()), reader.ready_target());
            noc_async_writes_flushed();
            landed = read_batch(0);
            noc_async_read_barrier();
        }
        noc_async_writes_flushed();
        const uint32_t in_flight = read_batch(landed);
        for (uint32_t n = landed; n > 0; --n) {
            send_packet(
                reinterpret_cast<uint32_t>(slots.header(send_slot)), Slots::packet_bytes(slot_tiles[send_slot]));
            if constexpr (trailing_incs) {
                if (slot_trailing_inc[send_slot] != 0) {
                    noc_async_writes_flushed();
                    inc_header->command_fields.unicast_seminc.noc_address = slot_trailing_inc[send_slot];
                    send_packet(reinterpret_cast<uint32_t>(inc_header), sizeof(PACKET_HEADER_TYPE));
                }
            }
            send_slot = send_slot + 1 == slot_packets ? 0 : send_slot + 1;
        }
        noc_async_read_barrier();
        landed = in_flight;
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
    tt::tt_fabric::common::experimental::close_connections(connection);
    noc_async_write_barrier();
}
