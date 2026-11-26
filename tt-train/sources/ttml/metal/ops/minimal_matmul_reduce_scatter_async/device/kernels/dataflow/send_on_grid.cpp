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

constexpr uint32_t kReadyStrideBytes = 32;

volatile tt_l1_ptr uint32_t* ready_word(uint32_t ready_base, uint32_t producer, uint32_t word) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        ready_base + (producer * kReadyStrideBytes) + (word * sizeof(uint32_t)));
}

volatile tt_l1_ptr uint32_t* ready_counter(uint32_t ready_base, uint32_t producer) {
    return ready_word(ready_base, producer, 0);
}

volatile tt_l1_ptr uint32_t* out_base(uint32_t ready_base, uint32_t producer) {
    return ready_word(ready_base, producer, 1);
}

volatile tt_l1_ptr uint32_t* taken_count(uint32_t ready_base, uint32_t producer) {
    return ready_word(ready_base, producer, 4);
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
    uint32_t dir>
class UnitCursor {
public:
    UnitCursor(const tt_l1_ptr ProducerArgs* producers, uint32_t num_producers, uint32_t ring_index) :
        producers_(producers), num_producers_(num_producers), ring_index_(ring_index) {
        if (num_producers_ == 0 || kUnitsPerProducer == 0) {
            done_ = true;
            return;
        }
        load_block();
        if (!load_unit()) {
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
    const BlockRegion& region() const {
        return region_;
    }
    uint32_t block() const {
        return block_;
    }
    void next_unit() {
        do {
            if (++p_ == num_producers_) {
                p_ = 0;
                if (++j_ == kUnitsPerProducer) {
                    done_ = true;
                    return;
                }
                load_block();
            }
        } while (!load_unit());
    }

private:
    static constexpr uint32_t kBlocksPerStep = mmrs::blocks_in_direction(N_blocks, num_dirs, dir) * M_blocks;
    static constexpr uint32_t kUnitsPerProducer = (ring_size - 1) * kBlocksPerStep;

    void load_block() {
        const uint32_t step = j_ / kBlocksPerStep;
        const uint32_t within = j_ % kBlocksPerStep;
        slice_ = mmrs::step_slice(step, ring_index_, ring_size, dir);
        nb_ = mmrs::nth_block_in_direction(within / M_blocks, num_dirs, dir);
        mb_ = within % M_blocks;
        block_ = (step * N_blocks * M_blocks) + (nb_ * M_blocks) + mb_;
    }

    bool load_unit() {
        region_ = block_region<slice_tiles, m_piece, M_blocks, N_blocks>(slice_, mb_, nb_, producers_[p_].share);
        return !region_.empty();
    }

    const tt_l1_ptr ProducerArgs* producers_;
    uint32_t num_producers_;
    uint32_t ring_index_;
    uint32_t p_ = 0;
    uint32_t j_ = 0;
    uint32_t slice_ = 0;
    uint32_t nb_ = 0;
    uint32_t mb_ = 0;
    uint32_t block_ = 0;
    BlockRegion region_;
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
    uint32_t packet_bytes(uint32_t slot) const {
        return sizeof(PACKET_HEADER_TYPE) + header(slot)->payload_size_bytes;
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
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t out_cb_blocks = get_compile_time_arg_val(15);
    constexpr uint32_t taken_semaphore_id = get_compile_time_arg_val(16);
    constexpr bool stage_in_l1 = get_compile_time_arg_val(17) == 1;
    constexpr uint32_t block_bytes = M_block_tiles * N_block_tiles * tile_size;
    static_assert(slot_packets >= 2 && slot_packets % 2 == 0 && slot_packets <= 32);
    constexpr uint32_t packets_per_batch = slot_packets / 2;
    constexpr ccl_routing_utils::line_unicast_route_info_t route =
        ccl_routing_utils::get_line_unicast_route_info_from_args<18>();
    constexpr auto buffer_args = TensorAccessorArgs<18 + ccl_routing_utils::num_line_unicast_args>();
    static_assert(tiles_per_packet >= NOC_SCATTER_WRITE_MIN_CHUNKS && tiles_per_packet <= NOC_SCATTER_WRITE_MAX_CHUNKS);
    constexpr bool trailing_incs = tiles_per_packet > kFusedIncMaxTiles;
    auto next_slot = [](uint32_t slot) { return slot + 1 == slot_packets ? 0 : slot + 1; };

    size_t argidx = 0;
    const uint32_t buffer_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t landed_sem_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_index = get_arg_val<uint32_t>(argidx++);
    tt::tt_fabric::RoutingPlaneConnectionManager connection;
    tt::tt_fabric::common::experimental::open_connections(connection, 1, argidx);
    const uint32_t num_producers = get_arg_val<uint32_t>(argidx++);
    const auto* producers = reinterpret_cast<const tt_l1_ptr ProducerArgs*>(get_arg_addr(argidx));

    const auto buffer = TensorAccessor(buffer_args, buffer_addr, tile_size);
    auto local_address = [&](uint32_t page) { return buffer.get_noc_addr(page); };
    auto neighbour_address = [&](uint32_t page) { return buffer.get_noc_addr(page, 0, edm_to_local_chip_noc); };
    constexpr bool banked = !buffer_args.is_sharded && buffer_args.is_dram;
    static_assert(!banked || buffer_args.get_aligned_page_size() == tile_size);
    BankedPages<tile_size> local_pages;
    BankedPages<tile_size> neighbour_pages;
    if constexpr (banked) {
        local_pages.fill(local_address);
        neighbour_pages.fill(neighbour_address);
    }
    const uint32_t ready_base = CircularBuffer(ready_cb).get_write_ptr();
    using Slots = SendSlots<tile_size, tiles_per_packet, slot_header_bytes>;
    constexpr uint32_t kAlign = NOC_DRAM_READ_ALIGNMENT_BYTES;
    const Slots slots{.base = (CircularBuffer(slots_cb).get_write_ptr() + kAlign - 1) & ~(kAlign - 1)};

    for (uint32_t p = 0; p < num_producers; p++) {
        *ready_counter(ready_base, p) = 0;
        *out_base(ready_base, p) = 0;
        *taken_count(ready_base, p) = 0;
    }
    const uint32_t go_sem_addr = get_semaphore(go_semaphore_id);
    const uint32_t taken_sem_addr = get_semaphore(taken_semaphore_id);
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
        set_scatter<tile_size>(slots.header(s), tiles_per_packet, 0);
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
    auto send_packet = [&](uint32_t header_addr, uint32_t bytes) __attribute__((always_inline)) {
        while (free_router_slots == 0) {
            free_router_slots = router.get_num_free_write_slots();
        }
        router.send_current_slot_stateful_non_blocking_from_address(header_addr, bytes);
        --free_router_slots;
    };

    using Cursor = UnitCursor<ring_size, slice_tiles, m_piece, M_blocks, N_blocks, num_dirs, dir>;
    Cursor cursor(producers, num_producers, ring_index);
    std::array<uint64_t, trailing_incs ? slot_packets : 1> slot_trailing_inc{};
    std::array<uint32_t, stage_in_l1 ? slot_packets : 1> slot_producer{};
    uint32_t trailing_slots = 0;
    uint32_t taken_slots = 0;
    uint32_t odd_slots = 0;
    uint32_t read_slot = 0;
    uint32_t send_slot = 0;
    uint32_t tiles_left = 0;
    uint32_t row_left = 0;
    uint32_t cols = 0;
    uint32_t page = 0;
    uint32_t row_step = 0;
    uint32_t source = 0;
    uint32_t source_x = 0;
    uint32_t source_y = 0;

    auto read_batch = [&]() __attribute__((always_inline)) -> uint32_t {
        uint32_t issued = 0;
        while (issued < packets_per_batch) {
            if (tiles_left == 0) {
                if (cursor.done()) {
                    break;
                }
                invalidate_l1_cache();
                if (*ready_counter(ready_base, cursor.producer()) < cursor.ready_target()) {
                    break;
                }
                const BlockRegion& region = cursor.region();
                cols = region.cols;
                tiles_left = region.rows * cols;
                row_left = cols;
                page = (region.row0 * N_tiles) + region.col0;
                row_step = N_tiles + 1 - cols;
                if constexpr (stage_in_l1) {
                    volatile tt_l1_ptr uint32_t* base = out_base(ready_base, cursor.producer());
                    while (*base == 0) {
                        invalidate_l1_cache();
                    }
                    source = *base + ((cursor.block() % out_cb_blocks) * block_bytes);
                    source_x = cursor.producer_args().noc_x;
                    source_y = cursor.producer_args().noc_y;
                }
            }
            const Header header = slots.header(read_slot);
            auto& scatter = header->command_fields.unicast_scatter_write;
            const uint32_t count = tiles_left < tiles_per_packet ? tiles_left : tiles_per_packet;
            uint32_t l1 = slots.payload(read_slot);
            if constexpr (stage_in_l1) {
                for (uint32_t i = 0; i < count;) {
                    const uint32_t run = row_left < count - i ? row_left : count - i;
                    noc_async_read(get_noc_addr(source_x, source_y, source), l1, run * tile_size);
                    for (uint32_t t = 0; t < run; ++t, ++i, ++page) {
                        if constexpr (banked) {
                            scatter.noc_address[i] = neighbour_pages(page);
                        } else {
                            scatter.noc_address[i] = neighbour_address(page);
                        }
                    }
                    l1 += run * tile_size;
                    source += run * tile_size;
                    row_left -= run;
                    if (row_left == 0) {
                        row_left = cols;
                        page += row_step - 1;
                        source += (N_block_tiles - cols) * tile_size;
                    }
                }
            } else {
                for (uint32_t i = 0; i < count; ++i, l1 += tile_size) {
                    uint64_t tile = 0;
                    if constexpr (banked) {
                        scatter.noc_address[i] = neighbour_pages(page);
                        tile = local_pages(page);
                    } else {
                        scatter.noc_address[i] = neighbour_address(page);
                        tile = local_address(page);
                    }
                    noc_async_read_one_packet(tile, l1, tile_size);
                    if (--row_left == 0) {
                        row_left = cols;
                        page += row_step;
                    } else {
                        ++page;
                    }
                }
            }
            tiles_left -= count;
            const uint32_t slot_bit = 1U << read_slot;
            uint64_t landed_inc = 0;
            if (tiles_left == 0) {
                const auto& producer = cursor.producer_args();
                landed_inc = safe_get_noc_addr(producer.noc_x, producer.noc_y, landed_sem_addr, edm_to_local_chip_noc);
                if constexpr (stage_in_l1) {
                    slot_producer[read_slot] = cursor.producer();
                    taken_slots |= slot_bit;
                }
                cursor.next_unit();
            }
            if constexpr (trailing_incs) {
                if (landed_inc != 0 && count > kFusedIncMaxTiles) {
                    slot_trailing_inc[read_slot] = landed_inc;
                    trailing_slots |= slot_bit;
                    landed_inc = 0;
                }
            }
            if (count != tiles_per_packet || landed_inc != 0) {
                set_scatter<tile_size>(header, count, landed_inc);
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

    uint32_t landed = 0;
    while (true) {
        noc_async_writes_flushed();
        if (landed == 0) {
            if (cursor.done()) {
                break;
            }
            noc_semaphore_wait_min(ready_counter(ready_base, cursor.producer()), cursor.ready_target());
        }
        const uint32_t in_flight = read_batch();
        for (uint32_t n = landed; n > 0; --n) {
            send_packet(reinterpret_cast<uint32_t>(slots.header(send_slot)), slots.packet_bytes(send_slot));
            if constexpr (trailing_incs) {
                const uint32_t slot_bit = 1U << send_slot;
                if ((trailing_slots & slot_bit) != 0) {
                    trailing_slots &= ~slot_bit;
                    noc_async_writes_flushed();
                    inc_header->command_fields.unicast_seminc.noc_address = slot_trailing_inc[send_slot];
                    send_packet(reinterpret_cast<uint32_t>(inc_header), sizeof(PACKET_HEADER_TYPE));
                }
            }
            send_slot = next_slot(send_slot);
        }
        noc_async_read_barrier();
        if constexpr (stage_in_l1) {
            uint32_t slot = read_slot;
            for (uint32_t n = in_flight; n > 0; --n) {
                slot = slot == 0 ? slot_packets - 1 : slot - 1;
                const uint32_t slot_bit = 1U << slot;
                if ((taken_slots & slot_bit) == 0) {
                    continue;
                }
                taken_slots &= ~slot_bit;
                const uint32_t p = slot_producer[slot];
                volatile tt_l1_ptr uint32_t* taken = taken_count(ready_base, p);
                *taken = *taken + 1;
                noc_async_write(
                    reinterpret_cast<uint32_t>(taken),
                    get_noc_addr(producers[p].noc_x, producers[p].noc_y, taken_sem_addr),
                    sizeof(uint32_t));
            }
        }
        landed = in_flight;
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();
    tt::tt_fabric::common::experimental::close_connections(connection);
    noc_async_write_barrier();
}
