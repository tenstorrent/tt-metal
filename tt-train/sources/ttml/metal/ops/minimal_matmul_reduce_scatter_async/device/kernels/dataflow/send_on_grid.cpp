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

namespace {
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

}  // namespace

void kernel_main() {
    Noc noc;
    constexpr uint32_t slots_cb = tt::CBIndex::c_4;
    constexpr uint32_t ready_cb = tt::CBIndex::c_6;
    constexpr uint32_t tile_size = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(1);
    constexpr uint32_t slot_packets = get_compile_time_arg_val(2);
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t slice_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t m_piece = get_compile_time_arg_val(6);
    constexpr uint32_t M_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t N_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t num_dirs = get_compile_time_arg_val(9);
    constexpr uint32_t dir = get_compile_time_arg_val(10);
    constexpr uint32_t ring_size = get_compile_time_arg_val(11);
    constexpr uint32_t go_semaphore_id = get_compile_time_arg_val(12);
    constexpr uint32_t packet_bytes = tiles_per_packet * tile_size;
    constexpr uint32_t packets_per_batch = slot_packets / 2;
    constexpr ccl_routing_utils::line_unicast_route_info_t route =
        ccl_routing_utils::get_line_unicast_route_info_from_args<13>();
    constexpr auto buffer_args = TensorAccessorArgs<13 + ccl_routing_utils::num_line_unicast_args>();

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
    constexpr uint32_t kAlign = NOC_DRAM_READ_ALIGNMENT_BYTES;
    const uint32_t slots_base = (CircularBuffer(slots_cb).get_write_ptr() + kAlign - 1) & ~(kAlign - 1);

    for (uint32_t p = 0; p < num_producers; p++) {
        *ready_counter(ready_base, p) = 0;
    }
    const uint32_t go_sem_addr = get_semaphore(go_semaphore_id);
    for (uint32_t p = 0; p < num_producers; p++) {
        noc_inline_dw_write(
            get_noc_addr(producers[p].noc_x, producers[p].noc_y, go_sem_addr), ready_base + (p * kReadyStrideBytes));
    }

    FabricWriter<tile_size, packet_size_bytes, tt::tt_fabric::WorkerToFabricEdmSender> fabric(
        noc, &connection.get(0).sender, route.dst_chip_id, route.dst_mesh_id);

    using Cursor = UnitCursor<ring_size, slice_tiles, m_piece, M_blocks, N_blocks, num_dirs, dir, tiles_per_packet>;
    Cursor reader(producers, num_producers, ring_index, N_tiles);
    Cursor sender(producers, num_producers, ring_index, N_tiles);
    uint32_t read_slot = 0;
    uint32_t send_slot = 0;
    uint32_t unsent = 0;

    auto read_batch = [&]() -> uint32_t {
        uint32_t issued = 0;
        while (!reader.done() && issued < packets_per_batch && unsent < slot_packets) {
            if (reader.unit_begin()) {
                invalidate_l1_cache();
                if (*ready_counter(ready_base, reader.producer()) < reader.ready_target()) {
                    break;
                }
            }
            uint32_t l1 = slots_base + (read_slot * packet_bytes);
            const uint32_t count = reader.count();
            for (uint32_t i = 0; i < count; ++i, l1 += tile_size) {
                noc.async_read(buffer, CoreLocalMem<uint32_t>(l1), tile_size, {.page_id = reader.page()}, {});
                reader.next_tile();
            }
            read_slot = (read_slot + 1) % slot_packets;
            ++unsent;
            ++issued;
            reader.next_packet();
        }
        return issued;
    };

    auto send_batch = [&](uint32_t n) {
        for (; n > 0; --n) {
            uint32_t l1 = slots_base + (send_slot * packet_bytes);
            const uint32_t count = sender.count();
            for (uint32_t i = 0; i < count; ++i, l1 += tile_size) {
                fabric.async_write(
                    l1, tt::tt_fabric::linear::addrgen_detail::get_noc_address(buffer, sender.page(), 0));
                sender.next_tile();
            }
            fabric.async_writes_flushed();
            if (sender.unit_end()) {
                const auto& producer = sender.producer_args();
                fabric.atomic_inc(
                    safe_get_noc_addr(
                        static_cast<uint8_t>(producer.noc_x), static_cast<uint8_t>(producer.noc_y), landed_sem_addr, 0),
                    1);
            }
            send_slot = (send_slot + 1) % slot_packets;
            --unsent;
            sender.next_packet();
        }
    };

    uint32_t landed = read_batch();
    noc.async_read_barrier();
    while (!sender.done()) {
        const uint32_t in_flight = read_batch();
        send_batch(landed);
        noc.async_read_barrier();
        landed = in_flight;
        if (landed == 0 && !sender.done()) {
            noc_semaphore_wait_min(ready_counter(ready_base, reader.producer()), reader.ready_target());
        }
    }

    noc.async_write_barrier();
    noc.async_atomic_barrier();
    tt::tt_fabric::common::experimental::close_connections(connection);
    noc.async_write_barrier();
}
