// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "dataflow_common.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"

using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(2);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t stripe_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t ring_size = get_compile_time_arg_val(6);
    constexpr bool transpose_b = get_compile_time_arg_val(7) == 1;
    constexpr ccl_routing_utils::line_unicast_route_info_t route =
        ccl_routing_utils::get_line_unicast_route_info_from_args<8>();
    constexpr auto pwb_args = TensorAccessorArgs<8 + ccl_routing_utils::num_line_unicast_args>();

    size_t argidx = 0;
    const uint32_t pwb_addr = get_arg_val<uint32_t>(argidx++);
    uint32_t stripe_sem_addrs[ring_size];
    for (uint32_t i = 0; i < ring_size; i++) {
        stripe_sem_addrs[i] = get_arg_val<uint32_t>(argidx++);
    }
    const uint32_t peer_sender_noc_x = get_arg_val<uint32_t>(argidx++);
    const uint32_t peer_sender_noc_y = get_arg_val<uint32_t>(argidx++);
    auto mux = tt::tt_fabric::FabricMuxV2Sender<false>::build_from_args(argidx);
    const uint32_t schedule_arg_idx = argidx;

    const auto pwb = TensorAccessor(pwb_args, pwb_addr, tile_size);
    Noc noc;
    CircularBuffer cb(cb_id);

    auto* scatter_hdr = PacketHeaderPool::allocate_header();
    auto* unicast_hdr = PacketHeaderPool::allocate_header();
    auto* inc_hdr = PacketHeaderPool::allocate_header();
    uint64_t dummy_addrs[NOC_SCATTER_WRITE_MAX_CHUNKS] = {0, 0, 0, 0};
    uint16_t chunk_sizes[NOC_SCATTER_WRITE_MAX_CHUNKS - 1] = {
        static_cast<uint16_t>(tile_size), static_cast<uint16_t>(tile_size), static_cast<uint16_t>(tile_size)};
    fabric_unicast_noc_scatter_write_set_state<
        UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
        scatter_hdr,
        static_cast<uint8_t>(route.distance_in_hops),
        NocUnicastScatterCommandHeader(dummy_addrs, chunk_sizes, tiles_per_packet),
        tile_size * tiles_per_packet);
    fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
        unicast_hdr, static_cast<uint8_t>(route.distance_in_hops), nullptr, tile_size);
    fabric_unicast_noc_unicast_atomic_inc_set_state<
        UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
        inc_hdr,
        static_cast<uint8_t>(route.distance_in_hops),
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{0, static_cast<uint16_t>(1)});
    ccl_routing_utils::fabric_set_line_unicast_route(scatter_hdr, route);
    ccl_routing_utils::fabric_set_line_unicast_route(unicast_hdr, route);
    ccl_routing_utils::fabric_set_line_unicast_route(inc_hdr, route);

    auto send_inc = [&](uint32_t noc_x, uint32_t noc_y, uint32_t addr) {
        const uint64_t sem_noc_addr =
            safe_get_noc_addr(static_cast<uint8_t>(noc_x), static_cast<uint8_t>(noc_y), addr, 0);
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            &mux, inc_hdr, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{sem_noc_addr, 0, 0});
        noc.async_writes_flushed();
    };

    mux.open();

    for_each_send_packet(
        schedule_arg_idx,
        K_tiles,
        chunk_tiles,
        tiles_per_packet,
        [](const SendSegment&, const SendPieceArgs&, uint32_t, uint32_t) {},
        [&](const SendSegment& seg,
            const SendPieceArgs& piece,
            uint32_t t0,
            uint32_t count,
            uint32_t k_start,
            uint32_t chunk) {
            cb.wait_front(tiles_per_packet);
            const uint32_t l1 = cb.get_read_ptr();
            uint64_t dst[NOC_SCATTER_WRITE_MAX_CHUNKS] = {0, 0, 0, 0};
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t t = t0 + i;
                const uint32_t row = (seg.stripe * stripe_tiles) + piece.row_offset + (t / chunk);
                const uint32_t page =
                    weight_page<transpose_b>(row, k_start + (t % chunk), K_tiles, stripe_tiles * ring_size);
                dst[i] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(pwb, page, 0);
            }
            if (count > 1) {
                fabric_unicast_noc_scatter_write_with_state<
                    UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                    UnicastScatterWriteUpdateMask::PayloadSize>(
                    &mux, scatter_hdr, l1, NocUnicastScatterCommandHeader(dst, chunk_sizes, count), tile_size * count);
            } else {
                fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                    &mux, unicast_hdr, l1, NocUnicastCommandHeader{dst[0]});
            }
            noc.async_writes_flushed();
            cb.pop_front(tiles_per_packet);
        },
        [&](const SendSegment& seg, const SendPieceArgs& piece, uint32_t, uint32_t) {
            send_inc(piece.inj_noc_x, piece.inj_noc_y, stripe_sem_addrs[seg.stripe]);
            if (piece.notify) {
                send_inc(peer_sender_noc_x, peer_sender_noc_y, stripe_sem_addrs[seg.stripe]);
            }
        });

    noc.async_write_barrier();
    mux.close();
}
