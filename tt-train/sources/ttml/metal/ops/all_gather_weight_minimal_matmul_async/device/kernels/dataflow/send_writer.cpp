// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "cpp/ttnn/operations/ccl/all_gather/device/kernels/unicast_common.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "dataflow_common.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t tile_size = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(2);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t stripe_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t ring_size = get_compile_time_arg_val(6);
    constexpr bool transpose_b = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t packet_size_bytes = get_compile_time_arg_val(8);
    constexpr ccl_routing_utils::line_unicast_route_info_t route =
        ccl_routing_utils::get_line_unicast_route_info_from_args<9>();
    constexpr auto pwb_args = TensorAccessorArgs<9 + ccl_routing_utils::num_line_unicast_args>();

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

    FabricWriter<tile_size, packet_size_bytes, tt::tt_fabric::FabricMuxV2Sender<false>> fabric(
        noc, &mux, route.dst_chip_id, route.dst_mesh_id);

    auto send_inc = [&](uint32_t noc_x, uint32_t noc_y, uint32_t addr) {
        fabric.atomic_inc(safe_get_noc_addr(static_cast<uint8_t>(noc_x), static_cast<uint8_t>(noc_y), addr, 0), 1);
        fabric.async_writes_flushed();
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
            uint32_t l1 = cb.get_read_ptr();
            for (uint32_t i = 0; i < count; ++i, l1 += tile_size) {
                const uint32_t t = t0 + i;
                const uint32_t row = (seg.stripe * stripe_tiles) + piece.row_offset + (t / chunk);
                const uint32_t page =
                    weight_page<transpose_b>(row, k_start + (t % chunk), K_tiles, stripe_tiles * ring_size);
                fabric.async_write(l1, tt::tt_fabric::linear::addrgen_detail::get_noc_address(pwb, page, 0));
            }
            fabric.async_writes_flushed();
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
