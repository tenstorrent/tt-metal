// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Gemma decode layer boundary, fabric sender + output writer (BRISC) (fused_decode/boundary.py: GemmaBoundary).
// The fabric part follows models/demos/gpt_oss/tt/fused_decode/kernels/boundary_sender.cpp.
//
// 1. (slots > 0) Publishes this device's flat partial (cb_part, `payload` bytes, gathered by bnd_reader.cpp) into
//    slot `ring index` of the receive buffer on every device of the 1-D line: locally by a NOC copy + one increment
//    of the local receive semaphore, remotely by a multicast of `fwd_hops` chips forward and `bwd_hops` chips
//    backward, `chunk` bytes per packet; the last packet of each direction is a fused write + semaphore increment
//    (packets of one route arrive in order), so the receiver counts one increment per device.
// 2. Writes the compute's flat outputs (cb_o0..2, `tiles` pages each) as row 0 of the consumers' TILE tensors
//    [1, 1, 32, hidden] (rows 1..31 are never written): BF16 (32 B face rows at bytes 0 / 512 of each 2 KB tile) or
//    FP32 (64 B face rows at bytes 0 / 1024 of each 4 KB tile).
//
// runtime args: [part_addr, slot_addr, recv_noc_x, recv_noc_y, sem_addr, out0_addr, out1_addr, out2_addr,
//                <fabric connection manager args>]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

void kernel_main() {
    const uint32_t part_addr = get_arg_val<uint32_t>(0);
    const uint32_t slot_addr = get_arg_val<uint32_t>(1);
    const uint32_t recv_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t recv_noc_y = get_arg_val<uint32_t>(3);
    const uint32_t sem_addr = get_arg_val<uint32_t>(4);
    const uint32_t out_addr[3] = {get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7)};

    constexpr uint32_t payload = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t fwd_hops = get_compile_time_arg_val(2);
    constexpr uint32_t bwd_hops = get_compile_time_arg_val(3);
    constexpr uint32_t slots = get_compile_time_arg_val(4);
    constexpr uint32_t cb_part = get_compile_time_arg_val(5);
    constexpr uint32_t part_pages = get_compile_time_arg_val(6);
    constexpr uint32_t tiles = get_compile_time_arg_val(7);
    constexpr uint32_t htiles = get_compile_time_arg_val(8);
    constexpr uint32_t outs = get_compile_time_arg_val(9);
    constexpr uint32_t cb_out0 = get_compile_time_arg_val(10);
    constexpr uint32_t fp32_mask = get_compile_time_arg_val(11);  // bit o: output o is FP32
    constexpr auto o0_args = TensorAccessorArgs<12>();
    constexpr auto o1_args = TensorAccessorArgs<o0_args.next_compile_time_args_offset()>();
    constexpr auto o2_args = TensorAccessorArgs<o1_args.next_compile_time_args_offset()>();
    constexpr uint32_t chunks = (payload + chunk - 1) / chunk;

    if constexpr (slots > 0) {
        size_t arg_idx = 8;
        auto fabric = FabricConnectionManager::build_from_args<
            FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx);
        PacketHeaderPool::reset();
        constexpr uint32_t ring_hdrs = chunks <= 4 ? chunks : 4;
        volatile PACKET_HEADER_TYPE* fwd_hdr[ring_hdrs];
        volatile PACKET_HEADER_TYPE* bwd_hdr[ring_hdrs];
        const uint64_t sem_noc = safe_get_noc_addr(recv_noc_x, recv_noc_y, sem_addr, 0);
        auto prepare = [&](uint32_t c, uint32_t h) {
            const uint32_t bytes = c + 1 < chunks ? chunk : payload - c * chunk;
            const uint64_t dst = safe_get_noc_addr(recv_noc_x, recv_noc_y, slot_addr + c * chunk, 0);
            if constexpr (fwd_hops > 0) {
                fwd_hdr[h]->to_chip_multicast(
                    tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(fwd_hops)});
                if (c + 1 == chunks) {
                    fwd_hdr[h]->to_noc_fused_unicast_write_atomic_inc(
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem_noc, 1, true}, bytes);
                } else {
                    fwd_hdr[h]->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst}, bytes);
                }
            }
            if constexpr (bwd_hops > 0) {
                bwd_hdr[h]->to_chip_multicast(
                    tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(bwd_hops)});
                if (c + 1 == chunks) {
                    bwd_hdr[h]->to_noc_fused_unicast_write_atomic_inc(
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem_noc, 1, true}, bytes);
                } else {
                    bwd_hdr[h]->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst}, bytes);
                }
            }
        };
        for (uint32_t h = 0; h < ring_hdrs; ++h) {
            if constexpr (fwd_hops > 0) {
                fwd_hdr[h] = PacketHeaderPool::allocate_header();
            }
            if constexpr (bwd_hops > 0) {
                bwd_hdr[h] = PacketHeaderPool::allocate_header();
            }
            prepare(h, h);
        }
        fabric.open_finish();
        cb_wait_front(cb_part, part_pages);
        noc_async_write(part_addr, get_noc_addr(slot_addr), payload);
        for (uint32_t c = 0; c < chunks; ++c) {
            const uint32_t bytes = c + 1 < chunks ? chunk : payload - c * chunk;
            const uint32_t src = part_addr + c * chunk;
            const uint32_t h = c % ring_hdrs;
            if (c >= ring_hdrs) {
                noc_async_writes_flushed();
                prepare(c, h);
            }
            if constexpr (fwd_hops > 0) {
                auto& conn = fabric.get_forward_connection();
                conn.wait_for_empty_write_slot();
                conn.send_payload_without_header_non_blocking_from_address(src, bytes);
                conn.send_payload_flush_non_blocking_from_address((uint32_t)fwd_hdr[h], sizeof(PACKET_HEADER_TYPE));
            }
            if constexpr (bwd_hops > 0) {
                auto& conn = fabric.get_backward_connection();
                conn.wait_for_empty_write_slot();
                conn.send_payload_without_header_non_blocking_from_address(src, bytes);
                conn.send_payload_flush_non_blocking_from_address((uint32_t)bwd_hdr[h], sizeof(PACKET_HEADER_TYPE));
            }
        }
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(sem_addr), 1);
        cb_pop_front(cb_part, part_pages);
        fabric.close();
        noc_async_atomic_barrier();
    }

    // Outputs: flat -> row 0 of the consumer's tiles.
    auto write_rows = [&](const auto& dst, uint32_t cb, bool fp32) {
        cb_wait_front(cb, tiles);
        const uint32_t src = get_read_ptr(cb);
        if (fp32) {
            for (uint32_t j = 0; j < htiles; ++j) {
                noc_async_write(src + 128 * j, dst.get_noc_addr(j, 0), 64);
                noc_async_write(src + 128 * j + 64, dst.get_noc_addr(j, 1024), 64);
            }
        } else {
            for (uint32_t j = 0; j < htiles; ++j) {
                noc_async_write(src + 64 * j, dst.get_noc_addr(j, 0), 32);
                noc_async_write(src + 64 * j + 32, dst.get_noc_addr(j, 512), 32);
            }
        }
        noc_async_writes_flushed();
    };
    if constexpr (outs > 0) {
        const auto d0 = TensorAccessor(o0_args, out_addr[0], (fp32_mask & 1) ? 4096 : 2048);
        write_rows(d0, cb_out0, fp32_mask & 1);
    }
    if constexpr (outs > 1) {
        const auto d1 = TensorAccessor(o1_args, out_addr[1], (fp32_mask & 2) ? 4096 : 2048);
        write_rows(d1, cb_out0 + 1, fp32_mask & 2);
    }
    if constexpr (outs > 2) {
        const auto d2 = TensorAccessor(o2_args, out_addr[2], (fp32_mask & 4) ? 4096 : 2048);
        write_rows(d2, cb_out0 + 2, fp32_mask & 4);
    }
    noc_async_write_barrier();
}
