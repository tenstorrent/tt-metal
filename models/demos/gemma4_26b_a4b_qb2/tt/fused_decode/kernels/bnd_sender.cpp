// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Gemma decode layer boundary, fabric sender + output writer (BRISC) (fused_decode/boundary.py: GemmaBoundary).
// The fabric part follows models/demos/gpt_oss/tt/fused_decode/kernels/boundary_sender.cpp.
//
// 0. (slots > 0, not pregathered) Gathers row 0 of tiles [0, htiles / 2) of the producer outputs into the flat partial
//    (bnd_reader.cpp gathers the rest and pushes cb_sync).
// 1. (slots > 0) Multicasts this device's flat partial (`payload` bytes at part_addr) to the other devices: forward
//    into slot me, backward into slot me - 1 (three slots per site; the own partial stays in the partial buffer).
//    Formerly published into
//    slot `ring index` of the receive buffer on every device of the 1-D line: locally by a NOC copy + one increment
//    of the local receive semaphore, remotely by a multicast of `fwd_hops` chips forward and `bwd_hops` chips
//    backward, `chunk` bytes per packet; the last packet of each direction is a fused write + semaphore increment
//    (packets of one route arrive in order), so the receiver counts one increment per device.
// 2. Writes tiles [0, htiles / 2) (bnd_reader.cpp the rest) of the compute's flat outputs (cb_o0..2) as row 0 of the
//    consumers' TILE tensors
//    [1, 1, 32, hidden] (rows 1..31 are never written): BF16 (32 B face rows at bytes 0 / 512 of each 2 KB tile) or
//    FP32 (64 B face rows at bytes 0 / 1024 of each 4 KB tile).
//
// runtime args: [part_addr, slot_addr, recv_noc_x, recv_noc_y, sem_addr, out0_addr, out1_addr, out2_addr, p0_addr,
//                p1_addr, (notify_x, notify_y) x notify, <fabric connection manager args>]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "api/debug/kernel_profiler.hpp"
#include "bnd_rows.hpp"

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
    constexpr uint32_t cb_sync = get_compile_time_arg_val(o2_args.next_compile_time_args_offset());
    constexpr uint32_t pregathered = get_compile_time_arg_val(o2_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t parts = get_compile_time_arg_val(o2_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t me = get_compile_time_arg_val(o2_args.next_compile_time_args_offset() + 3);  // ring index
    constexpr uint32_t ring = get_compile_time_arg_val(o2_args.next_compile_time_args_offset() + 4);
    constexpr auto p0_args = TensorAccessorArgs<o2_args.next_compile_time_args_offset() + 5>();
    constexpr auto p1_args = TensorAccessorArgs<p0_args.next_compile_time_args_offset()>();
    // a program joined on other cores (RouterB1): increment semaphore `ready_sem` on `notify` cores once flat output 0
    // (cb_flat0) is packed
    constexpr uint32_t cb_flat0 = get_compile_time_arg_val(p1_args.next_compile_time_args_offset());
    constexpr uint32_t notify = get_compile_time_arg_val(p1_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t ready_sem = get_compile_time_arg_val(p1_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t chunks = (payload + chunk - 1) / chunk;
    constexpr uint32_t half = htiles / 2;

    if constexpr (slots > 0) {
        size_t arg_idx = 10 + 2 * notify;  // after the notify NOC coords
        auto fabric = FabricConnectionManager::build_from_args<
            FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx);
        PacketHeaderPool::reset();
        volatile PACKET_HEADER_TYPE* fwd_hdr[2];
        volatile PACKET_HEADER_TYPE* bwd_hdr[2];
        for (uint32_t h = 0; h < 2; ++h) {
            if constexpr (fwd_hops > 0) {
                fwd_hdr[h] = PacketHeaderPool::allocate_header();
            }
            if constexpr (bwd_hops > 0) {
                bwd_hdr[h] = PacketHeaderPool::allocate_header();
            }
        }
        fabric.open_finish();
        if constexpr (!pregathered) {
            gather_rows(TensorAccessor(p0_args, get_arg_val<uint32_t>(8), 2048), part_addr, 0, half);
            if constexpr (parts > 1) {
                gather_rows(TensorAccessor(p1_args, get_arg_val<uint32_t>(9), 2048), part_addr + tiles * 2048, 0, half);
            }
            noc_async_read_barrier();
            cb_wait_front(cb_sync, 1);
        }
        // the own partial is complete: the compute reads it straight from the partial buffer
        cb_reserve_back(cb_part, part_pages);
        cb_push_back(cb_part, part_pages);
        // Multicast forward (to the ring - 1 - me devices after this one) into slot me, and backward (to the me devices
        // before it) into slot me - 1: receiver r then holds device g's partial in slot g (g < r) or g - 1 (g > r),
        // three slots, no two senders on one slot. The last packet of each direction increments the semaphores.
        const uint64_t sem_noc = safe_get_noc_addr(recv_noc_x, recv_noc_y, sem_addr, 0);
        auto send = [&](auto& conn, volatile PACKET_HEADER_TYPE** hdr, uint32_t hops, uint32_t slot) {
            for (uint32_t c = 0; c < chunks; ++c) {
                const uint32_t bytes = c + 1 < chunks ? chunk : payload - c * chunk;
                volatile PACKET_HEADER_TYPE* h = hdr[c & 1];
                if (c >= 2) {
                    noc_async_writes_flushed();  // the header's previous packet has left L1
                }
                h->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(hops)});
                const uint64_t dst =
                    safe_get_noc_addr(recv_noc_x, recv_noc_y, slot_addr + slot * payload + c * chunk, 0);
                if (c + 1 == chunks) {
                    h->to_noc_fused_unicast_write_atomic_inc(
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem_noc, 1, true}, bytes);
                } else {
                    h->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_payload_without_header_non_blocking_from_address(part_addr + c * chunk, bytes);
                conn.send_payload_flush_non_blocking_from_address((uint32_t)h, sizeof(PACKET_HEADER_TYPE));
            }
        };
        if constexpr (fwd_hops > 0) {
            send(fabric.get_forward_connection(), fwd_hdr, fwd_hops, me);
        }
        if constexpr (bwd_hops > 0) {
            send(fabric.get_backward_connection(), bwd_hdr, bwd_hops, me - 1);
        }
        noc_async_writes_flushed();
        fabric.close();
        noc_async_atomic_barrier();
    } else if constexpr (!pregathered) {
        (void)cb_part;
    }

    if constexpr (notify > 0) {
        // the joined program's cores read flat output 0 from this core once their semaphore is raised
        cb_wait_front(cb_flat0, tiles);
        const uint32_t sem_l1 = get_semaphore(ready_sem);
        for (uint32_t i = 0; i < notify; ++i) {
            const uint32_t x = get_arg_val<uint32_t>(10 + 2 * i);
            const uint32_t y = get_arg_val<uint32_t>(11 + 2 * i);
            noc_semaphore_inc(get_noc_addr(x, y, sem_l1), 1);
        }
        noc_async_atomic_barrier();
    }
    // Outputs: flat -> row 0 of the consumer's tiles [0, half) (bnd_reader.cpp writes the rest).
    if constexpr (outs > 0) {
        DeviceZoneScopedN("BND_WRITE");
        cb_wait_front(cb_out0, tiles);
        write_rows(
            TensorAccessor(o0_args, out_addr[0], (fp32_mask & 1) ? 4096 : 2048),
            get_read_ptr(cb_out0),
            0,
            half,
            fp32_mask & 1);
    }
    if constexpr (outs > 1) {
        cb_wait_front(cb_out0 + 1, tiles);
        write_rows(
            TensorAccessor(o1_args, out_addr[1], (fp32_mask & 2) ? 4096 : 2048),
            get_read_ptr(cb_out0 + 1),
            0,
            half,
            fp32_mask & 2);
    }
    if constexpr (outs > 2) {
        cb_wait_front(cb_out0 + 2, tiles);
        write_rows(
            TensorAccessor(o2_args, out_addr[2], (fp32_mask & 4) ? 4096 : 2048),
            get_read_ptr(cb_out0 + 2),
            0,
            half,
            fp32_mask & 4);
    }
    noc_async_write_barrier();
}
