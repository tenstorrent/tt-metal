// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode layer boundary, fabric sender (BRISC) (tt/decode_boundary.py: DecodeBoundary).
//
// Publishes this device's flat partial sum (the row-parallel o_proj / MoE down output, `payload` bytes in the local
// partial buffer) into slot `slot` of the receive buffer on every device of the TP ring:
//   - locally by a NOC copy, followed by one increment of the local receive semaphore;
//   - to the other devices through the fabric: a multicast of `fwd_hops` chips on the forward connection and of
//     `bwd_hops` chips on the backward connection, `chunk` bytes per packet, each packet a fused write + atomic
//     increment of the receive semaphore (same core and addresses on every device).
// Every device then holds all TP partial sums in slot order, and the receiver sums them in the same order.
// producers > 0 (fused into the producer op, DecodeBoundary.sending_program): the fabric connections open while the
// producers run; the partial is sent once `producers` cores have incremented program semaphore `sem_id`.
// With two fabric links, two senders (BRISC on link 0, NCRISC on link 1) send chunks first_chunk, first_chunk +
// chunk_step, ...; only the one with `local` makes the local copy.
//
// runtime args: [partial_addr, slot_addr, recv_noc_x, recv_noc_y, sem_addr, <fabric connection manager args>]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

void kernel_main() {
    const uint32_t partial_addr = get_arg_val<uint32_t>(0);
    const uint32_t slot_addr = get_arg_val<uint32_t>(1);
    const uint32_t recv_noc_x = get_arg_val<uint32_t>(2);
    const uint32_t recv_noc_y = get_arg_val<uint32_t>(3);
    const uint32_t sem_addr = get_arg_val<uint32_t>(4);

    constexpr uint32_t payload = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t fwd_hops = get_compile_time_arg_val(2);
    constexpr uint32_t bwd_hops = get_compile_time_arg_val(3);
    constexpr uint32_t producers = get_compile_time_arg_val(4);
    constexpr uint32_t sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t first_chunk = get_compile_time_arg_val(6);
    constexpr uint32_t chunk_step = get_compile_time_arg_val(7);
    constexpr uint32_t local = get_compile_time_arg_val(8);
    constexpr uint32_t chunks = (payload + chunk - 1) / chunk;

    DeviceZoneScopedN("BND_S_SEND");
    size_t arg_idx = 5;
    auto fabric = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx);

    PacketHeaderPool::reset();
    volatile PACKET_HEADER_TYPE* fwd_hdr[chunks];
    volatile PACKET_HEADER_TYPE* bwd_hdr[chunks];
    const uint64_t sem_noc = safe_get_noc_addr(recv_noc_x, recv_noc_y, sem_addr, 0);
    for (uint32_t c = first_chunk; c < chunks; c += chunk_step) {
        const uint32_t bytes = c + 1 < chunks ? chunk : payload - c * chunk;
        const uint64_t dst = safe_get_noc_addr(recv_noc_x, recv_noc_y, slot_addr + c * chunk, 0);
        if constexpr (fwd_hops > 0) {
            fwd_hdr[c] = PacketHeaderPool::allocate_header();
            fwd_hdr[c]->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(fwd_hops)});
            fwd_hdr[c]->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem_noc, 1, true}, bytes);
        }
        if constexpr (bwd_hops > 0) {
            bwd_hdr[c] = PacketHeaderPool::allocate_header();
            bwd_hdr[c]->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(bwd_hops)});
            bwd_hdr[c]->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem_noc, 1, true}, bytes);
        }
    }
    fabric.open_finish();
    if constexpr (producers > 0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id)), producers);
    }
    if constexpr (local) {
        noc_async_write(partial_addr, get_noc_addr(slot_addr), payload);
    }

    for (uint32_t c = first_chunk; c < chunks; c += chunk_step) {
        const uint32_t bytes = c + 1 < chunks ? chunk : payload - c * chunk;
        const uint32_t src = partial_addr + c * chunk;
        if constexpr (fwd_hops > 0) {
            auto& conn = fabric.get_forward_connection();
            conn.wait_for_empty_write_slot();
            conn.send_payload_without_header_non_blocking_from_address(src, bytes);
            conn.send_payload_flush_non_blocking_from_address((uint32_t)fwd_hdr[c], sizeof(PACKET_HEADER_TYPE));
        }
        if constexpr (bwd_hops > 0) {
            auto& conn = fabric.get_backward_connection();
            conn.wait_for_empty_write_slot();
            conn.send_payload_without_header_non_blocking_from_address(src, bytes);
            conn.send_payload_flush_non_blocking_from_address((uint32_t)bwd_hdr[c], sizeof(PACKET_HEADER_TYPE));
        }
    }

    if constexpr (local) {
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(sem_addr), 1);
    }
    fabric.close();
    noc_async_atomic_barrier();
}
