// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode layer boundary, fabric sender (BRISC) (fused_decode/boundary.py: DecodeBoundary).
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
    // mc: multi-core boundary (DecodeBoundary helpers): chunk c goes to helper c's receive slots (runtime args 5..:
    // helper noc x, y per chunk), each chunk with its own receive-semaphore increment; slot_addr is then the slot
    // offset in the helpers' receive buffer.
    constexpr uint32_t mc = get_compile_time_arg_val(9);
    // balance (mc, fwd_hops = bwd_hops + 1): the second half of the chunks takes the long way backward (hop counts
    // swapped), so both directions of a link carry the same bytes (each destination still gets every chunk once)
    constexpr uint32_t balance = get_compile_time_arg_val(10);
    constexpr uint32_t chunks = (payload + chunk - 1) / chunk;
    auto hx = [](uint32_t c) -> uint32_t { return get_arg_val<uint32_t>(5 + 2 * c); };
    auto hy = [](uint32_t c) -> uint32_t { return get_arg_val<uint32_t>(6 + 2 * c); };

    DeviceZoneScopedN("BND_S_SEND");
    size_t arg_idx = 5 + (mc ? 2 * chunks : 0);
    auto fabric = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx);

    // Packet headers: one per chunk when they fit this RISC's share of the header pool (all prepared while the
    // producers run); a multi-row payload has more chunks, so it cycles a ring of `ring_hdrs` headers, each rewritten
    // for its next chunk once the previous send out of it has left L1.
    PacketHeaderPool::reset();
    constexpr uint32_t my_chunks = (chunks - first_chunk + chunk_step - 1) / chunk_step;
    constexpr uint32_t ring_hdrs = my_chunks <= 4 ? my_chunks : 4;
    volatile PACKET_HEADER_TYPE* fwd_hdr[ring_hdrs];
    volatile PACKET_HEADER_TYPE* bwd_hdr[ring_hdrs];
    const uint64_t sem_noc0 = safe_get_noc_addr(recv_noc_x, recv_noc_y, sem_addr, 0);
    // Plain writes; only this sender's last chunk carries the receive-semaphore increment (packets of one route
    // arrive in order), so a receiver counts one increment per sender and source device (as ttnn's all-reduce does:
    // a fused atomic per packet costs more than the write).
    constexpr uint32_t last_chunk = first_chunk + (my_chunks - 1) * chunk_step;
    auto prepare = [&](uint32_t c, uint32_t h) {
        const uint32_t bytes = c + 1 < chunks ? chunk : payload - c * chunk;
        const uint64_t dst = mc ? safe_get_noc_addr(hx(c), hy(c), slot_addr, 0)
                                : safe_get_noc_addr(recv_noc_x, recv_noc_y, slot_addr + c * chunk, 0);
        const uint64_t sem_noc = mc ? safe_get_noc_addr(hx(c), hy(c), sem_addr, 0) : sem_noc0;
        const bool swap = balance && 2 * c >= chunks;
        if constexpr (fwd_hops > 0) {
            fwd_hdr[h]->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(swap ? bwd_hops : fwd_hops)});
            if (mc || c == last_chunk) {
                fwd_hdr[h]->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem_noc, 1, true}, bytes);
            } else {
                fwd_hdr[h]->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst}, bytes);
            }
        }
        if constexpr (bwd_hops > 0) {
            bwd_hdr[h]->to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(swap ? fwd_hops : bwd_hops)});
            if (mc || c == last_chunk) {
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
        prepare(first_chunk + h * chunk_step, h);
    }
    fabric.open_finish();
    if constexpr (producers > 0) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id)), producers);
    }
    if constexpr (local) {
        if constexpr (mc) {
            for (uint32_t c = 0; c < chunks; ++c) {
                const uint32_t bytes = c + 1 < chunks ? chunk : payload - c * chunk;
                noc_async_write(partial_addr + c * chunk, get_noc_addr(hx(c), hy(c), slot_addr), bytes);
            }
        } else {
            noc_async_write(partial_addr, get_noc_addr(slot_addr), payload);
        }
    }

    uint32_t i = 0;
    for (uint32_t c = first_chunk; c < chunks; c += chunk_step, ++i) {
        const uint32_t bytes = c + 1 < chunks ? chunk : payload - c * chunk;
        const uint32_t src = partial_addr + c * chunk;
        const uint32_t h = i % ring_hdrs;
        if (i >= ring_hdrs) {
            noc_async_writes_flushed();  // the header's previous packet has left L1
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

    if constexpr (local) {
        noc_async_write_barrier();
        if constexpr (mc) {
            for (uint32_t c = 0; c < chunks; ++c) {
                noc_semaphore_inc(get_noc_addr(hx(c), hy(c), sem_addr), 1);
            }
        } else {
            noc_semaphore_inc(get_noc_addr(sem_addr), 1);
        }
    }
    fabric.close();
    noc_async_atomic_barrier();
}
