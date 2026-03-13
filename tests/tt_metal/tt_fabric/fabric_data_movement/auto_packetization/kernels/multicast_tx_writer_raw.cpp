// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unified multicast TX kernel for auto-packetization integration tests.
// Dispatches to the correct fabric multicast API based on the send_op RT arg.
//
// RT args:
//   0: src_l1_addr, 1: total_size, 2: dst_base_addr,
//   3: rx_noc_x, 4: rx_noc_y, 5: sem_l1_addr
//   6: scatter_offset (u32, 0 for non-scatter ops)
//   7: send_op        (u32, one of TX_OP_*)
//   [FABRIC_2D] next: dir_mask, senderW/E/N/S args, e/w/n/s_hops
//   [!FABRIC_2D] next: start_distance, range, sender args

#include "tx_kernel_common.h"

void kernel_main() {
    size_t idx = 0;
    const uint32_t src_l1_addr   = get_arg_val<uint32_t>(idx++);
    const uint32_t total_size    = get_arg_val<uint32_t>(idx++);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_x      = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y      = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr   = get_arg_val<uint32_t>(idx++);
    const uint32_t scatter_offset = get_arg_val<uint32_t>(idx++);
    const uint32_t send_op        = get_arg_val<uint32_t>(idx++);

    const uint64_t dst_noc_addr  = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr, 0);
    const uint64_t dst_noc_addr1 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr + scatter_offset, 0);
    const uint64_t sem_noc_addr  = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, 0);
    const uint16_t scatter_chunk_size = static_cast<uint16_t>(total_size / 2);

#ifdef FABRIC_2D
    // ===== 2D Mesh Mode =====
    const uint32_t dir_mask = get_arg_val<uint32_t>(idx++);
    const bool hasW = (dir_mask & 0x1u) != 0;
    const bool hasE = (dir_mask & 0x2u) != 0;
    const bool hasN = (dir_mask & 0x4u) != 0;
    const bool hasS = (dir_mask & 0x8u) != 0;

    WorkerToFabricEdmSender senderW{}, senderE{}, senderN{}, senderS{};
    if (hasW) { senderW = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx); }
    if (hasE) { senderE = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx); }
    if (hasN) { senderN = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx); }
    if (hasS) { senderS = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx); }

    const uint16_t e_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t w_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t n_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t s_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));

    volatile tt_l1_ptr PACKET_HEADER_TYPE* left_packet_header  = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* right_packet_header = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* north_packet_header = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* south_packet_header = PacketHeaderPool::allocate_header();

    if (hasW) { senderW.open<true>(); }
    if (hasE) { senderE.open<true>(); }
    if (hasN) { senderN.open<true>(); }
    if (hasS) { senderS.open<true>(); }

    auto send_dir = [&](
        uint16_t hops,
        WorkerToFabricEdmSender& s,
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr,
        MeshMcastRange ranges) {
        if (hops == 0) return;
        if (send_op == TX_OP_WRITE) {
            fabric_multicast_noc_unicast_write(
                &s, hdr, 0, 0, ranges, src_l1_addr, total_size,
                tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr});
        } else if (send_op == TX_OP_SCATTER_WRITE) {
            fabric_multicast_noc_scatter_write(
                &s, hdr, 0, 0, ranges, src_l1_addr, total_size,
                tt::tt_fabric::NocUnicastScatterCommandHeader{{dst_noc_addr, dst_noc_addr1}, {scatter_chunk_size}});
        } else if (send_op == TX_OP_FUSED_ATOMIC_INC) {
            fabric_multicast_noc_fused_unicast_with_atomic_inc(
                &s, hdr, 0, 0, ranges, src_l1_addr, total_size,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, 1, true});
        } else {  // TX_OP_FUSED_SCATTER_ATOMIC_INC
            fabric_multicast_noc_fused_scatter_write_atomic_inc(
                &s, hdr, 0, 0, ranges, src_l1_addr, total_size,
                tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                    {dst_noc_addr, dst_noc_addr1}, sem_noc_addr, {scatter_chunk_size}, 1, true});
        }
    };

    send_dir(w_hops, senderW, left_packet_header,  {0, static_cast<uint8_t>(w_hops), 0, 0});
    send_dir(e_hops, senderE, right_packet_header, {static_cast<uint8_t>(e_hops), 0, 0, 0});
    send_dir(n_hops, senderN, north_packet_header, {static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), static_cast<uint8_t>(n_hops), 0});
    send_dir(s_hops, senderS, south_packet_header, {static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), 0, static_cast<uint8_t>(s_hops)});

    // Non-fused ops need a separate completion atomic_inc per direction
    if (send_op == TX_OP_WRITE || send_op == TX_OP_SCATTER_WRITE) {
        noc_async_writes_flushed();
        auto send_completion = [&](
            uint16_t hops, WorkerToFabricEdmSender& s,
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr,
            uint16_t e_r, uint16_t w_r, uint16_t n_r, uint16_t s_r) {
            if (hops == 0) return;
            fabric_set_mcast_route(hdr, 0, 0, e_r, w_r, n_r, s_r);
            hdr->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc_addr, 1, 32));
            s.wait_for_empty_write_slot();
            s.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
        };
        send_completion(w_hops, senderW, left_packet_header,  0,      w_hops, 0,      0);
        send_completion(e_hops, senderE, right_packet_header, e_hops, 0,      0,      0);
        send_completion(n_hops, senderN, north_packet_header, e_hops, w_hops, n_hops, 0);
        send_completion(s_hops, senderS, south_packet_header, e_hops, w_hops, 0,      s_hops);
    }

    if (hasW) { senderW.close(); }
    if (hasE) { senderE.close(); }
    if (hasN) { senderN.close(); }
    if (hasS) { senderS.close(); }

#else
    // ===== 1D Linear Mode =====
    const uint8_t start_distance = static_cast<uint8_t>(get_arg_val<uint32_t>(idx++));
    const uint8_t range          = static_cast<uint8_t>(get_arg_val<uint32_t>(idx++));

    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::allocate_header();
    sender.open<true>();

    if (send_op == TX_OP_WRITE) {
        fabric_multicast_noc_unicast_write(
            &sender, packet_header, src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, start_distance, range);
        noc_async_writes_flushed();
        fabric_multicast_noc_unicast_atomic_inc(
            &sender, packet_header,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_addr, 1, 32),
            start_distance, range);
    } else if (send_op == TX_OP_SCATTER_WRITE) {
        fabric_multicast_noc_scatter_write(
            &sender, packet_header, src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterCommandHeader{{dst_noc_addr, dst_noc_addr1}, {scatter_chunk_size}},
            start_distance, range);
        noc_async_writes_flushed();
        fabric_multicast_noc_unicast_atomic_inc(
            &sender, packet_header,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_addr, 1, 32),
            start_distance, range);
    } else if (send_op == TX_OP_FUSED_ATOMIC_INC) {
        fabric_multicast_noc_fused_unicast_with_atomic_inc(
            &sender, packet_header, src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, 1, true},
            start_distance, range);
    } else {  // TX_OP_FUSED_SCATTER_ATOMIC_INC
        fabric_multicast_noc_fused_scatter_write_atomic_inc(
            &sender, packet_header, src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                {dst_noc_addr, dst_noc_addr1}, sem_noc_addr, {scatter_chunk_size}, 1, true},
            start_distance, range);
    }

    sender.close();
#endif
}
