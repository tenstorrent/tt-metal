// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Multicast scatter sender kernel for auto-packetization integration tests.
// Sends a scatter payload via fabric_multicast_noc_scatter_write per active
// direction, then sends a separate atomic_inc per direction for completion.
// Scatter is passthrough (no chunking) -- payloads must fit in a single packet.
//
// RT args:
//   0: src_l1_addr         (u32) - L1 address of source data
//   1: total_size           (u32) - total bytes to send (must be <= FABRIC_MAX_PACKET_SIZE)
//   2: dst_base_addr        (u32) - destination buffer base address
//   3: rx_noc_x             (u32) - receiver worker NOC X coordinate
//   4: rx_noc_y             (u32) - receiver worker NOC Y coordinate
//   5: sem_l1_addr          (u32) - receiver semaphore L1 address
//   6: dir_mask             (u32) - bitmask: bit0=W, bit1=E, bit2=N, bit3=S
//   7: scatter_offset       (u32) - offset for second scatter destination
//   ... per-direction fabric connection args (W, E, N, S order) ...
//   ... then: e_hops, w_hops, n_hops, s_hops (u32 each)

#include "tx_kernel_common.h"

void kernel_main() {
    size_t idx = 0;
    const uint32_t src_l1_addr     = get_arg_val<uint32_t>(idx++);
    const uint32_t total_size      = get_arg_val<uint32_t>(idx++);
    const uint32_t dst_base_addr   = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_x        = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y        = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr     = get_arg_val<uint32_t>(idx++);

    const uint64_t dst_noc_addr0 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr, 0);
    const uint64_t sem_noc_addr  = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, 0);

#ifdef FABRIC_2D
    // ===== 2D Mesh Mode =====
    const uint32_t dir_mask        = get_arg_val<uint32_t>(idx++);
    const uint32_t scatter_offset  = get_arg_val<uint32_t>(idx++);

    const bool hasW = (dir_mask & 0x1u) != 0;
    const bool hasE = (dir_mask & 0x2u) != 0;
    const bool hasN = (dir_mask & 0x4u) != 0;
    const bool hasS = (dir_mask & 0x8u) != 0;

    // Build per-direction fabric senders in fixed order: W, E, N, S
    WorkerToFabricEdmSender senderW{};
    WorkerToFabricEdmSender senderE{};
    WorkerToFabricEdmSender senderN{};
    WorkerToFabricEdmSender senderS{};

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

    // Compute NOC addresses for scatter: two destinations
    const uint64_t dst_noc_addr1 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr + scatter_offset, 0);

    const uint16_t scatter_chunk_size = static_cast<uint16_t>(total_size / 2);

    // Send scatter payload in each active direction
    if (w_hops > 0) {
        MeshMcastRange ranges_w{0, static_cast<uint8_t>(w_hops), 0, 0};
        fabric_multicast_noc_scatter_write(
            &senderW, left_packet_header, 0, 0, ranges_w,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterCommandHeader{{dst_noc_addr0, dst_noc_addr1}, {scatter_chunk_size}});
    }
    if (e_hops > 0) {
        MeshMcastRange ranges_e{static_cast<uint8_t>(e_hops), 0, 0, 0};
        fabric_multicast_noc_scatter_write(
            &senderE, right_packet_header, 0, 0, ranges_e,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterCommandHeader{{dst_noc_addr0, dst_noc_addr1}, {scatter_chunk_size}});
    }
    if (n_hops > 0) {
        MeshMcastRange ranges_n{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops),
            static_cast<uint8_t>(n_hops), 0};
        fabric_multicast_noc_scatter_write(
            &senderN, north_packet_header, 0, 0, ranges_n,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterCommandHeader{{dst_noc_addr0, dst_noc_addr1}, {scatter_chunk_size}});
    }
    if (s_hops > 0) {
        MeshMcastRange ranges_s{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops),
            0, static_cast<uint8_t>(s_hops)};
        fabric_multicast_noc_scatter_write(
            &senderS, south_packet_header, 0, 0, ranges_s,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterCommandHeader{{dst_noc_addr0, dst_noc_addr1}, {scatter_chunk_size}});
    }

    noc_async_writes_flushed();

    // Scatter is NOT fused -- send separate atomic_inc per direction
    auto send_completion = [&](
        uint16_t hops,
        WorkerToFabricEdmSender& sender_dir,
        volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr,
        uint16_t e_r, uint16_t w_r, uint16_t n_r, uint16_t s_r) {
        if (hops > 0) {
            fabric_set_mcast_route(hdr, 0, 0, e_r, w_r, n_r, s_r);
            hdr->to_noc_unicast_atomic_inc(
                NocUnicastAtomicIncCommandHeader(sem_noc_addr, /*inc=*/1, /*width_bits=*/32));
            sender_dir.wait_for_empty_write_slot();
            sender_dir.send_payload_flush_non_blocking_from_address(
                (uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
        }
    };

    send_completion(w_hops, senderW, left_packet_header,  0,      w_hops, 0,      0);
    send_completion(e_hops, senderE, right_packet_header, e_hops, 0,      0,      0);
    send_completion(n_hops, senderN, north_packet_header, e_hops, w_hops, n_hops, 0);
    send_completion(s_hops, senderS, south_packet_header, e_hops, w_hops, 0,      s_hops);

    if (hasW) { senderW.close(); }
    if (hasE) { senderE.close(); }
    if (hasN) { senderN.close(); }
    if (hasS) { senderS.close(); }

#else
    // ===== 1D Linear Mode =====
    const uint8_t start_distance = static_cast<uint8_t>(get_arg_val<uint32_t>(idx++));
    const uint8_t range          = static_cast<uint8_t>(get_arg_val<uint32_t>(idx++));
    const uint32_t scatter_offset = get_arg_val<uint32_t>(idx++);

    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::allocate_header();

    sender.open<true>();

    const uint64_t dst_noc_addr1 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr + scatter_offset, 0);
    const uint16_t scatter_chunk_size = static_cast<uint16_t>(total_size / 2);

    // Linear multicast scatter write with auto-packetization
    fabric_multicast_noc_scatter_write(
        &sender,
        packet_header,
        src_l1_addr,
        total_size,
        tt::tt_fabric::NocUnicastScatterCommandHeader{{dst_noc_addr0, dst_noc_addr1}, {scatter_chunk_size}},
        start_distance,
        range);

    noc_async_writes_flushed();

    // Completion: linear multicast atomic_inc to all receivers
    fabric_multicast_noc_unicast_atomic_inc(
        &sender,
        packet_header,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_addr, /*inc=*/1, /*width_bits=*/32),
        start_distance,
        range);

    sender.close();
#endif
}
