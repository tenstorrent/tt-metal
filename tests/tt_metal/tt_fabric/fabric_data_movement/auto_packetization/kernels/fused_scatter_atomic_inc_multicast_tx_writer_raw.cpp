// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Multicast fused scatter write + atomic_inc sender kernel for auto-packetization tests.
// Sends a scatter payload via fabric_multicast_noc_fused_scatter_write_atomic_inc per
// active direction. The fused wrapper distributes data to TWO destination NOC addresses
// AND fires an atomic_inc on completion. Payloads must fit in a single packet (passthrough).
//
// Does NOT send separate atomic_inc -- the fused wrapper handles it.
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

    const uint64_t dst_noc_addr1 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr + scatter_offset, 0);

    const uint16_t scatter_chunk_size = static_cast<uint16_t>(total_size / 2);

    // Fused scatter + atomic_inc per direction (passthrough, no chunking)
    if (w_hops > 0) {
        MeshMcastRange ranges_w{0, static_cast<uint8_t>(w_hops), 0, 0};
        fabric_multicast_noc_fused_scatter_write_atomic_inc(
            &senderW, left_packet_header, 0, 0, ranges_w,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                {dst_noc_addr0, dst_noc_addr1}, sem_noc_addr, {scatter_chunk_size}, /*val=*/1, /*flush=*/true});
    }
    if (e_hops > 0) {
        MeshMcastRange ranges_e{static_cast<uint8_t>(e_hops), 0, 0, 0};
        fabric_multicast_noc_fused_scatter_write_atomic_inc(
            &senderE, right_packet_header, 0, 0, ranges_e,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                {dst_noc_addr0, dst_noc_addr1}, sem_noc_addr, {scatter_chunk_size}, /*val=*/1, /*flush=*/true});
    }
    if (n_hops > 0) {
        MeshMcastRange ranges_n{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops),
            static_cast<uint8_t>(n_hops), 0};
        fabric_multicast_noc_fused_scatter_write_atomic_inc(
            &senderN, north_packet_header, 0, 0, ranges_n,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                {dst_noc_addr0, dst_noc_addr1}, sem_noc_addr, {scatter_chunk_size}, /*val=*/1, /*flush=*/true});
    }
    if (s_hops > 0) {
        MeshMcastRange ranges_s{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops),
            0, static_cast<uint8_t>(s_hops)};
        fabric_multicast_noc_fused_scatter_write_atomic_inc(
            &senderS, south_packet_header, 0, 0, ranges_s,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                {dst_noc_addr0, dst_noc_addr1}, sem_noc_addr, {scatter_chunk_size}, /*val=*/1, /*flush=*/true});
    }

    // No separate completion signaling -- fused wrapper handles it.

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

    // Linear multicast fused scatter + atomic_inc (passthrough, no chunking)
    fabric_multicast_noc_fused_scatter_write_atomic_inc(
        &sender,
        packet_header,
        src_l1_addr,
        total_size,
        tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
            {dst_noc_addr0, dst_noc_addr1},
            sem_noc_addr,
            {scatter_chunk_size},
            /*val=*/1,
            /*flush=*/true},
        start_distance,
        range);

    // No separate completion signaling -- fused wrapper handles it.

    sender.close();
#endif
}
