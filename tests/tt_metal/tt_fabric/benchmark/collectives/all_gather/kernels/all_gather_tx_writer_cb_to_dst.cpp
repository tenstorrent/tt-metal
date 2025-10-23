// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "accessor/tensor_accessor.h"
#include "accessor/tensor_accessor_args.h"

using namespace tt;
using namespace tt::tt_fabric;

using eth_chan_directions::EAST;
using eth_chan_directions::NORTH;
using eth_chan_directions::SOUTH;
using eth_chan_directions::WEST;

static inline void set_mcast_header(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    eth_chan_directions trunk_direction,
    uint16_t trunk_hops,
    uint16_t e_hops,
    uint16_t w_hops) {
    uint16_t n_hops = 0;
    uint16_t s_hops = 0;
    if (trunk_direction == eth_chan_directions::NORTH) {
        n_hops = trunk_hops;
    } else if (trunk_direction == eth_chan_directions::SOUTH) {
        s_hops = trunk_hops;
    }
    fabric_set_mcast_route(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(packet_header),
        0,
        0,
        e_hops,
        w_hops,
        n_hops,
        s_hops);
}

//
// Writer (fabric sender) kernel — sends pages from CB c_0 to the dst device.
// Per page: wait for one CB page → build header to dst → send payload → send header.
// After all pages: flush, then atomic-inc the receiver’s global semaphore (completion signal).
//
// CT args:
//   0: TOTAL_PAGES
//   1: PAGE_SIZE
//
// RT args (must match host):
//   0:  dst_base       (u32)
//   1:  rx_noc_x       (u32)   // same worker on every chip
//   2:  rx_noc_y       (u32)
//   3:  sem_l1_addr    (u32)   // same L1 offset on every chip
//   … fabric-connection args … (inserted by append_fabric_connection_rt_args on host)
//   … then optional Phase-A diagnostics:
//      e_hops (u32), w_hops (u32), n_hops (u32), s_hops (u32)

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr uint32_t TOTAL_PAGES = get_compile_time_arg_val(CTA_BASE + 0);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 1);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    size_t idx = 0;
    const uint32_t dst_base = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr = get_arg_val<uint32_t>(idx++);

    // Build the fabric connection next (these args were appended by the host
    // right after the fixed 6 args).
    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    // Phase A diagnostics (optional): hops were appended by the host
    // AFTER the fabric-connection args, so read them now.
    const uint16_t e_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t w_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t n_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t s_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));

    volatile tt_l1_ptr PACKET_HEADER_TYPE* left_packet_header = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* right_packet_header = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* north_packet_header = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* south_packet_header = PacketHeaderPool::allocate_header();

    sender.open<true>();

    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/PAGE_SIZE);

    for (uint32_t i = 0; i < TOTAL_PAGES; ++i) {
        cb_wait_front(CB_ID, 1);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        // Pace transmissions so we don’t overrun the fabric send queue.
        sender.wait_for_empty_write_slot();

        // Compute destination NOC address (DRAM or L1 interleaved)
        uint64_t dest_noc_addr = dst_acc.get_noc_addr(/*page_id=*/i, /*offset=*/0, /*noc=*/0);

        // Build the NOC header for this page (mcast route already set above)
        // --- Branch 1: direct WEST fanout (left) ---
        if (w_hops > 0) {
            // Program header for WEST-only branch (no trunk)
            set_mcast_header(left_packet_header, eth_chan_directions::WEST, /*trunk_hops=*/0, /*E*/ 0, /*W*/ w_hops);
            // Build the NOC header for this page
            left_packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            // Pace & send
            sender.wait_for_empty_write_slot();
            sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, PAGE_SIZE);
            sender.send_payload_blocking_from_address((uint32_t)left_packet_header, sizeof(PACKET_HEADER_TYPE));
        }

        // --- Branch 2: direct EAST fanout (right) ---
        if (e_hops > 0) {
            set_mcast_header(right_packet_header, eth_chan_directions::EAST, /*trunk_hops=*/0, /*E*/ e_hops, /*W*/ 0);
            right_packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            sender.wait_for_empty_write_slot();
            sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, PAGE_SIZE);
            sender.send_payload_blocking_from_address((uint32_t)right_packet_header, sizeof(PACKET_HEADER_TYPE));
        }

        // --- Branch 3: NORTH trunk (optional) ---
        if (n_hops > 0) {
            set_mcast_header(
                north_packet_header, eth_chan_directions::NORTH, /*trunk*/ n_hops, /*E*/ e_hops, /*W*/ w_hops);
            north_packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            sender.wait_for_empty_write_slot();
            sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, PAGE_SIZE);
            sender.send_payload_blocking_from_address((uint32_t)north_packet_header, sizeof(PACKET_HEADER_TYPE));
        }

        // --- Branch 4: SOUTH trunk (optional) ---
        if (s_hops > 0) {
            set_mcast_header(
                south_packet_header, eth_chan_directions::SOUTH, /*trunk*/ s_hops, /*E*/ e_hops, /*W*/ w_hops);
            south_packet_header->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            sender.wait_for_empty_write_slot();
            sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, PAGE_SIZE);
            sender.send_payload_blocking_from_address((uint32_t)south_packet_header, sizeof(PACKET_HEADER_TYPE));
        }

        cb_pop_front(CB_ID, 1);
    }

    noc_async_writes_flushed();

    // === Single multicast completion to identical mailboxes on all destination chips ===
    ASSERT(sem_l1_addr != 0);
    const uint64_t sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);

    // Send a completion per active branch so every sub-tree gets the semaphore bump.
    if (w_hops > 0) {
        set_mcast_header(left_packet_header, eth_chan_directions::WEST, /*trunk*/ 0, /*E*/ 0, /*W*/ w_hops);
        left_packet_header->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader(sem_noc, /*inc=*/1, /*width_bits=*/32));
        sender.wait_for_empty_write_slot();
        sender.send_payload_flush_non_blocking_from_address((uint32_t)left_packet_header, sizeof(PACKET_HEADER_TYPE));
    }
    if (e_hops > 0) {
        set_mcast_header(right_packet_header, eth_chan_directions::EAST, /*trunk*/ 0, /*E*/ e_hops, /*W*/ 0);
        right_packet_header->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader(sem_noc, /*inc=*/1, /*width_bits=*/32));
        sender.wait_for_empty_write_slot();
        sender.send_payload_flush_non_blocking_from_address((uint32_t)right_packet_header, sizeof(PACKET_HEADER_TYPE));
    }
    if (n_hops > 0) {
        set_mcast_header(north_packet_header, eth_chan_directions::NORTH, /*trunk*/ n_hops, /*E*/ e_hops, /*W*/ w_hops);
        north_packet_header->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader(sem_noc, /*inc=*/1, /*width_bits=*/32));
        sender.wait_for_empty_write_slot();
        sender.send_payload_flush_non_blocking_from_address((uint32_t)north_packet_header, sizeof(PACKET_HEADER_TYPE));
    }
    if (s_hops > 0) {
        set_mcast_header(south_packet_header, eth_chan_directions::SOUTH, /*trunk*/ s_hops, /*E*/ e_hops, /*W*/ w_hops);
        south_packet_header->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader(sem_noc, /*inc=*/1, /*width_bits=*/32));
        sender.wait_for_empty_write_slot();
        sender.send_payload_flush_non_blocking_from_address((uint32_t)south_packet_header, sizeof(PACKET_HEADER_TYPE));
    }

    sender.close();
}
