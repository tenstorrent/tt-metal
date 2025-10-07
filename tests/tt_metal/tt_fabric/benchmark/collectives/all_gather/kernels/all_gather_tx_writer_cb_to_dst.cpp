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

    // TEMP (2D API): manual packet header. Post-uplift this becomes implicit.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();

    // We'll program the multicast route per "stripe" (row) below.
    auto mh = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header);

    sender.open<true>();

    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/PAGE_SIZE);

    for (uint32_t i = 0; i < TOTAL_PAGES; ++i) {
        cb_wait_front(CB_ID, 1);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        // Compute destination NOC address (DRAM or L1 interleaved)
        uint64_t dest_noc_addr = dst_acc.get_noc_addr(/*page_id=*/i, /*offset=*/0, /*noc=*/0);

        // -------- Stripe 0: source row (E/W only) --------
        sender.wait_for_empty_write_slot();
        fabric_set_mcast_route(
            reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh),
            /*dst_dev_id (ignored)*/ 0,
            /*dst_mesh_id (ignored)*/ 0,
            /*e_num_hops*/ e_hops,
            /*w_num_hops*/ w_hops,
            /*n_num_hops*/ 0,
            /*s_num_hops*/ 0);
        header->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
        sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, PAGE_SIZE);
        sender.send_payload_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));

        // -------- Stripes north of source row (k = 1..n_hops) --------
        for (uint16_t k = 1; k <= n_hops; ++k) {
            sender.wait_for_empty_write_slot();
            fabric_set_mcast_route(
                reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh),
                /*dst_dev_id (ignored)*/ 0,
                /*dst_mesh_id (ignored)*/ 0,
                /*e_num_hops*/ e_hops,
                /*w_num_hops*/ w_hops,
                /*n_num_hops*/ k,
                /*s_num_hops*/ 0);
            header->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, PAGE_SIZE);
            sender.send_payload_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
        }

        // -------- Stripes south of source row (k = 1..s_hops) --------
        for (uint16_t k = 1; k <= s_hops; ++k) {
            sender.wait_for_empty_write_slot();
            fabric_set_mcast_route(
                reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh),
                /*dst_dev_id (ignored)*/ 0,
                /*dst_mesh_id (ignored)*/ 0,
                /*e_num_hops*/ e_hops,
                /*w_num_hops*/ w_hops,
                /*n_num_hops*/ 0,
                /*s_num_hops*/ k);
            header->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, PAGE_SIZE);
            sender.send_payload_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
        }

        cb_pop_front(CB_ID, 1);
    }

    noc_async_writes_flushed();

    // === Single multicast completion to identical mailboxes on all destination chips ===
    ASSERT(sem_l1_addr != 0);
    const uint64_t sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);

    // Send completion INC along the same stripes.
    const uint32_t total_groups = 1u + (uint32_t)n_hops + (uint32_t)s_hops;
    uint32_t group_idx = 0;

    // Source row
    sender.wait_for_empty_write_slot();
    fabric_set_mcast_route(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh),
        /*dst_dev_id (ignored)*/ 0,
        /*dst_mesh_id (ignored)*/ 0,
        /*e_num_hops*/ e_hops,
        /*w_num_hops*/ w_hops,
        /*n_num_hops*/ 0,
        /*s_num_hops*/ 0);
    header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, /*inc=*/1, /*width_bits=*/32));
    // Block or flush depending if this is the last group
    if (++group_idx == total_groups) {
        sender.send_payload_flush_non_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
    } else {
        sender.send_payload_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
    }

    // North rows
    for (uint16_t k = 1; k <= n_hops; ++k) {
        sender.wait_for_empty_write_slot();
        fabric_set_mcast_route(
            reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh),
            /*dst_dev_id (ignored)*/ 0,
            /*dst_mesh_id (ignored)*/ 0,
            /*e_num_hops*/ e_hops,
            /*w_num_hops*/ w_hops,
            /*n_num_hops*/ k,
            /*s_num_hops*/ 0);
        header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, /*inc=*/1, /*width_bits=*/32));
        if (++group_idx == total_groups) {
            sender.send_payload_flush_non_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
        } else {
            sender.send_payload_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
        }
    }

    // South rows
    for (uint16_t k = 1; k <= s_hops; ++k) {
        sender.wait_for_empty_write_slot();
        fabric_set_mcast_route(
            reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh),
            /*dst_dev_id (ignored)*/ 0,
            /*dst_mesh_id (ignored)*/ 0,
            /*e_num_hops*/ e_hops,
            /*w_num_hops*/ w_hops,
            /*n_num_hops*/ 0,
            /*s_num_hops*/ k);
        header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, /*inc=*/1, /*width_bits=*/32));
        if (++group_idx == total_groups) {
            sender.send_payload_flush_non_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
        } else {
            sender.send_payload_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
        }
    }

    sender.close();
}
