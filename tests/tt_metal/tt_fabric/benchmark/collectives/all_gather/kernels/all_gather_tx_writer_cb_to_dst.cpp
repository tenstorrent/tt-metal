// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/dprint.h"
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

    DPRINT << "writer: rx=(" << rx_noc_x << "," << rx_noc_y << ") sem_l1=0x" << sem_l1_addr << ENDL();

    // Four fabric connections in fixed order: W, E, N, S
    auto conn_W = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    auto conn_E = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    auto conn_N = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    auto conn_S = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    // First-hop start node IDs per leg (must match host order W,E,N,S).
    const uint16_t dst_dev_W = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_mesh_W = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_dev_E = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_mesh_E = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_dev_N = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_mesh_N = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_dev_S = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_mesh_S = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));

    // Hops + leg mask
    const uint16_t e_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t w_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t n_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t s_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint32_t leg_mask = get_arg_val<uint32_t>(idx++);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_W = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_E = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_N = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_S = PacketHeaderPool::allocate_header();

    // Clear headers so fabric_set_mcast_route writes into a clean slate
    zero_l1_buf(reinterpret_cast<uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdr_W)), sizeof(PACKET_HEADER_TYPE));
    zero_l1_buf(reinterpret_cast<uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdr_E)), sizeof(PACKET_HEADER_TYPE));
    zero_l1_buf(reinterpret_cast<uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdr_N)), sizeof(PACKET_HEADER_TYPE));
    zero_l1_buf(reinterpret_cast<uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdr_S)), sizeof(PACKET_HEADER_TYPE));

    auto mh_W = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_W);
    auto mh_E = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_E);
    auto mh_N = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_N);
    auto mh_S = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_S);

    const bool use_W = (w_hops > 0) && (leg_mask & 1u);
    const bool use_E = (e_hops > 0) && (leg_mask & 2u);
    const bool use_N = (n_hops > 0) && (leg_mask & 4u);
    const bool use_S = (s_hops > 0) && (leg_mask & 8u);

    if (use_W) {
        conn_W.open<true>();
    }
    if (use_E) {
        conn_E.open<true>();
    }
    if (use_N) {
        conn_N.open<true>();
    }
    if (use_S) {
        conn_S.open<true>();
    }

    // --- One-shot config dump ---
    DPRINT << "writer:init pages=" << TOTAL_PAGES << " page=" << PAGE_SIZE << " rx=(" << rx_noc_x << "," << rx_noc_y
           << ")"
           << " legs[W,E,N,S]=" << (int)use_W << "," << (int)use_E << "," << (int)use_N << "," << (int)use_S
           << " hops E/W/N/S=" << e_hops << "/" << w_hops << "/" << n_hops << "/" << s_hops << ENDL();
    if (use_W) {
        DPRINT << "writer:W start (mesh,dev)=(" << (int)dst_mesh_W << "," << (int)dst_dev_W << ")" << ENDL();
    }
    if (use_E) {
        DPRINT << "writer:E start (mesh,dev)=(" << (int)dst_mesh_E << "," << (int)dst_dev_E << ")" << ENDL();
    }
    if (use_N) {
        DPRINT << "writer:N start (mesh,dev)=(" << (int)dst_mesh_N << "," << (int)dst_dev_N << ")" << ENDL();
    }
    if (use_S) {
        DPRINT << "writer:S start (mesh,dev)=(" << (int)dst_mesh_S << "," << (int)dst_dev_S << ")" << ENDL();
    }

    auto should_log = [&](uint32_t i) -> bool { return (i == 0) || (i + 1 == TOTAL_PAGES) || ((i & 31u) == 0); };

    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/PAGE_SIZE);

    for (uint32_t i = 0; i < TOTAL_PAGES; ++i) {
        cb_wait_front(CB_ID, 1);
        const uint32_t page_l1_addr = get_read_ptr(CB_ID);
        uint64_t dest_noc_addr = dst_acc.get_noc_addr(i, rx_noc_x, rx_noc_y);
        if (should_log(i)) {
            DPRINT << "writer:page " << i << " page_l1=0x" << page_l1_addr << ENDL();
        }

        // NORTH trunk (fan-out E/W on each north row)
        if (use_N) {
            conn_N.wait_for_empty_write_slot();
            if (should_log(i)) {
                DPRINT << "writer:N route&send page " << i << ENDL();
            }
            fabric_set_mcast_route(
                reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh_N),
                dst_dev_N,
                dst_mesh_N,
                e_hops,
                w_hops,
                (uint16_t)(n_hops ? (n_hops - 1) : 0),
                0);
            hdr_N->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            conn_N.send_payload_without_header_non_blocking_from_address(page_l1_addr, PAGE_SIZE);
            conn_N.send_payload_flush_non_blocking_from_address((uint32_t)hdr_N, sizeof(PACKET_HEADER_TYPE));
        }
        // SOUTH trunk
        if (use_S) {
            conn_S.wait_for_empty_write_slot();
            if (should_log(i)) {
                DPRINT << "writer:S route&send page " << i << ENDL();
            }
            fabric_set_mcast_route(
                reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh_S),
                dst_dev_S,
                dst_mesh_S,
                e_hops,
                w_hops,
                0,
                (uint16_t)(s_hops ? (s_hops - 1) : 0));
            hdr_S->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            conn_S.send_payload_without_header_non_blocking_from_address(page_l1_addr, PAGE_SIZE);
            conn_S.send_payload_flush_non_blocking_from_address((uint32_t)hdr_S, sizeof(PACKET_HEADER_TYPE));
        }
        // WEST branch on source row
        if (use_W) {
            conn_W.wait_for_empty_write_slot();
            if (should_log(i)) {
                DPRINT << "writer:W route&send page " << i << ENDL();
            }
            fabric_set_mcast_route(
                reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh_W),
                dst_dev_W,
                dst_mesh_W,
                0,
                (uint16_t)(w_hops ? (w_hops - 1) : 0),
                0,
                0);
            hdr_W->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            conn_W.send_payload_without_header_non_blocking_from_address(page_l1_addr, PAGE_SIZE);
            conn_W.send_payload_flush_non_blocking_from_address((uint32_t)hdr_W, sizeof(PACKET_HEADER_TYPE));
        }
        // EAST branch on source row
        if (use_E) {
            DPRINT << "writer:E before wait for empty write slot" << i << ENDL();
            conn_E.wait_for_empty_write_slot();
            DPRINT << "writer:E after wait for empty write slot" << i << ENDL();
            if (should_log(i)) {
                DPRINT << "writer:E route&send page " << i << ENDL();
            }
            fabric_set_mcast_route(
                reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh_E),
                dst_dev_E,
                dst_mesh_E,
                (uint16_t)(e_hops ? (e_hops - 1) : 0),
                0,
                0,
                0);
            DPRINT << "writer:E after fabric_set_mcast_rout" << i << ENDL();
            hdr_E->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            DPRINT << "writer:E after to_noc_unicast_write" << i << ENDL();
            conn_E.send_payload_without_header_non_blocking_from_address(page_l1_addr, PAGE_SIZE);
            DPRINT << "writer:E after send_payload_without_header_non_blocking_from_address" << i << ENDL();
            conn_E.send_payload_flush_non_blocking_from_address((uint32_t)hdr_E, sizeof(PACKET_HEADER_TYPE));
            DPRINT << "writer:E after send_payload_blocking_from_address" << i << ENDL();
        }
        DPRINT << "writer: Before cb_pop_front" << i << ENDL();
        cb_pop_front(CB_ID, 1);
        DPRINT << "writer: After cb_pop_front" << i << ENDL();
    }

    DPRINT << "writer:payload loop done; flushing async writes" << ENDL();
    noc_async_write_barrier();
    DPRINT << "writer:payloads flushed; sending completion atomics" << ENDL();

    // === Single multicast completion to identical mailboxes on all destination chips ===
    ASSERT(sem_l1_addr != 0);
    const uint64_t sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, 0);
    const uint32_t sem_noc_hi = static_cast<uint32_t>(sem_noc >> 32);
    const uint32_t sem_noc_lo = static_cast<uint32_t>(sem_noc & 0xffffffffu);
    DPRINT << "writer: sem_noc[hi:lo]=0x" << sem_noc_hi << ":0x" << sem_noc_lo << ENDL();

    uint32_t legs = (use_N ? 1u : 0u) + (use_S ? 1u : 0u) + (use_W ? 1u : 0u) + (use_E ? 1u : 0u);
    uint32_t sent = 0;
    auto mark_last = [&]() { return ++sent == legs; };

    if (use_N) {
        conn_N.wait_for_empty_write_slot();
        DPRINT << "writer:N atomic_inc" << ENDL();
        fabric_set_mcast_route(
            reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh_N),
            dst_dev_N,
            dst_mesh_N,
            e_hops,
            w_hops,
            (uint16_t)(n_hops ? (n_hops - 1) : 0),
            0);
        hdr_N->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, 1, 32));
        bool last = mark_last();
        if (last) {
            conn_N.send_payload_flush_non_blocking_from_address((uint32_t)hdr_N, sizeof(PACKET_HEADER_TYPE));
        } else {
            conn_N.send_payload_blocking_from_address((uint32_t)hdr_N, sizeof(PACKET_HEADER_TYPE));
        }
    }
    if (use_S) {
        conn_S.wait_for_empty_write_slot();
        DPRINT << "writer:S atomic_inc" << ENDL();
        fabric_set_mcast_route(
            reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh_S),
            dst_dev_S,
            dst_mesh_S,
            e_hops,
            w_hops,
            0,
            (uint16_t)(s_hops ? (s_hops - 1) : 0));
        hdr_S->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, 1, 32));
        bool last = mark_last();
        if (last) {
            conn_S.send_payload_flush_non_blocking_from_address((uint32_t)hdr_S, sizeof(PACKET_HEADER_TYPE));
        } else {
            conn_S.send_payload_blocking_from_address((uint32_t)hdr_S, sizeof(PACKET_HEADER_TYPE));
        }
    }
    if (use_W) {
        conn_W.wait_for_empty_write_slot();
        DPRINT << "writer:W atomic_inc" << ENDL();
        fabric_set_mcast_route(
            reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh_W),
            dst_dev_W,
            dst_mesh_W,
            0,
            (uint16_t)(w_hops ? (w_hops - 1) : 0),
            0,
            0);
        hdr_W->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, 1, 32));
        bool last = mark_last();
        if (last) {
            conn_W.send_payload_flush_non_blocking_from_address((uint32_t)hdr_W, sizeof(PACKET_HEADER_TYPE));
        } else {
            conn_W.send_payload_blocking_from_address((uint32_t)hdr_W, sizeof(PACKET_HEADER_TYPE));
        }
    }
    if (use_E) {
        conn_E.wait_for_empty_write_slot();
        DPRINT << "writer:E atomic_inc" << ENDL();
        fabric_set_mcast_route(
            reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh_E),
            dst_dev_E,
            dst_mesh_E,
            (uint16_t)(e_hops ? (e_hops - 1) : 0),
            0,
            0,
            0);
        hdr_E->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, 1, 32));
        bool last = mark_last();
        if (last) {
            conn_E.send_payload_flush_non_blocking_from_address((uint32_t)hdr_E, sizeof(PACKET_HEADER_TYPE));
        } else {
            conn_E.send_payload_blocking_from_address((uint32_t)hdr_E, sizeof(PACKET_HEADER_TYPE));
        }
    }

    if (use_W) {
        conn_W.close();
    }
    if (use_E) {
        conn_E.close();
    }
    if (use_N) {
        conn_N.close();
    }
    if (use_S) {
        conn_S.close();
    }
    DPRINT << "writer:done" << ENDL();
}
