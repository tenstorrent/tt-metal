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
    const uint32_t sem_noc_index = get_arg_val<uint32_t>(idx++);

    // DPRINT << "writer: rx=(" << rx_noc_x << "," << rx_noc_y << ") sem_l1=0x" << sem_l1_addr << ENDL();

    // Four fabric connections in fixed order: W, E, N, S
    auto conn_W = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    auto conn_E = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    auto conn_N = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    auto conn_S = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    // Hops + leg mask
    const uint16_t e_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t w_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t n_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t s_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint32_t leg_mask = get_arg_val<uint32_t>(idx++);

    constexpr uint32_t TAG_PAYLOAD = 0xEA000000u;
    constexpr uint32_t TAG_SEM = 0xEB000000u;
    constexpr uint32_t TAG_HDR = 0xEE000000u;
    constexpr uint32_t BR_W = 0x01, BR_E = 0x02, BR_N = 0x04, BR_S = 0x08;

    DPRINT << "P0 pre: hops E/W/N/S=" << (uint32_t)e_hops << "/" << (uint32_t)w_hops << "/" << (uint32_t)n_hops << "/"
           << (uint32_t)s_hops << " leg_mask=" << leg_mask << ENDL();

    // DPRINT << "P1 alloc: begin" << ENDL();

    // DPRINT << "P1W alloc try" << ENDL();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_W = PacketHeaderPool::allocate_header();
    // DPRINT << "P1W alloc ok ptr=0x" << (uint32_t)((uintptr_t)hdr_W & 0xffffffffu) << ENDL();

    // DPRINT << "P1E alloc try" << ENDL();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_E = PacketHeaderPool::allocate_header();
    // DPRINT << "P1E alloc ok ptr=0x" << (uint32_t)((uintptr_t)hdr_E & 0xffffffffu) << ENDL();

    // DPRINT << "P1N alloc try" << ENDL();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_N = PacketHeaderPool::allocate_header();
    // DPRINT << "P1N alloc ok ptr=0x" << (uint32_t)((uintptr_t)hdr_N & 0xffffffffu) << ENDL();

    // DPRINT << "P1S alloc try" << ENDL();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_S = PacketHeaderPool::allocate_header();
    // DPRINT << "P1S alloc ok ptr=0x" << (uint32_t)((uintptr_t)hdr_S & 0xffffffffu) << ENDL();

    // DPRINT << "P1 alloc: done" << ENDL();

    // Clear headers so fabric_set_mcast_route writes into a clean slate
    zero_l1_buf(reinterpret_cast<uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdr_W)), sizeof(PACKET_HEADER_TYPE));
    zero_l1_buf(reinterpret_cast<uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdr_E)), sizeof(PACKET_HEADER_TYPE));
    zero_l1_buf(reinterpret_cast<uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdr_N)), sizeof(PACKET_HEADER_TYPE));
    zero_l1_buf(reinterpret_cast<uint32_t*>(const_cast<PACKET_HEADER_TYPE*>(hdr_S)), sizeof(PACKET_HEADER_TYPE));

    auto mh_W = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_W);
    auto mh_E = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_E);
    auto mh_N = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_N);
    auto mh_S = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(hdr_S);
    // Low-latency mesh header view for route programming
    auto mh_W_ll = reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(hdr_W);
    auto mh_E_ll = reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(hdr_E);
    auto mh_N_ll = reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(hdr_N);
    auto mh_S_ll = reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(hdr_S);

    const bool use_W = (w_hops > 0) && (leg_mask & 1u);
    const bool use_E = (e_hops > 0) && (leg_mask & 2u);
    const bool use_N = (n_hops > 0) && (leg_mask & 4u);
    const bool use_S = (s_hops > 0) && (leg_mask & 8u);

    // DPRINT << "P0 use: [W,E,N,S]=" << (int)use_W << "," << (int)use_E << "," << (int)use_N << "," << (int)use_S
    //        << ENDL();

    if (use_W) {
        // DPRINT << "P2W open enter" << ENDL();
        conn_W.open<true>();
        // DPRINT << "P2W open ok" << ENDL();
    }
    if (use_E) {
        // DPRINT << "P2E open enter" << ENDL();
        conn_E.open<true>();
        // DPRINT << "P2E open ok" << ENDL();
    }
    if (use_N) {
        // DPRINT << "P2N open enter" << ENDL();
        conn_N.open<true>();
        // DPRINT << "P2N open ok" << ENDL();
    }
    if (use_S) {
        // DPRINT << "P2S open enter" << ENDL();
        conn_S.open<true>();
        // DPRINT << "P2S open ok" << ENDL();
    }

    // --- One-shot config dump ---
    // DPRINT << "writer:init pages=" << TOTAL_PAGES << " page=" << PAGE_SIZE << " rx=(" << rx_noc_x << "," << rx_noc_y
    //        << ")"
    //        << " legs[W,E,N,S]=" << (int)use_W << "," << (int)use_E << "," << (int)use_N << "," << (int)use_S
    //        << " hops E/W/N/S=" << e_hops << "/" << w_hops << "/" << n_hops << "/" << s_hops << ENDL();
    // if (use_W) {
    //     DPRINT << "writer:W start (mesh,dev)=(" << (int)dst_mesh_W << "," << (int)dst_dev_W << ")" << ENDL();
    // }
    // if (use_E) {
    //     DPRINT << "writer:E start (mesh,dev)=(" << (int)dst_mesh_E << "," << (int)dst_dev_E << ")" << ENDL();
    // }
    // if (use_N) {
    //     DPRINT << "writer:N start (mesh,dev)=(" << (int)dst_mesh_N << "," << (int)dst_dev_N << ")" << ENDL();
    // }
    // if (use_S) {
    //     DPRINT << "writer:S start (mesh,dev)=(" << (int)dst_mesh_S << "," << (int)dst_dev_S << ")" << ENDL();
    // }

    auto should_log = [&](uint32_t i) -> bool { return (i == 0) || (i + 1 == TOTAL_PAGES) || ((i & 31u) == 0); };

    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/PAGE_SIZE);

    for (uint32_t i = 0; i < TOTAL_PAGES; ++i) {
        // DPRINT << "L" << i << " A cb_wait_front enter" << ENDL();
        cb_wait_front(CB_ID, 1);
        // DPRINT << "L" << i << " B cb_wait_front done; get_read_ptr" << ENDL();

        const uint32_t page_l1_addr = get_read_ptr(CB_ID);
        // DPRINT << "L" << i << " C got ptr=0x" << page_l1_addr << ENDL();
        constexpr uint8_t DRAM_NOC = 1;  // use NOC-1 for DRAM
        uint64_t dest_noc_addr = dst_acc.get_noc_addr(/*page_id=*/i, /*offset=*/0, /*noc=*/DRAM_NOC);
        if (should_log(i)) {
            // DPRINT << "writer:page " << i << " page_l1=0x" << page_l1_addr << ENDL();
        }

        // NORTH trunk (fan-out E/W on each north row)
        if (use_N) {
            conn_N.wait_for_empty_write_slot();
            if (should_log(i)) {
                DPRINT << "writer:N route&send page " << i << ENDL();
            }
            fabric_set_mcast_route(mh_N_ll, 0, 0, e_hops, w_hops, n_hops, 0);
            volatile tt_l1_ptr uint32_t* __p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mh_N);
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
            fabric_set_mcast_route(mh_S_ll, 0, 0, e_hops, w_hops, 0, s_hops);
            volatile tt_l1_ptr uint32_t* __p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mh_S);
            // Push first 8 words of header
            hdr_S->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            DPRINT << "writer:S route&send page " << i << ENDL();
            DPRINT << "S: header prepared" << ENDL();
            conn_S.send_payload_without_header_non_blocking_from_address(page_l1_addr, PAGE_SIZE);
            DPRINT << "S: payload queued" << ENDL();
            conn_S.send_payload_flush_non_blocking_from_address((uint32_t)hdr_S, sizeof(PACKET_HEADER_TYPE));
            DPRINT << "S: header sent" << ENDL();
        }
        // WEST branch on source row
        if (use_W) {
            conn_W.wait_for_empty_write_slot();
            fabric_set_mcast_route(mh_W_ll, 0, 0, 0, w_hops, 0, 0);
            hdr_W->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            if (should_log(i)) {
                DPRINT << "writer:W route&send page " << i << ENDL();
                volatile tt_l1_ptr uint32_t* __p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mh_W);
            }
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
            fabric_set_mcast_route(mh_E_ll, 0, 0, e_hops, 0, 0, 0);
            DPRINT << "writer:E after fabric_set_mcast_rout" << i << ENDL();
            hdr_E->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);
            DPRINT << "writer:E after to_noc_unicast_write" << i << ENDL();
            conn_E.send_payload_without_header_non_blocking_from_address(page_l1_addr, PAGE_SIZE);
            DPRINT << "writer:E after send_payload_without_header_non_blocking_from_address" << i << ENDL();
            conn_E.send_payload_flush_non_blocking_from_address((uint32_t)hdr_E, sizeof(PACKET_HEADER_TYPE));
            DPRINT << "writer:E after send_payload_blocking_from_address" << i << ENDL();
        }
        // --- Loopback on this chip: write to receiver DRAM via NOC-1 (not worker L1) ---
        // Reuse the accessor so we target DRAM banking correctly on NOC1.
        const uint64_t self_dram_noc1_addr = dst_acc.get_noc_addr(/*page_id=*/i, /*offset=*/0, /*noc=*/DRAM_NOC);
        noc_async_write(page_l1_addr, self_dram_noc1_addr, PAGE_SIZE);

        DPRINT << "writer: Before cb_pop_front" << i << ENDL();
        cb_pop_front(CB_ID, 1);
        DPRINT << "writer: After cb_pop_front" << i << ENDL();
    }

    DPRINT << "writer:payload loop done; flushing async writes" << ENDL();
    noc_async_write_barrier();
    DPRINT << "writer:payloads flushed; sending completion atomics" << ENDL();

    // === Multicast completion: exactly one atomic route that covers all receivers ===
    ASSERT(sem_l1_addr != 0);
    const uint64_t sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, sem_noc_index);
    const uint32_t sem_noc_hi = static_cast<uint32_t>(sem_noc >> 32);
    const uint32_t sem_noc_lo = static_cast<uint32_t>(sem_noc & 0xffffffffu);
    DPRINT << "writer: sem_noc[hi:lo]=0x" << sem_noc_hi << ":0x" << sem_noc_lo << ENDL();

    // If a trunk (N or S) exists, send the atomic on the trunk only (it fans out E/W as needed).
    // Otherwise (single-row), send two atomics: one on E and one on W (each side of the row).
    if (use_N) {
        conn_N.wait_for_empty_write_slot();
        DPRINT << "writer:N atomic_inc" << ENDL();
        fabric_set_mcast_route(mh_N_ll, 0, 0, e_hops, w_hops, n_hops, 0);
        hdr_N->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, 1, 32));
        conn_N.send_payload_flush_non_blocking_from_address((uint32_t)hdr_N, sizeof(PACKET_HEADER_TYPE));
    } else if (use_S) {
        conn_S.wait_for_empty_write_slot();
        DPRINT << "writer:S atomic_inc" << ENDL();
        fabric_set_mcast_route(mh_S_ll, 0, 0, e_hops, w_hops, 0, s_hops);
        hdr_S->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, 1, 32));
        conn_S.send_payload_flush_non_blocking_from_address((uint32_t)hdr_S, sizeof(PACKET_HEADER_TYPE));
    } else {
        // Single-row: bump both directions so every chip on the row gets exactly one bump
        if (use_W) {
            conn_W.wait_for_empty_write_slot();
            DPRINT << "writer:W atomic_inc" << ENDL();
            fabric_set_mcast_route(mh_W_ll, 0, 0, 0, w_hops, 0, 0);
            hdr_W->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, 1, 32));
            {
                volatile tt_l1_ptr uint32_t* __p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mh_W);
            }
            conn_W.send_payload_blocking_from_address((uint32_t)hdr_W, sizeof(PACKET_HEADER_TYPE));
        }
        if (use_E) {
            conn_E.wait_for_empty_write_slot();
            DPRINT << "writer:E atomic_inc" << ENDL();
            fabric_set_mcast_route(mh_E_ll, 0, 0, e_hops, 0, 0, 0);
            hdr_E->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, 1, 32));
            // Use flush on the last send so fabric pushes promptly
            conn_E.send_payload_flush_non_blocking_from_address((uint32_t)hdr_E, sizeof(PACKET_HEADER_TYPE));
        }
    }

    noc_semaphore_inc(sem_noc, 1);
    noc_async_atomic_barrier();

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
