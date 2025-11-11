// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "accessor/tensor_accessor.h"
#include "accessor/tensor_accessor_args.h"

using namespace tt;
using namespace tt::tt_fabric;
using namespace tt::tt_fabric::mesh::experimental;

//
// Writer (fabric sender) kernel — sends pages from CB c_0 to the dst device.
// Per page: wait for one CB page → build header to dst → send payload → send header.
// After all pages: flush, then atomic-inc the receiver's global semaphore (completion signal).
//

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

    // directional connections — parse a bitmask and build up to four senders in fixed order: W,E,N,S
    const uint32_t dir_mask = get_arg_val<uint32_t>(idx++);
    const bool hasW = (dir_mask & 0x1u) != 0;
    const bool hasE = (dir_mask & 0x2u) != 0;
    const bool hasN = (dir_mask & 0x4u) != 0;
    const bool hasS = (dir_mask & 0x8u) != 0;

    WorkerToFabricEdmSender senderW{};
    WorkerToFabricEdmSender senderE{};
    WorkerToFabricEdmSender senderN{};
    WorkerToFabricEdmSender senderS{};

    if (hasW) {
        senderW = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    }
    if (hasE) {
        senderE = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    }
    if (hasN) {
        senderN = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    }
    if (hasS) {
        senderS = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    }

    // Phase A diagnostics (optional): hops were appended by the host
    // AFTER the fabric-connection args, so read them now.
    const uint16_t e_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t w_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t n_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t s_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));

    // TensorAccessor for destination address computation
    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/PAGE_SIZE);

    // Packet headers for each direction
    volatile PACKET_HEADER_TYPE* left_packet_header = PacketHeaderPool::allocate_header();
    volatile PACKET_HEADER_TYPE* right_packet_header = PacketHeaderPool::allocate_header();
    volatile PACKET_HEADER_TYPE* north_packet_header = PacketHeaderPool::allocate_header();
    volatile PACKET_HEADER_TYPE* south_packet_header = PacketHeaderPool::allocate_header();

    // Set multicast routes for each active direction (requires 7 args: header, dev_id, mesh_id, e, w, n, s)
    if (w_hops > 0) {
        fabric_set_mcast_route(left_packet_header, 0, 0, 0, static_cast<uint16_t>(w_hops), 0, 0);
    }
    if (e_hops > 0) {
        fabric_set_mcast_route(right_packet_header, 0, 0, static_cast<uint16_t>(e_hops), 0, 0, 0);
    }
    if (n_hops > 0) {
        fabric_set_mcast_route(north_packet_header, 0, 0, 0, 0, static_cast<uint16_t>(n_hops), 0);
    }
    if (s_hops > 0) {
        fabric_set_mcast_route(south_packet_header, 0, 0, 0, 0, 0, static_cast<uint16_t>(s_hops));
    }

    for (uint32_t i = 0; i < TOTAL_PAGES; ++i) {
        cb_wait_front(CB_ID, 1);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        // --- Branch 1: direct WEST fanout (left) ---
        if (w_hops > 0) {
            MeshMcastRange ranges_w{0, static_cast<uint8_t>(w_hops), 0, 0};  // e, w, n, s
            fabric_multicast_noc_unicast_write(&senderW, left_packet_header, 0, 0, ranges_w, src_l1_addr, dst_acc, i);
        }

        // --- Branch 2: direct EAST fanout (right) ---
        if (e_hops > 0) {
            MeshMcastRange ranges_e{static_cast<uint8_t>(e_hops), 0, 0, 0};  // e, w, n, s
            fabric_multicast_noc_unicast_write(&senderE, right_packet_header, 0, 0, ranges_e, src_l1_addr, dst_acc, i);
        }

        // --- Branch 3: NORTH trunk (optional) ---
        if (n_hops > 0) {
            MeshMcastRange ranges_n{
                static_cast<uint8_t>(e_hops),
                static_cast<uint8_t>(w_hops),
                static_cast<uint8_t>(n_hops),
                0};  // e, w, n, s
            fabric_multicast_noc_unicast_write(&senderN, north_packet_header, 0, 0, ranges_n, src_l1_addr, dst_acc, i);
        }

        // --- Branch 4: SOUTH trunk (optional) ---
        if (s_hops > 0) {
            MeshMcastRange ranges_s{
                static_cast<uint8_t>(e_hops),
                static_cast<uint8_t>(w_hops),
                0,
                static_cast<uint8_t>(s_hops)};  // e, w, n, s
            fabric_multicast_noc_unicast_write(&senderS, south_packet_header, 0, 0, ranges_s, src_l1_addr, dst_acc, i);
        }

        cb_pop_front(CB_ID, 1);
    }

    noc_async_writes_flushed();

    // Final atomic increment to signal completion (7 args: sender, header, dev_id, mesh_id, src_addr, size, noc_cmd)
    uint64_t sem_noc_addr = get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr);

    if (w_hops > 0) {
        fabric_unicast_noc_fused_unicast_with_atomic_inc(
            &senderW,
            left_packet_header,
            0,
            0,
            0,
            0,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{sem_noc_addr});
    }
    if (e_hops > 0) {
        fabric_unicast_noc_fused_unicast_with_atomic_inc(
            &senderE,
            right_packet_header,
            0,
            0,
            0,
            0,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{sem_noc_addr});
    }
    if (n_hops > 0) {
        fabric_unicast_noc_fused_unicast_with_atomic_inc(
            &senderN,
            north_packet_header,
            0,
            0,
            0,
            0,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{sem_noc_addr});
    }
    if (s_hops > 0) {
        fabric_unicast_noc_fused_unicast_with_atomic_inc(
            &senderS,
            south_packet_header,
            0,
            0,
            0,
            0,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{sem_noc_addr});
    }

    noc_async_writes_flushed();
}
