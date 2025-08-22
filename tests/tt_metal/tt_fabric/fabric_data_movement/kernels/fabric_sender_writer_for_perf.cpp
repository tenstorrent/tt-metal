// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"  // InterleavedAddrGen, safe_get_noc_addr

using namespace tt;
using namespace tt::tt_fabric;

// CT args:
//   0: TOTAL_PAGES
//   1: PAGE_SIZE
//
// RT args (must match host):
//   0: dst_base       (u32)  // receiver buffer base (L1 offset or DRAM base)
//   1: dst_is_dram    (u32)  // 0=L1, 1=DRAM
//   2: dst_mesh_id    (u32)  // logical (truncated to u16)
//   3: dst_dev_id     (u32)  // logical (truncated to u16)
//   4: rx_noc_x       (u32)  // receiver worker XY
//   5: rx_noc_y       (u32)
//   6: sem_l1_addr    (u32)  // receiver L1 semaphore address
//   [then]: append_fabric_connection_rt_args(...)

void kernel_main() {
    constexpr uint32_t TOTAL_PAGES = get_compile_time_arg_val(0);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(1);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;
    constexpr uint32_t k_noc_index = 0;  // normalize to NOC0

    size_t idx = 0;
    const uint32_t dst_base = get_arg_val<uint32_t>(idx++);
    const bool dst_is_dram = (get_arg_val<uint32_t>(idx++) != 0);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_dev_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint32_t rx_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr = get_arg_val<uint32_t>(idx++);

    // Build fabric connection from remaining RT args
    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    // Reusable packet header in L1
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
    zero_l1_buf((uint32_t*)header, sizeof(PACKET_HEADER_TYPE));

    // Fabric header (2D dynamic routing): route to (dst_mesh_id, dst_dev_id)
    fabric_set_unicast_route(
        reinterpret_cast<volatile tt_l1_ptr MeshPacketHeader*>(header),
        eth_chan_directions::EAST,  // ignored for dynamic routing
        /*my_dev_id*/ 0,            // ignored
        /*dst_dev_id*/ dst_dev_id,
        /*dst_mesh_id*/ dst_mesh_id,
        /*ew_dim*/ 0  // ignored
    );

    sender.open<true>();

    for (uint32_t i = 0; i < TOTAL_PAGES; ++i) {
        cb_wait_front(CB_ID, 1);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        sender.wait_for_empty_write_slot();

        // Compute destination NOC address (DRAM interleaved vs L1 + XY)
        uint64_t dest_noc_addr;
        if (dst_is_dram) {
            const InterleavedAddrGen<true> gen{.bank_base_address = dst_base, .page_size = PAGE_SIZE};
            dest_noc_addr = get_noc_addr(/*page_idx=*/i, gen, /*bank=*/0, /*noc_index=*/k_noc_index);
        } else {
            const uint32_t l1_off = dst_base + i * PAGE_SIZE;
            dest_noc_addr = safe_get_noc_addr(rx_noc_x, rx_noc_y, l1_off, /*noc_index=*/k_noc_index);
        }

        // Build the NOC header for this page
        header->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);

        // 1) send payload (no header)
        sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, PAGE_SIZE);
        // 2) send header (completes the packet)
        sender.send_payload_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));

        cb_pop_front(CB_ID, 1);
    }

    noc_async_writes_flushed();

    // Final signal: bump receiver semaphore so the receiver kernel exits
    if (sem_l1_addr != 0) {
        const uint64_t sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*noc_index=*/k_noc_index);
        header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, /*inc=*/1, /*width_bits=*/32));
        sender.wait_for_empty_write_slot();
        sender.send_payload_flush_non_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
    }

    sender.close();
}
