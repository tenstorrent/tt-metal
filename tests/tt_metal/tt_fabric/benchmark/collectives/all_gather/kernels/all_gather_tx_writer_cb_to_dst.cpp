// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
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
//   0: dst_base            (u32)  // receiver buffer base (start of concat buffer)
//   1: dst_mesh_id         (u32)  // logical (truncated to u16)
//   2: dst_dev_id          (u32)  // logical (truncated to u16)
//   3: rank_offset_bytes   (u32)  // this sender's slot offset = rank*S
//   … fabric-connection args … (inserted by append_fabric_connection_rt_args on host)
//   … 2D hops (Phase-A diag) …
//   … completion fan-out:
//       num_ranks (u32),
//       then for each r in [0..num_ranks-1]:
//         rx_noc_x (u32), rx_noc_y (u32), sem_l1_addr (u32)

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr uint32_t TOTAL_PAGES = get_compile_time_arg_val(CTA_BASE + 0);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 1);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    size_t idx = 0;
    const uint32_t dst_base = get_arg_val<uint32_t>(idx++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_dev_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint32_t rank_offset_bytes = get_arg_val<uint32_t>(idx++);

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

    // Set multicast route once using Phase-A hop counts (2D temporary API).
    auto mh = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header);
    fabric_set_mcast_route(
        reinterpret_cast<volatile tt_l1_ptr LowLatencyMeshPacketHeader*>(mh),
        /*dst_dev_id (ignored)*/ 0,
        /*dst_mesh_id (ignored)*/ 0,
        /*e_num_hops*/ e_hops,
        /*w_num_hops*/ w_hops,
        /*n_num_hops*/ n_hops,
        /*s_num_hops*/ s_hops);

    sender.open<true>();

    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/PAGE_SIZE);

    for (uint32_t i = 0; i < TOTAL_PAGES; ++i) {
        cb_wait_front(CB_ID, 1);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        // Pace transmissions so we don’t overrun the fabric send queue.
        sender.wait_for_empty_write_slot();

        // Place this sender's slice at [rank_offset_bytes .. rank_offset_bytes+S)
        uint64_t dest_noc_addr = dst_acc.get_noc_addr(/*page_id=*/i, /*offset=*/rank_offset_bytes, /*noc=*/0);

        // Build the NOC header for this page (mcast route already set above)
        header->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);

        // TEMP (2D API): payload then header. Will be a single call after uplift
        // 1) send payload (no header)
        sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, PAGE_SIZE);
        // 2) send header (completes the packet)
        sender.send_payload_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));

        cb_pop_front(CB_ID, 1);
    }

    noc_async_writes_flushed();

    // Final signal fan-out: bump EVERY receiver's semaphore (expected=num_ranks).
    const uint32_t num_ranks = get_arg_val<uint32_t>(idx++);
    for (uint32_t r = 0; r < num_ranks; ++r) {
        const uint32_t rx_x = get_arg_val<uint32_t>(idx++);
        const uint32_t rx_y = get_arg_val<uint32_t>(idx++);
        const uint32_t sem_l1 = get_arg_val<uint32_t>(idx++);
        const uint64_t sem_noc = safe_get_noc_addr(rx_x, rx_y, sem_l1, /*NOC_INDEX=*/0);
        // Keep completion unicast per receiver
        fabric_set_unicast_route(mh, eth_chan_directions::EAST, /*my_dev_id*/ 0, dst_dev_id, dst_mesh_id, /*ew_dim*/ 0);
        header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, /*inc=*/1, /*width_bits=*/32));
        sender.wait_for_empty_write_slot();
        sender.send_payload_flush_non_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));
    }
    sender.close();
}
