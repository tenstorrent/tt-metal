// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"

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
//   0: dst_base       (u32)  // receiver buffer base (L1 offset or DRAM base)
//   1: dst_mesh_id    (u32)  // logical (truncated to u16)
//   2: dst_dev_id     (u32)  // logical (truncated to u16)
//   3: rx_noc_x       (u32)  // receiver worker XY
//   4: rx_noc_y       (u32)
//   5: sem_l1_addr    (u32)  // receiver L1 (forward/completion) semaphore address
//   6: return_sem_addr(u32)  // SOURCE-local L1 addr of the return semaphore (round-trip wait target)
//   7: ts_out_l1_addr (u32)  // SOURCE-local L1 scratch for the round-trip cycle delta (2x u32)

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr uint32_t TOTAL_PAGES = get_compile_time_arg_val(CTA_BASE + 0);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 1);
    constexpr bool is_2d_fabric =
        get_compile_time_arg_val(CTA_BASE + 2) != 0;  // route by dev/mesh (2D) vs num_hops (1D)
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    size_t idx = 0;
    const uint32_t dst_base = get_arg_val<uint32_t>(idx++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_dev_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint32_t rx_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t return_sem_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t ts_out_l1_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t fwd_num_hops = get_arg_val<uint32_t>(idx++);  // 1D routing distance (unused in 2D)

    // Build a fabric send adapter from the runtime args that the host packed.
    // Needed before sending over fabric: binds this core to a specific routing/link.
    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    // TEMP (2D API): manual packet header. Post-uplift this becomes implicit.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();

    // Fabric route setup (temporary 2D API):
    // Program a fixed unicast route to (dst_mesh_id, dst_dev_id). Dynamic routing is not
    // supported in this path (see guard below). This API will change soon. The future 2D
    // interface will mirror the 1D style. See linear/api.h for the reference shape.
    if constexpr (is_2d_fabric) {
        (void)fabric_set_unicast_route(
            (HybridMeshPacketHeader*)header, /*dst_dev_id=*/dst_dev_id, /*dst_mesh_id=*/dst_mesh_id);
    } else {
        (void)fabric_set_unicast_route<false>((LowLatencyPacketHeader*)header, (uint16_t)fwd_num_hops);
    }

    sender.open<true>();

    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base);

    for (uint32_t i = 0; i < TOTAL_PAGES; ++i) {
        cb_wait_front(CB_ID, 1);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        // Pace transmissions so we don’t overrun the fabric send queue.
        sender.wait_for_empty_write_slot();

        // Compute destination NOC address (DRAM or L1 interleaved)
        uint64_t dest_noc_addr = dst_acc.get_noc_addr(/*page_id=*/i, /*offset=*/0, /*noc=*/0);

        // Build the NOC header for this page
        header->to_noc_unicast_write(NocUnicastCommandHeader{dest_noc_addr}, PAGE_SIZE);

        // TEMP (2D API): payload then header. Will be a single call after uplift
        // 1) send payload (no header)
        sender.send_payload_without_header_non_blocking_from_address(src_l1_addr, PAGE_SIZE);
        // 2) send header (completes the packet)
        sender.send_payload_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));

        cb_pop_front(CB_ID, 1);
    }

    noc_async_writes_flushed();

    // Final signal: bump receiver semaphore so the receiver kernel exits.
    // In this benchmark we always have a completion semaphore.
    ASSERT(sem_l1_addr != 0);

    const uint64_t sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);

    if constexpr (is_2d_fabric) {
        (void)fabric_set_unicast_route(
            (HybridMeshPacketHeader*)header, /*dst_dev_id=*/dst_dev_id, /*dst_mesh_id=*/dst_mesh_id);
    } else {
        (void)fabric_set_unicast_route<false>((LowLatencyPacketHeader*)header, (uint16_t)fwd_num_hops);
    }
    header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(sem_noc, /*inc=*/1));

    sender.wait_for_empty_write_slot();
    sender.send_payload_flush_non_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));

    // ---- Round-trip latency: time the control-packet RTT on the SOURCE clock only ----
    // t0 is taken AFTER the forward inc is handed to fabric (excludes the local issue cost,
    // matching tt_fabric_latency_sender.cpp; the per-hop slope is invariant to this choice).
    // The receiver bounces an atomic-inc back to `return_sem_addr` on this core; t1 is taken
    // when we observe it. Skew and per-chip AICLK cancel because both timestamps come from
    // this chip's free-running wall clock.
    noc_async_writes_flushed();
    const uint64_t t0 = get_timestamp();

    volatile tt_l1_ptr uint32_t* return_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(return_sem_addr);
    noc_semaphore_wait(return_sem_ptr, 1);
    const uint64_t t1 = get_timestamp();
    noc_semaphore_set(return_sem_ptr, 0);

    const uint64_t rtt_cycles = t1 - t0;
    volatile tt_l1_ptr uint32_t* ts_out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_out_l1_addr);
    ts_out[0] = static_cast<uint32_t>(rtt_cycles & 0xFFFFFFFFu);
    ts_out[1] = static_cast<uint32_t>(rtt_cycles >> 32);

    sender.close();
}
