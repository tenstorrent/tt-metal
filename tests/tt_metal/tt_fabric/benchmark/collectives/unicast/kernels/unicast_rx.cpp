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

using namespace tt;
using namespace tt::tt_fabric;

// Receiver-side completion wait + return-path bounce (for round-trip latency).
//
// Runs on the destination device. Blocks until the forward completion semaphore reaches
// `expected_value` (the sender atomically increments it after sending all payload, and
// fabric guarantees the semaphore signal is delivered after payload data). It then sends a
// single atomic-inc back over fabric to a semaphore on the SOURCE sender core, so the sender
// can time the src->dst->src round trip entirely on its own clock (no cross-chip skew).
//
// CT args: none
// RT args:
//   0: fwd_sem_addr     (u32)  // L1 addr of the forward (completion) semaphore on this dst core
//   1: expected_value   (u32)  // e.g. 1
//   2: src_mesh_id      (u32)  // logical (truncated to u16) — return route target
//   3: src_dev_id       (u32)  // logical (truncated to u16)
//   4: src_noc_x        (u32)  // source sender-core NOC coords (return-sem owner)
//   5: src_noc_y        (u32)
//   6: return_sem_addr  (u32)  // L1 addr of the return semaphore on the source sender core
//   then: fabric-connection args consumed by WorkerToFabricEdmSender::build_from_args (appended last)
void kernel_main() {
    size_t idx = 0;
    const uint32_t fwd_sem_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t expected_value = get_arg_val<uint32_t>(idx++);
    const uint16_t src_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t src_dev_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint32_t src_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t src_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t return_sem_addr = get_arg_val<uint32_t>(idx++);

    // Build the return fabric send adapter (dst -> src). The host appended the
    // fabric-connection args last, so build_from_args consumes them from here.
    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    volatile tt_l1_ptr uint32_t* fwd_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_sem_addr);

    noc_semaphore_wait(fwd_sem_ptr, expected_value);

    // Reset for next iteration so we always observe a fresh 0->1 transition.
    noc_semaphore_set(fwd_sem_ptr, 0);

    // Bounce a single atomic-inc back to the source's return semaphore.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
    auto mh = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header);
    (void)fabric_set_unicast_route(mh, /*dst_dev_id=*/src_dev_id, /*dst_mesh_id=*/src_mesh_id);

    sender.open<true>();

    const uint64_t ret_noc = safe_get_noc_addr(src_noc_x, src_noc_y, return_sem_addr, /*NOC_INDEX=*/0);
    header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader(ret_noc, /*inc=*/1));

    sender.wait_for_empty_write_slot();
    sender.send_payload_flush_non_blocking_from_address((uint32_t)header, sizeof(PACKET_HEADER_TYPE));

    sender.close();
}
