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
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "accessor/tensor_accessor.h"
#include "accessor/tensor_accessor_args.h"

using namespace tt;
using namespace tt::tt_fabric;
using namespace tt::tt_fabric::mesh::experimental;

//
// Writer (fabric sender) kernel — sends pages from CB c_0 to the dst device using scatter write.
// This version uses the addrgen overload API for cleaner, more maintainable code.
// Per pair of pages: wait for two CB pages → send via scatter addrgen overload (sends to two destinations).
// After all pages: flush, then atomic-inc the receiver's global semaphore (completion signal).
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
//   5: sem_l1_addr    (u32)  // receiver L1 semaphore address

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
    const uint32_t rx_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr = get_arg_val<uint32_t>(idx++);

    // Build a fabric send adapter from the runtime args that the host packed.
    // Needed before sending over fabric: binds this core to a specific routing/link.
    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    // TEMP (2D API): manual packet header. Post-uplift this becomes implicit.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();

    // Fabric route setup (temporary 2D API):
    // Program a fixed unicast route to (dst_mesh_id, dst_dev_id). Dynamic routing is not
    // supported in this path (see guard below). This API will change soon. The future 2D
    // interface will mirror the 1D style. See linear/api.h for the reference shape.
    auto mh = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header);
#if defined(DYNAMIC_ROUTING_ENABLED)
    static_assert(false, "Dynamic routing is not supported");
#endif
    (void)fabric_set_unicast_route(mh, /*dst_dev_id=*/dst_dev_id, /*dst_mesh_id=*/dst_mesh_id);

    sender.open<true>();

    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/PAGE_SIZE);

    // Scatter write requires even number of pages
    ASSERT((TOTAL_PAGES % 2) == 0);

    // Process pages in pairs for scatter write
    for (uint32_t i = 0; i < TOTAL_PAGES; i += 2) {
        // Read 2 pages from CB
        cb_wait_front(CB_ID, 2);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        // Use scatter addrgen overload - sends to two destinations (consecutive pages)
        // First chunk goes to page i, second chunk goes to page i+1
        fabric_unicast_noc_scatter_write(
            &sender,
            header,
            dst_dev_id,
            dst_mesh_id,
            src_l1_addr,
            dst_acc,  // TensorAccessor as addrgen
            i,        // page_id0
            i + 1,    // page_id1
            0,        // offset0
            0         // offset1
        );

        noc_async_write_barrier();
        cb_pop_front(CB_ID, 2);
    }

    noc_async_writes_flushed();

    // Final signal: bump receiver semaphore so the receiver kernel exits.
    // In this test we always have a completion semaphore.
    ASSERT(sem_l1_addr != 0);

    const uint64_t sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);

    fabric_unicast_noc_unicast_atomic_inc(
        &sender,
        header,
        dst_dev_id,
        dst_mesh_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc, /*inc=*/1, /*width_bits=*/32));

    sender.close();
}
