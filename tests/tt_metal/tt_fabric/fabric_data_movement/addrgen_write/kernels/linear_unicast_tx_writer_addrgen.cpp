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
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "accessor/tensor_accessor.h"
#include "accessor/tensor_accessor_args.h"

using namespace tt;
using namespace tt::tt_fabric;
using namespace tt::tt_fabric::linear::experimental;

//
// Linear (1D) unicast writer kernel using addrgen overload.
// Sends pages from CB c_0 to the dst device using the linear fabric API.
//
// CT args:
//   TensorAccessorArgs at offset 0
//   0: TOTAL_PAGES
//   1: PAGE_SIZE (actual data size to transfer)
//   2: ALIGNED_PAGE_SIZE (destination buffer spacing for address calculation)
//
// RT args (must match host):
//   0: dst_base       (u32)  // receiver buffer base (L1 offset or DRAM base)
//   1: rx_noc_x       (u32)  // receiver worker XY
//   2: rx_noc_y       (u32)
//   3: sem_l1_addr    (u32)  // receiver L1 semaphore address
//   4: num_hops       (u32)  // unicast hop count
//   ... fabric connection args ...

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr uint32_t TOTAL_PAGES = get_compile_time_arg_val(CTA_BASE + 0);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 1);
    constexpr uint32_t ALIGNED_PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 2);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    size_t idx = 0;
    const uint32_t dst_base = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr = get_arg_val<uint32_t>(idx++);
    const uint8_t num_hops = static_cast<uint8_t>(get_arg_val<uint32_t>(idx++));

    // Build a fabric send adapter from the runtime args that the host packed.
    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    // Allocate packet header
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();

    sender.open<true>();

    // Create TensorAccessor for destination address generation
    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/ALIGNED_PAGE_SIZE);

    // Main loop - process pages
    for (uint32_t i = 0; i < TOTAL_PAGES; i++) {
        cb_wait_front(CB_ID, 1);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        // Use the linear addrgen overload
        fabric_unicast_noc_unicast_write(
            &sender,
            header,
            src_l1_addr,
            dst_acc,
            i,         // page_id
            num_hops,  // unicast hop count
            0          // offset
        );

        noc_async_writes_flushed();
        cb_pop_front(CB_ID, 1);
    }

    noc_async_writes_flushed();

    // Post-loop completion: send atomic inc to signal receiver
    ASSERT(sem_l1_addr != 0);
    const uint64_t sem_noc_final = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);

    fabric_unicast_noc_unicast_atomic_inc(
        &sender,
        header,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_final, /*inc=*/1, /*width_bits=*/32),
        num_hops);

    sender.close();
}
