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
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"
#include "accessor/tensor_accessor.h"
#include "accessor/tensor_accessor_args.h"
#include "kernel_common.hpp"

using namespace tt;
using namespace tt::tt_fabric;
using namespace tt::tt_fabric::linear::experimental;
using namespace tt::tt_fabric::common::experimental;

//
// Linear (1D) unicast writer kernel for connection manager variants.
// Sends pages from CB c_0 to the dst device using RoutingPlaneConnectionManager.
// Currently supports BasicWrite with ConnMgrBasic variant only.
//
// CT args:
//   TensorAccessorArgs at offset 0
//   0: OPERATION_TYPE (OperationType enum: BasicWrite only for now)
//   1: API_VARIANT (ApiVariant enum: ConnMgrBasic only for now)
//   2: TOTAL_PAGES
//   3: PAGE_SIZE (actual data size to transfer)
//   4: ALIGNED_PAGE_SIZE (destination buffer spacing for address calculation)
//   5: SRC_ALIGNED_PAGE_SIZE (source CB stride - unused for BasicWrite, but kept for consistency)
//
// RT args (must match host):
//   0: dst_base       (u32)  // receiver buffer base (L1 offset or DRAM base)
//   1: rx_noc_x       (u32)  // receiver worker XY
//   2: rx_noc_y       (u32)
//   3: sem_l1_addr    (u32)  // receiver L1 semaphore address
//   4: num_connections (u32) // number of connections in route
//   5+: num_hops[0..num_connections-1] (packed as uint32_t)
//   ... routing plane connection manager args ...

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr uint32_t OPERATION_TYPE = get_compile_time_arg_val(CTA_BASE + 0);
    constexpr uint32_t API_VARIANT = get_compile_time_arg_val(CTA_BASE + 1);
    constexpr uint32_t TOTAL_PAGES = get_compile_time_arg_val(CTA_BASE + 2);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 3);
    constexpr uint32_t ALIGNED_PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 4);
    [[maybe_unused]] constexpr uint32_t SRC_ALIGNED_PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 5);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    // Cast to enum types for clearer comparisons
    constexpr auto operation_type = static_cast<OperationType>(OPERATION_TYPE);
    constexpr auto api_variant = static_cast<ApiVariant>(API_VARIANT);

    // Currently only supports BasicWrite with ConnMgrBasic
    static_assert(operation_type == OperationType::BasicWrite, "Only BasicWrite supported for linear conn mgr");
    static_assert(api_variant == ApiVariant::ConnMgrBasic, "Only ConnMgrBasic supported for linear conn mgr");

    size_t idx = 0;
    const uint32_t dst_base = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr = get_arg_val<uint32_t>(idx++);

    // Connection manager variant: allocate route and build connection manager
    const uint32_t num_connections = get_arg_val<uint32_t>(idx++);

    // Read num_hops array (one per connection)
    uint8_t num_hops[8];  // Max 8 connections supported
    for (uint32_t i = 0; i < num_connections; ++i) {
        num_hops[i] = static_cast<uint8_t>(get_arg_val<uint32_t>(idx++));
    }

    tt::tt_fabric::RoutingPlaneConnectionManager connection_manager;
    uint8_t route_id = PacketHeaderPool::allocate_header_n(num_connections);
    open_connections(connection_manager, num_connections, idx);

    // Use ALIGNED_PAGE_SIZE (dst) for address calculation
    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/ALIGNED_PAGE_SIZE);

    // Main loop - process pages
    for (uint32_t i = 0; i < TOTAL_PAGES; ++i) {
        cb_wait_front(CB_ID, 1);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        // Use linear connection manager addrgen overload
        fabric_unicast_noc_unicast_write(
            connection_manager,
            route_id,
            src_l1_addr,
            dst_acc,
            i,         // page_id
            num_hops,  // per-connection hop counts
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
        connection_manager,
        route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_final, /*inc=*/1, /*width_bits=*/32),
        num_hops);

    close_connections(connection_manager);
}
