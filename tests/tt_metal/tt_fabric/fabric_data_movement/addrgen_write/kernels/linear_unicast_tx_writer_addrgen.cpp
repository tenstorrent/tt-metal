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
#include "kernel_common.hpp"

using namespace tt;
using namespace tt::tt_fabric;
using namespace tt::tt_fabric::linear::experimental;

//
// Linear (1D) unicast writer kernel using addrgen overload.
// Sends pages from CB c_0 to the dst device using compile-time parameters to select:
//   - OPERATION_TYPE: BasicWrite (only BasicWrite is supported for this linear test)
//   - API_VARIANT: Basic, WithState, or SetState
//
// CT args:
//   TensorAccessorArgs at offset 0
//   0: OPERATION_TYPE (OperationType enum: BasicWrite)
//   1: API_VARIANT (ApiVariant enum: Basic, WithState, SetState)
//   2: TOTAL_PAGES
//   3: PAGE_SIZE (actual data size to transfer)
//   4: ALIGNED_PAGE_SIZE (destination buffer spacing for address calculation)
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
    constexpr uint32_t OPERATION_TYPE = get_compile_time_arg_val(CTA_BASE + 0);
    constexpr uint32_t API_VARIANT = get_compile_time_arg_val(CTA_BASE + 1);
    constexpr uint32_t TOTAL_PAGES = get_compile_time_arg_val(CTA_BASE + 2);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 3);
    constexpr uint32_t ALIGNED_PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 4);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    // Cast to enum types for clearer comparisons
    constexpr auto operation_type = static_cast<OperationType>(OPERATION_TYPE);
    constexpr auto api_variant = static_cast<ApiVariant>(API_VARIANT);

    // This kernel is simplified for BasicWrite and Basic/WithState/SetState API variants only
    static_assert(operation_type == OperationType::BasicWrite, "Linear unicast writer only supports BasicWrite");

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

    // Route setup - required for Basic and WithState variants
    if constexpr (api_variant == ApiVariant::Basic || api_variant == ApiVariant::WithState) {
        header->to_chip_unicast(num_hops);
    }

    // WithState pattern: manually set send type
    if constexpr (api_variant == ApiVariant::WithState) {
        header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
    }

    sender.open<true>();

    // Create TensorAccessor for destination address generation
    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/ALIGNED_PAGE_SIZE);

    // Pre-loop setup for WithState and SetState variants
    if constexpr (api_variant == ApiVariant::WithState) {
        auto initial_noc_addr = tt::tt_fabric::linear::addrgen_detail::get_noc_address(dst_acc, 0, 0);
        header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{initial_noc_addr}, PAGE_SIZE);
    } else if constexpr (api_variant == ApiVariant::SetState) {
        fabric_unicast_noc_unicast_write_set_state(
            header,
            num_hops,
            dst_acc,
            0  // page_id for initial configuration
        );
    }

    // Main loop - process pages
    for (uint32_t i = 0; i < TOTAL_PAGES; i++) {
        cb_wait_front(CB_ID, 1);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        if constexpr (api_variant == ApiVariant::Basic) {
            // Use the linear addrgen overload (Basic variant)
            fabric_unicast_noc_unicast_write(
                &sender,
                header,
                src_l1_addr,
                dst_acc,
                i,         // page_id
                num_hops,  // unicast hop count
                0          // offset
            );
        } else {  // WithState or SetState
            // Use the linear addrgen overload with state
            fabric_unicast_noc_unicast_write_with_state(
                &sender,
                header,
                src_l1_addr,
                dst_acc,
                i,         // page_id
                num_hops,  // unicast hop count
                0          // offset
            );
        }

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
