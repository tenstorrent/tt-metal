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
#include "kernel_common.hpp"

using namespace tt;
using namespace tt::tt_fabric;
using namespace tt::tt_fabric::mesh::experimental;

//
// Unified unicast writer (fabric sender) kernel — consolidates 9 variants.
// Sends pages from CB c_0 to the dst device using compile-time parameters to select:
//   - OPERATION_TYPE: BasicWrite, Scatter, or FusedAtomicInc
//   - API_VARIANT: Basic, WithState, or SetState
//
// CT args:
//   0: OPERATION_TYPE (OperationType enum: BasicWrite, Scatter, FusedAtomicInc)
//   1: API_VARIANT (ApiVariant enum: Basic, WithState, SetState)
//   2: TOTAL_PAGES
//   3: PAGE_SIZE (actual data size to transfer)
//   4: ALIGNED_PAGE_SIZE (destination buffer spacing for address calculation)
//   5: SRC_ALIGNED_PAGE_SIZE (source CB stride for scatter operations)
//
// RT args (must match host):
//   0: dst_base       (u32)  // receiver buffer base (L1 offset or DRAM base)
//   1: dst_mesh_id    (u32)  // logical (truncated to u16)
//   2: dst_dev_id     (u32)  // logical (truncated to u16)
//   3: rx_noc_x       (u32)  // receiver worker XY
//   4: rx_noc_y       (u32)
//   5: sem_l1_addr    (u32)  // receiver L1 semaphore address
//   ... fabric connection args ...

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr uint32_t OPERATION_TYPE = get_compile_time_arg_val(CTA_BASE + 0);
    constexpr uint32_t API_VARIANT = get_compile_time_arg_val(CTA_BASE + 1);
    constexpr uint32_t TOTAL_PAGES = get_compile_time_arg_val(CTA_BASE + 2);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 3);
    constexpr uint32_t ALIGNED_PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 4);
    constexpr uint32_t SRC_ALIGNED_PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 5);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    // Cast to enum types for clearer comparisons
    constexpr auto operation_type = static_cast<OperationType>(OPERATION_TYPE);
    constexpr auto api_variant = static_cast<ApiVariant>(API_VARIANT);

    // Scatter requires even number of pages
    if constexpr (operation_type == OperationType::Scatter) {
        ASSERT((TOTAL_PAGES % 2) == 0);
    }

    size_t idx = 0;
    const uint32_t dst_base = get_arg_val<uint32_t>(idx++);
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_dev_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint32_t rx_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr = get_arg_val<uint32_t>(idx++);

    // Build a fabric send adapter from the runtime args that the host packed.
    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    // TEMP (2D API): manual packet header.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();

    // Fabric route setup (temporary 2D API):
    auto mh = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header);
#if defined(DYNAMIC_ROUTING_ENABLED)
    static_assert(false, "Dynamic routing is not supported");
#endif

    // Route setup - required for Basic and WithState variants
    if constexpr (api_variant == ApiVariant::Basic || api_variant == ApiVariant::WithState) {
        (void)fabric_set_unicast_route(mh, /*dst_dev_id=*/dst_dev_id, /*dst_mesh_id=*/dst_mesh_id);
    }

    // WithState pattern: manually set send type
    if constexpr (api_variant == ApiVariant::WithState) {
        if constexpr (operation_type == OperationType::BasicWrite) {
            header->noc_send_type = tt::tt_fabric::NOC_UNICAST_WRITE;
        } else if constexpr (operation_type == OperationType::Scatter) {
            header->noc_send_type = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
        } else if constexpr (operation_type == OperationType::FusedAtomicInc) {
            header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
        }
    }

    sender.open<true>();

    // For non-scatter: Use ALIGNED_PAGE_SIZE (dst) for address calculation
    // For scatter: Use SRC_ALIGNED_PAGE_SIZE to match CB stride (less BW efficient but correct)
    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/ALIGNED_PAGE_SIZE);
    const auto scatter_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/SRC_ALIGNED_PAGE_SIZE);

    // FusedAtomicInc: compute semaphore NOC address before loop
    uint64_t sem_noc = 0;
    if constexpr (operation_type == OperationType::FusedAtomicInc) {
        ASSERT(sem_l1_addr != 0);
        sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);
    }

    // Pre-loop setup for WithState and SetState variants
    if constexpr (api_variant == ApiVariant::WithState) {
        if constexpr (operation_type == OperationType::BasicWrite) {
            auto initial_noc_addr = tt::tt_fabric::addrgen_detail::get_noc_address(dst_acc, 0, 0);
            header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{initial_noc_addr}, PAGE_SIZE);
        } else if constexpr (operation_type == OperationType::Scatter) {
            // Use scatter_acc with SRC_ALIGNED_PAGE_SIZE to match CB stride
            auto noc_addr0 = tt::tt_fabric::addrgen_detail::get_noc_address(scatter_acc, 0, 0);
            auto noc_addr1 = tt::tt_fabric::addrgen_detail::get_noc_address(scatter_acc, 1, 0);
            header->to_noc_unicast_scatter_write(
                tt::tt_fabric::NocUnicastScatterCommandHeader(
                    {noc_addr0, noc_addr1}, {static_cast<uint16_t>(SRC_ALIGNED_PAGE_SIZE)}),
                SRC_ALIGNED_PAGE_SIZE * 2);
        } else if constexpr (operation_type == OperationType::FusedAtomicInc) {
            auto initial_noc_addr = tt::tt_fabric::addrgen_detail::get_noc_address(dst_acc, 0, 0);
            header->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{initial_noc_addr, sem_noc, 1, true}, PAGE_SIZE);
        }
    } else if constexpr (api_variant == ApiVariant::SetState) {
        if constexpr (operation_type == OperationType::BasicWrite) {
            fabric_unicast_noc_unicast_write_set_state(
                header,
                dst_dev_id,
                dst_mesh_id,
                dst_acc,
                0  // page_id for initial configuration
            );
        } else if constexpr (operation_type == OperationType::Scatter) {
            // Use scatter_acc with SRC_ALIGNED_PAGE_SIZE to match CB stride
            fabric_unicast_noc_scatter_write_set_state(
                header,
                dst_dev_id,
                dst_mesh_id,
                scatter_acc,
                0,  // page_id0
                1,  // page_id1
                0,  // offset0
                0   // offset1
            );
        } else if constexpr (operation_type == OperationType::FusedAtomicInc) {
            fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
                header,
                dst_dev_id,
                dst_mesh_id,
                dst_acc,
                0,  // page_id for initial configuration
                sem_noc,
                1,    // val (increment by 1)
                0,    // offset
                true  // flush
            );
        }
    }

    // Main loop - process pages
    constexpr uint32_t loop_increment = (operation_type == OperationType::Scatter) ? 2 : 1;
    constexpr uint32_t cb_wait_count = (operation_type == OperationType::Scatter) ? 2 : 1;

    for (uint32_t i = 0; i < TOTAL_PAGES; i += loop_increment) {
        cb_wait_front(CB_ID, cb_wait_count);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        // Write operation - branch on operation_type and api_variant
        if constexpr (operation_type == OperationType::BasicWrite) {
            if constexpr (api_variant == ApiVariant::Basic) {
                fabric_unicast_noc_unicast_write(
                    &sender,
                    header,
                    dst_dev_id,
                    dst_mesh_id,
                    src_l1_addr,
                    dst_acc,
                    i,
                    0  // offset
                );
            } else {  // WithState or SetState
                fabric_unicast_noc_unicast_write_with_state(
                    &sender,
                    header,
                    dst_dev_id,
                    dst_mesh_id,
                    src_l1_addr,
                    dst_acc,
                    i,
                    0  // offset
                );
            }
        } else if constexpr (operation_type == OperationType::Scatter) {
            // Use scatter_acc with SRC_ALIGNED_PAGE_SIZE to match CB stride
            if constexpr (api_variant == ApiVariant::Basic) {
                fabric_unicast_noc_scatter_write(
                    &sender,
                    header,
                    dst_dev_id,
                    dst_mesh_id,
                    src_l1_addr,
                    scatter_acc,
                    i,      // page_id0
                    i + 1,  // page_id1
                    0,      // offset0
                    0       // offset1
                );
            } else {  // WithState or SetState
                fabric_unicast_noc_scatter_write_with_state(
                    &sender,
                    header,
                    dst_dev_id,
                    dst_mesh_id,
                    src_l1_addr,
                    scatter_acc,
                    i,      // page_id0
                    i + 1,  // page_id1
                    0,      // offset0
                    0       // offset1
                );
            }
        } else if constexpr (operation_type == OperationType::FusedAtomicInc) {
            if constexpr (api_variant == ApiVariant::Basic) {
                fabric_unicast_noc_fused_unicast_with_atomic_inc(
                    &sender,
                    header,
                    dst_dev_id,
                    dst_mesh_id,
                    src_l1_addr,
                    dst_acc,
                    i,
                    sem_noc,
                    1,    // val (increment by 1)
                    0,    // offset
                    true  // flush
                );
            } else {  // WithState or SetState
                fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
                    &sender,
                    header,
                    dst_dev_id,
                    dst_mesh_id,
                    src_l1_addr,
                    dst_acc,
                    i,
                    sem_noc,
                    1,    // val (increment by 1)
                    0,    // offset
                    true  // flush
                );
            }
        }

        // Synchronization - Scatter uses write_barrier, others can use writes_flushed
        if constexpr (operation_type == OperationType::Scatter) {
            noc_async_write_barrier();
        } else {
            noc_async_writes_flushed();
        }

        cb_pop_front(CB_ID, cb_wait_count);
    }

    noc_async_writes_flushed();

    // Post-loop completion: BasicWrite and Scatter need separate atomic inc
    // FusedAtomicInc does NOT (it's fused with each write)
    if constexpr (operation_type != OperationType::FusedAtomicInc) {
        ASSERT(sem_l1_addr != 0);
        const uint64_t sem_noc_final = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);

        fabric_unicast_noc_unicast_atomic_inc(
            &sender,
            header,
            dst_dev_id,
            dst_mesh_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_final, /*inc=*/1, /*width_bits=*/32));
    }

    sender.close();
}
