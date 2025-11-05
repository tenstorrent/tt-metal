// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <type_traits>
#include "dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/addrgen_api_common.h"
#include "accessor/tensor_accessor.h"
#include "accessor/tensor_accessor_args.h"

using namespace tt;
using namespace tt::tt_fabric;

// UpdateMask enum for fused atomic inc stateful variants
enum class UnicastFusedAtomicIncUpdateMask : uint32_t {
    None = 0,
    WriteDstAddr = 1u << 0,
    SemaphoreAddr = 1u << 1,
    Wrap = 1u << 2,
    Val = 1u << 3,
    Flush = 1u << 4,
    PayloadSize = 1u << 5,
};

// Bitwise OR operator for combining mask flags
constexpr inline UnicastFusedAtomicIncUpdateMask operator|(
    UnicastFusedAtomicIncUpdateMask a, UnicastFusedAtomicIncUpdateMask b) {
    return static_cast<UnicastFusedAtomicIncUpdateMask>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

// Helper to check if a flag is set in the mask
template <UnicastFusedAtomicIncUpdateMask UpdateMask, UnicastFusedAtomicIncUpdateMask Flag>
constexpr bool has_flag() {
    return (static_cast<uint32_t>(UpdateMask) & static_cast<uint32_t>(Flag)) != 0u;
}

// Kernel-compatible populate helper that uses to_noc_fused_unicast_write_atomic_inc
template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename CommandHeaderT>
FORCE_INLINE void populate_unicast_fused_atomic_inc_fields(
    volatile PACKET_HEADER_TYPE* packet_header, uint16_t packet_size_bytes, const CommandHeaderT& command_header) {
    if constexpr (has_flag<UpdateMask, UnicastFusedAtomicIncUpdateMask::PayloadSize>()) {
        packet_header->payload_size_bytes = packet_size_bytes;
    }
    if constexpr (has_flag<UpdateMask, UnicastFusedAtomicIncUpdateMask::WriteDstAddr>()) {
        if constexpr (!std::is_same_v<CommandHeaderT, std::nullptr_t>) {
            // Use kernel-compatible API to set write destination and other fields
            packet_header->to_noc_fused_unicast_write_atomic_inc(command_header, packet_header->payload_size_bytes);
        }
    } else {
        // If not updating WriteDstAddr, we still need to update other individual fields if needed
        // (This path would be for masks that don't include WriteDstAddr but include other fields)
        if constexpr (has_flag<UpdateMask, UnicastFusedAtomicIncUpdateMask::SemaphoreAddr>()) {
            packet_header->command_fields.unicast_seminc_fused.semaphore_noc_address =
                command_header.semaphore_noc_address;
        }
        if constexpr (has_flag<UpdateMask, UnicastFusedAtomicIncUpdateMask::Wrap>()) {
            packet_header->command_fields.unicast_seminc_fused.wrap = command_header.wrap;
        }
        if constexpr (has_flag<UpdateMask, UnicastFusedAtomicIncUpdateMask::Val>()) {
            packet_header->command_fields.unicast_seminc_fused.val = command_header.val;
        }
        if constexpr (has_flag<UpdateMask, UnicastFusedAtomicIncUpdateMask::Flush>()) {
            packet_header->command_fields.unicast_seminc_fused.flush = command_header.flush;
        }
    }
}

// Base _with_state function (needed by addrgen overload)
template <UnicastFusedAtomicIncUpdateMask UpdateMask, typename FabricSenderType>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint32_t src_addr,
    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader noc_fused_unicast_atomic_inc_command_header = {},
    uint16_t packet_size_bytes = 0) {
    populate_unicast_fused_atomic_inc_fields<UpdateMask>(
        packet_header, packet_size_bytes, noc_fused_unicast_atomic_inc_command_header);
    client_interface->wait_for_empty_write_slot();
    client_interface->send_payload_without_header_non_blocking_from_address(
        src_addr, packet_header->payload_size_bytes);
    client_interface->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
}

// Addrgen overload for _with_state (matches mesh/api.h)
template <typename FabricSenderType, typename AddrGenType>
FORCE_INLINE void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt_l1_ptr FabricSenderType* client_interface,
    volatile PACKET_HEADER_TYPE* packet_header,
    uint8_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint32_t src_addr,
    const AddrGenType& addrgen,
    uint32_t page_id,
    uint64_t semaphore_noc_address,
    uint16_t val,
    uint16_t wrap,
    uint32_t offset = 0,
    bool flush = true) {
    auto page_size = tt::tt_fabric::addrgen_detail::get_page_size(addrgen);
    auto noc_address = tt::tt_fabric::addrgen_detail::get_noc_address(addrgen, page_id, offset);

    // Call base _with_state with UpdateMask to update WriteDstAddr and PayloadSize only
    // The route and send type should already be set for optimal performance
    // Note: dst_dev_id/dst_mesh_id are kept for API symmetry but not used by base _with_state
    constexpr auto UpdateMask =
        UnicastFusedAtomicIncUpdateMask::WriteDstAddr | UnicastFusedAtomicIncUpdateMask::PayloadSize;
    fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<UpdateMask>(
        client_interface,
        packet_header,
        src_addr,
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{noc_address, semaphore_noc_address, val, wrap, flush},
        page_size);
}

//
// Writer (fabric sender) kernel — sends pages from CB c_0 to the dst device using fused atomic inc with state.
// This version uses the fused atomic inc _with_state addrgen overload API for more efficient repeated sends.
// Route and send type are set once before the loop, then _with_state is called for each page.
// The fused operation combines unicast write + atomic increment in ONE fabric operation per page.
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
    // Set route once before the loop (for _with_state variant)
    (void)fabric_set_unicast_route(mh, /*dst_dev_id=*/dst_dev_id, /*dst_mesh_id=*/dst_mesh_id);
    // Set the send type to fused atomic inc once
    header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;

    sender.open<true>();

    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/PAGE_SIZE);

    // Compute semaphore NOC address (used in fused write+atomic_inc)
    ASSERT(sem_l1_addr != 0);
    const uint64_t sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);

    for (uint32_t i = 0; i < TOTAL_PAGES; ++i) {
        cb_wait_front(CB_ID, 1);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        // Use fused addrgen overload with_state variant - updates only WriteDstAddr and PayloadSize
        // The route and send type are already set, so this is more efficient than the base variant
        fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
            &sender,
            header,
            dst_dev_id,
            dst_mesh_id,
            src_l1_addr,
            dst_acc,  // TensorAccessor as addrgen
            i,        // page_id
            sem_noc,  // semaphore NOC address
            1,        // val (increment by 1)
            0,        // wrap (no wrap)
            0,        // offset
            true      // flush
        );

        cb_pop_front(CB_ID, 1);
    }

    noc_async_writes_flushed();

    // No separate atomic inc needed - it's fused with each write operation!

    sender.close();
}
