// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(PROFILE_NOC_EVENTS) && \
    (!defined(DISPATCH_KERNEL) || (defined(DISPATCH_KERNEL) && (PROFILE_KERNEL == PROFILER_OPT_DO_DISPATCH_CORES)))

#include "tools/profiler/noc_event_profiler.hpp"
#include "fabric/fabric_edm_packet_header.hpp"

namespace kernel_profiler {

// how slow is this? alternative is sotring entire route buffer which isn't ideal either...
template <uint32_t STATIC_ID = 12345>
FORCE_INLINE void recordRoutingFields2D(
    const volatile tt::tt_fabric::LowLatencyMeshRoutingFields& routing_fields, const volatile uint8_t* route_buffer) {
    KernelProfilerNocEventMetadata ev_md;
    auto& routing_fields_2d = ev_md.data.fabric_routing_fields_2d;
    routing_fields_2d.noc_xfer_type = KernelProfilerNocEventMetadata::NocEventType::FABRIC_ROUTING_FIELDS_2D;

    // dimension order routing: first we have N/S forwarding with possible branching/local writes for mcast
    uint8_t total_hops = 0;
    while (route_buffer[total_hops] & tt::tt_fabric::LowLatencyMeshRoutingFields::WRITE_AND_FORWARD_NS) {
        total_hops++;
    }

    routing_fields_2d.ns_hops = total_hops;

    // compute e/w hops and check for e/w line mcast
    while (route_buffer[total_hops] != tt::tt_fabric::LowLatencyMeshRoutingFields::NOOP) {
        total_hops++;
    }

    // Look at last entry in buffer to check if west branch exists
    // If west branch exists, compute west hops and east hops as remaining
    // Otherwise, we only have east hops (which is trivially to 0 if we have no e/w hops at all)
    if (route_buffer[total_hops - 1] == tt::tt_fabric::LowLatencyMeshRoutingFields::FORWARD_EAST) {
        routing_fields_2d.w_hops = total_hops - routing_fields.branch_west_offset;
        routing_fields_2d.e_hops = routing_fields.branch_west_offset - routing_fields_2d.ns_hops;
    } else {
        routing_fields_2d.e_hops = total_hops - routing_fields_2d.ns_hops;
    }

    // look at first entries of trunk/branches in buffer to check for N/S line mcast, E/W line mcast, or 2d mcast
    // the last check is for the 1N1E1W edge case (which is a 2d mcast)
    if (routing_fields_2d.ns_hops > 0 &&
            (route_buffer[0] & tt::tt_fabric::LowLatencyMeshRoutingFields::WRITE_AND_FORWARD_NS) ==
                tt::tt_fabric::LowLatencyMeshRoutingFields::WRITE_AND_FORWARD_NS ||
        routing_fields_2d.e_hops > 0 && route_buffer[routing_fields.branch_east_offset] ==
                                            tt::tt_fabric::LowLatencyMeshRoutingFields::WRITE_AND_FORWARD_EW ||
        routing_fields_2d.w_hops > 0 && route_buffer[routing_fields.branch_west_offset] ==
                                            tt::tt_fabric::LowLatencyMeshRoutingFields::WRITE_AND_FORWARD_EW ||
        routing_fields_2d.e_hops > 0 && routing_fields_2d.w_hops > 0) {
        routing_fields_2d.is_mcast = true;
    }

    kernel_profiler::flush_to_dram_if_full<kernel_profiler::DoingDispatch::DISPATCH>();
    kernel_profiler::timeStampedData<STATIC_ID, kernel_profiler::DoingDispatch::DISPATCH>(ev_md.asU64());
}

void record_fabric_header(const volatile PACKET_HEADER_TYPE* fabric_header_ptr) {
    // determine routing fields type at compile time
    KernelProfilerNocEventMetadata::FabricPacketType routing_fields_type;
    if constexpr (std::is_base_of_v<tt::tt_fabric::LowLatencyMeshRoutingFields, ROUTING_FIELDS_TYPE>) {
        routing_fields_type = KernelProfilerNocEventMetadata::FabricPacketType::LOW_LATENCY_MESH;
    } else if constexpr (std::is_base_of_v<tt::tt_fabric::LowLatencyRoutingFields, ROUTING_FIELDS_TYPE>) {
        routing_fields_type = KernelProfilerNocEventMetadata::FabricPacketType::LOW_LATENCY;
    } else if constexpr (std::is_base_of_v<tt::tt_fabric::RoutingFields, ROUTING_FIELDS_TYPE>) {
        routing_fields_type = KernelProfilerNocEventMetadata::FabricPacketType::REGULAR;
    }

    // first profiler packet stores XY address data as well as packet type tag (used to decode routing fields)
    auto noc_send_type = fabric_header_ptr->get_noc_send_type();
    switch (noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            const volatile auto& unicast_write_cmd = fabric_header_ptr->get_command_fields().unicast_write;
            noc_event_profiler::recordFabricNocEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_WRITE,
                routing_fields_type,
                unicast_write_cmd.noc_address);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            const volatile auto& unicast_write_cmd = fabric_header_ptr->get_command_fields().unicast_seminc;
            noc_event_profiler::recordFabricNocEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_ATOMIC_INC,
                routing_fields_type,
                unicast_write_cmd.noc_address);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC: {
            const volatile auto& unicast_write_cmd = fabric_header_ptr->get_command_fields().unicast_seminc_fused;
            noc_event_profiler::recordFabricNocEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_FUSED_UNICAST_ATOMIC_INC,
                routing_fields_type,
                unicast_write_cmd.noc_address);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE: {
            const volatile auto& unicast_write_cmd = fabric_header_ptr->get_command_fields().unicast_inline_write;
            noc_event_profiler::recordFabricNocEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_INLINE_WRITE,
                routing_fields_type,
                unicast_write_cmd.noc_address);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE: {
            const volatile auto& unicast_write_cmd = fabric_header_ptr->get_command_fields().unicast_scatter_write;
            noc_event_profiler::recordFabricScatterEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_SCATTER_WRITE,
                routing_fields_type,
                unicast_write_cmd.noc_address,
                unicast_write_cmd.chunk_size,
                NOC_SCATTER_WRITE_MAX_CHUNKS);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_MULTICAST_WRITE: {
            const volatile auto& mcast_write_cmd = fabric_header_ptr->get_command_fields().mcast_write;
            noc_event_profiler::recordFabricNocEventMulticast(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_MULTICAST_WRITE,
                routing_fields_type,
                mcast_write_cmd.noc_x_start,
                mcast_write_cmd.noc_y_start,
                mcast_write_cmd.mcast_rect_size_x,
                mcast_write_cmd.mcast_rect_size_y);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_MULTICAST_ATOMIC_INC: {
            const volatile auto& mcast_write_cmd = fabric_header_ptr->get_command_fields().mcast_seminc;
            noc_event_profiler::recordFabricNocEventMulticast(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_MULTICAST_ATOMIC_INC,
                routing_fields_type,
                mcast_write_cmd.noc_x_start,
                mcast_write_cmd.noc_y_start,
                mcast_write_cmd.size_x,
                mcast_write_cmd.size_y);
            break;
        }
    }

    // following profiler event just stores the routing fields
    if constexpr (std::is_base_of_v<tt::tt_fabric::LowLatencyMeshRoutingFields, ROUTING_FIELDS_TYPE>) {
#if defined(FABRIC_2D) && !defined(DYNAMIC_ROUTING_ENABLED)
        recordRoutingFields2D(fabric_header_ptr->routing_fields, fabric_header_ptr->route_buffer);
#endif
    } else if constexpr (std::is_base_of_v<tt::tt_fabric::LowLatencyRoutingFields, ROUTING_FIELDS_TYPE>) {
        noc_event_profiler::recordRoutingFields1D(fabric_header_ptr->routing_fields.value);
    } else if constexpr (std::is_base_of_v<tt::tt_fabric::RoutingFields, ROUTING_FIELDS_TYPE>) {
        noc_event_profiler::recordRoutingFields1D(fabric_header_ptr->routing_fields.value);
    }
}
}  // namespace kernel_profiler

#define RECORD_FABRIC_HEADER(_fabric_header_ptr)                   \
    {                                                              \
        kernel_profiler::record_fabric_header(_fabric_header_ptr); \
    }

#else

// null macros when noc tracing is disabled
#define RECORD_FABRIC_HEADER(_fabric_header_ptr)

#endif
