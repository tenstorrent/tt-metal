// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tools/profiler/noc_event_profiler.hpp"
#include "api/tt-metalium/fabric_edm_packet_header.hpp"

namespace kernel_profiler {

void record_fabric_header(const volatile PACKET_HEADER_TYPE* fabric_header_ptr) {
#ifdef PROFILE_NOC_EVENTS

    // determine routing fields type at compile time
    KernelProfilerNocEventMetadata::FabricPacketType routing_fields_type;
    if constexpr (std::is_base_of_v<tt::tt_fabric::LowLatencyMeshRoutingFields, ROUTING_FIELDS_TYPE>) {
        routing_fields_type = KernelProfilerNocEventMetadata::FabricPacketType::LOW_LATENCY_MESH;
    } else if constexpr (std::is_base_of_v<tt::tt_fabric::LowLatencyRoutingFields, ROUTING_FIELDS_TYPE>) {
        routing_fields_type = KernelProfilerNocEventMetadata::FabricPacketType::LOW_LATENCY;
    } else if constexpr (std::is_base_of_v<tt::tt_fabric::RoutingFields, ROUTING_FIELDS_TYPE>) {
        routing_fields_type = KernelProfilerNocEventMetadata::FabricPacketType::REGULAR;
    }

    auto noc_send_type = fabric_header_ptr->get_noc_send_type();
    switch (noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            const volatile auto& unicast_write_cmd = fabric_header_ptr->get_command_fields().unicast_write;
            noc_event_profiler::recordFabricNocEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_WRITE,
                routing_fields_type,
                unicast_write_cmd.noc_address,
                fabric_header_ptr->routing_fields.value);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            const volatile auto& unicast_write_cmd = fabric_header_ptr->get_command_fields().unicast_seminc;
            noc_event_profiler::recordFabricNocEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_ATOMIC_INC,
                routing_fields_type,
                unicast_write_cmd.noc_address,
                fabric_header_ptr->routing_fields.value);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC: {
            const volatile auto& unicast_write_cmd = fabric_header_ptr->get_command_fields().unicast_seminc_fused;
            noc_event_profiler::recordFabricNocEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_FUSED_UNICAST_ATOMIC_INC,
                routing_fields_type,
                unicast_write_cmd.noc_address,
                fabric_header_ptr->routing_fields.value);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE: {
            const volatile auto& unicast_write_cmd = fabric_header_ptr->get_command_fields().unicast_inline_write;
            noc_event_profiler::recordFabricNocEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_INLINE_WRITE,
                routing_fields_type,
                unicast_write_cmd.noc_address,
                fabric_header_ptr->routing_fields.value);
            break;
        }
        case tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE: {
            const volatile auto& unicast_write_cmd = fabric_header_ptr->get_command_fields().unicast_scatter_write;
            noc_event_profiler::recordFabricScatterEvent(
                KernelProfilerNocEventMetadata::NocEventType::FABRIC_UNICAST_SCATTER_WRITE,
                routing_fields_type,
                unicast_write_cmd.noc_address,
                unicast_write_cmd.chunk_size,
                NOC_SCATTER_WRITE_MAX_CHUNKS,
                fabric_header_ptr->routing_fields.value);
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
                mcast_write_cmd.mcast_rect_size_y,
                fabric_header_ptr->routing_fields.value);
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
                mcast_write_cmd.size_y,
                fabric_header_ptr->routing_fields.value);
            break;
        }
    }
#endif
}
}  // namespace kernel_profiler

#define RECORD_FABRIC_HEADER(_fabric_header_ptr)                   \
    {                                                              \
        kernel_profiler::record_fabric_header(_fabric_header_ptr); \
    }
