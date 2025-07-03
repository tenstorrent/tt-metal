// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "fabric_edm_packet_header.hpp"
#include "edm_fabric_worker_adapters.hpp"
#include "fabric_edm_types.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/1d_fabric_constants.hpp"
#include <cstdint>

// If the hop/distance counter equals to the below value, it indicates that it has
// arrived at (atleast one of) the intended destination(s)
static constexpr size_t DESTINATION_HOP_COUNT = 1;
// TODO: make 0 and the associated field to num mcast destinations
static constexpr size_t LAST_MCAST_DESTINATION = 1;

FORCE_INLINE void print_pkt_hdr_routing_fields(volatile tt::tt_fabric::PacketHeader* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    switch (packet_start->chip_send_type) {
        case tt::tt_fabric::CHIP_UNICAST: {
            DPRINT << "C_UNI: dist:"
                   << (uint32_t)(packet_start->routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK)
                   << "\n";
            break;
        }
        case tt::tt_fabric::CHIP_MULTICAST: {
            DPRINT << "C_MCST: dist:"
                   << (uint32_t)(packet_start->routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK)
                   << ", rng:"
                   << (uint32_t)((packet_start->routing_fields.value & tt::tt_fabric::RoutingFields::RANGE_MASK) >>
                                 tt::tt_fabric::RoutingFields::START_DISTANCE_FIELD_BIT_WIDTH)
                   << "\n";
            break;
        }
    };
#endif
}

FORCE_INLINE void print_pkt_hdr_routing_fields(volatile tt::tt_fabric::LowLatencyPacketHeader* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    DPRINT << "ROUTE:" << packet_start->routing_fields.value << "\n";
#endif
}

template <typename T>
FORCE_INLINE void print_pkt_header_noc_fields(volatile T* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    switch (packet_start->noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            DPRINT << "N_WR addr:" << (uint64_t)packet_start->command_fields.unicast_write.noc_address << "\n";
        } break;
        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            DPRINT << "N_WR addr:" << (uint64_t)packet_start->command_fields.unicast_seminc.noc_address
                   << ", val:" << (uint32_t)packet_start->command_fields.unicast_seminc.val << "\n";

        } break;
        default:
            ASSERT(false);  // unimplemented
            break;
    };
#endif
}

FORCE_INLINE void print_pkt_header(volatile tt::tt_fabric::PacketHeader* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    auto const& header = *packet_start;
    DPRINT << "PKT: nsnd_t:" << (uint32_t)packet_start->noc_send_type
           << ", csnd_t:" << (uint32_t)packet_start->chip_send_type
           << ", src_chip:" << (uint32_t)packet_start->src_ch_id
           << ", payload_size_bytes:" << (uint32_t)packet_start->payload_size_bytes << "\n";
    print_pkt_hdr_routing_fields(packet_start);
    print_pkt_header_noc_fields(packet_start);
#endif
}

FORCE_INLINE void print_pkt_header(volatile tt::tt_fabric::LowLatencyPacketHeader* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    auto const& header = *packet_start;
    DPRINT << "PKT: nsnd_t:" << (uint32_t)packet_start->noc_send_type
           << ", src_chip:" << (uint32_t)packet_start->src_ch_id
           << ", payload_size_bytes:" << (uint32_t)packet_start->payload_size_bytes << "\n";
    print_pkt_hdr_routing_fields(packet_start);
    print_pkt_header_noc_fields(packet_start);
#endif
}

FORCE_INLINE void flush_write_to_noc_pipeline(uint8_t rx_channel_id) {
    if constexpr (enable_ring_support) {
        auto start_trid = RX_CH_TRID_STARTS[rx_channel_id];
        auto end_trid = start_trid + NUM_TRANSACTION_IDS;
        for (int i = start_trid; i < end_trid; i++) {
            if constexpr (tt::tt_fabric::local_chip_noc_equals_downstream_noc) {
                while (
                    !ncrisc_noc_nonposted_write_with_transaction_id_flushed(tt::tt_fabric::edm_to_local_chip_noc, i));
            } else {
                while (
                    !ncrisc_noc_nonposted_write_with_transaction_id_flushed(tt::tt_fabric::edm_to_downstream_noc, i));
                while (
                    !ncrisc_noc_nonposted_write_with_transaction_id_flushed(tt::tt_fabric::edm_to_local_chip_noc, i));
            }
        }
    } else {
        for (size_t i = 0; i < NUM_TRANSACTION_IDS; i++) {
            if constexpr (tt::tt_fabric::local_chip_noc_equals_downstream_noc) {
                while (
                    !ncrisc_noc_nonposted_write_with_transaction_id_flushed(tt::tt_fabric::edm_to_local_chip_noc, i));
            } else {
                while (
                    !ncrisc_noc_nonposted_write_with_transaction_id_flushed(tt::tt_fabric::edm_to_downstream_noc, i));
                while (
                    !ncrisc_noc_nonposted_write_with_transaction_id_flushed(tt::tt_fabric::edm_to_local_chip_noc, i));
            }
        }
    }
}

// Since we unicast to local, we must omit the packet header
// This function only does reads, and within scope there are no modifications to the packet header
__attribute__((optimize("jump-tables"))) FORCE_INLINE void execute_chip_unicast_to_local_chip(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    uint32_t transaction_id,
    uint8_t rx_channel_id) {
    const auto& header = *packet_start;
    uint32_t payload_start_address = reinterpret_cast<size_t>(packet_start) + sizeof(PACKET_HEADER_TYPE);

    tt::tt_fabric::NocSendType noc_send_type = header.noc_send_type;
    if (noc_send_type > tt::tt_fabric::NocSendType::NOC_SEND_TYPE_LAST) {
        __builtin_unreachable();
    }
    switch (noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            const auto dest_address = header.command_fields.unicast_write.noc_address;
            noc_async_write_one_packet_with_trid<false, false>(
                payload_start_address,
                dest_address,
                payload_size_bytes,
                transaction_id,
                tt::tt_fabric::local_chip_data_cmd_buf,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);
        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            const uint64_t dest_address = header.command_fields.unicast_seminc.noc_address;
            const auto increment = header.command_fields.unicast_seminc.val;
            if (header.command_fields.unicast_seminc.flush) {
                flush_write_to_noc_pipeline(rx_channel_id);
            }
            noc_semaphore_inc<true>(
                dest_address,
                increment,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);

        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE: {
            const auto dest_address = header.command_fields.unicast_inline_write.noc_address;
            const auto value = header.command_fields.unicast_inline_write.value;
            noc_inline_dw_write<false, true>(
                dest_address,
                value,
                0xF,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);
        } break;

        case tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC: {
            const auto dest_address = header.command_fields.unicast_seminc_fused.noc_address;
            noc_async_write_one_packet_with_trid<false, false>(
                payload_start_address,
                dest_address,
                payload_size_bytes,
                transaction_id,
                tt::tt_fabric::local_chip_data_cmd_buf,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);

            const uint64_t semaphore_dest_address = header.command_fields.unicast_seminc_fused.semaphore_noc_address;
            const auto increment = header.command_fields.unicast_seminc_fused.val;
            if (header.command_fields.unicast_seminc_fused.flush) {
                flush_write_to_noc_pipeline(rx_channel_id);
            }
            noc_semaphore_inc<true>(
                semaphore_dest_address,
                increment,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);
        } break;

#ifdef ARCH_WORMHOLE
        case tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE: {
            size_t offset = 0;
            size_t chunk_size;
            for (size_t i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS; ++i) {
                if (i == NOC_SCATTER_WRITE_MAX_CHUNKS - 1) {
                    chunk_size = payload_size_bytes - offset;
                } else {
                    chunk_size = header.command_fields.unicast_scatter_write.chunk_size[i];
                }
                const auto dest_address = header.command_fields.unicast_scatter_write.noc_address[i];
                noc_async_write_one_packet_with_trid<false, false>(
                    payload_start_address + offset,
                    dest_address,
                    chunk_size,
                    transaction_id,
                    tt::tt_fabric::local_chip_data_cmd_buf,
                    tt::tt_fabric::edm_to_local_chip_noc);
                offset += chunk_size;
            }
        } break;
#else
        case tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE:
#endif
        case tt::tt_fabric::NocSendType::NOC_MULTICAST_WRITE:
        case tt::tt_fabric::NocSendType::NOC_MULTICAST_ATOMIC_INC:
        default: {
            ASSERT(false);
        } break;
    };
}

FORCE_INLINE void update_packet_header_for_next_hop(
    volatile tt_l1_ptr tt::tt_fabric::PacketHeader* packet_header, tt::tt_fabric::RoutingFields cached_routing_fields) {
    // if the distance field is one, it means the range field decrements, else the start distance field decrements
    // TODO [optimization]: If we can make the terminal value 0, then we can save an instruction on the eq insn
    bool decrement_range = (cached_routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK) ==
                           tt::tt_fabric::RoutingFields::LAST_HOP_DISTANCE_VAL;
    uint8_t decrement_val = static_cast<uint8_t>(1)
                            << (decrement_range * tt::tt_fabric::RoutingFields::RANGE_HOPS_FIELD_BIT_WIDTH);
    packet_header->routing_fields.value = cached_routing_fields.value - decrement_val;
}

FORCE_INLINE void update_packet_header_for_next_hop(
    volatile tt_l1_ptr tt::tt_fabric::LowLatencyPacketHeader* packet_header,
    tt::tt_fabric::LowLatencyRoutingFields cached_routing_fields) {
    packet_header->routing_fields.value =
        cached_routing_fields.value >> tt::tt_fabric::LowLatencyRoutingFields::FIELD_WIDTH;
}

FORCE_INLINE void update_packet_header_for_next_hop(
    volatile tt_l1_ptr tt::tt_fabric::LowLatencyMeshPacketHeader* packet_header,
    tt::tt_fabric::LowLatencyMeshRoutingFields cached_routing_fields) {
    // This is the hop index. At every ethernet hop, we increment by 1
    // so that the next receiver indexes into its respecive hop command
    // in packet_header.route_buffer[]
    packet_header->routing_fields.value = cached_routing_fields.value + 1;
}

FORCE_INLINE void update_packet_header_for_next_hop(
    volatile tt_l1_ptr tt::tt_fabric::MeshPacketHeader* packet_header,
    tt::tt_fabric::LowLatencyMeshRoutingFields cached_routing_fields) {}

// This function forwards a packet to the downstream EDM channel for eventual sending
// to the next chip in the line/ring
//
// Modifies the packet header (decrements hop counts) so ...
//
// !!!WARNING!!!
// !!!WARNING!!! * do NOT call before determining if the packet should be consumed locally or forwarded
// !!!WARNING!!! * ENSURE DOWNSTREAM EDM HAS SPACE FOR PACKET BEFORE CALLING
// !!!WARNING!!!
// This function does a write, so needs to be volatile to avoid compiler optimizations
template <bool enable_ring_support, bool stateful_api, uint8_t NUM_SENDER_BUFFERS>
FORCE_INLINE void forward_payload_to_downstream_edm(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint16_t payload_size_bytes,
    ROUTING_FIELDS_TYPE cached_routing_fields,
    tt::tt_fabric::EdmToEdmSender<NUM_SENDER_BUFFERS>& downstream_edm_interface,
    uint8_t transaction_id) {
    // TODO: PERF - this should already be getting checked by the caller so this should be redundant make it an ASSERT
    ASSERT(downstream_edm_interface.edm_has_space_for_packet());  // best effort check

    // This is a good place to print the packet header for debug if you are trying to inspect packets
    // because it is before we start manipulating the header for forwarding
    update_packet_header_for_next_hop(packet_header, cached_routing_fields);
    downstream_edm_interface.template send_payload_non_blocking_from_address_with_trid<
        enable_ring_support,
        tt::tt_fabric::edm_to_downstream_noc,
        stateful_api>(
        reinterpret_cast<size_t>(packet_header), payload_size_bytes + sizeof(PACKET_HEADER_TYPE), transaction_id);
}
