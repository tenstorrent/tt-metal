// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_adapter.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"

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
    if constexpr (enable_deadlock_avoidance) {
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
__attribute__((optimize("jump-tables")))
#ifndef FABRIC_2D
FORCE_INLINE
#endif
    void
    execute_chip_unicast_to_local_chip(
        tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
        uint16_t payload_size_bytes,
        uint32_t transaction_id,
        uint8_t rx_channel_id) {
    const auto& header = *packet_start;
    uint32_t payload_start_address = reinterpret_cast<size_t>(packet_start) + sizeof(PACKET_HEADER_TYPE);

    constexpr bool update_counter = false;

    tt::tt_fabric::NocSendType noc_send_type = header.noc_send_type;
    if (noc_send_type > tt::tt_fabric::NocSendType::NOC_SEND_TYPE_LAST) {
        __builtin_unreachable();
    }
    switch (noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            const auto dest_address = header.command_fields.unicast_write.noc_address;
            noc_async_write_one_packet_with_trid<update_counter, false>(
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
            noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(
                dest_address,
                value,
                0xF,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);
        } break;

        case tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC: {
            const auto dest_address = header.command_fields.unicast_seminc_fused.noc_address;
            noc_async_write_one_packet_with_trid<update_counter, false>(
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

        case tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE: {
            const auto& scatter = header.command_fields.unicast_scatter_write;
            const uint8_t chunk_count = scatter.chunk_count;

            // NOTE: when chunk_count < 4, chunk_size[n-2] can be used without calculating final_chunk_size.
            //       However the perf (n == 2) is much worse than implementation below.
            //       Need to check perf with 2 <= n <= 4
            size_t offset = 0;
            const uint8_t last_chunk_index = chunk_count - 1;
            uint16_t chunk_size = scatter.chunk_size[0];
            noc_async_write_one_packet_with_trid<update_counter, false>(
                payload_start_address + offset,
                scatter.noc_address[0],
                chunk_size,
                transaction_id,
                tt::tt_fabric::local_chip_data_cmd_buf,
                tt::tt_fabric::edm_to_local_chip_noc);
            offset += chunk_size;
            if (chunk_count > 2) {
                chunk_size = scatter.chunk_size[1];
                noc_async_write_one_packet_with_trid<update_counter, false>(
                    payload_start_address + offset,
                    scatter.noc_address[1],
                    chunk_size,
                    transaction_id,
                    tt::tt_fabric::local_chip_data_cmd_buf,
                    tt::tt_fabric::edm_to_local_chip_noc);
                offset += chunk_size;
                if (chunk_count == 4) [[likely]] {
                    chunk_size = scatter.chunk_size[2];
                    noc_async_write_one_packet_with_trid<update_counter, false>(
                        payload_start_address + offset,
                        scatter.noc_address[2],
                        chunk_size,
                        transaction_id,
                        tt::tt_fabric::local_chip_data_cmd_buf,
                        tt::tt_fabric::edm_to_local_chip_noc);
                    offset += chunk_size;
                }
            }

            const uint16_t final_chunk_size = static_cast<uint16_t>(payload_size_bytes - offset);
            noc_async_write_one_packet_with_trid<update_counter, false>(
                payload_start_address + offset,
                scatter.noc_address[last_chunk_index],
                final_chunk_size,
                transaction_id,
                tt::tt_fabric::local_chip_data_cmd_buf,
                tt::tt_fabric::edm_to_local_chip_noc);
        } break;
        case tt::tt_fabric::NocSendType::NOC_MULTICAST_WRITE:
        case tt::tt_fabric::NocSendType::NOC_MULTICAST_ATOMIC_INC:
        default: {
            ASSERT(false);
        } break;
    };
}

// Forward packet to local relay in UDM mode
// Unlike execute_chip_unicast_to_local_chip, this sends the FULL packet (header + payload)
// to the relay, which will then handle forwarding to local chip workers
//
// !!!WARNING!!! * ENSURE RELAY HAS SPACE FOR PACKET BEFORE CALLING
template <typename LocalRelayInterfaceT>
__attribute__((optimize("jump-tables"))) void execute_chip_unicast_to_relay(
    LocalRelayInterfaceT& local_relay_interface,
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    uint32_t transaction_id,
    uint8_t rx_channel_id) {
    // Assert that relay has space (best effort check)
    ASSERT(local_relay_interface.edm_has_space_for_packet());

    // Send the full packet (header + payload) to relay
    // The relay will handle the local chip forwarding
    uint32_t packet_address = reinterpret_cast<size_t>(packet_start);
    uint32_t total_size_bytes = payload_size_bytes + sizeof(PACKET_HEADER_TYPE);

    // Send to relay using the same mechanism as router-to-router forwarding
    local_relay_interface.template send_payload_non_blocking_from_address_with_trid<
        enable_deadlock_avoidance,
        tt::tt_fabric::edm_to_downstream_noc,
        false,  // stateful_api
        true    // increment_pointers
        >(packet_address, total_size_bytes, transaction_id);
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

/**
 * Update packet header for next hop (1D Low Latency routing)
 *
 * ExtensionWords=0 (≤16 hops): Compiles to single shift instruction
 * ExtensionWords>0 (>16 hops): Includes refill logic
 */
FORCE_INLINE void update_packet_header_for_next_hop(
    volatile tt_l1_ptr tt::tt_fabric::LowLatencyPacketHeader* packet_header,
    tt::tt_fabric::LowLatencyRoutingFields cached_routing_fields) {
    using LowLatencyFields = tt::tt_fabric::RoutingFieldsConstants::LowLatency;

    // Shift to consume current hop (always happens)
    uint32_t new_value = cached_routing_fields.value >> LowLatencyFields::FIELD_WIDTH;

    // Refill logic - only included when ExtensionWords > 0 (>16 hops)
#if defined(FABRIC_1D_PKT_HDR_EXTENSION_WORDS) && (FABRIC_1D_PKT_HDR_EXTENSION_WORDS > 0)
    // route_buffer exists: include refill logic for >16 hop packets
    constexpr uint32_t EXT = FABRIC_1D_PKT_HDR_EXTENSION_WORDS;

    if (new_value == 0) [[unlikely]] {
        // Refill from buffer[0]
        new_value = cached_routing_fields.route_buffer[0];

// Shift buffer left
#pragma unroll
        for (uint32_t i = 0; i < EXT - 1; i++) {
            const_cast<uint32_t*>(packet_header->routing_fields.route_buffer)[i] =
                cached_routing_fields.route_buffer[i + 1];
        }
        const_cast<uint32_t*>(packet_header->routing_fields.route_buffer)[EXT - 1] = 0;
    } else {
// No refill needed - just copy buffer as-is
#pragma unroll
        for (uint32_t i = 0; i < EXT; i++) {
            const_cast<uint32_t*>(packet_header->routing_fields.route_buffer)[i] =
                cached_routing_fields.route_buffer[i];
        }
    }
#endif

    // Write new value (always happens)
    packet_header->routing_fields.value = new_value;
}

FORCE_INLINE void update_packet_header_for_next_hop(
    volatile tt_l1_ptr tt::tt_fabric::HybridMeshPacketHeader* packet_header,
    tt::tt_fabric::LowLatencyMeshRoutingFields cached_routing_fields) {
    if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
        packet_header->routing_fields.value = cached_routing_fields.value + 1;
    }
}

template <uint8_t NUM_SENDER_BUFFERS>
void update_packet_header_for_next_hop(
    tt::tt_fabric::EdmToEdmSender<NUM_SENDER_BUFFERS>& downstream_edm_interface, uint32_t value) {
    if constexpr (UPDATE_PKT_HDR_ON_RX_CH) {
        tt::tt_fabric::HybridMeshPacketHeader* packet_base = nullptr;
        std::uintptr_t offset = reinterpret_cast<std::uintptr_t>(&(packet_base->routing_fields));
        downstream_edm_interface.template update_edm_buffer_slot_word(
            offset, value, tt::tt_fabric::edm_to_downstream_noc);
    }
}

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
template <bool enable_deadlock_avoidance, bool stateful_api, bool increment_pointers = true, uint8_t NUM_SENDER_BUFFERS>
#if !defined(FABRIC_2D) && !defined(ARCH_BLACKHOLE)
FORCE_INLINE
#endif
    void
    forward_payload_to_downstream_edm(
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
        uint16_t payload_size_bytes,
        ROUTING_FIELDS_TYPE cached_routing_fields,
        tt::tt_fabric::EdmToEdmSender<NUM_SENDER_BUFFERS>& downstream_edm_interface,
        uint8_t transaction_id) {
    // TODO: PERF - this should already be getting checked by the caller so this should be redundant make it an ASSERT
    ASSERT(downstream_edm_interface.template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>());  // best effort check

    // This is a good place to print the packet header for debug if you are trying to inspect packets
    // because it is before we start manipulating the header for forwarding
    if constexpr (increment_pointers) {
        update_packet_header_for_next_hop(packet_header, cached_routing_fields);
    }
    downstream_edm_interface.template send_payload_non_blocking_from_address_with_trid<
        enable_deadlock_avoidance,
        tt::tt_fabric::edm_to_downstream_noc,
        stateful_api,
        increment_pointers>(
        reinterpret_cast<size_t>(packet_header), payload_size_bytes + sizeof(PACKET_HEADER_TYPE), transaction_id);
    if constexpr (!increment_pointers) {
        update_packet_header_for_next_hop(downstream_edm_interface, cached_routing_fields.value);
    }
}
