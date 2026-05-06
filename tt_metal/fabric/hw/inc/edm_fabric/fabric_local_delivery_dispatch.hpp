// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// fabric_local_delivery_dispatch.hpp
// =============================================================================
//
// Outlined Phase-7 local-delivery dispatch + packet-header debug helpers,
// extracted from `fabric_edm_packet_transmission.hpp` so callers (in
// particular the CRAQ-Fabric generated kernel) can include the shared logic
// WITHOUT transitively pulling in `fabric_erisc_router_ct_args.hpp`.
//
// This is part of the structural decoupling work required by:
//   feedback_no_upstream_ct_args.md
//   "Generated kernel must NOT use upstream EDM CT args"
//
// The CT-arg-derived constants used by `execute_chip_unicast_to_local_chip_impl`
// (NOC id, command buffer id, NOC VC, TRID range, deadlock-avoidance flag,
// channel-trimming usage recorder) are surfaced as template parameters of
// `LocalDeliveryDispatchConfig` so each caller binds them from its own
// CT-arg infrastructure.
//
// `fabric_edm_packet_transmission.hpp` is now a thin compatibility shim that
// includes this header AND `fabric_erisc_router_ct_args.hpp`, then re-exposes
// the legacy free-function names bound to upstream CT-arg constants.
//
// Backwards-compatibility invariant: upstream callers (fabric_erisc_router.cpp,
// fabric_erisc_router_speedy_path.hpp, etc.) must keep working unchanged.
// =============================================================================

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_adapter.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

// If the hop/distance counter equals to the below value, it indicates that it has
// arrived at (at least one of) the intended destination(s)
static constexpr size_t LOCAL_DELIVERY_DESTINATION_HOP_COUNT = 1;
// TODO: make 0 and the associated field to num mcast destinations
static constexpr size_t LOCAL_DELIVERY_LAST_MCAST_DESTINATION = 1;

// -----------------------------------------------------------------------------
// Packet-header DPRINT helpers (CT-arg-free).
// -----------------------------------------------------------------------------

FORCE_INLINE void print_pkt_hdr_routing_fields_outlined(
    volatile tt::tt_fabric::PacketHeader* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    switch (packet_start->chip_send_type) {
        case tt::tt_fabric::CHIP_UNICAST: {
            DPRINT << "C_UNI: dist:"
                   << (uint32_t)(packet_start->routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK)
                   << "\n";
            DEVICE_PRINT(
                "C_UNI: dist:{}\n",
                (uint32_t)(packet_start->routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK));
            break;
        }
        case tt::tt_fabric::CHIP_MULTICAST: {
            DPRINT << "C_MCST: dist:"
                   << (uint32_t)(packet_start->routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK)
                   << ", rng:"
                   << (uint32_t)((packet_start->routing_fields.value & tt::tt_fabric::RoutingFields::RANGE_MASK) >>
                                 tt::tt_fabric::RoutingFields::START_DISTANCE_FIELD_BIT_WIDTH)
                   << "\n";
            DEVICE_PRINT(
                "C_MCST: dist:{}, rng:{}\n",
                (uint32_t)(packet_start->routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK),
                (uint32_t)((packet_start->routing_fields.value & tt::tt_fabric::RoutingFields::RANGE_MASK) >>
                           tt::tt_fabric::RoutingFields::START_DISTANCE_FIELD_BIT_WIDTH));
            break;
        }
    };
#endif
}

FORCE_INLINE void print_pkt_hdr_routing_fields_outlined(
    volatile tt::tt_fabric::LowLatencyPacketHeader* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    DPRINT << "ROUTE:" << packet_start->routing_fields.value << "\n";
    DEVICE_PRINT("ROUTE:{}\n", packet_start->routing_fields.value);
#endif
}

template <typename T>
FORCE_INLINE void print_pkt_header_noc_fields_outlined(volatile T* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    switch (packet_start->noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            DPRINT << "N_WR addr:" << (uint64_t)packet_start->command_fields.unicast_write.noc_address << "\n";
            DEVICE_PRINT("N_WR addr:{}\n", (uint64_t)packet_start->command_fields.unicast_write.noc_address);
        } break;
        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            DPRINT << "N_WR addr:" << (uint64_t)packet_start->command_fields.unicast_seminc.noc_address
                   << ", val:" << (uint32_t)packet_start->command_fields.unicast_seminc.val << "\n";
            DEVICE_PRINT(
                "N_WR addr:{}, val:{}\n",
                (uint64_t)packet_start->command_fields.unicast_seminc.noc_address,
                (uint32_t)packet_start->command_fields.unicast_seminc.val);

        } break;
        default:
            ASSERT(false);  // unimplemented
            break;
    };
#endif
}

FORCE_INLINE void print_pkt_header_outlined(volatile tt::tt_fabric::PacketHeader* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    auto const& header = *packet_start;
    DPRINT << "PKT: nsnd_t:" << (uint32_t)packet_start->noc_send_type
           << ", csnd_t:" << (uint32_t)packet_start->chip_send_type
           << ", src_chip:" << (uint32_t)packet_start->src_ch_id
           << ", payload_size_bytes:" << (uint32_t)packet_start->payload_size_bytes << "\n";
    DEVICE_PRINT(
        "PKT: nsnd_t:{} csnd_t:{} src_chip:{} payload_size_bytes:{}\n",
        (uint32_t)packet_start->noc_send_type,
        (uint32_t)packet_start->chip_send_type,
        (uint32_t)packet_start->src_ch_id,
        (uint32_t)packet_start->payload_size_bytes);
    print_pkt_hdr_routing_fields_outlined(packet_start);
    print_pkt_header_noc_fields_outlined(packet_start);
#endif
}

FORCE_INLINE void print_pkt_header_outlined(volatile tt::tt_fabric::LowLatencyPacketHeader* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    auto const& header = *packet_start;
    DPRINT << "PKT: nsnd_t:" << (uint32_t)packet_start->noc_send_type
           << ", src_chip:" << (uint32_t)packet_start->src_ch_id
           << ", payload_size_bytes:" << (uint32_t)packet_start->payload_size_bytes << "\n";
    DEVICE_PRINT(
        "PKT: nsnd_t:{} src_chip:{} payload_size_bytes:{}\n",
        (uint32_t)packet_start->noc_send_type,
        (uint32_t)packet_start->src_ch_id,
        (uint32_t)packet_start->payload_size_bytes);
    print_pkt_hdr_routing_fields_outlined(packet_start);
    print_pkt_header_noc_fields_outlined(packet_start);
#endif
}

// Shifts the chunk encoding in a scatter write packet to the next chunk
FORCE_INLINE void shift_to_next_chunk_outlined(uint8_t& chunk_encodings) { chunk_encodings >>= 2; }

// -----------------------------------------------------------------------------
// LocalDeliveryDispatchConfig
// -----------------------------------------------------------------------------
//
// Compile-time configuration bundle binding the constants previously taken
// from `fabric_erisc_router_ct_args.hpp` directly. Each caller (upstream's
// EDM router, the CRAQ-Fabric generated kernel, etc.) builds one of these
// from its own CT-arg layer and passes it as a template parameter to
// `execute_chip_unicast_to_local_chip_impl_outlined`.
//
// The defaults below are NOT meaningful for any real device — every caller
// must specify them. They are provided only so the type can be default-
// instantiated in headers that do not have CT args resolved at parse time.
template <
    uint8_t LocalChipNoc,
    uint8_t LocalChipDataCmdBuf,
    uint8_t ForwardAndLocalWriteNocVc,
    bool LocalChipNocEqualsDownstreamNoc,
    uint8_t DownstreamNoc,
    bool EnableDeadlockAvoidance,
    uint8_t NumTransactionIds>
struct LocalDeliveryDispatchConfig {
    static constexpr uint8_t local_chip_noc = LocalChipNoc;
    static constexpr uint8_t local_chip_data_cmd_buf = LocalChipDataCmdBuf;
    static constexpr uint8_t forward_and_local_write_noc_vc = ForwardAndLocalWriteNocVc;
    static constexpr bool local_chip_noc_equals_downstream_noc = LocalChipNocEqualsDownstreamNoc;
    static constexpr uint8_t downstream_noc = DownstreamNoc;
    static constexpr bool enable_deadlock_avoidance = EnableDeadlockAvoidance;
    static constexpr uint8_t num_transaction_ids = NumTransactionIds;
};

// -----------------------------------------------------------------------------
// flush_write_to_noc_pipeline (outlined)
// -----------------------------------------------------------------------------
//
// CT-arg-free counterpart to upstream's `flush_write_to_noc_pipeline`.
// The TRID start array (`RX_CH_TRID_STARTS` upstream) is passed as an explicit
// argument because callers compute it from their own CT-arg layer.
template <typename Config, typename TridStartsArray>
FORCE_INLINE void flush_write_to_noc_pipeline_outlined(
    uint8_t rx_channel_id, const TridStartsArray& rx_ch_trid_starts) {
    if constexpr (Config::enable_deadlock_avoidance) {
        auto start_trid = rx_ch_trid_starts[rx_channel_id];
        auto end_trid = start_trid + Config::num_transaction_ids;
        for (int i = start_trid; i < end_trid; i++) {
            if constexpr (Config::local_chip_noc_equals_downstream_noc) {
                while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(Config::local_chip_noc, i));
            } else {
                while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(Config::downstream_noc, i));
                while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(Config::local_chip_noc, i));
            }
        }
    } else {
        for (size_t i = 0; i < Config::num_transaction_ids; i++) {
            if constexpr (Config::local_chip_noc_equals_downstream_noc) {
                while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(Config::local_chip_noc, i));
            } else {
                while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(Config::downstream_noc, i));
                while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(Config::local_chip_noc, i));
            }
        }
    }
}

// -----------------------------------------------------------------------------
// execute_chip_unicast_to_local_chip_impl (outlined)
// -----------------------------------------------------------------------------
//
// Core implementation of unicast-to-local-chip dispatch, parameterized on the
// per-caller LocalDeliveryDispatchConfig instead of reading global CT-arg
// constants. `usage_recorder` is taken by reference so the caller binds it to
// either upstream's `channel_trimming_usage_recorder` or a local no-op stub.
//
// The function body is a verbatim port of upstream's
// `execute_chip_unicast_to_local_chip_impl`; only the constant references and
// the NOC-pipeline flush call change.
template <typename Config, typename UsageRecorder, typename TridStartsArray>
__attribute__((optimize("jump-tables")))
#ifndef FABRIC_2D
FORCE_INLINE
#endif
    void
    execute_chip_unicast_to_local_chip_impl_outlined(
        tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
        uint16_t payload_size_bytes,
        tt::tt_fabric::NocSendType noc_send_type,
        uint32_t transaction_id,
        uint8_t rx_channel_id,
        UsageRecorder& usage_recorder,
        const TridStartsArray& rx_ch_trid_starts) {
    const auto& header = *packet_start;
    uint32_t payload_start_address = reinterpret_cast<size_t>(packet_start) + sizeof(PACKET_HEADER_TYPE);

    constexpr bool update_counter = false;

    usage_recorder.set_noc_send_type_used(rx_channel_id, noc_send_type);
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
                Config::local_chip_data_cmd_buf,
                Config::local_chip_noc,
                Config::forward_and_local_write_noc_vc);
        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            const uint64_t dest_address = header.command_fields.unicast_seminc.noc_address;
            const auto increment = header.command_fields.unicast_seminc.val;
            if (header.command_fields.unicast_seminc.flush) {
                flush_write_to_noc_pipeline_outlined<Config>(rx_channel_id, rx_ch_trid_starts);
            }
            noc_semaphore_inc<true>(
                dest_address,
                increment,
                Config::local_chip_noc,
                Config::forward_and_local_write_noc_vc);

        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE: {
            const auto dest_address = header.command_fields.unicast_inline_write.noc_address;
            const auto value = header.command_fields.unicast_inline_write.value;
            noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(
                dest_address,
                value,
                0xF,
                Config::local_chip_noc,
                Config::forward_and_local_write_noc_vc);
        } break;

        case tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC: {
            const auto dest_address = header.command_fields.unicast_seminc_fused.noc_address;
            noc_async_write_one_packet_with_trid<update_counter, false>(
                payload_start_address,
                dest_address,
                payload_size_bytes,
                transaction_id,
                Config::local_chip_data_cmd_buf,
                Config::local_chip_noc,
                Config::forward_and_local_write_noc_vc);

            const uint64_t semaphore_dest_address = header.command_fields.unicast_seminc_fused.semaphore_noc_address;
            const auto increment = header.command_fields.unicast_seminc_fused.val;
            if (header.command_fields.unicast_seminc_fused.flush) {
                flush_write_to_noc_pipeline_outlined<Config>(rx_channel_id, rx_ch_trid_starts);
            }
            noc_semaphore_inc<true>(
                semaphore_dest_address,
                increment,
                Config::local_chip_noc,
                Config::forward_and_local_write_noc_vc);
        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE: {
            using ChunkEncoding = tt::tt_fabric::NocScatterWriteChunkEncoding;
            const auto& scatter = header.command_fields.unicast_scatter_write;
            const uint8_t chunk_count = scatter.chunk_count;
            uint8_t packet_encoding = scatter.chunk_encoding;

            size_t offset = 0;
            const uint8_t last_chunk_index = chunk_count - 1;

            uint16_t chunk_size = scatter.chunk_size[0];
            noc_async_write_one_packet_with_trid<update_counter, false>(
                payload_start_address + offset,
                scatter.noc_address[0],
                chunk_size,
                transaction_id,
                Config::local_chip_data_cmd_buf,
                Config::local_chip_noc);
            offset += chunk_size;
            shift_to_next_chunk_outlined(packet_encoding);

            if (chunk_count > 2) {
                chunk_size = scatter.chunk_size[1];
                noc_async_write_one_packet_with_trid<update_counter, false>(
                    payload_start_address + offset,
                    scatter.noc_address[1],
                    chunk_size,
                    transaction_id,
                    Config::local_chip_data_cmd_buf,
                    Config::local_chip_noc);
                offset += chunk_size;
                shift_to_next_chunk_outlined(packet_encoding);

                if (chunk_count == 4) [[likely]] {
                    chunk_size = scatter.chunk_size[2];
                    noc_async_write_one_packet_with_trid<update_counter, false>(
                        payload_start_address + offset,
                        scatter.noc_address[2],
                        chunk_size,
                        transaction_id,
                        Config::local_chip_data_cmd_buf,
                        Config::local_chip_noc);
                    offset += chunk_size;
                    shift_to_next_chunk_outlined(packet_encoding);
                }
            }

            constexpr uint8_t CHUNK_ENCODING_MASK = 0b11;
            const ChunkEncoding chunk_encoding = static_cast<ChunkEncoding>(packet_encoding & CHUNK_ENCODING_MASK);
            const uint64_t final_destination_noc_address = scatter.noc_address[last_chunk_index];
            if (chunk_encoding == ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {
                const uint16_t final_chunk_size = static_cast<uint16_t>(payload_size_bytes - offset);
                noc_async_write_one_packet_with_trid<update_counter, false>(
                    payload_start_address + offset,
                    final_destination_noc_address,
                    final_chunk_size,
                    transaction_id,
                    Config::local_chip_data_cmd_buf,
                    Config::local_chip_noc);
            } else if (chunk_encoding != ChunkEncoding::CHUNK_ENCODING_NOP) {
                if (chunk_encoding == ChunkEncoding::CHUNK_ENCODING_SEMINC_FLUSH) {
                    flush_write_to_noc_pipeline_outlined<Config>(rx_channel_id, rx_ch_trid_starts);
                }
                noc_semaphore_inc<true>(
                    final_destination_noc_address,
                    scatter.chunk_size[last_chunk_index],
                    Config::local_chip_noc,
                    Config::forward_and_local_write_noc_vc);
            } else {
                ASSERT(false);
            }
        } break;

        case tt::tt_fabric::NocSendType::NOC_MULTICAST_WRITE:
        case tt::tt_fabric::NocSendType::NOC_MULTICAST_ATOMIC_INC:
        default: {
            ASSERT(false);
        } break;
    };
}

// Wrapper that resolves noc_send_type from the packet header via a packed 4B
// load (payload_size_bytes + noc_send_type in one read) then delegates to
// `execute_chip_unicast_to_local_chip_impl_outlined`.
template <typename Config, typename UsageRecorder, typename TridStartsArray>
__attribute__((optimize("jump-tables")))
#ifndef FABRIC_2D
FORCE_INLINE
#endif
    void
    execute_chip_unicast_to_local_chip_outlined(
        tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
        uint16_t payload_size_bytes,
        uint32_t transaction_id,
        uint8_t rx_channel_id,
        UsageRecorder& usage_recorder,
        const TridStartsArray& rx_ch_trid_starts) {
    auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_start);
    execute_chip_unicast_to_local_chip_impl_outlined<Config>(
        packet_start,
        payload_size_bytes,
        packed.noc_send_type,
        transaction_id,
        rx_channel_id,
        usage_recorder,
        rx_ch_trid_starts);
}

// -----------------------------------------------------------------------------
// NoOpUsageRecorder
// -----------------------------------------------------------------------------
//
// Minimal stub for callers that do not enable channel-trimming usage capture
// (e.g. the CRAQ-Fabric generated kernel). Models the same `set_noc_send_type_used`
// API as upstream's `tt::tt_fabric::FabricDatapathUsageL1Ptr`.
struct NoOpUsageRecorder {
    FORCE_INLINE void set_noc_send_type_used(uint8_t /*rx_channel_id*/,
                                             tt::tt_fabric::NocSendType /*noc_send_type*/) const {}
};
