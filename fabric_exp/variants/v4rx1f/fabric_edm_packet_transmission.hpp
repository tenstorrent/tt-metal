// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_adapter.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"

// If the hop/distance counter equals to the below value, it indicates that it has
// arrived at (at least one of) the intended destination(s)
static constexpr size_t DESTINATION_HOP_COUNT = 1;
// TODO: make 0 and the associated field to num mcast destinations
static constexpr size_t LAST_MCAST_DESTINATION = 1;

FORCE_INLINE void print_pkt_hdr_routing_fields(volatile tt::tt_fabric::PacketHeader* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    switch (packet_start->chip_send_type) {
        case tt::tt_fabric::CHIP_UNICAST: {
            DPRINT(
                "C_UNI: dist:{}\n",
                (uint32_t)(packet_start->routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK));
            break;
        }
        case tt::tt_fabric::CHIP_MULTICAST: {
            DPRINT(
                "C_MCST: dist:{}, rng:{}\n",
                (uint32_t)(packet_start->routing_fields.value & tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK),
                (uint32_t)((packet_start->routing_fields.value & tt::tt_fabric::RoutingFields::RANGE_MASK) >>
                           tt::tt_fabric::RoutingFields::START_DISTANCE_FIELD_BIT_WIDTH));
            break;
        }
    };
#endif
}

FORCE_INLINE void print_pkt_hdr_routing_fields(volatile tt::tt_fabric::LowLatencyPacketHeader* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    DPRINT("ROUTE:{}\n", packet_start->routing_fields.value);
#endif
}

template <typename T>
FORCE_INLINE void print_pkt_header_noc_fields(volatile T* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    switch (packet_start->noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            DPRINT("N_WR addr:{}\n", (uint64_t)packet_start->command_fields.unicast_write.noc_address);
        } break;
        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            DPRINT(
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

FORCE_INLINE void print_pkt_header(volatile tt::tt_fabric::PacketHeader* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    auto const& header = *packet_start;
    DPRINT(
        "PKT: nsnd_t:{} csnd_t:{} src_chip:{} payload_size_bytes:{}\n",
        (uint32_t)packet_start->noc_send_type,
        (uint32_t)packet_start->chip_send_type,
        (uint32_t)packet_start->src_ch_id,
        (uint32_t)packet_start->payload_size_bytes);
    print_pkt_hdr_routing_fields(packet_start);
    print_pkt_header_noc_fields(packet_start);
#endif
}

FORCE_INLINE void print_pkt_header(volatile tt::tt_fabric::LowLatencyPacketHeader* const packet_start) {
#ifdef DEBUG_PRINT_ENABLED
    auto const& header = *packet_start;
    DPRINT(
        "PKT: nsnd_t:{} src_chip:{} payload_size_bytes:{}\n",
        (uint32_t)packet_start->noc_send_type,
        (uint32_t)packet_start->src_ch_id,
        (uint32_t)packet_start->payload_size_bytes);
    print_pkt_hdr_routing_fields(packet_start);
    print_pkt_header_noc_fields(packet_start);
#endif
}

// Flushes prior non-posted writes on this receiver channel so that all prior data writes (e.g., to DRAM or L1)
// have been acknowledged at their destinations before a signaling atomic increment is issued.
// NOTE: This intentionally flushes prior WRITES only; prior atomics are NOT waited on (matching Wormhole's
// posted-atomic behavior and Blackhole VC1). On Blackhole, noc_fast_atomic_increment forces posted=false on
// write_at_cmd_buf (which defaults to transaction ID 0), so the per-ID loop below incidentally serialized
// against in-flight atomics on ID 0, whereas ncrisc_noc_nonposted_writes_flushed checks NIU_MST_WR_ACK_RECEIVED
// and allows back-to-back atomics to pipeline. If a workload ever requires cross-core atomic-to-atomic ordering,
// that should be exposed as an explicit opt-in check (ncrisc_noc_nonposted_atomics_flushed).
FORCE_INLINE void flush_write_to_noc_pipeline(uint8_t rx_channel_id) {
#if defined(ARCH_BLACKHOLE)
    if constexpr (
        !enable_deadlock_avoidance && (
#if defined(COMPILE_FOR_DRISC)
                                          true
#else
                                          rx_batch_completion &&
                                          tt::tt_fabric::local_chip_noc_equals_downstream_noc &&
                                          tt::tt_fabric::edm_to_local_chip_noc == noc_index
#endif
                                          )) {
        while (!ncrisc_noc_nonposted_writes_flushed(tt::tt_fabric::edm_to_local_chip_noc));
        return;
    }
#endif
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

// Shifts the chunk encoding in a scatter write packet to the next chunk
FORCE_INLINE void shift_to_next_chunk(uint8_t& chunk_encodings) { chunk_encodings >>= 2; }

FORCE_INLINE void complete_pending_atomic_flush(uint8_t rx_channel_id) {
    flush_write_to_noc_pipeline(rx_channel_id);
    noc_semaphore_inc_with_state<true, /*skip_cmdbuf_chk=*/true>(tt::tt_fabric::edm_to_local_chip_noc);
}

#if defined(ARCH_BLACKHOLE)
FORCE_INLINE void complete_relaxed_atomic_flush(uint32_t trid_b, uint32_t trid_s) {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, trid_s));
    if (trid_b != trid_s) {
        while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, trid_b));
    }
    noc_semaphore_inc_with_state<true, /*skip_cmdbuf_chk=*/true>(noc);
}
#endif

FORCE_INLINE void drain_pending_atomic_flush(uint8_t rx_channel_id, [[maybe_unused]] uint32_t& pending_atomic_flush) {
#if defined(ARCH_BLACKHOLE)
    if (pending_atomic_flush < 0x100u) {
        complete_pending_atomic_flush(rx_channel_id);
    } else {
        complete_relaxed_atomic_flush((pending_atomic_flush >> 4) & 0xFu, pending_atomic_flush & 0xFu);
    }
    pending_atomic_flush = 0;
#else
    complete_pending_atomic_flush(rx_channel_id);
#endif
}

FORCE_INLINE void execute_local_semaphore_inc(
    uint64_t addr,
    uint32_t incr,
    uint8_t flush,
    uint8_t rx_channel_id,
    [[maybe_unused]] uint32_t* pending_atomic_flush = nullptr,
    [[maybe_unused]] uint8_t last_write_trid = 0) {
#if defined(FABRIC_ERISC_ROUTER_V2) || defined(FABRIC_ERISC_ROUTER_V3)
    // Routers v2 and v3 stage a flushed atomic before its flush, and can defer it (pending_atomic_flush != nullptr).
    if (flush != tt::tt_fabric::NocFlushType::FLUSH_NONE) {
        noc_semaphore_inc_set_state<true>(
            addr, incr, tt::tt_fabric::edm_to_local_chip_noc, tt::tt_fabric::forward_and_local_write_noc_vc);
#if defined(ARCH_BLACKHOLE)
        if (pending_atomic_flush != nullptr) {
            if (!ncrisc_noc_nonposted_writes_flushed(tt::tt_fabric::edm_to_local_chip_noc)) {
                *pending_atomic_flush =
                    (flush == tt::tt_fabric::NocFlushType::FLUSH_RELAXED) ? (0x10u | (last_write_trid & 0xFu)) : 1u;
                return;
            }
            noc_semaphore_inc_with_state<true, /*skip_cmdbuf_chk=*/true>(tt::tt_fabric::edm_to_local_chip_noc);
            return;
        }
#endif
        complete_pending_atomic_flush(rx_channel_id);
    } else {
        noc_semaphore_inc<true>(
            addr, incr, tt::tt_fabric::edm_to_local_chip_noc, tt::tt_fabric::forward_and_local_write_noc_vc);
    }
#else
    if (flush != tt::tt_fabric::NocFlushType::FLUSH_NONE) {
        flush_write_to_noc_pipeline(rx_channel_id);
    }
    noc_semaphore_inc<true>(
        addr, incr, tt::tt_fabric::edm_to_local_chip_noc, tt::tt_fabric::forward_and_local_write_noc_vc);
#endif
}

#if defined(COMPILE_FOR_DRISC) && defined(ARCH_BLACKHOLE) && !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
// On DRISC, NoC1 local_chip_data_cmd_buf (write_cmd_buf = 0) is dedicated to local-chip unicast writes on
// forward_and_local_write_noc_vc, and write_at_cmd_buf (3) is dedicated to local-chip semaphore increments.
// Pre-programming NOC_CTRL and MID=0 once at init (and restoring them after the rare NOC_UNICAST_INLINE_WRITE
// path) avoids rewriting those registers on every write/atomic and lets flushed atomics pre-stage their
// target address and data registers while waiting for the payload write ack.
FORCE_INLINE void drisc_local_write_init_cmd_buf() {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    constexpr uint8_t cmd_buf = tt::tt_fabric::local_chip_data_cmd_buf;
    constexpr uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC |
                                       NOC_CMD_STATIC_VC(tt::tt_fabric::forward_and_local_write_noc_vc) |
                                       NOC_CMD_RESP_MARKED;
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, 0x0);

    constexpr uint8_t at_cmd_buf = write_at_cmd_buf;
    constexpr uint32_t noc_at_cmd_field = NOC_CMD_VC_STATIC |
                                          NOC_CMD_STATIC_VC(tt::tt_fabric::forward_and_local_write_noc_vc) |
                                          NOC_CMD_RESP_MARKED | NOC_CMD_AT;
    while (!noc_cmd_buf_ready(noc, at_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, at_cmd_buf, NOC_CTRL, noc_at_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, at_cmd_buf, NOC_TARG_ADDR_MID, 0x0);
}

template <bool set_trid = true, bool update_counters = true>
FORCE_INLINE void drisc_local_write_one_packet(
    uint32_t src_addr, uint64_t dest_addr, uint32_t len_bytes, uint32_t trid) {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    constexpr uint8_t cmd_buf = tt::tt_fabric::local_chip_data_cmd_buf;
    const uint32_t dest_lo = static_cast<uint32_t>(dest_addr);
    const uint32_t dest_coord = static_cast<uint32_t>(dest_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    if constexpr (set_trid) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(trid));
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_addr);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dest_lo);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_COORDINATE, dest_coord);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    if constexpr (update_counters) {
        noc_nonposted_writes_num_issued[noc] += 1;
        noc_nonposted_writes_acked[noc] += 1;
    }
}

FORCE_INLINE void drisc_local_semaphore_inc(uint64_t addr, uint32_t incr, bool flush, uint8_t rx_channel_id) {
    constexpr uint8_t noc = tt::tt_fabric::edm_to_local_chip_noc;
    constexpr uint8_t cmd_buf = write_at_cmd_buf;
    const uint32_t addr_lo = static_cast<uint32_t>(addr);
    const uint32_t addr_coord = static_cast<uint32_t>(addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
    const uint32_t at_len_be =
        NOC_AT_INS(NOC_AT_INS_INCR_GET) | NOC_AT_WRAP(31) | NOC_AT_IND_32((addr_lo >> 2) & 0x3) | NOC_AT_IND_32_SRC(0);
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, addr_lo);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, addr_coord);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, at_len_be);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_DATA, incr);
    if (flush) {
        flush_write_to_noc_pipeline(rx_channel_id);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_atomics_acked[noc] += 1;
}
#endif

// Core implementation of unicast-to-local-chip dispatch.
// Accepts pre-resolved payload_size_bytes and noc_send_type to avoid redundant uncached L1 reads
// when the caller has already loaded these (e.g. via a packed 4B load).
__attribute__((optimize("jump-tables")))
// #ifndef FABRIC_2D
FORCE_INLINE
    // #endif
    void
    execute_chip_unicast_to_local_chip_impl(
        tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
        uint16_t payload_size_bytes,
        tt::tt_fabric::NocSendType noc_send_type,
        uint32_t transaction_id,
        uint8_t rx_channel_id,
        uint32_t* pending_atomic_flush = nullptr) {
    const auto& header = *packet_start;
    uint32_t payload_start_address = reinterpret_cast<size_t>(packet_start) + sizeof(PACKET_HEADER_TYPE);

    constexpr bool update_counter = false;

    channel_trimming_usage_recorder.set_noc_send_type_used(rx_channel_id, noc_send_type);
    // NOC_SPARSE_MCAST_WRITE (enum value 8) sits above the contiguous standard range. Handling it as
    // an early, unlikely branch keeps the standard switch below bounded to the dense
    // 0..NOC_UNICAST_SCATTER_WRITE range, so its jump table is unchanged.
    if (noc_send_type == tt::tt_fabric::NocSendType::NOC_SPARSE_MCAST_WRITE) [[unlikely]] {
        // chip_idx/write_idx select this chip's page group; the router advances both before
        // forwarding, so the values read here are guaranteed to be this chip's (see
        // receiver_forward_packet). The same payload is written to each of this chip's pages.
        const auto& sparse = header.command_fields.sparse_mcast_write;
        const uint8_t count = sparse.counts[sparse.chip_idx];
        const uint8_t base = sparse.write_idx;
        for (uint8_t j = 0; j < count; j++) {
            noc_async_write_one_packet_with_trid<update_counter, false>(
                payload_start_address,
                sparse.noc_address[base + j],
                payload_size_bytes,
                transaction_id,
                tt::tt_fabric::local_chip_data_cmd_buf,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);
        }
        return;
    }
    // Standard local-write types are 0..NOC_UNICAST_SCATTER_WRITE; nothing above that range reaches this
    // point (sparse returned above, and the gap types between it and sparse never hit the local-write path).
    if (noc_send_type > tt::tt_fabric::NocSendType::NOC_SEND_TYPE_LAST) {
        __builtin_unreachable();
    }
    switch (noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            const auto dest_address = header.command_fields.unicast_write.noc_address;
#if defined(COMPILE_FOR_DRISC) && defined(ARCH_BLACKHOLE) && !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
            drisc_local_write_one_packet<true, true>(
                payload_start_address, dest_address, payload_size_bytes, transaction_id);
#else
            noc_async_write_one_packet_with_trid<update_counter, false>(
                payload_start_address,
                dest_address,
                payload_size_bytes,
                transaction_id,
                tt::tt_fabric::local_chip_data_cmd_buf,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);
#endif
        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC: {
            const uint64_t dest_address = header.command_fields.unicast_seminc.noc_address;
            const auto increment = header.command_fields.unicast_seminc.val;
#if defined(COMPILE_FOR_DRISC) && defined(ARCH_BLACKHOLE) && !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
            drisc_local_semaphore_inc(
                dest_address,
                increment,
                header.command_fields.unicast_seminc.flush != tt::tt_fabric::NocFlushType::FLUSH_NONE,
                rx_channel_id);
#else
            execute_local_semaphore_inc(
                dest_address,
                increment,
                header.command_fields.unicast_seminc.flush,
                rx_channel_id,
                pending_atomic_flush,
                static_cast<uint8_t>((transaction_id - 1u) & 0xFu));
#endif
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
#if defined(COMPILE_FOR_DRISC) && defined(ARCH_BLACKHOLE) && !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
            drisc_local_write_init_cmd_buf();
#endif
        } break;

        case tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC: {
            const auto dest_address = header.command_fields.unicast_seminc_fused.noc_address;
            const uint64_t semaphore_dest_address = header.command_fields.unicast_seminc_fused.semaphore_noc_address;
            const auto increment = header.command_fields.unicast_seminc_fused.val;
            const uint8_t flush = header.command_fields.unicast_seminc_fused.flush;
#if defined(COMPILE_FOR_DRISC) && defined(ARCH_BLACKHOLE) && !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
            drisc_local_write_one_packet<true, true>(
                payload_start_address, dest_address, payload_size_bytes, transaction_id);
            drisc_local_semaphore_inc(
                semaphore_dest_address, increment, flush != tt::tt_fabric::NocFlushType::FLUSH_NONE, rx_channel_id);
#else
            noc_async_write_one_packet_with_trid<update_counter, false>(
                payload_start_address,
                dest_address,
                payload_size_bytes,
                transaction_id,
                tt::tt_fabric::local_chip_data_cmd_buf,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);
            execute_local_semaphore_inc(
                semaphore_dest_address,
                increment,
                flush,
                rx_channel_id,
                pending_atomic_flush,
                static_cast<uint8_t>(transaction_id));
#endif
        } break;

        case tt::tt_fabric::NocSendType::NOC_UNICAST_SCATTER_WRITE: {
            using ChunkEncoding = tt::tt_fabric::NocScatterWriteChunkEncoding;
            const auto& scatter = header.command_fields.unicast_scatter_write;
#if defined(COMPILE_FOR_DRISC) && defined(ARCH_BLACKHOLE) && !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
            static_assert(offsetof(tt::tt_fabric::NocUnicastScatterCommandHeader, chunk_size) == 32);
            static_assert(offsetof(tt::tt_fabric::NocUnicastScatterCommandHeader, chunk_count) == 38);
            static_assert(offsetof(tt::tt_fabric::NocUnicastScatterCommandHeader, chunk_encoding) == 39);
            const auto* const scatter_words = reinterpret_cast<const uint32_t*>(&scatter.chunk_size[0]);
            const uint32_t sizes_01 = scatter_words[0];
            const uint32_t size2_cnt_enc = scatter_words[1];
            const uint8_t chunk_count = static_cast<uint8_t>(size2_cnt_enc >> 16);
            uint8_t packet_encoding = static_cast<uint8_t>(size2_cnt_enc >> 24);

            size_t offset = 0;
            const uint8_t last_chunk_index = chunk_count - 1;
            uint16_t chunk_size = static_cast<uint16_t>(sizes_01);
            const uint64_t addr0 = scatter.noc_address[0];
            drisc_local_write_one_packet</*set_trid=*/true, /*update_counters=*/false>(
                payload_start_address, addr0, chunk_size, transaction_id);
            offset += chunk_size;
            shift_to_next_chunk(packet_encoding);
            uint32_t num_write_chunks = 1;

            if (chunk_count > 2) {
                chunk_size = static_cast<uint16_t>(sizes_01 >> 16);
                const uint64_t addr1 = scatter.noc_address[1];
                drisc_local_write_one_packet</*set_trid=*/false, /*update_counters=*/false>(
                    payload_start_address + offset, addr1, chunk_size, transaction_id);
                offset += chunk_size;
                shift_to_next_chunk(packet_encoding);
                num_write_chunks++;

                if (chunk_count == 4) [[likely]] {
                    chunk_size = static_cast<uint16_t>(size2_cnt_enc);
                    const uint64_t addr2 = scatter.noc_address[2];
                    drisc_local_write_one_packet</*set_trid=*/false, /*update_counters=*/false>(
                        payload_start_address + offset, addr2, chunk_size, transaction_id);
                    offset += chunk_size;
                    shift_to_next_chunk(packet_encoding);
                    num_write_chunks++;
                }
            }

            constexpr uint8_t CHUNK_ENCODING_MASK = 0b11;
            const ChunkEncoding chunk_encoding = static_cast<ChunkEncoding>(packet_encoding & CHUNK_ENCODING_MASK);
            const uint64_t final_destination_noc_address = scatter.noc_address[last_chunk_index];
            if (chunk_encoding == ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {
                const uint16_t final_chunk_size = static_cast<uint16_t>(payload_size_bytes - offset);
                drisc_local_write_one_packet</*set_trid=*/false, /*update_counters=*/false>(
                    payload_start_address + offset, final_destination_noc_address, final_chunk_size, transaction_id);
                num_write_chunks++;
                noc_nonposted_writes_num_issued[tt::tt_fabric::edm_to_local_chip_noc] += num_write_chunks;
                noc_nonposted_writes_acked[tt::tt_fabric::edm_to_local_chip_noc] += num_write_chunks;
            } else if (chunk_encoding != ChunkEncoding::CHUNK_ENCODING_NOP) {
                noc_nonposted_writes_num_issued[tt::tt_fabric::edm_to_local_chip_noc] += num_write_chunks;
                noc_nonposted_writes_acked[tt::tt_fabric::edm_to_local_chip_noc] += num_write_chunks;
                drisc_local_semaphore_inc(
                    final_destination_noc_address,
                    scatter.chunk_size[last_chunk_index],
                    chunk_encoding == ChunkEncoding::CHUNK_ENCODING_SEMINC_FLUSH,
                    rx_channel_id);
            } else {
                ASSERT(false);
            }
#else
            const uint8_t chunk_count = scatter.chunk_count;
            uint8_t packet_encoding = scatter.chunk_encoding;

            // NOTE: when chunk_count < 4, chunk_size[n-2] can be used without calculating final_chunk_size.
            //       However the perf (n == 2) is much worse than implementation below.
            //       Need to check perf with 2 <= n <= 4
            size_t offset = 0;
            const uint8_t last_chunk_index = chunk_count - 1;

            // Unicast Scatter Write currently supports 2 modes:
            // 1. Up to 4 unicast writes
            // 2. 2 unicast writes followed by a semaphore increment
            // First chunk is guaranteed to be a unicast write
            uint16_t chunk_size = scatter.chunk_size[0];
            noc_async_write_one_packet_with_trid<update_counter, false>(
                payload_start_address + offset,
                scatter.noc_address[0],
                chunk_size,
                transaction_id,
                tt::tt_fabric::local_chip_data_cmd_buf,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);
            offset += chunk_size;
            shift_to_next_chunk(packet_encoding);

            if (chunk_count > 2) {
                // Second chunk is guaranteed to be a unicast write
                chunk_size = scatter.chunk_size[1];
                noc_async_write_one_packet_with_trid<update_counter, false>(
                    payload_start_address + offset,
                    scatter.noc_address[1],
                    chunk_size,
                    transaction_id,
                    tt::tt_fabric::local_chip_data_cmd_buf,
                    tt::tt_fabric::edm_to_local_chip_noc,
                    tt::tt_fabric::forward_and_local_write_noc_vc);
                offset += chunk_size;
                shift_to_next_chunk(packet_encoding);

                if (chunk_count == 4) [[likely]] {
                    // If there are 4 chunks, the third chunk is guaranteed to be a unicast write
                    chunk_size = scatter.chunk_size[2];
                    noc_async_write_one_packet_with_trid<update_counter, false>(
                        payload_start_address + offset,
                        scatter.noc_address[2],
                        chunk_size,
                        transaction_id,
                        tt::tt_fabric::local_chip_data_cmd_buf,
                        tt::tt_fabric::edm_to_local_chip_noc,
                        tt::tt_fabric::forward_and_local_write_noc_vc);
                    offset += chunk_size;
                    shift_to_next_chunk(packet_encoding);
                }
            }

            // Last chunk is either a unicast write or a semaphore increment
            constexpr uint8_t CHUNK_ENCODING_MASK = 0b11;  // Gets us the encoding of the least significant chunk
            const ChunkEncoding chunk_encoding = static_cast<ChunkEncoding>(packet_encoding & CHUNK_ENCODING_MASK);
            const uint64_t final_destination_noc_address = scatter.noc_address[last_chunk_index];
            if (chunk_encoding == ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {
                const uint16_t final_chunk_size = static_cast<uint16_t>(payload_size_bytes - offset);
                noc_async_write_one_packet_with_trid<update_counter, false>(
                    payload_start_address + offset,
                    final_destination_noc_address,
                    final_chunk_size,
                    transaction_id,
                    tt::tt_fabric::local_chip_data_cmd_buf,
                    tt::tt_fabric::edm_to_local_chip_noc,
                    tt::tt_fabric::forward_and_local_write_noc_vc);
            } else if (chunk_encoding != ChunkEncoding::CHUNK_ENCODING_NOP) {
                // If a semaphore increment is being performed, the increment value is fed in in the chunk size, to
                // reduce payload size
                execute_local_semaphore_inc(
                    final_destination_noc_address,
                    scatter.chunk_size[last_chunk_index],
                    chunk_encoding == ChunkEncoding::CHUNK_ENCODING_SEMINC_FLUSH
                        ? static_cast<uint8_t>(
                              tt::tt_fabric::NocFlushType::FLUSH_STRICT + ((packet_encoding >> 2) & 0x1u))
                        : static_cast<uint8_t>(tt::tt_fabric::NocFlushType::FLUSH_NONE),
                    rx_channel_id,
                    pending_atomic_flush,
                    static_cast<uint8_t>(transaction_id));
            } else {
                // Every scatter write packet contains a chunk_count, and the unicast_writes above are only called for
                // chunk_count-1 times. Therefore, we should never get here (seeing a NOP).
                ASSERT(false);
            }
#endif
        } break;

        case tt::tt_fabric::NocSendType::NOC_MULTICAST_WRITE:
        case tt::tt_fabric::NocSendType::NOC_MULTICAST_ATOMIC_INC:
        default: {
            ASSERT(false);
        } break;
    };
}

// Wrapper that resolves noc_send_type from the packet header via a packed 4B load
// (payload_size_bytes + noc_send_type in one read), then delegates to the core implementation.
// The caller-provided payload_size_bytes is still used (not the packed copy) to preserve
// existing call-site semantics.
__attribute__((optimize("jump-tables")))
// #ifndef FABRIC_2D
FORCE_INLINE
    // #endif
    void
    execute_chip_unicast_to_local_chip(
        tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
        uint16_t payload_size_bytes,
        uint32_t transaction_id,
        uint8_t rx_channel_id) {
    auto packed = PACKET_HEADER_TYPE::PackedPayloadAndSendType::load(packet_start);
    execute_chip_unicast_to_local_chip_impl(
        packet_start, payload_size_bytes, packed.noc_send_type, transaction_id, rx_channel_id);
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
    ASSERT(local_relay_interface.template edm_has_space_for_packet<ENABLE_RISC_CPU_DATA_CACHE>());

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
#pragma GCC unroll 16
        for (uint32_t i = 0; i < EXT - 1; i++) {
            const_cast<uint32_t*>(packet_header->routing_fields.route_buffer)[i] =
                cached_routing_fields.route_buffer[i + 1];
        }
        const_cast<uint32_t*>(packet_header->routing_fields.route_buffer)[EXT - 1] = 0;
    } else {
// No refill needed - just copy buffer as-is
#pragma GCC unroll 16
        for (uint32_t i = 0; i < EXT; i++) {
            const_cast<uint32_t*>(packet_header->routing_fields.route_buffer)[i] =
                cached_routing_fields.route_buffer[i];
        }
    }
#endif

    // Write new value (always happens)
    packet_header->routing_fields.value = new_value;
}

// Destination-major 2D action maps are complete at injection and immutable in transit. These no-op
// overloads keep the shared 1D/2D forwarding path well-formed when it resolves updates by header type;
// the 1D overloads above still advance their routes.
FORCE_INLINE void update_packet_header_for_next_hop(
    volatile tt_l1_ptr tt::tt_fabric::HybridMeshPacketHeader* /*packet_header*/,
    tt::tt_fabric::LowLatencyMeshRoutingFields /*cached_routing_fields*/) {}

template <uint8_t NUM_SENDER_BUFFERS>
void update_packet_header_for_next_hop(
    tt::tt_fabric::EdmToEdmSender<NUM_SENDER_BUFFERS>& /*downstream_edm_interface*/, uint32_t /*value*/) {}

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
template <
    bool enable_deadlock_avoidance,
    bool stateful_api,
    bool increment_pointers = true,
    bool update_local_free_slots = true,
    uint8_t NUM_SENDER_BUFFERS>
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
        increment_pointers,
        update_local_free_slots>(
        reinterpret_cast<size_t>(packet_header), payload_size_bytes + sizeof(PACKET_HEADER_TYPE), transaction_id);
    if constexpr (!increment_pointers) {
        update_packet_header_for_next_hop(downstream_edm_interface, cached_routing_fields.value);
    }
}
