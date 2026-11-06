// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Local delivery: a packet's NoC commands on this chip. Plain writes, scatter writes, atomic increments and fused write
// + atomic increments (the common send types) are issued here directly, with the NoC commands of the matching arm of
// execute_chip_unicast_to_local_chip_impl but without its send-type dispatch (LEDGER E1-8, E1-10). Every other send
// type goes out of line (deliver_other): inline writes to local_inline_write, the rest through
// execute_chip_unicast_to_local_chip_impl.
//
// Local writes use one command buffer, programmed once with the write state that never changes (local_write_init),
// so each write sets only its transaction ID, source, destination and length (LEDGER E1-17). Nothing else on this
// ERISC uses that command buffer while the loop runs (static_assert below) but an inline write, which restores it
// after. With DPRINT or the watcher, whose traffic shares command buffers, each write programs its whole state
// instead.
//
// Atomics are strict: a flushed atomic waits for every earlier write of this ERISC to be acknowledged first.

#include "router_common.hpp"

namespace fabric_router {

constexpr uint8_t LOCAL_NOC = tt::tt_fabric::edm_to_local_chip_noc;
constexpr uint8_t LOCAL_WRITE_CMD_BUF = tt::tt_fabric::local_chip_data_cmd_buf;
constexpr uint8_t LOCAL_NOC_VC = tt::tt_fabric::forward_and_local_write_noc_vc;

constexpr bool local_write_state_preset =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
    command_buffer_presets;
#else
    false;
#endif

constexpr bool local_write_cmd_buf_is_exclusive = [] {
    for (size_t ch = 0; ch < NUM_RECEIVER_CHANNELS; ch++) {
        if (receiver_channel_forwarding_data_cmd_buf_ids[ch] == LOCAL_WRITE_CMD_BUF ||
            receiver_channel_forwarding_sync_cmd_buf_ids[ch] == LOCAL_WRITE_CMD_BUF) {
            return false;
        }
    }
    return LOCAL_WRITE_CMD_BUF != write_at_cmd_buf;
}();

FORCE_INLINE void local_write_init() {
    if constexpr (local_write_state_preset) {
        static_assert(!local_write_state_preset || local_write_cmd_buf_is_exclusive);  // checked even when discarded
        while (!noc_cmd_buf_ready(LOCAL_NOC, LOCAL_WRITE_CMD_BUF));
        noc_write_init_state<LOCAL_WRITE_CMD_BUF, CQ_NOC_mkp>(LOCAL_NOC, LOCAL_NOC_VC);
    }
}

// One local write of `size` bytes tagged with `trid`. COUNTED adds it to the software write counters
// (noc_nonposted_writes_num_issued/acked), as the write API always does on Blackhole; a scatter packet counts its writes
// once, after the last one.
template <bool COUNTED>
FORCE_INLINE void local_write(uint32_t src_addr, uint64_t dst_noc_addr, uint32_t size, uint32_t trid) {
    WAYPOINT("NAWW");
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID, src_addr, dst_noc_addr, size, -1, false, LOCAL_NOC);
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(LOCAL_NOC, dst_noc_addr, src_addr, size);
    if constexpr (local_write_state_preset) {
        ncrisc_noc_set_transaction_id(LOCAL_NOC, LOCAL_WRITE_CMD_BUF, trid);  // waits for the command buffer
        noc_write_with_state<
            noc_mode,
            LOCAL_WRITE_CMD_BUF,
            CQ_NOC_SNDL,
            CQ_NOC_SEND,
            CQ_NOC_wait,
            /*update_counter=*/COUNTED,
            /*posted=*/false>(LOCAL_NOC, src_addr, dst_noc_addr, size);
    } else {
        while (!noc_cmd_buf_ready(LOCAL_NOC, LOCAL_WRITE_CMD_BUF));
        ncrisc_noc_fast_write<noc_mode, /*use_trid=*/true, /*update_counter=*/COUNTED>(
            LOCAL_NOC,
            LOCAL_WRITE_CMD_BUF,
            src_addr,
            dst_noc_addr,
            size,
            LOCAL_NOC_VC,
            /*mcast=*/false,
            /*linked=*/false,
            /*num_dests=*/1,
            /*multicast_path_reserve=*/true,
            /*posted=*/false,
            trid);
    }
    WAYPOINT("NWPD");
}

// Adds `writes` local writes to the software write counters (noc_nonposted_writes_num_issued/acked), as the write API
// does for each write.
FORCE_INLINE void count_local_writes(uint32_t writes) {
    noc_nonposted_writes_num_issued[LOCAL_NOC] += writes;
    noc_nonposted_writes_acked[LOCAL_NOC] += writes;
}

// An atomic increment; a flushed one first waits for every earlier write to be acknowledged.
FORCE_INLINE void local_atomic_inc(uint64_t addr, uint32_t value, bool flush, uint8_t channel) {
    if (flush) {
        flush_write_to_noc_pipeline(channel);
    }
    noc_semaphore_inc<true>(addr, value, LOCAL_NOC, LOCAL_NOC_VC);
}

// An inline write (one word) to this chip, on the local-write command buffer, then that buffer's preset restored: on
// Blackhole a write to L1 is emulated there by a posted write, which leaves the buffer posted.
FORCE_INLINE void local_inline_write(uint64_t dst_noc_addr, uint32_t value) {
    noc_inline_dw_write<InlineWriteDst::DEFAULT, /*posted=*/true, /*flush=*/true, /*emulated_write_on_cmd_buf=*/true>(
        dst_noc_addr, value, 0xF, LOCAL_NOC, LOCAL_NOC_VC, /*customized_src_addr=*/0, LOCAL_WRITE_CMD_BUF);
    local_write_init();
}

// Send types without a direct arm, out of line (rare: inline writes, multicast, sparse multicast).
__attribute__((noinline)) void deliver_other(
    tt_l1_ptr PACKET_HEADER_TYPE* header, uint16_t payload_size, tt::tt_fabric::NocSendType type, uint8_t trid, uint8_t channel) {
    if (type == tt::tt_fabric::NocSendType::NOC_UNICAST_INLINE_WRITE) {
        channel_trimming_usage_recorder.set_noc_send_type_used(channel, type);
        const auto& inline_write = header->command_fields.unicast_inline_write;
        local_inline_write(inline_write.noc_address, inline_write.value);
        return;
    }
    execute_chip_unicast_to_local_chip_impl(header, payload_size, type, trid, channel);
}

// Writes the packet to this chip; every write that reads the slot is tagged with `trid`. CH: the host's index of the
// receiver channel (receiver_channel_of), for the channel-trimming capture and the channel's flush.
template <size_t CH>
FORCE_INLINE void deliver(
    tt_l1_ptr PACKET_HEADER_TYPE* header, uint16_t payload_size, tt::tt_fabric::NocSendType type, uint8_t trid) {
    using tt::tt_fabric::NocSendType;
    const uint32_t payload = reinterpret_cast<uint32_t>(header) + sizeof(PACKET_HEADER_TYPE);
    if (type == NocSendType::NOC_UNICAST_WRITE) {
        channel_trimming_usage_recorder.set_noc_send_type_used(CH, type);
        local_write<true>(payload, header->command_fields.unicast_write.noc_address, payload_size, trid);
        return;
    }
    if (type == NocSendType::NOC_UNICAST_SCATTER_WRITE) {
        channel_trimming_usage_recorder.set_noc_send_type_used(CH, type);
        using ChunkEncoding = tt::tt_fabric::NocScatterWriteChunkEncoding;
        constexpr uint8_t chunk_encoding_mask = 0b11;  // the current chunk's encoding (main: CHUNK_ENCODING_MASK)
        const auto& scatter = header->command_fields.unicast_scatter_write;
        const uint8_t chunks = scatter.chunk_count;
        uint8_t encoding = scatter.chunk_encoding;
        // Up to three writes, then the last chunk: a write of the rest of the payload, or a semaphore increment (its
        // value carried in the chunk size).
        uint32_t offset = scatter.chunk_size[0];
        uint32_t writes = 1;
        local_write<false>(payload, scatter.noc_address[0], offset, trid);
        shift_to_next_chunk(encoding);
        if (chunks > 2) {
            const uint32_t chunk1_size = scatter.chunk_size[1];
            local_write<false>(payload + offset, scatter.noc_address[1], chunk1_size, trid);
            offset += chunk1_size;
            shift_to_next_chunk(encoding);
            writes++;
            if (chunks == 4) {
                const uint32_t chunk2_size = scatter.chunk_size[2];
                local_write<false>(payload + offset, scatter.noc_address[2], chunk2_size, trid);
                offset += chunk2_size;
                shift_to_next_chunk(encoding);
                writes++;
            }
        }
        const auto last = static_cast<ChunkEncoding>(encoding & chunk_encoding_mask);
        const uint64_t last_noc_addr = scatter.noc_address[chunks - 1];
        if (last == ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {
            local_write<false>(payload + offset, last_noc_addr, static_cast<uint16_t>(payload_size - offset), trid);
            writes++;
        }
        count_local_writes(writes);
        if (last != ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {
            ASSERT(last != ChunkEncoding::CHUNK_ENCODING_NOP);
            local_atomic_inc(
                last_noc_addr, scatter.chunk_size[chunks - 1], last == ChunkEncoding::CHUNK_ENCODING_SEMINC_FLUSH, CH);
        }
        return;
    }
    if (type == NocSendType::NOC_UNICAST_ATOMIC_INC) {
        channel_trimming_usage_recorder.set_noc_send_type_used(CH, type);
        const auto& inc = header->command_fields.unicast_seminc;
        local_atomic_inc(inc.noc_address, inc.val, inc.flush != 0, CH);
        return;
    }
    if (type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {
        channel_trimming_usage_recorder.set_noc_send_type_used(CH, type);
        const auto& fused = header->command_fields.unicast_seminc_fused;
        local_write<true>(payload, fused.noc_address, payload_size, trid);
        local_atomic_inc(fused.semaphore_noc_address, fused.val, fused.flush != 0, CH);
        return;
    }
    deliver_other(header, payload_size, type, trid, CH);
}

}  // namespace fabric_router
