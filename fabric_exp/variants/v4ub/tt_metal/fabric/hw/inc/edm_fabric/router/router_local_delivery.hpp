// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Local delivery: a packet's NoC commands on this chip. Plain writes, scatter writes, atomic increments and fused write
// + atomic increments (the common send types) are issued here directly, with the NoC commands of the matching arm of
// execute_chip_unicast_to_local_chip_impl but without its send-type dispatch (LEDGER E1-8, E1-10: scatter -7 to -12,
// atomic -4 to -11). Every other send type goes through execute_chip_unicast_to_local_chip_impl, out of line.
//
// Local writes use one command buffer, programmed once with the write state that never changes (local_write_init),
// so each write sets only its transaction ID, source, destination and length (LEDGER E1-17: scatter -6). Nothing else
// on this ERISC uses that command buffer while the loop runs (static_assert below). With DPRINT or the watcher, whose
// traffic shares command buffers, each write programs its whole state instead.
//
// Atomics are strict: a flushed atomic waits for every earlier write of this ERISC to be acknowledged first.

#include "router_common.hpp"
#include "router_completions.hpp"

namespace fabric_router {

constexpr uint8_t local_noc = tt::tt_fabric::edm_to_local_chip_noc;
constexpr uint8_t local_write_cmd_buf = tt::tt_fabric::local_chip_data_cmd_buf;
constexpr uint8_t local_noc_vc = tt::tt_fabric::forward_and_local_write_noc_vc;

constexpr bool local_write_state_preset =
#if !defined(DEBUG_PRINT_ENABLED) && !defined(WATCHER_ENABLED)
    true;
#else
    false;
#endif

constexpr bool local_write_cmd_buf_is_exclusive = [] {
    for (size_t ch = 0; ch < NUM_RECEIVER_CHANNELS; ch++) {
        if (receiver_channel_forwarding_data_cmd_buf_ids[ch] == local_write_cmd_buf ||
            receiver_channel_forwarding_sync_cmd_buf_ids[ch] == local_write_cmd_buf) {
            return false;
        }
    }
    return local_write_cmd_buf != write_at_cmd_buf;
}();

FORCE_INLINE void local_write_init() {
    if constexpr (local_write_state_preset) {
        static_assert(local_write_cmd_buf_is_exclusive);
        while (!noc_cmd_buf_ready(local_noc, local_write_cmd_buf));
        noc_write_init_state<local_write_cmd_buf, CQ_NOC_mkp>(local_noc, local_noc_vc);
    }
}

// One local write of `size` bytes tagged with `trid`. COUNTED adds it to the software write counters
// (noc_nonposted_writes_num_issued/acked), as the write API always does on Blackhole; a scatter packet counts its writes
// once, after the last one. READY: the caller has seen the local-write command buffer ready and issued nothing on it
// since (receive's packet reads), so the write skips its own check.
template <bool COUNTED, bool READY = false>
FORCE_INLINE void local_write(uint32_t src_addr, uint64_t dst_noc_addr, uint32_t size, uint32_t trid) {
    WAYPOINT("NAWW");
    RECORD_NOC_EVENT_WITH_ADDR(NocEventType::WRITE_WITH_TRID, src_addr, dst_noc_addr, size, -1, false, local_noc);
    DEBUG_SANITIZE_NOC_WRITE_TRANSACTION(local_noc, dst_noc_addr, src_addr, size);
    if constexpr (local_write_state_preset) {
        ncrisc_noc_set_transaction_id<READY>(local_noc, local_write_cmd_buf, trid);  // else waits for the buffer
        noc_write_with_state<noc_mode, local_write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_wait, COUNTED, false>(
            local_noc, src_addr, dst_noc_addr, size);
    } else {
        if constexpr (!READY) {
            while (!noc_cmd_buf_ready(local_noc, local_write_cmd_buf));
        }
        ncrisc_noc_fast_write<noc_mode, /*use_trid=*/true, COUNTED>(
            local_noc, local_write_cmd_buf, src_addr, dst_noc_addr, size, local_noc_vc, false, false, 1, true, false, trid);
    }
    WAYPOINT("NWPD");
}

// An atomic increment; a flushed one first waits for every earlier write to be acknowledged.
FORCE_INLINE void local_atomic_inc(uint64_t addr, uint32_t value, bool flush, uint8_t channel) {
    if (flush) {
        flush_write_to_noc_pipeline(channel);
    }
    noc_semaphore_inc<true>(addr, value, local_noc, local_noc_vc);
}

// Send types without a direct arm, out of line (rare: inline writes, multicast, sparse multicast).
__attribute__((noinline)) void deliver_other(
    tt_l1_ptr PACKET_HEADER_TYPE* header, uint16_t payload_size, tt::tt_fabric::NocSendType type, uint8_t trid, uint8_t channel) {
    execute_chip_unicast_to_local_chip_impl(header, payload_size, type, trid, channel);
}

// The send types whose NoC commands read the slot's data (a local write).
FORCE_INLINE uint32_t reads_slot(tt::tt_fabric::NocSendType type) {
    using tt::tt_fabric::NocSendType;
    constexpr uint32_t types = (1u << NocSendType::NOC_UNICAST_WRITE) | (1u << NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) |
                               (1u << NocSendType::NOC_UNICAST_SCATTER_WRITE) | (1u << NocSendType::NOC_SPARSE_MCAST_WRITE);
    static_assert(NocSendType::NOC_SEND_TYPE_LAST < 32);
    return ((types >> (static_cast<uint32_t>(type) & 31u)) & 1u) * SlotRecord::READ_LOCAL;
}

// Writes the packet to this chip. Returns the slot-read bits for the slot's record. LOCAL_READY: the local-write command
// buffer was seen ready (receive's packet reads): the packet's first local write skips its check.
template <uint8_t CHANNEL, bool LOCAL_READY = false>
FORCE_INLINE uint32_t deliver(
    tt_l1_ptr PACKET_HEADER_TYPE* header, uint16_t payload_size, tt::tt_fabric::NocSendType type, uint8_t trid) {
    using tt::tt_fabric::NocSendType;
    const uint32_t payload = reinterpret_cast<uint32_t>(header) + sizeof(PACKET_HEADER_TYPE);
    if (type == NocSendType::NOC_UNICAST_WRITE) {
        channel_trimming_usage_recorder.set_noc_send_type_used(CHANNEL, type);
        local_write<true, LOCAL_READY>(payload, header->command_fields.unicast_write.noc_address, payload_size, trid);
        return SlotRecord::READ_LOCAL;
    }
    if (type == NocSendType::NOC_UNICAST_SCATTER_WRITE) {
        channel_trimming_usage_recorder.set_noc_send_type_used(CHANNEL, type);
        using ChunkEncoding = tt::tt_fabric::NocScatterWriteChunkEncoding;
        const auto& scatter = header->command_fields.unicast_scatter_write;
        const uint8_t chunks = scatter.chunk_count;
        uint8_t encoding = scatter.chunk_encoding;
        // Up to three writes, then the last chunk: a write of the rest of the payload, or a semaphore increment (its
        // value carried in the chunk size).
        uint32_t offset = scatter.chunk_size[0];
        uint32_t writes = 1;
        local_write<false, LOCAL_READY>(payload, scatter.noc_address[0], offset, trid);
        shift_to_next_chunk(encoding);
        if (chunks > 2) {
            const uint32_t size1 = scatter.chunk_size[1];
            local_write<false>(payload + offset, scatter.noc_address[1], size1, trid);
            offset += size1;
            shift_to_next_chunk(encoding);
            writes++;
            if (chunks == 4) {
                const uint32_t size2 = scatter.chunk_size[2];
                local_write<false>(payload + offset, scatter.noc_address[2], size2, trid);
                offset += size2;
                shift_to_next_chunk(encoding);
                writes++;
            }
        }
        const auto last = static_cast<ChunkEncoding>(encoding & 0b11);
        const uint64_t last_addr = scatter.noc_address[chunks - 1];
        if (last == ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {
            local_write<false>(payload + offset, last_addr, static_cast<uint16_t>(payload_size - offset), trid);
            writes++;
        }
        noc_nonposted_writes_num_issued[local_noc] += writes;
        noc_nonposted_writes_acked[local_noc] += writes;
        if (last != ChunkEncoding::CHUNK_ENCODING_UNICAST_WRITE) {
            ASSERT(last != ChunkEncoding::CHUNK_ENCODING_NOP);
            local_atomic_inc(
                last_addr, scatter.chunk_size[chunks - 1], last == ChunkEncoding::CHUNK_ENCODING_SEMINC_FLUSH, CHANNEL);
        }
        return SlotRecord::READ_LOCAL;
    }
    if (type == NocSendType::NOC_UNICAST_ATOMIC_INC) {
        channel_trimming_usage_recorder.set_noc_send_type_used(CHANNEL, type);
        const auto& inc = header->command_fields.unicast_seminc;
        local_atomic_inc(inc.noc_address, inc.val, inc.flush != 0, CHANNEL);
        return 0;
    }
    if (type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {
        channel_trimming_usage_recorder.set_noc_send_type_used(CHANNEL, type);
        const auto& fused = header->command_fields.unicast_seminc_fused;
        local_write<true, LOCAL_READY>(payload, fused.noc_address, payload_size, trid);
        local_atomic_inc(fused.semaphore_noc_address, fused.val, fused.flush != 0, CHANNEL);
        return SlotRecord::READ_LOCAL;
    }
    deliver_other(header, payload_size, type, trid, CHANNEL);
    return reads_slot(type);
}

}  // namespace fabric_router
