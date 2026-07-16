// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// [debug] Producer/consumer CONTRACT for the fabric detailed flow-control traces ([rxlog]/[txlog]). This is the
// ONLY thing the on-device fabric router (producer) and the host reader (consumer) share: the byte layout of
// the DRAM-resident log HEADER and the packed record arrays. The router appends delta-encoded records to an L1
// region, flushes them to a per-router DRAM buffer, and at the STOP marker also flushes a small header to the
// front of that buffer; the host reader reads the header + records straight back from DRAM. Because both sides
// use these exact struct definitions there is no hand-maintained mirror to drift.
//
// Everything that is producer-side-only (the L1 working state used for delta encoding: last_* baselines, live
// counters, per-pass stashes) lives in the producer header (detailed_fabric_log_producer.hpp), NOT here -- it is
// not part of the contract and the consumer never sees it.
//
// Device-safe: only <cstdint>, plain-old-data structs, no host-only headers, so it compiles into eRisc kernels.
// Record layout notes: every field is a delta since the previous recorded row (the reset snapshot is the
// implicit zero baseline) except the absolute levels called out below. Records are kept 4-aligned and their
// arrays a 16 B multiple so the NOC DRAM flush start addresses stay 16 B-aligned.

namespace tt::tt_fabric {

// DRAM buffer layout per log per router: [header region | packed record array]. The header region is a fixed
// 16 B-aligned reservation at the front of the buffer that holds the small *LogHeader (flushed at STOP); the
// records begin right after it. A fixed reservation (rather than sizeof(header)) keeps the record start 16 B-
// aligned for the NOC flush and lets a header grow up to this size without moving the records. Both headers
// below must fit (static_assert'd).
inline constexpr uint32_t DETAILED_FABRIC_LOG_DRAM_HEADER_REGION = 64;

// ============================================================================
// Receiver flow-control trace ([rxlog])
// ============================================================================

struct ReceiverLogRecord {     // 16 bytes (see static_assert)
    uint32_t ts_delta;         // cycles since previous record (get_timestamp_32b domain), full 32b (no saturation)
    uint32_t iter_delta;       // main-loop passes since previous record, full 32b (no saturation)
    uint8_t ready;             // to_receiver_pkts_sent doorbell level, absolute (bounded by receiver slot count)
    uint8_t ack_delta;         // ack_counter increments since previous record
    uint8_t wr_sent_delta;     // wr_sent_counter increments since previous record
    uint8_t wr_flush_delta;    // wr_flush_counter increments since previous record (always 0 in fused builds)
    uint8_t completion_delta;  // completion_counter increments since previous record
    // 3 B tail padding (struct alignment 4): keeps records 16 B / 4-aligned so the uint32 deltas above are
    // never accessed unaligned on the eRisc, and CAPACITY*16 stays a 16 B multiple for the DRAM flush.
};
static_assert(sizeof(ReceiverLogRecord) == 16, "ReceiverLogRecord must be 16 bytes");

// Cap the number of records so the log is bounded and never runs past the carved region. 252 * 16 B of records
// + a 64 B L1 working header = 4096 B exactly; the builder carves RECEIVER_LOG_BUFFER_SIZE = 4096 B to hold it
// (keep the two in sync). 252 is the most 16 B records that fit alongside the header without growing the carve.
inline constexpr uint32_t RECEIVER_LOG_CAPACITY = 252;
inline constexpr uint32_t RECEIVER_LOG_MAGIC = 0xC0FFEE02;

// Shared receiver-log HEADER: the framing the router flushes to the front of its DRAM buffer at STOP and the
// reader reads back to frame + reconstruct the record array. Every field here is consumer-visible; producer-
// only working state (last_*, count) is NOT here (see detailed_fabric_log_producer.hpp).
struct ReceiverLogHeader {
    uint32_t magic;      // sanity tag (RECEIVER_LOG_MAGIC); 0 / garbage => no valid trace at this address
    uint32_t dropped;    // records skipped after both L1 and the DRAM buffer filled (honest reporting on dump)
    uint32_t window_id;  // correlation/batch id from start_detailed_logging (the marker value); 0 => none
    // Initial in-flight backlog at window open, expressed as each counter's gap above the completion counter
    // (ack >= wr_sent >= wr_flush >= completion, so all >= 0). The window may open NON-quiescent; seeding the
    // reconstruction accumulators with these gaps rebases all counters to a COMMON baseline (completion at
    // open), restoring ack >= cmpl and the correct occupancy.
    uint32_t base_ack_gap;
    uint32_t base_wr_sent_gap;
    uint32_t base_wr_flush_gap;
    // DRAM record-array state: dram_write_offset is the total bytes of records written to the DRAM buffer (the
    // record array size the reader reads back), dram_records the running count of records flushed to DRAM.
    uint32_t dram_write_offset;
    uint32_t dram_records;
};
static_assert(
    sizeof(ReceiverLogHeader) <= DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
    "ReceiverLogHeader must fit the DRAM header region");

// ============================================================================
// Sender flow-control trace ([txlog])
// ============================================================================

inline constexpr uint32_t SENDER_LOG_NUM_CHANNELS = 2;  // the two VC0 sender channels (indices 0 and 1 on 1D EDM)

// Per-channel block-reason code: what the pass did / why it did not transmit (see record_sender_flow_state).
enum SenderSendReason : uint8_t {
    SENDER_REASON_IDLE = 0,     // no send and no obvious block (nothing pending / turn skipped)
    SENDER_REASON_SENT = 1,     // transmitted a packet this pass
    SENDER_REASON_STARVED = 2,  // downstream credit + txq free, but no unsent packet from the producer
    SENDER_REASON_RXFULL = 3,   // unsent packet + txq free, but no downstream receiver credit
    SENDER_REASON_TXQBUSY = 4,  // unsent packet + downstream credit, but the eth txq is busy
};

struct SenderLogRecord {    // 20 bytes (see static_assert)
    uint32_t ts_delta;      // cycles since previous record (get_timestamp_32b), full 32b (no saturation)
    uint32_t iter_delta;    // main-loop passes since previous record, full 32b (no saturation)
    uint8_t dn_credits;     // outbound.num_free_slots (VC-shared downstream headroom), absolute
    uint8_t ch_flags;       // low nibble = ch0 [conn:bit3 | reason:bits0-2]; high nibble = ch1 (same layout)
    uint8_t ch0_local_occ;  // num_buffers[0] - free_slots[0] (packets waiting to transmit), absolute
    uint8_t ch0_sent_delta;
    uint8_t ch0_acked_delta;
    uint8_t ch0_cmpl_delta;
    uint8_t ch1_local_occ;
    uint8_t ch1_sent_delta;
    uint8_t ch1_acked_delta;
    uint8_t ch1_cmpl_delta;
    // 2 B tail padding (struct alignment 4): keeps records 20 B / 4-aligned so the uint32 deltas above are
    // never accessed unaligned on the eRisc, and CAPACITY*20 stays a 16 B multiple for the DRAM flush.
};
static_assert(sizeof(SenderLogRecord) == 20, "SenderLogRecord must be 20 bytes");

// 128 * 20 B records + a 112 B L1 working header = 2672 B, under the builder's SENDER_LOG_BUFFER_SIZE (4096).
inline constexpr uint32_t SENDER_LOG_CAPACITY = 128;
inline constexpr uint32_t SENDER_LOG_MAGIC = 0xC0FFEE03;

// Shared sender-log HEADER: the DRAM-flushed framing the reader reads. Consumer-visible fields only; producer-
// only working state (sent/acked/cmpl totals, last_*, per-pass occ/reason/conn, count) lives in the producer.
struct SenderLogHeader {
    uint32_t magic;      // sanity tag (SENDER_LOG_MAGIC); 0 / garbage => no valid trace at this address
    uint32_t dropped;    // records skipped after both L1 and the DRAM buffer filled
    uint32_t window_id;  // correlation/batch id from start_detailed_logging (the marker value); 0 => none
    // Initial backlog at window open, each counter's gap above completion (sent >= acked >= cmpl). Same
    // rationale as ReceiverLogHeader::base_*_gap: rebases all three to a common completion baseline.
    uint32_t base_sent_gap[SENDER_LOG_NUM_CHANNELS];
    uint32_t base_acked_gap[SENDER_LOG_NUM_CHANNELS];
    // DRAM record-array state: total record bytes written to DRAM and the running flushed-record count.
    uint32_t dram_write_offset;
    uint32_t dram_records;
};
static_assert(
    sizeof(SenderLogHeader) <= DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
    "SenderLogHeader must fit the DRAM header region");

}  // namespace tt::tt_fabric
