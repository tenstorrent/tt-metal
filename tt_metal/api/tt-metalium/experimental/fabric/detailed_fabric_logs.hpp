// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// [debug] Producer/consumer CONTRACT for the fabric detailed flow-control traces ([rxlog]/[txlog]). This is the
// ONLY thing the on-device fabric router (producer) and the host reader (consumer) share: the byte layout of the
// DRAM-resident log HEADER and the packed 8-byte log WORDS. The router appends words to an L1 region, flushes them
// to a per-router DRAM buffer, and at the STOP marker also flushes a small header to the front of that buffer; the
// host reader reads the header + words straight back from DRAM. Because both sides use these exact definitions
// there is no hand-maintained mirror to drift.
//
// Everything that is producer-side-only (the L1 working state: per-channel last_* baselines, running totals,
// per-pass stashes, the word-buffer cursor) lives in the producer header (detailed_fabric_log_producer.hpp), NOT
// here -- it is not part of the contract and the consumer never sees it.
//
// Device-safe: only <cstdint>, plain-old-data + free functions, no host-only headers, so it compiles into eRisc
// kernels.
//
// ============================================================================
// Word stream model (topology/channel agnostic -- 1D and 2D use the SAME format)
// ============================================================================
// Each log is a stream of fixed 8-byte WORDS, of two kinds, distinguished by byte 0:
//
//   COMMON word  (byte0 = count, ALWAYS >= 1):
//       one per hot-loop iteration in which at least one serviced channel's state changed. Carries the shared
//       per-iteration timeline (iter_delta, ts_delta) and `count` = how many CHANNEL words follow it.
//
//   CHANNEL word (byte0 = 0, the reserved sentinel):
//       one per changed channel, emitted right after the common word for that iteration. Carries that channel's
//       delta-encoded flow-control state. byte1 is the nibble-packed channel id (see make_channel_id): high
//       nibble = VC id, low nibble = channel index within the VC.
//
// Because a common word's count is always >= 1, byte0 == 0 unambiguously marks a channel word. The stream is
// therefore self-describing and INDEPENDENT of L1->DRAM flush boundaries: a full-buffer flush may land between a
// common word and its channel words, or mid-run, without affecting reconstruction -- words are simply appended to
// DRAM in order. The reader walks the words in order: a common word starts a new iteration sample and says how
// many channel words belong to it.
//
// Alignment: every word is 8 bytes; the L1 word buffer capacity is EVEN, so every full-buffer flush moves a 16 B
// multiple and the DRAM record start stays 16 B-aligned (only the final partial tail flush may move an odd word
// count, and nothing follows it). See detailed_fabric_log_producer.hpp.

namespace tt::tt_fabric {

// DRAM buffer layout per log per router: [header region | packed 8 B word array]. The header region is a fixed
// 16 B-aligned reservation at the front holding the small *LogHeader (flushed at STOP); words begin right after
// it. A fixed reservation (rather than sizeof(header)) keeps the word-array start 16 B-aligned for the NOC flush
// and lets a header grow up to this size without moving the words. Both headers below must fit (static_assert'd).
inline constexpr uint32_t DETAILED_FABRIC_LOG_DRAM_HEADER_REGION = 64;

// ============================================================================
// Channel geometry maxima (KEEP IN SYNC with builder_config in fabric_builder_config.hpp)
// ============================================================================
// The header carries per-channel baseline gaps indexed by [vc][local], so it must be sized to the worst case.
// These mirror builder_config: MAX_NUM_VCS = 3; VC0 up to 5 sender channels (Z-router), VC1 up to 4, VC2 1 -> 5
// max per VC; one receiver channel per VC. channel ids are nibble-packed so both VC and local index must be <= 15.
inline constexpr uint32_t DETAILED_FABRIC_MAX_VCS = 3;
inline constexpr uint32_t DETAILED_FABRIC_MAX_SENDER_CH_PER_VC = 5;
inline constexpr uint32_t DETAILED_FABRIC_MAX_RECEIVER_CH_PER_VC = 1;

// ============================================================================
// channel_id byte: high nibble = VC id, low nibble = channel index within the VC
// ============================================================================
constexpr uint8_t make_channel_id(uint32_t vc, uint32_t local) {
    return static_cast<uint8_t>(((vc & 0xF) << 4) | (local & 0xF));
}
constexpr uint32_t channel_id_vc(uint8_t channel_id) { return static_cast<uint32_t>(channel_id >> 4); }
constexpr uint32_t channel_id_local(uint8_t channel_id) { return static_cast<uint32_t>(channel_id & 0xF); }

// ============================================================================
// Byte-field clamps (deltas are 1 byte; iter_delta is 24 bit). Values that would overflow saturate rather than
// wrap, so a reconstructed column can only under-count a long idle gap, never jump backwards. Real per-pass
// counter deltas are 0/1 so they never reach the byte clamp; iter_delta only saturates across multi-ms stalls
// (ts_delta remains the accurate time axis there).
// ============================================================================
inline constexpr uint32_t DETAILED_FABRIC_LOG_U8_MAX = 0xFF;
inline constexpr uint32_t DETAILED_FABRIC_LOG_U24_MAX = 0xFFFFFF;
constexpr uint8_t log_clamp_u8(uint32_t v) {
    return static_cast<uint8_t>(v > DETAILED_FABRIC_LOG_U8_MAX ? DETAILED_FABRIC_LOG_U8_MAX : v);
}
constexpr uint32_t log_clamp_u24(uint32_t v) {
    return v > DETAILED_FABRIC_LOG_U24_MAX ? DETAILED_FABRIC_LOG_U24_MAX : v;
}

// ============================================================================
// Word codec. Words are stored/transported as little-endian uint64_t; byte i of the word is bits [8i, 8i+8).
// Both host and device are little-endian, so a uint64_t written to memory lays byte 0 at the lowest address and
// the reader recovers the same fields. All packers/unpackers are free functions so kernel and reader share them.
// ============================================================================

// A common word's byte0 (count) is always >= 1; a channel word's byte0 is 0. So this cleanly discriminates them.
inline bool log_word_is_channel(uint64_t word) { return (word & 0xFF) == 0; }

// ---- Common word: byte0 = count | bytes1-3 = iter_delta (24b) | bytes4-7 = ts_delta (32b) ----
inline uint64_t pack_common_log_word(uint8_t count, uint32_t iter_delta, uint32_t ts_delta) {
    return static_cast<uint64_t>(count) | (static_cast<uint64_t>(log_clamp_u24(iter_delta)) << 8) |
           (static_cast<uint64_t>(ts_delta) << 32);
}
inline uint8_t common_log_word_count(uint64_t w) { return static_cast<uint8_t>(w & 0xFF); }
inline uint32_t common_log_word_iter_delta(uint64_t w) { return static_cast<uint32_t>((w >> 8) & 0xFFFFFF); }
inline uint32_t common_log_word_ts_delta(uint64_t w) { return static_cast<uint32_t>(w >> 32); }

// The channel id lives in byte1 of every channel word (both rx and tx), so it can be read before dispatching on
// the log kind.
inline uint8_t channel_log_word_channel_id(uint64_t w) { return static_cast<uint8_t>((w >> 8) & 0xFF); }

// ============================================================================
// Receiver flow-control trace ([rxlog])
// ============================================================================
// Receiver CHANNEL word: byte0 = 0 | byte1 = channel_id | byte2 = ready (absolute doorbell level) |
// byte3 = ack_delta | byte4 = wr_sent_delta | byte5 = wr_flush_delta | byte6 = completion_delta | byte7 = pad.
inline uint64_t pack_receiver_channel_log_word(
    uint8_t channel_id,
    uint8_t ready,
    uint8_t ack_delta,
    uint8_t wr_sent_delta,
    uint8_t wr_flush_delta,
    uint8_t completion_delta) {
    return (static_cast<uint64_t>(channel_id) << 8) | (static_cast<uint64_t>(ready) << 16) |
           (static_cast<uint64_t>(ack_delta) << 24) | (static_cast<uint64_t>(wr_sent_delta) << 32) |
           (static_cast<uint64_t>(wr_flush_delta) << 40) | (static_cast<uint64_t>(completion_delta) << 48);
}
inline uint8_t receiver_channel_log_word_ready(uint64_t w) { return static_cast<uint8_t>((w >> 16) & 0xFF); }
inline uint8_t receiver_channel_log_word_ack_delta(uint64_t w) { return static_cast<uint8_t>((w >> 24) & 0xFF); }
inline uint8_t receiver_channel_log_word_wr_sent_delta(uint64_t w) { return static_cast<uint8_t>((w >> 32) & 0xFF); }
inline uint8_t receiver_channel_log_word_wr_flush_delta(uint64_t w) { return static_cast<uint8_t>((w >> 40) & 0xFF); }
inline uint8_t receiver_channel_log_word_completion_delta(uint64_t w) { return static_cast<uint8_t>((w >> 48) & 0xFF); }

inline constexpr uint32_t RECEIVER_LOG_MAGIC = 0xC0FFEE04;  // format v2 (8 B word stream); bump on layout change

// Shared receiver-log HEADER: the framing the router flushes to the front of its DRAM buffer at STOP and the
// reader reads back to frame + reconstruct the word stream. Consumer-visible fields only; producer-only working
// state lives in detailed_fabric_log_producer.hpp. Per-channel baseline gaps are indexed [vc][local] straight
// from the nibble-packed channel id -- no VC-boundary reconstruction needed on the host.
struct ReceiverLogHeader {
    uint32_t magic;              // sanity tag (RECEIVER_LOG_MAGIC); 0 / garbage => no valid trace at this address
    uint32_t dropped;            // words skipped after both L1 and the DRAM buffer filled (honest reporting)
    uint32_t window_id;          // correlation/batch id from start_detailed_logging (the marker value); 0 => none
    uint32_t dram_write_offset;  // total bytes of words written to the DRAM buffer (what the reader reads back)
    uint32_t dram_words;         // running count of words flushed to DRAM
    // Initial in-flight backlog at window open per channel, each counter's gap above its completion counter
    // (ack >= wr_sent >= wr_flush >= completion, so all >= 0; bounded by channel slot count so a byte suffices).
    // Seeding the reconstruction accumulators with these rebases every counter to a common completion baseline,
    // so occupancy is correct even when the window opens non-quiescent.
    uint8_t base_ack_gap[DETAILED_FABRIC_MAX_VCS][DETAILED_FABRIC_MAX_RECEIVER_CH_PER_VC];
    uint8_t base_wr_sent_gap[DETAILED_FABRIC_MAX_VCS][DETAILED_FABRIC_MAX_RECEIVER_CH_PER_VC];
    uint8_t base_wr_flush_gap[DETAILED_FABRIC_MAX_VCS][DETAILED_FABRIC_MAX_RECEIVER_CH_PER_VC];
};
static_assert(
    sizeof(ReceiverLogHeader) <= DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
    "ReceiverLogHeader must fit the DRAM header region");

// ============================================================================
// Sender flow-control trace ([txlog])
// ============================================================================

// Per-channel block-reason code: what the pass did / why it did not transmit (see record path in the router).
enum SenderSendReason : uint8_t {
    SENDER_REASON_IDLE = 0,     // no send and no obvious block (nothing pending / turn skipped)
    SENDER_REASON_SENT = 1,     // transmitted a packet this pass
    SENDER_REASON_STARVED = 2,  // downstream credit + txq free, but no unsent packet from the producer
    SENDER_REASON_RXFULL = 3,   // unsent packet + txq free, but no downstream receiver credit
    SENDER_REASON_TXQBUSY = 4,  // unsent packet + downstream credit, but the eth txq is busy
};

// Sender per-channel flags byte: bits0-2 = reason, bit3 = conn (an upstream worker holds an open connection).
constexpr uint8_t pack_sender_flags(uint8_t reason, bool conn) {
    return static_cast<uint8_t>((reason & 0x7) | (conn ? 0x8 : 0));
}
constexpr uint8_t sender_flags_reason(uint8_t flags) { return static_cast<uint8_t>(flags & 0x7); }
constexpr bool sender_flags_conn(uint8_t flags) { return (flags & 0x8) != 0; }

// Sender CHANNEL word: byte0 = 0 | byte1 = channel_id | byte2 = flags | byte3 = local_occ | byte4 = sent_delta |
// byte5 = acked_delta | byte6 = cmpl_delta | byte7 = dn_credits (downstream free slots at this channel's remote
// receiver -- shared by all channels of the same VC, so channels with the same high nibble report one credit pool).
inline uint64_t pack_sender_channel_log_word(
    uint8_t channel_id,
    uint8_t flags,
    uint8_t local_occ,
    uint8_t sent_delta,
    uint8_t acked_delta,
    uint8_t cmpl_delta,
    uint8_t dn_credits) {
    return (static_cast<uint64_t>(channel_id) << 8) | (static_cast<uint64_t>(flags) << 16) |
           (static_cast<uint64_t>(local_occ) << 24) | (static_cast<uint64_t>(sent_delta) << 32) |
           (static_cast<uint64_t>(acked_delta) << 40) | (static_cast<uint64_t>(cmpl_delta) << 48) |
           (static_cast<uint64_t>(dn_credits) << 56);
}
inline uint8_t sender_channel_log_word_flags(uint64_t w) { return static_cast<uint8_t>((w >> 16) & 0xFF); }
inline uint8_t sender_channel_log_word_local_occ(uint64_t w) { return static_cast<uint8_t>((w >> 24) & 0xFF); }
inline uint8_t sender_channel_log_word_sent_delta(uint64_t w) { return static_cast<uint8_t>((w >> 32) & 0xFF); }
inline uint8_t sender_channel_log_word_acked_delta(uint64_t w) { return static_cast<uint8_t>((w >> 40) & 0xFF); }
inline uint8_t sender_channel_log_word_cmpl_delta(uint64_t w) { return static_cast<uint8_t>((w >> 48) & 0xFF); }
inline uint8_t sender_channel_log_word_dn_credits(uint64_t w) { return static_cast<uint8_t>((w >> 56) & 0xFF); }

inline constexpr uint32_t SENDER_LOG_MAGIC = 0xC0FFEE05;  // format v2 (8 B word stream); bump on layout change

// Shared sender-log HEADER: the DRAM-flushed framing the reader reads. Consumer-visible fields only. Per-channel
// baseline gaps indexed [vc][local] from the nibble-packed channel id (same rationale as ReceiverLogHeader).
struct SenderLogHeader {
    uint32_t magic;              // sanity tag (SENDER_LOG_MAGIC); 0 / garbage => no valid trace at this address
    uint32_t dropped;            // words skipped after both L1 and the DRAM buffer filled
    uint32_t window_id;          // correlation/batch id from start_detailed_logging (the marker value); 0 => none
    uint32_t dram_write_offset;  // total bytes of words written to DRAM
    uint32_t dram_words;         // running flushed-word count
    // Initial backlog at window open per channel, each counter's gap above completion (sent >= acked >= cmpl;
    // bounded by slot count so a byte suffices). Same rationale as ReceiverLogHeader::base_*_gap.
    uint8_t base_sent_gap[DETAILED_FABRIC_MAX_VCS][DETAILED_FABRIC_MAX_SENDER_CH_PER_VC];
    uint8_t base_acked_gap[DETAILED_FABRIC_MAX_VCS][DETAILED_FABRIC_MAX_SENDER_CH_PER_VC];
};
static_assert(
    sizeof(SenderLogHeader) <= DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
    "SenderLogHeader must fit the DRAM header region");

}  // namespace tt::tt_fabric
