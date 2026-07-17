// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>  // offsetof
#include <cstdint>

#include <tt-metalium/experimental/fabric/detailed_fabric_log.hpp>
#include <tt-metalium/experimental/fabric/detailed_fabric_logs.hpp>

// [debug] PRODUCER-side machinery for the fabric detailed flow-control traces ([rxlog]/[txlog]).
//
// This header owns everything only the on-device fabric router needs to GENERATE the traces: the L1 working
// structs (the shared DRAM header from detailed_fabric_logs.hpp, the per-channel diff state, and the 8-byte word
// buffer), and the functions that init / diff / append / flush / finalize them. None of this is visible to the
// host reader, which only ever sees the shared header + word arrays in DRAM. The fabric router kernel
// (fabric_erisc_router.cpp) just CALLS into these functions.
//
// Not a standalone translation unit: include it ONLY from the fabric router kernel TU, AFTER its compile-time
// args and channel-serviced tables are in scope. It relies on those ambient symbols (same convention as the other
// edm_fabric kernel headers):
//   - CT-arg constexprs: receiver_log_buffer_addr, sender_log_buffer_addr, {receiver,sender}_log_dram_buffer_base,
//     {receiver,sender}_log_dram_buffer_size, log_dram_bank_id, NUM_SENDER_CHANNELS, NUM_RECEIVER_CHANNELS,
//     ACTUAL_VC0/VC1/VC2_SENDER_CHANNELS, VC1/VC2_SENDER_CHANNEL_START
//   - channel tables: is_receiver_channel_serviced[], is_sender_channel_serviced[], VC0_RECEIVER_CHANNEL
//   - device intrinsics: get_timestamp_32b, get_noc_addr_from_bank_id, noc_async_write[_barrier], FORCE_INLINE
//
// The trace is a self-describing stream of 8-byte words (see detailed_fabric_logs.hpp): per hot-loop iteration in
// which any serviced channel changed, one COMMON word (shared iter/ts timeline + a count) followed by one CHANNEL
// word per changed channel. DRAM buffer layout per log is [header region | packed word array]: words flush during
// the window (whole buffer at a time -> even word count -> 16 B-aligned), the header flushes to `base` at STOP.

namespace tt::tt_fabric {

// [debug] Window gate: set while a detailed-logging window is open (the router's telemetry marker is nonzero).
// A 1-byte .bss global (not in the carved L1 region) because it is read on *every* send attempt to gate the
// instrumentation, and a local-RAM read is far cheaper than a volatile L1 read.
bool detailed_log_window_active = false;

// [debug] Transient per-iteration scratch: the channel words produced this pass, before they are framed by a
// common word and appended. .bss globals (local RAM, not the carved L1 region) so building them costs no L1
// traffic; sized to the max serviced channel count for each domain. Reset to count=0 at the start of each pass.
// Receiver and sender have separate scratch so a single-eRisc (Wormhole) router can build both in one pass.
struct PendingLogWords {
    uint32_t count;
    uint64_t words[NUM_SENDER_CHANNELS > NUM_RECEIVER_CHANNELS ? NUM_SENDER_CHANNELS : NUM_RECEIVER_CHANNELS];
};
PendingLogWords receiver_pending_words;
PendingLogWords sender_pending_words;

// [debug] True iff this eRisc drives the receiver / sender trace. A router eRisc services either all receiver
// channels or all sender channels (or both, single-eRisc), so keying off VC0's serviced flag / sender channel 0
// tells us which trace(s) this eRisc owns. VC0 (receiver channel 0, sender channel 0) always exists.
constexpr bool receiver_log_enabled() { return is_receiver_channel_serviced[VC0_RECEIVER_CHANNEL]; }
constexpr bool sender_log_enabled() { return is_sender_channel_serviced[0]; }
// True iff sender channel `ch` feeds the [txlog] trace: in range and serviced (all serviced senders are logged).
constexpr bool sender_log_channel_enabled(uint32_t ch) {
    return ch < NUM_SENDER_CHANNELS && is_sender_channel_serviced[ch];
}

// [debug] Flat sender channel index -> (VC id, index within VC). Pure function of the per-VC sender counts, which
// are contiguous ranges [0,VC0)|[VC0,VC0+VC1)|[VC0+VC1,...). Used to stamp the nibble-packed channel id so the
// host can group same-VC senders (they share a downstream credit pool) with no host-side build state.
constexpr uint32_t sender_flat_to_vc(uint32_t flat) {
    if (flat < ACTUAL_VC0_SENDER_CHANNELS) {
        return 0;
    }
    if (flat < VC2_SENDER_CHANNEL_START) {
        return 1;
    }
    return 2;
}
constexpr uint32_t sender_flat_to_local(uint32_t flat) {
    if (flat < ACTUAL_VC0_SENDER_CHANNELS) {
        return flat;
    }
    if (flat < VC2_SENDER_CHANNEL_START) {
        return flat - VC1_SENDER_CHANNEL_START;
    }
    return flat - VC2_SENDER_CHANNEL_START;
}

// ============================================================================
// L1 word-buffer geometry. Even capacities so every full-buffer flush moves a 16 B multiple (16 B-aligned DRAM
// writes); the final partial tail flush at STOP may be an odd word count (nothing follows it). Sized to leave the
// per-log carve (RECEIVER/SENDER_LOG_BUFFER_SIZE = 4096 B) room for the header + diff state (static_assert'd).
// ============================================================================
inline constexpr uint32_t RECEIVER_LOG_WORD_CAPACITY = 480;
inline constexpr uint32_t SENDER_LOG_WORD_CAPACITY = 448;
static_assert(RECEIVER_LOG_WORD_CAPACITY % 2 == 0, "receiver word capacity must be even for 16 B-aligned flushes");
static_assert(SENDER_LOG_WORD_CAPACITY % 2 == 0, "sender word capacity must be even for 16 B-aligned flushes");

// ============================================================================
// [debug] Receiver flow-control trace ([rxlog]) -- producer side.
// ============================================================================

// Per-channel diff state (kernel-only): the immutable nibble-packed channel id plus the previous recorded row's
// absolute values, seeded by init_receiver_channel_state() from a live snapshot at window open. The receiver
// reads live hardware counters each pass, so there are no running totals here.
struct ReceiverChannelState {
    uint8_t channel_id;
    uint32_t last_ready;
    uint32_t last_ack;
    uint32_t last_wr_sent;
    uint32_t last_wr_flush;
    uint32_t last_completion;
};

// L1 working struct for the receiver trace: shared DRAM header, then producer-only working state (the shared
// timeline baseline + word-buffer cursor + per-channel diff state), then the 8-byte word buffer.
struct ReceiverLog {
    static constexpr uint32_t WORD_CAPACITY = RECEIVER_LOG_WORD_CAPACITY;
    ReceiverLogHeader header;  // shared contract (flushed to DRAM at STOP)
    uint32_t last_ts;          // ts of the previous common word (shared timeline delta baseline)
    uint32_t last_iter;        // iter of the previous common word
    uint32_t word_count;       // words currently resident in `words` (the tail; the rest are in DRAM)
    ReceiverChannelState channels[NUM_RECEIVER_CHANNELS];
    // alignas(16): the word array is the NOC flush SOURCE, whose start must be 16 B-aligned. The carve base
    // (receiver_log_buffer_addr) is 16 B-aligned by the builder, so a 16 B-aligned offset makes the absolute
    // source address 16 B-aligned (static_assert'd below).
    alignas(16) uint64_t words[WORD_CAPACITY];
};
static_assert(sizeof(ReceiverLog) <= RECEIVER_LOG_BUFFER_SIZE, "ReceiverLog exceeds the carved receiver_log_buffer");
static_assert(offsetof(ReceiverLog, words) % 16 == 0, "receiver word array must be 16 B-aligned for NOC flush");

// Accessor folds to the constant base address (receiver_log_buffer_addr is a CT-arg constexpr).
FORCE_INLINE ReceiverLog* receiver_log() { return reinterpret_cast<ReceiverLog*>(receiver_log_buffer_addr); }

// ============================================================================
// [debug] Sender flow-control trace ([txlog]) -- producer side.
// ============================================================================

// Per-channel diff state (kernel-only): the immutable nibble-packed channel id, the running monotonic totals
// (bumped by the accumulate hooks in the sender step while the window is open), the per-pass stash (occ / reason /
// conn / dn_credits, written by stash_sender_flow_state), and the previous recorded row's baseline (last_*).
struct SenderChannelState {
    uint8_t channel_id;
    uint8_t occ;         // this-pass local backlog level (stashed)
    uint8_t reason;      // this-pass block-reason annotation (stashed)
    uint8_t conn;        // this-pass connection flag (stashed)
    uint8_t dn_credits;  // this-pass downstream free slots at the remote receiver (stashed; per-VC pool)
    uint8_t last_occ;    // previous recorded backlog level
    uint8_t last_dn_credits;
    uint32_t sent;  // running monotonic totals (bumped at the event sites in the sender step)
    uint32_t acked;
    uint32_t cmpl;
    uint32_t last_sent;  // previous recorded totals
    uint32_t last_acked;
    uint32_t last_cmpl;
};

struct SenderLog {
    static constexpr uint32_t WORD_CAPACITY = SENDER_LOG_WORD_CAPACITY;
    SenderLogHeader header;
    uint32_t last_ts;
    uint32_t last_iter;
    uint32_t word_count;
    SenderChannelState channels[NUM_SENDER_CHANNELS];
    alignas(16) uint64_t words[WORD_CAPACITY];  // NOC flush source -- 16 B-aligned (see ReceiverLog::words)
};
static_assert(sizeof(SenderLog) <= SENDER_LOG_BUFFER_SIZE, "SenderLog exceeds the carved sender_log_buffer");
static_assert(offsetof(SenderLog, words) % 16 == 0, "sender word array must be 16 B-aligned for NOC flush");

FORCE_INLINE SenderLog* sender_log() { return reinterpret_cast<SenderLog*>(sender_log_buffer_addr); }

// ============================================================================
// [debug] Shared word-buffer append / flush machinery (domain-agnostic: operates on any log that exposes
// header.{dram_write_offset,dram_words,dropped}, word_count, words[], and WORD_CAPACITY).
// ============================================================================

// Blocking bulk-flush of `num_words` words (from the front of the L1 word buffer) to this router's DRAM word
// array. The array lives in a single bank at a fixed per-bank byte offset; the write targets exactly that bank
// via get_noc_addr_from_bank_id, so it never straddles a bank boundary. Full-buffer flushes move WORD_CAPACITY
// (even) words = a 16 B multiple, keeping the DRAM offset 16 B-aligned; only the final tail flush may move an odd
// word count (nothing follows it). Accounts the words as dropped WITHOUT writing once the array would overflow.
template <typename LogT>
__attribute__((noinline)) void log_flush_words_to_dram(
    LogT* log, uint32_t num_words, uint32_t records_dram_base, uint32_t records_dram_size, uint32_t bank) {
    if (num_words == 0) {
        return;
    }
    uint32_t bytes = num_words * sizeof(uint64_t);
    uint32_t off = log->header.dram_write_offset;
    if (off + bytes > records_dram_size) {
        log->header.dropped += num_words;  // DRAM word array full -> account the drop
        return;
    }
    uint64_t dst = get_noc_addr_from_bank_id<true>(bank, records_dram_base + off);
    noc_async_write(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&log->words[0])), dst, bytes);
    noc_async_write_barrier();
    log->header.dram_write_offset = off + bytes;
    log->header.dram_words += num_words;
}

// Append one word to the L1 buffer, first bulk-flushing (and wrapping) the full buffer to DRAM if it is full.
// Flushing when full means the flush always moves WORD_CAPACITY (even) words -> 16 B-aligned. A flush may land
// between a common word and its channel words; that is fine because the stream is self-describing.
template <typename LogT>
FORCE_INLINE void log_append_word(
    LogT* log, uint64_t word, uint32_t records_dram_base, uint32_t records_dram_size, uint32_t bank) {
    if (log->word_count >= LogT::WORD_CAPACITY) {
        log_flush_words_to_dram(log, LogT::WORD_CAPACITY, records_dram_base, records_dram_size, bank);
        log->word_count = 0;
    }
    log->words[log->word_count++] = word;
}

// Blocking flush of the shared log header to the FRONT of this router's DRAM buffer (offset 0), so the host reader
// can frame + reconstruct the word array without ever touching L1. Called once at STOP after the tail flush.
template <typename HeaderT>
FORCE_INLINE void flush_log_header_to_dram(HeaderT* header, uint32_t dram_buffer_base, uint32_t bank) {
    uint64_t dst = get_noc_addr_from_bank_id<true>(bank, dram_buffer_base);
    noc_async_write(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(header)), dst, static_cast<uint32_t>(sizeof(HeaderT)));
    noc_async_write_barrier();
}

// [debug] Build the per-iteration COMMON word (shared timeline) and advance the delta baseline. Called once per
// pass that produced >=1 channel word, just before dumping. `count` is that channel-word count (always >= 1, so
// it never collides with the channel-word sentinel).
template <typename LogT>
FORCE_INLINE uint64_t build_common_log_word(LogT* log, uint32_t iter, uint32_t count) {
    uint32_t now = get_timestamp_32b();
    uint32_t iter_delta = iter - log->last_iter;  // full range; pack clamps to 24 bit across multi-ms stalls
    uint32_t ts_delta = now - log->last_ts;       // 32 bit (wraps, pre-existing)
    log->last_iter = iter;
    log->last_ts = now;
    return pack_common_log_word(static_cast<uint8_t>(count), iter_delta, ts_delta);
}

// ============================================================================
// [debug] Receiver: init (#6/#7), per-channel diff (#4), dump (#5), finalize (#8).
// ============================================================================

// #6: seed a receiver channel's diff baseline from a live snapshot at window open.
FORCE_INLINE void init_receiver_channel_state(
    ReceiverChannelState& st,
    uint8_t channel_id,
    uint32_t ready,
    uint32_t ack,
    uint32_t wr_sent,
    uint32_t wr_flush,
    uint32_t completion) {
    st.channel_id = channel_id;
    st.last_ready = ready;
    st.last_ack = ack;
    st.last_wr_sent = wr_sent;
    st.last_wr_flush = wr_flush;
    st.last_completion = completion;
}

// #7: capture a receiver channel's initial in-flight backlog into the header (indexed [vc][local] from the id).
// Clamped: wr_flush is frozen at 0 in fused builds while completion advances, so guard the unsigned subtraction.
FORCE_INLINE void capture_receiver_base_gaps(
    ReceiverLogHeader& h, uint8_t channel_id, uint32_t ack, uint32_t wr_sent, uint32_t wr_flush, uint32_t completion) {
    uint32_t vc = channel_id_vc(channel_id);
    uint32_t local = channel_id_local(channel_id);
    h.base_ack_gap[vc][local] = log_clamp_u8(ack >= completion ? ack - completion : 0);
    h.base_wr_sent_gap[vc][local] = log_clamp_u8(wr_sent >= completion ? wr_sent - completion : 0);
    h.base_wr_flush_gap[vc][local] = log_clamp_u8(wr_flush >= completion ? wr_flush - completion : 0);
}

// #7 (scalars): reset the receiver header framing and zero the per-channel base-gap table at window open.
FORCE_INLINE void reset_receiver_header(ReceiverLogHeader& h, uint32_t window_id) {
    h.magic = RECEIVER_LOG_MAGIC;
    h.dropped = 0;
    h.window_id = window_id;
    h.dram_write_offset = 0;
    h.dram_words = 0;
    for (uint32_t vc = 0; vc < DETAILED_FABRIC_MAX_VCS; vc++) {
        for (uint32_t c = 0; c < DETAILED_FABRIC_MAX_RECEIVER_CH_PER_VC; c++) {
            h.base_ack_gap[vc][c] = 0;
            h.base_wr_sent_gap[vc][c] = 0;
            h.base_wr_flush_gap[vc][c] = 0;
        }
    }
}

// #4: append a receiver channel word iff its flow-control state changed since the last recorded row. ready is
// stored absolute; the monotonic counters as clamped byte deltas (0/1 per pass, never clamped in practice).
FORCE_INLINE void make_diff_against(
    ReceiverChannelState& st,
    PendingLogWords& pending,
    uint32_t ready,
    uint32_t ack,
    uint32_t wr_sent,
    uint32_t wr_flush,
    uint32_t completion) {
    if (ready == st.last_ready && ack == st.last_ack && wr_sent == st.last_wr_sent && wr_flush == st.last_wr_flush &&
        completion == st.last_completion) {
        return;  // no change -> collapse
    }
    pending.words[pending.count++] = pack_receiver_channel_log_word(
        st.channel_id,
        log_clamp_u8(ready),
        log_clamp_u8(ack - st.last_ack),
        log_clamp_u8(wr_sent - st.last_wr_sent),
        log_clamp_u8(wr_flush - st.last_wr_flush),
        log_clamp_u8(completion - st.last_completion));
    st.last_ready = ready;
    st.last_ack = ack;
    st.last_wr_sent = wr_sent;
    st.last_wr_flush = wr_flush;
    st.last_completion = completion;
}

// #5: frame the pending channel words with a common word and append the batch to L1 (flushing to DRAM as needed).
FORCE_INLINE void dump_receiver_log_words(ReceiverLog* log, uint64_t common_word, PendingLogWords& pending) {
    const uint32_t base = static_cast<uint32_t>(receiver_log_dram_buffer_base) + DETAILED_FABRIC_LOG_DRAM_HEADER_REGION;
    const uint32_t size = static_cast<uint32_t>(receiver_log_dram_buffer_size) - DETAILED_FABRIC_LOG_DRAM_HEADER_REGION;
    log_append_word(log, common_word, base, size, log_dram_bank_id);
    for (uint32_t i = 0; i < pending.count; i++) {
        log_append_word(log, pending.words[i], base, size, log_dram_bank_id);
    }
}

// #8: flush the L1 word tail then the completed header to DRAM at STOP, so the DRAM buffer holds the COMPLETE
// trace ([header | packed words]). No-op on eRiscs that do not service the receiver channels.
static __attribute__((noinline)) void finalize_receiver_log() {
    if constexpr (receiver_log_enabled()) {
        ReceiverLog* log = receiver_log();
        const uint32_t base =
            static_cast<uint32_t>(receiver_log_dram_buffer_base) + DETAILED_FABRIC_LOG_DRAM_HEADER_REGION;
        const uint32_t size =
            static_cast<uint32_t>(receiver_log_dram_buffer_size) - DETAILED_FABRIC_LOG_DRAM_HEADER_REGION;
        if (log->word_count > 0) {
            log_flush_words_to_dram(log, log->word_count, base, size, log_dram_bank_id);
            log->word_count = 0;
        }
        flush_log_header_to_dram(&log->header, static_cast<uint32_t>(receiver_log_dram_buffer_base), log_dram_bank_id);
    }
}

// ============================================================================
// [debug] Sender: init (#6/#7), per-channel diff (#4), dump (#5), finalize (#8). Sender channel state lives in
// the log struct (hooks maintain the running totals; stash writes the per-pass annotation), so diff/init walk it
// with a compile-time unroll over the serviced channels.
// ============================================================================

// [debug] Stash this pass's per-channel level + annotation into the log (called from run_sender_channel_step_impl
// while the window is active). The monotonic totals are bumped at their event sites, not here.
FORCE_INLINE void stash_sender_flow_state(
    uint32_t ch, uint32_t local_occ, uint32_t dn_credits, uint8_t reason, bool conn) {
    SenderChannelState& st = sender_log()->channels[ch];
    st.occ = log_clamp_u8(local_occ);
    st.reason = reason;
    st.conn = conn ? 1 : 0;
    st.dn_credits = log_clamp_u8(dn_credits);
}

// #6: seed a sender channel's diff baseline from its current running totals + stash at window open.
FORCE_INLINE void init_sender_channel_state(SenderChannelState& st, uint8_t channel_id) {
    st.channel_id = channel_id;
    st.last_occ = st.occ;
    st.last_dn_credits = st.dn_credits;
    st.last_sent = st.sent;
    st.last_acked = st.acked;
    st.last_cmpl = st.cmpl;
}

// #7: capture a sender channel's initial backlog gaps into the header (indexed [vc][local] from the id). Clamped:
// sent/acked are in-window-only event counts, so cmpl can exceed them across window boundaries -> clamp to 0.
FORCE_INLINE void capture_sender_base_gaps(SenderLogHeader& h, const SenderChannelState& st) {
    uint32_t vc = channel_id_vc(st.channel_id);
    uint32_t local = channel_id_local(st.channel_id);
    h.base_sent_gap[vc][local] = log_clamp_u8(st.sent >= st.cmpl ? st.sent - st.cmpl : 0);
    h.base_acked_gap[vc][local] = log_clamp_u8(st.acked >= st.cmpl ? st.acked - st.cmpl : 0);
}

// #7 (scalars): reset the sender header framing and zero the per-channel base-gap table at window open.
FORCE_INLINE void reset_sender_header(SenderLogHeader& h, uint32_t window_id) {
    h.magic = SENDER_LOG_MAGIC;
    h.dropped = 0;
    h.window_id = window_id;
    h.dram_write_offset = 0;
    h.dram_words = 0;
    for (uint32_t vc = 0; vc < DETAILED_FABRIC_MAX_VCS; vc++) {
        for (uint32_t c = 0; c < DETAILED_FABRIC_MAX_SENDER_CH_PER_VC; c++) {
            h.base_sent_gap[vc][c] = 0;
            h.base_acked_gap[vc][c] = 0;
        }
    }
}

// #4: append a sender channel word iff its state changed since the last recorded row. reason/conn flip every pass,
// so they are excluded from the change test and stored as a snapshot annotation. Reads the current values from the
// channel state (hooks/stash maintain them); the counters go out as clamped byte deltas.
FORCE_INLINE void make_diff_against(SenderChannelState& st, PendingLogWords& pending) {
    if (st.sent == st.last_sent && st.acked == st.last_acked && st.cmpl == st.last_cmpl && st.occ == st.last_occ &&
        st.dn_credits == st.last_dn_credits) {
        return;  // no change -> collapse
    }
    pending.words[pending.count++] = pack_sender_channel_log_word(
        st.channel_id,
        pack_sender_flags(st.reason, st.conn != 0),
        st.occ,
        log_clamp_u8(st.sent - st.last_sent),
        log_clamp_u8(st.acked - st.last_acked),
        log_clamp_u8(st.cmpl - st.last_cmpl),
        st.dn_credits);
    st.last_sent = st.sent;
    st.last_acked = st.acked;
    st.last_cmpl = st.cmpl;
    st.last_occ = st.occ;
    st.last_dn_credits = st.dn_credits;
}

// Compile-time unroll: diff every serviced sender channel into the pending scratch (#4 over all channels).
template <uint32_t CH>
FORCE_INLINE void diff_serviced_sender_channels(SenderLog* log, PendingLogWords& pending) {
    if constexpr (CH < NUM_SENDER_CHANNELS) {
        if constexpr (sender_log_channel_enabled(CH)) {
            make_diff_against(log->channels[CH], pending);
        }
        diff_serviced_sender_channels<CH + 1>(log, pending);
    }
}

// Compile-time unroll: init every serviced sender channel's diff state + base gaps at window open (#6 + #7).
template <uint32_t CH>
FORCE_INLINE void init_serviced_sender_channels(SenderLog* log) {
    if constexpr (CH < NUM_SENDER_CHANNELS) {
        if constexpr (sender_log_channel_enabled(CH)) {
            constexpr uint8_t cid = make_channel_id(sender_flat_to_vc(CH), sender_flat_to_local(CH));
            init_sender_channel_state(log->channels[CH], cid);
            capture_sender_base_gaps(log->header, log->channels[CH]);
        }
        init_serviced_sender_channels<CH + 1>(log);
    }
}

// #5: frame the pending sender channel words with a common word and append the batch to L1 (flush as needed).
FORCE_INLINE void dump_sender_log_words(SenderLog* log, uint64_t common_word, PendingLogWords& pending) {
    const uint32_t base = static_cast<uint32_t>(sender_log_dram_buffer_base) + DETAILED_FABRIC_LOG_DRAM_HEADER_REGION;
    const uint32_t size = static_cast<uint32_t>(sender_log_dram_buffer_size) - DETAILED_FABRIC_LOG_DRAM_HEADER_REGION;
    log_append_word(log, common_word, base, size, log_dram_bank_id);
    for (uint32_t i = 0; i < pending.count; i++) {
        log_append_word(log, pending.words[i], base, size, log_dram_bank_id);
    }
}

// #8: flush the L1 word tail then the completed header to DRAM at STOP. No-op on eRiscs that do not service senders.
static __attribute__((noinline)) void finalize_sender_log() {
    if constexpr (sender_log_enabled()) {
        SenderLog* log = sender_log();
        const uint32_t base =
            static_cast<uint32_t>(sender_log_dram_buffer_base) + DETAILED_FABRIC_LOG_DRAM_HEADER_REGION;
        const uint32_t size =
            static_cast<uint32_t>(sender_log_dram_buffer_size) - DETAILED_FABRIC_LOG_DRAM_HEADER_REGION;
        if (log->word_count > 0) {
            log_flush_words_to_dram(log, log->word_count, base, size, log_dram_bank_id);
            log->word_count = 0;
        }
        flush_log_header_to_dram(&log->header, static_cast<uint32_t>(sender_log_dram_buffer_base), log_dram_bank_id);
    }
}

}  // namespace tt::tt_fabric
