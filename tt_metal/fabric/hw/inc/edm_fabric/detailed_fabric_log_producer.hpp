// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/experimental/fabric/detailed_fabric_log.hpp>
#include <tt-metalium/experimental/fabric/detailed_fabric_logs.hpp>

// [debug] PRODUCER-side machinery for the fabric detailed flow-control traces ([rxlog]/[txlog]).
//
// This header owns everything only the on-device fabric router needs to GENERATE the traces: the L1 working
// structs (the shared header from detailed_fabric_logs.hpp plus the delta-encoding working state -- last_*
// baselines, live per-channel totals, per-pass stashes), and the functions that reset / append / flush them.
// None of this is visible to the host reader, which only ever sees the shared header + record arrays in DRAM.
// The fabric router kernel (fabric_erisc_router.cpp) should just CALL into these functions.
//
// Not a standalone translation unit: include it ONLY from the fabric router kernel TU, AFTER its compile-time
// args and channel-serviced tables are in scope. It relies on those ambient symbols (same convention as the
// other edm_fabric kernel headers):
//   - CT-arg constexprs: receiver_log_buffer_addr, sender_log_buffer_addr, {receiver,sender}_log_dram_buffer_base,
//     {receiver,sender}_log_dram_buffer_size, log_dram_bank_id
//   - channel tables: is_receiver_channel_serviced[], is_sender_channel_serviced[], VC0_RECEIVER_CHANNEL,
//     NUM_SENDER_CHANNELS
//   - device intrinsics: get_timestamp_32b, get_noc_addr_from_bank_id, noc_async_write[_barrier], FORCE_INLINE
//
// DRAM buffer layout per log per router is [header region | packed record array] (see detailed_fabric_logs.hpp):
// records are flushed during the window to (base + HEADER_REGION); the header is flushed to `base` at STOP.

namespace tt::tt_fabric {

// [debug] Window gate: set while a detailed-logging window is open (the router's telemetry marker is nonzero).
// A 1-byte .bss global (not in the carved L1 region) because it is read on *every* send attempt to gate the
// instrumentation, and a local-RAM read is far cheaper than a volatile L1 read. Log writes only happen while
// the window is open, so their L1 cost is confined to the active region.
bool detailed_log_window_active = false;

// ============================================================================
// [debug] Receiver flow-control trace ([rxlog]) -- producer side.
// ============================================================================

// L1 working struct for the receiver trace: the shared (DRAM-flushed) header, then producer-only working state
// used for the delta encoding, then the record ring. Kept in the carved L1 log region rather than in kernel
// .bss so it does not eat the eRisc's tight data/stack budget. Records land at offset 64 (32 B header + 32 B
// of working state), matching the builder's 4096 B RECEIVER_LOG_BUFFER_SIZE carve (64 + 252*16 = 4096).
struct ReceiverLog {
    ReceiverLogHeader header;  // shared contract (flushed to DRAM at STOP)
    // Producer-only working state (never seen by the reader): the previous recorded row's absolute values,
    // seeded by reset_receiver_log() from a live snapshot at the window start, plus the L1-resident tail count.
    uint32_t count;  // number of valid records still resident in `records` (the tail; the rest are in DRAM)
    uint32_t last_ts;
    uint32_t last_iter;
    uint32_t last_ready;
    uint32_t last_ack;
    uint32_t last_wr_sent;
    uint32_t last_wr_flush;
    uint32_t last_completion;
    ReceiverLogRecord records[RECEIVER_LOG_CAPACITY];
};
static_assert(
    sizeof(ReceiverLog) <= RECEIVER_LOG_BUFFER_SIZE, "ReceiverLog exceeds the carved receiver_log_buffer region");

// Accessor folds to the constant base address (receiver_log_buffer_addr is a CT-arg constexpr), so it
// occupies no storage in local RAM.
FORCE_INLINE volatile ReceiverLog* receiver_log() {
    return reinterpret_cast<volatile ReceiverLog*>(receiver_log_buffer_addr);
}

// [debug] Blocking bulk-flush of `bytes` of packed records (starting at records_l1_src in L1) to this router's
// DRAM record array. The array lives in a single bank at a fixed per-bank byte offset (records_dram_base); the
// write targets exactly that bank via get_noc_addr_from_bank_id, so it never straddles a bank boundary. The
// caller passes bytes = num_records * sizeof(record) and we advance *dram_write_offset by exactly that, so
// successive flushes lay their records down contiguously -- DRAM ends up holding one packed record array, not a
// series of fixed-size chunks. Returns false WITHOUT writing once the array region would overflow
// (records_dram_size), so the caller can account the drop.
//
// Alignment: records_l1_src is &records[0] (16 B-aligned: the region base and the record-array offset are 16 B
// multiples) and records_dram_base is 16 B-aligned (buffer base + the 16 B-aligned HEADER_REGION). Every full-
// buffer flush moves CAPACITY*sizeof(record) -- a 16 B multiple for both logs (252*16, 128*20) -- so the DRAM
// offset stays 16 B-aligned and each write's start addresses satisfy the NOC 16 B write-alignment requirement
// (only the start addresses must be aligned, not the length). Only the final partial flush can move a
// non-multiple, and nothing follows it.
//
// Plain blocking noc_async_write: flushes happen only when an L1 buffer fills (every few hundred logged
// state-changes = dozens+ of tokens), so the stall is rare and coarse enough not to materially perturb the
// flow-control timing being measured.
FORCE_INLINE bool flush_log_records_to_dram(
    uint32_t records_l1_src,
    uint32_t bytes,
    uint32_t records_dram_base,
    uint32_t records_dram_size,
    uint32_t bank,
    volatile uint32_t* dram_write_offset) {
    uint32_t off = *dram_write_offset;
    if (off + bytes > records_dram_size) {
        return false;  // DRAM record array full -> caller accounts the drop
    }
    uint64_t dst = get_noc_addr_from_bank_id<true>(bank, records_dram_base + off);
    noc_async_write(records_l1_src, dst, bytes);
    noc_async_write_barrier();
    *dram_write_offset = off + bytes;
    return true;
}

// [debug] Blocking flush of the shared log header to the FRONT of this router's DRAM buffer (offset 0 in the
// header region), so the host reader can frame + reconstruct the record array without ever touching L1. Called
// once at STOP, after the record tail has been flushed (so dram_write_offset/dram_records are final). Only the
// start address must satisfy NOC 16 B alignment (buffer base is 16 B-aligned); the length may be any size.
FORCE_INLINE void flush_log_header_to_dram(
    uint32_t header_l1_src, uint32_t bytes, uint32_t dram_buffer_base, uint32_t bank) {
    uint64_t dst = get_noc_addr_from_bank_id<true>(bank, dram_buffer_base);
    noc_async_write(header_l1_src, dst, bytes);
    noc_async_write_barrier();
}

// [debug] Reset the receiver trace at the start marker: clear the log and snapshot the current flow-control
// state as the delta baseline. Every row in the window is stored as a delta against this snapshot. Counters are
// rebased to the completion snapshot (base_*_gap), so reconstructed values are 0-based on completion and stay
// correctly ordered even if the window opens with packets still in flight.
FORCE_INLINE void reset_receiver_log(
    uint32_t window_id,
    uint32_t window_start_cycles,
    uint32_t iter,
    uint32_t ready,
    uint32_t ack,
    uint32_t wr_sent,
    uint32_t wr_flush,
    uint32_t completion) {
    volatile ReceiverLog* log = receiver_log();
    log->header.magic = RECEIVER_LOG_MAGIC;
    log->header.dropped = 0;
    log->header.window_id = window_id;  // correlation/batch id from start_detailed_logging (the marker value)
    log->header.dram_write_offset = 0;
    log->header.dram_records = 0;
    log->count = 0;
    log->last_ts = window_start_cycles;
    log->last_iter = iter;
    log->last_ready = ready;
    log->last_ack = ack;
    log->last_wr_sent = wr_sent;
    log->last_wr_flush = wr_flush;
    log->last_completion = completion;
    // Initial in-flight backlog: each counter's gap above completion at window open. ack/wr_sent always lead
    // completion, but wr_flush is frozen at 0 in fused builds (fuse_receiver_flush_and_completion_ptr) while
    // completion advances -- so clamp to avoid an unsigned underflow that would poison the wflush column.
    log->header.base_ack_gap = ack >= completion ? ack - completion : 0;
    log->header.base_wr_sent_gap = wr_sent >= completion ? wr_sent - completion : 0;
    log->header.base_wr_flush_gap = wr_flush >= completion ? wr_flush - completion : 0;
}

// [debug] Append a record iff the flow-control state changed since the last recorded row. Gated by the caller
// on detailed_log_window_active. Cheap when nothing changed: 5 compares and an early return. Everything is
// stored as a delta against the previous recorded row (reset seeds the baseline): the monotonic counter deltas
// are 0/1 per pass so they never overflow a byte; ts_delta/iter_delta are full 32b so a long idle gap is
// captured exactly (no saturation); `ready` is stored absolute.
FORCE_INLINE void record_receiver_flow_state(
    uint32_t iter, uint32_t ready, uint32_t ack, uint32_t wr_sent, uint32_t wr_flush, uint32_t completion) {
    volatile ReceiverLog* log = receiver_log();
    if (ready == log->last_ready && ack == log->last_ack && wr_sent == log->last_wr_sent &&
        wr_flush == log->last_wr_flush && completion == log->last_completion) {
        return;  // no change -> collapse
    }

    uint32_t idx = log->count;
    if (idx >= RECEIVER_LOG_CAPACITY) {
        // L1 buffer full: bulk-flush its CAPACITY records to DRAM and wrap in place, preserving the delta
        // baseline (last_*/base_*_gap) so the reconstructed counter chain continues seamlessly across the
        // flushed boundary. If the DRAM buffer is also full, we cannot persist these CAPACITY records, so
        // account them as dropped (symmetric to the flushed case's dram_records += CAPACITY) and wrap anyway;
        // later fills will keep dropping CAPACITY at a time until the window ends.
        if (flush_log_records_to_dram(
                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&log->records[0])),
                RECEIVER_LOG_CAPACITY * sizeof(ReceiverLogRecord),
                static_cast<uint32_t>(receiver_log_dram_buffer_base) + DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
                static_cast<uint32_t>(receiver_log_dram_buffer_size) - DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
                log_dram_bank_id,
                &log->header.dram_write_offset)) {
            log->header.dram_records += RECEIVER_LOG_CAPACITY;
        } else {
            log->header.dropped += RECEIVER_LOG_CAPACITY;
        }
        log->count = 0;
        idx = 0;
    }
    uint32_t now = get_timestamp_32b();

    // Deltas against the previous recorded row. Counters are monotonic so these are non-negative. ts/iter are
    // full 32b (stored verbatim, no saturation); the single-byte counter deltas are 0/1 per pass so they never
    // reach their clamp.
    uint32_t ts_d = now - log->last_ts;
    uint32_t iter_d = iter - log->last_iter;
    uint32_t ack_d = ack - log->last_ack;
    uint32_t wr_sent_d = wr_sent - log->last_wr_sent;
    uint32_t wr_flush_d = wr_flush - log->last_wr_flush;
    uint32_t completion_d = completion - log->last_completion;

    volatile ReceiverLogRecord* rec = &log->records[idx];
    rec->ts_delta = ts_d;      // full 32b: capture long idle gaps without saturating
    rec->iter_delta = iter_d;  // full 32b
    rec->ready = ready > 0xFF ? 0xFF : static_cast<uint8_t>(ready);
    rec->ack_delta = ack_d > 0xFF ? 0xFF : static_cast<uint8_t>(ack_d);
    rec->wr_sent_delta = wr_sent_d > 0xFF ? 0xFF : static_cast<uint8_t>(wr_sent_d);
    rec->wr_flush_delta = wr_flush_d > 0xFF ? 0xFF : static_cast<uint8_t>(wr_flush_d);
    rec->completion_delta = completion_d > 0xFF ? 0xFF : static_cast<uint8_t>(completion_d);

    log->last_ts = now;
    log->last_iter = iter;
    log->last_ready = ready;
    log->last_ack = ack;
    log->last_wr_sent = wr_sent;
    log->last_wr_flush = wr_flush;
    log->last_completion = completion;

    log->count = idx + 1;
}

// ============================================================================
// [debug] Sender flow-control trace ([txlog]) -- producer side. Sender-side analog of the receiver trace, on
// the sender eRisc: one COMBINED record per main-loop pass covers BOTH serviced VC0 sender channels
// (ch0 = local worker, ch1 = forward). Per channel: monotonic sent/acked/completed totals + local occupancy;
// dn_credits is VC-shared (stored once); reason/conn are a per-pass annotation.
// ============================================================================

// L1 working struct for the sender trace: shared header, then producer-only working state (per-channel
// monotonic totals, the per-pass level stash + reason/conn annotation + VC-shared dn_credits, the previous-
// recorded-row baseline last_*, and the L1-resident tail count), then the record ring. Records land at offset
// 112, matching the builder's SENDER_LOG_BUFFER_SIZE (4096) carve (112 + 128*20 = 2672 <= 4096).
struct SenderLog {
    SenderLogHeader header;  // shared contract (flushed to DRAM at STOP)
    uint32_t count;          // records still resident in `records` (the tail; the rest are in DRAM)
    uint32_t sent[SENDER_LOG_NUM_CHANNELS];
    uint32_t acked[SENDER_LOG_NUM_CHANNELS];
    uint32_t cmpl[SENDER_LOG_NUM_CHANNELS];
    uint32_t dn_credits;  // VC-shared downstream free slots (last writer in a pass wins)
    uint32_t last_ts;
    uint32_t last_iter;
    uint32_t last_dn_credits;
    uint32_t last_sent[SENDER_LOG_NUM_CHANNELS];
    uint32_t last_acked[SENDER_LOG_NUM_CHANNELS];
    uint32_t last_cmpl[SENDER_LOG_NUM_CHANNELS];
    uint8_t occ[SENDER_LOG_NUM_CHANNELS];       // per-pass local backlog level
    uint8_t last_occ[SENDER_LOG_NUM_CHANNELS];  // previous recorded backlog level
    uint8_t reason[SENDER_LOG_NUM_CHANNELS];    // per-pass block-reason annotation
    uint8_t conn[SENDER_LOG_NUM_CHANNELS];      // per-pass connection flag
    SenderLogRecord records[SENDER_LOG_CAPACITY];
};
static_assert(sizeof(SenderLog) <= SENDER_LOG_BUFFER_SIZE, "SenderLog exceeds the carved sender_log_buffer region");

FORCE_INLINE volatile SenderLog* sender_log() { return reinterpret_cast<volatile SenderLog*>(sender_log_buffer_addr); }

// [debug] True iff sender channel `ch` should feed the [txlog] trace: serviced, and one of the two VC0 senders.
constexpr bool sender_log_channel_enabled(uint32_t ch) {
    return ch < SENDER_LOG_NUM_CHANNELS && is_sender_channel_serviced[ch];
}
constexpr bool sender_log_enabled() { return sender_log_channel_enabled(0) || sender_log_channel_enabled(1); }
// The [txlog] dump indexes SENDER_NUM_BUFFERS_ARRAY[0..1], so require >= 2 sender channels when it is active.
static_assert(
    !sender_log_enabled() || NUM_SENDER_CHANNELS >= SENDER_LOG_NUM_CHANNELS,
    "sender [txlog] trace assumes at least 2 sender channels");

// [debug] Stash this pass's per-channel levels + annotation into the L1 log struct (called from
// run_sender_channel_step_impl while the window is active). The monotonic totals are bumped at their event
// sites (see the accumulate hooks in the sender step), not here.
FORCE_INLINE void stash_sender_flow_state(
    uint32_t ch, uint32_t local_occ, uint32_t dn_credits, uint8_t reason, bool conn) {
    volatile SenderLog* log = sender_log();
    log->occ[ch] = local_occ > 0xFF ? 0xFF : static_cast<uint8_t>(local_occ);
    log->reason[ch] = reason;
    log->conn[ch] = conn ? 1 : 0;
    log->dn_credits = dn_credits;
}

// [debug] Reset the sender trace at the start marker: clear the log and snapshot the current per-channel totals
// as the delta baseline, so reconstructed counters read 0-based within the window. noinline (non-static so it
// never trips -Wunused-function on the receiver eRisc, where it is not called): keeps its locals out of
// kernel_main's frame. Called once per window.
__attribute__((noinline)) void reset_sender_log(uint32_t window_id, uint32_t window_start_cycles, uint32_t iter) {
    volatile SenderLog* log = sender_log();
    log->header.magic = SENDER_LOG_MAGIC;
    log->header.dropped = 0;
    log->header.window_id = window_id;  // correlation/batch id from start_detailed_logging (the marker value)
    log->header.dram_write_offset = 0;
    log->header.dram_records = 0;
    log->count = 0;
    log->last_ts = window_start_cycles;
    log->last_iter = iter;
    log->last_dn_credits = log->dn_credits;
    for (uint32_t ch = 0; ch < SENDER_LOG_NUM_CHANNELS; ch++) {
        log->last_occ[ch] = log->occ[ch];
        log->last_sent[ch] = log->sent[ch];
        log->last_acked[ch] = log->acked[ch];
        log->last_cmpl[ch] = log->cmpl[ch];
        // Initial backlog gaps above completion at window open. Clamped: sent/acked are in-window-only event
        // counts, so across window boundaries a completion counted here for an event NOT counted (its send/ack
        // fell outside any window) could push cmpl above them -- clamp to 0 rather than underflow.
        log->header.base_sent_gap[ch] = log->sent[ch] >= log->cmpl[ch] ? log->sent[ch] - log->cmpl[ch] : 0;
        log->header.base_acked_gap[ch] = log->acked[ch] >= log->cmpl[ch] ? log->acked[ch] - log->cmpl[ch] : 0;
    }
}

// [debug] Emit one combined record iff any channel's occupancy/counters (or the shared dn_credits) changed
// since the last recorded row. reason/conn flip every pass, so they are excluded from the change test and
// stored as a snapshot annotation. Called once per pass (after both sender steps) while the window is active.
// noinline (non-static, see reset_sender_log): its delta/reconstruction locals stay out of kernel_main's frame.
// The call is gated on detailed_log_window_active, so it costs nothing outside the monitored window.
__attribute__((noinline)) void record_sender_flow_state(uint32_t iter) {
    volatile SenderLog* log = sender_log();
    bool changed = log->dn_credits != log->last_dn_credits;
    for (uint32_t ch = 0; ch < SENDER_LOG_NUM_CHANNELS; ch++) {
        changed = changed || log->occ[ch] != log->last_occ[ch] || log->sent[ch] != log->last_sent[ch] ||
                  log->acked[ch] != log->last_acked[ch] || log->cmpl[ch] != log->last_cmpl[ch];
    }
    if (!changed) {
        return;  // no change -> collapse
    }

    uint32_t idx = log->count;
    if (idx >= SENDER_LOG_CAPACITY) {
        // L1 full: bulk-flush its CAPACITY records to DRAM and wrap in place, preserving the delta baseline so
        // the reconstructed chain stays continuous across the boundary. If DRAM is full too, account the
        // CAPACITY unflushable records as dropped (symmetric to dram_records += CAPACITY) and wrap anyway.
        if (flush_log_records_to_dram(
                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&log->records[0])),
                SENDER_LOG_CAPACITY * sizeof(SenderLogRecord),
                static_cast<uint32_t>(sender_log_dram_buffer_base) + DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
                static_cast<uint32_t>(sender_log_dram_buffer_size) - DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
                log_dram_bank_id,
                &log->header.dram_write_offset)) {
            log->header.dram_records += SENDER_LOG_CAPACITY;
        } else {
            log->header.dropped += SENDER_LOG_CAPACITY;
        }
        log->count = 0;
        idx = 0;
    }
    uint32_t now = get_timestamp_32b();
    uint32_t ts_d = now - log->last_ts;
    uint32_t iter_d = iter - log->last_iter;
    uint32_t sent0_d = log->sent[0] - log->last_sent[0];
    uint32_t acked0_d = log->acked[0] - log->last_acked[0];
    uint32_t cmpl0_d = log->cmpl[0] - log->last_cmpl[0];
    uint32_t sent1_d = log->sent[1] - log->last_sent[1];
    uint32_t acked1_d = log->acked[1] - log->last_acked[1];
    uint32_t cmpl1_d = log->cmpl[1] - log->last_cmpl[1];

    volatile SenderLogRecord* rec = &log->records[idx];
    rec->ts_delta = ts_d;      // full 32b: capture long idle gaps without saturating
    rec->iter_delta = iter_d;  // full 32b
    rec->dn_credits = log->dn_credits > 0xFF ? 0xFF : static_cast<uint8_t>(log->dn_credits);
    rec->ch_flags = static_cast<uint8_t>(
        (log->reason[0] & 0x7) | (log->conn[0] ? 0x8 : 0) | ((log->reason[1] & 0x7) << 4) | (log->conn[1] ? 0x80 : 0));
    rec->ch0_local_occ = log->occ[0];
    rec->ch0_sent_delta = sent0_d > 0xFF ? 0xFF : static_cast<uint8_t>(sent0_d);
    rec->ch0_acked_delta = acked0_d > 0xFF ? 0xFF : static_cast<uint8_t>(acked0_d);
    rec->ch0_cmpl_delta = cmpl0_d > 0xFF ? 0xFF : static_cast<uint8_t>(cmpl0_d);
    rec->ch1_local_occ = log->occ[1];
    rec->ch1_sent_delta = sent1_d > 0xFF ? 0xFF : static_cast<uint8_t>(sent1_d);
    rec->ch1_acked_delta = acked1_d > 0xFF ? 0xFF : static_cast<uint8_t>(acked1_d);
    rec->ch1_cmpl_delta = cmpl1_d > 0xFF ? 0xFF : static_cast<uint8_t>(cmpl1_d);

    log->last_ts = now;
    log->last_iter = iter;
    log->last_dn_credits = log->dn_credits;
    for (uint32_t ch = 0; ch < SENDER_LOG_NUM_CHANNELS; ch++) {
        log->last_occ[ch] = log->occ[ch];
        log->last_sent[ch] = log->sent[ch];
        log->last_acked[ch] = log->acked[ch];
        log->last_cmpl[ch] = log->cmpl[ch];
    }
    log->count = idx + 1;
}

// [debug] Finalize the receiver flow-control trace ([rxlog]) at the STOP marker: flush the final partial run of
// records still resident in L1 to DRAM, then flush the shared header to the front of the DRAM buffer, so this
// router's DRAM buffer holds the COMPLETE trace ([header | packed record array]). The host reader
// (dump_detailed_fabric_logs) then reads header + records straight from DRAM -- it never touches L1, and there
// is NO on-device print. Empty body on eRiscs that do not service the VC0 receiver channel. noinline: keeps its
// frame out of kernel_main (bounded by -Werror=stack-usage). Runs once per window.
static __attribute__((noinline)) void finalize_receiver_log() {
    if constexpr (is_receiver_channel_serviced[VC0_RECEIVER_CHANNEL]) {
        volatile ReceiverLog* rlog = receiver_log();
        if (rlog->count > 0) {
            if (flush_log_records_to_dram(
                    static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&rlog->records[0])),
                    rlog->count * sizeof(ReceiverLogRecord),
                    static_cast<uint32_t>(receiver_log_dram_buffer_base) + DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
                    static_cast<uint32_t>(receiver_log_dram_buffer_size) - DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
                    log_dram_bank_id,
                    &rlog->header.dram_write_offset)) {
                rlog->header.dram_records += rlog->count;
            } else {
                rlog->header.dropped += rlog->count;
            }
        }
        flush_log_header_to_dram(
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&rlog->header)),
            sizeof(ReceiverLogHeader),
            static_cast<uint32_t>(receiver_log_dram_buffer_base),
            log_dram_bank_id);
    }
}

// [debug] Finalize the sender flow-control trace ([txlog]) at the STOP marker: flush the L1 record tail then the
// shared header to DRAM, same as finalize_receiver_log. Empty body on eRiscs that do not service the VC0 sender
// channels.
static __attribute__((noinline)) void finalize_sender_log() {
    if constexpr (sender_log_enabled()) {
        volatile SenderLog* slog = sender_log();
        if (slog->count > 0) {
            if (flush_log_records_to_dram(
                    static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&slog->records[0])),
                    slog->count * sizeof(SenderLogRecord),
                    static_cast<uint32_t>(sender_log_dram_buffer_base) + DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
                    static_cast<uint32_t>(sender_log_dram_buffer_size) - DETAILED_FABRIC_LOG_DRAM_HEADER_REGION,
                    log_dram_bank_id,
                    &slog->header.dram_write_offset)) {
                slog->header.dram_records += slog->count;
            } else {
                slog->header.dropped += slog->count;
            }
        }
        flush_log_header_to_dram(
            static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&slog->header)),
            sizeof(SenderLogHeader),
            static_cast<uint32_t>(sender_log_dram_buffer_base),
            log_dram_bank_id);
    }
}

}  // namespace tt::tt_fabric
