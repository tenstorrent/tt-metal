// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TEMPORARY BENCHMARK INSTRUMENTATION -- delete with the RELAY_LINEAR copy benchmark.
// Layout, host decode, and validation protocol: fd_section_timeline_benchmark_design.md.

#pragma once

#include <cstdint>

namespace fd_copy_bench {

constexpr uint32_t kBaseQuasar = 3 * 1024 * 1024;  // 0x300000
constexpr uint32_t kBaseTt1xx = 1200 * 1024;       // 0x12C000

// Layout
constexpr uint32_t kRowsCapacity = 256;
constexpr uint32_t kPfRowWords = 16;
constexpr uint32_t kDpRowWords = 16;
constexpr uint32_t kPfDoneFlagOff = 0x0000 / 4;
constexpr uint32_t kDpDoneFlagOff = 0x0040 / 4;
constexpr uint32_t kPfStagingOff = 0x0080 / 4;
constexpr uint32_t kPfCountersOff = 0x8080 / 4;
constexpr uint32_t kDpStagingOff = 0x80C0 / 4;
constexpr uint32_t kDpCountersOff = 0xC0C0 / 4;
constexpr uint32_t kTotalBytes = 0xC100;
constexpr uint32_t kDoneMagic = 0x464C5348;  // 'FLSH'

// Counter block indices (relative to kPfCountersOff / kDpCountersOff)
constexpr uint32_t kCtrRowCount = 0;
constexpr uint32_t kCtrDroppedRow = 1;
constexpr uint32_t kCtrDroppedChunk = 2;
constexpr uint32_t kCtrUnexpectedCmd = 3;

// Prefetcher row word indices -- EXACTLY 16 words = one 64-byte cache line. The row is written once per
// command and never re-read, so every line it spans is a compulsory miss served from slow TL1; at 32 words
// that was two fills and ~45 cyc per stamp. Words 12-15 are therefore per-command DURATIONS accumulated in
// registers and stored once, not timestamp pairs -- half the stores and half the words, at the cost of
// per-chunk visibility we no longer need (read-hiding and O(1)-publish are answered).
//
// Timestamps (host subtracts):
constexpr uint32_t kPfHeaderEnter = 0, kPfHeaderExit = 1, kPfProcessEnter = 8, kPfLinearEnter = 9, kPfLinearExit = 10,
                   kPfProcessExit = 11;
// acquire_pages() blocks on dispatcher page credit and sits inside BOTH the header and publish brackets,
// so it is attributable to neither unless split out. A duration, not a pair -- one store instead of two.
constexpr uint32_t kPfHeaderAcq = 3;
// Counts and accumulated durations. Accumulators are immune to the multi-traversal blending that made the
// original timestamp-based fetch group unusable (97/99 rows out of order): they sum correctly however many
// times the site runs. The fetch words cover fetch_q_get_cmds, which runs before the command it feeds, so
// they are stored and reset in pf_commit_row(); the rest reset at kPfLinearEnter.
// kPfHeaderWrite isolates the NoC write inside the header handler -- a fixed ~32 B transfer, so it is the
// one apples-to-apples point where both arches run exactly one engine-programming sequence. Retire it once
// the I$ experiment has used it as its falsification test (it should barely move; the rest should).
//
// Words 2, 4, 6 are free -- probes retired once their question was answered, each ~46 cyc/command on
// Quasar: fetch_read_wait / fetch_ptr_ops (CQ read 14 cyc, uncached ptr stores 12 -- 2.4% of
// prefetch_external, so command fetch is neither memory- nor uncached-bound); fetch_traversals (exactly
// 2.0 every run on both arches). Re-add any of them temporarily if a change could plausibly move them.
// Words 2 and 4 re-added 2026-08-13 (FD_BENCH_PF_FETCH_WAYPOINTS) to re-verify fetch_read_wait/fetch_ptr_ops
// on dispatch-engine, since that data was Tensix-dispatch-only and never re-confirmed on the new core type.
constexpr uint32_t kPfFetchReadWait = 2, kPfFetchPtrOps = 4;
constexpr uint32_t kPfHeaderWrite = 5, kPfPubAcqTotal = 7;
// Issue-to-completion latency of the PRIME read (the one issued before the double-buffer loop). Distinct
// from kPfDramReadExposed, which is only the part the command blocks on: a low exposed value can mean the
// read was hidden behind the previous chunk's write rather than fast, so the cross-arch read comparison
// needs this one. The prime read is used because it is the only read with no predecessor to hide behind,
// making it comparable across arches regardless of chunk count.
constexpr uint32_t kPfReadLatency = 14;
// kPfDramReadExposed is EXPOSED wait (blocked at the barrier) -- the only read term the period pays.
// kPfEntryFlush drains the PREVIOUS command's writes, so producer cost is publish[i] + entry_flush[i+1].
constexpr uint32_t kPfDramReadExposed = 12, kPfPublish = 13, kPfEntryFlush = 15;
static_assert(kPfEntryFlush < kPfRowWords, "prefetcher row must stay within one cache line");

// Dispatcher row word indices
constexpr uint32_t kDpLoopWaitStart = 0, kDpLoopWaitEnd = 1, kDpCmdStart = 2, kDpCmdEnd = 3;
constexpr uint32_t kDpChunkBase = 4, kDpChunkStride = 2, kDpChunkSlots = 4;
constexpr uint32_t kDpChunkAcqStart = 0, kDpChunkAcqEnd = 1;

}  // namespace fd_copy_bench

// Device-side accessor. Guard mirrors the idiom at the top of tools/profiler/kernel_profiler.hpp --
// these macros are defined only for kernel/firmware compilation, never for host builds.
#if defined(COMPILE_FOR_DM) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || \
    defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_AERISC)

namespace fd_copy_bench {

// Control uses the uncached alias on Quasar so the host sees the done-flag without a flush; staging
// does not, so per-waypoint stores stay cheap. Do NOT also touch either region through the other view.
//
// Staging is deliberately NOT volatile. A waypoint's correctness comes from when bench_cycle() executes,
// not from when its value reaches L1 -- nothing reads staging until the flush at the end-of-window marker.
// Volatile made every stamp a scheduling barrier, which grew kernel_main_hd's frame by 64 B and cost ~597
// cyc/iteration (measured, 64 KB). Dead-store elimination is therefore legal and desirable here: it
// collapses a waypoint restamped across loop traversals to its last write, which is the documented
// last-writer-wins semantics. control() stays volatile -- the done-flag is a real handshake.
#if defined(ARCH_QUASAR)
constexpr uint32_t kStagingBase = kBaseQuasar;
constexpr uint32_t kControlBase = kBaseQuasar + MEM_L1_UNCACHED_BASE;
#else
constexpr uint32_t kStagingBase = kBaseTt1xx;
constexpr uint32_t kControlBase = kBaseTt1xx;
#endif

inline uint32_t tt_l1_ptr* staging() { return reinterpret_cast<uint32_t tt_l1_ptr*>(kStagingBase); }
inline volatile uint32_t tt_l1_ptr* control() { return reinterpret_cast<volatile uint32_t tt_l1_ptr*>(kControlBase); }

#ifndef FD_BENCH_PF_TIMELINE
#define FD_BENCH_PF_TIMELINE 0
#endif
#ifndef FD_BENCH_PF_CHUNK_WAYPOINTS
#define FD_BENCH_PF_CHUNK_WAYPOINTS 0
#endif
#ifndef FD_BENCH_DP_TIMELINE
#define FD_BENCH_DP_TIMELINE 0
#endif
#ifndef FD_BENCH_DP_CHUNK_WAYPOINTS
#define FD_BENCH_DP_CHUNK_WAYPOINTS 0
#endif
#ifndef FD_BENCH_NULL_PROBE
#define FD_BENCH_NULL_PROBE 0
#endif
// Re-runs the SAME instrumentation counting retired instructions instead of cycles, so every section is
// reported in instructions. Divide a cycles run by an instret run, section by section, to get IPC directly
// -- no static/dynamic guessing, no residual arithmetic. `instret` is CSR 0xc02 on both arches (Zicntr on
// tt-1xx, Zihpm/Zicsr on the Quasar DM). The period line stays in cycles either way; the lockstep check
// compares prefetcher instructions against dispatcher cycles in this mode and will warn -- expected, ignore.
#ifndef FD_BENCH_COUNT_INSTRET
#define FD_BENCH_COUNT_INSTRET 0
#endif
#ifndef FD_BENCH_VALIDATE_STREAM
#define FD_BENCH_VALIDATE_STREAM 0
#endif
// Re-verifies fetch_read_wait/fetch_ptr_ops (see kPfFetchReadWait/kPfFetchPtrOps above) on whichever
// Quasar core type is running. Independent of FD_BENCH_COUNT_INSTRET; both accumulate in whatever unit
// bench_cycle() currently returns.
#ifndef FD_BENCH_PF_FETCH_WAYPOINTS
#define FD_BENCH_PF_FETCH_WAYPOINTS 0
#endif

#if FD_BENCH_PF_CHUNK_WAYPOINTS && !FD_BENCH_PF_TIMELINE
#error "FD_BENCH_PF_CHUNK_WAYPOINTS requires FD_BENCH_PF_TIMELINE"
#endif
#if FD_BENCH_DP_CHUNK_WAYPOINTS && !FD_BENCH_DP_TIMELINE
#error "FD_BENCH_DP_CHUNK_WAYPOINTS requires FD_BENCH_DP_TIMELINE"
#endif

FORCE_INLINE uint32_t bench_cycle_raw() {
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

FORCE_INLINE uint32_t bench_instret_raw() {
    uint32_t c;
    asm volatile("rdinstret %0" : "=r"(c));
    return c;
}

// Flag-gated waypoints. FD_BENCH_NULL_PROBE compiles the CSR read out while leaving every store, branch
// and register-pressure effect intact, so the two builds differ only in rdcycle.
FORCE_INLINE uint32_t bench_cycle() {
#if FD_BENCH_NULL_PROBE
    return 0;
#elif FD_BENCH_COUNT_INSTRET
    return bench_instret_raw();
#else
    return bench_cycle_raw();
#endif
}

// The period baseline (the unconditional dispatcher pair) must NEVER be nulled or switched to instret --
// it is the measurement the other modes are trying to attribute, and nulling it reports period=0.
FORCE_INLINE uint32_t bench_cycle_period() { return bench_cycle_raw(); }

// Word offset of the row currently being filled. Constant-initialized on purpose: a pointer initializer
// would need a namespace-scope reinterpret_cast, i.e. dynamic init, which may never run in a kernel.
inline uint32_t g_pf_row_base = kPfStagingOff;
inline uint32_t g_pf_rows = 0;
// Accumulated in registers, stored once. The publish/read ones live inside shared helpers that take no
// chunk index, so a running total is the only cheap way in; process_relay_linear_cmd resets them on entry,
// which discards anything other command types contributed.
inline uint32_t g_pf_pub_acq = 0;
inline uint32_t g_pf_dram_read = 0;
inline uint32_t g_pf_publish = 0;
// Prime-read issue timestamp, and its issue-to-completion latency. The latency is captured at whichever
// barrier retires the prime read -- the in-loop one when the transfer is multi-chunk, the final one when
// it is single-chunk -- so both sites guard on it still being zero.
inline uint32_t g_pf_read_issue = 0;
inline uint32_t g_pf_read_latency = 0;
// Accumulate across both fetch_q_get_cmds calls that feed one row (header + payload), reset in
// pf_commit_row() -- these calls happen before the row they feed even starts, so they can't be reset
// at kPfLinearEnter the way the relay_cmd-scoped accumulators are.
inline uint32_t g_pf_fetch_read_wait = 0;
inline uint32_t g_pf_fetch_ptr_ops = 0;

FORCE_INLINE void pf_mark(uint32_t word, uint32_t t) { staging()[g_pf_row_base + word] = t; }

// At exactly kRowsCapacity rows the array is full but nothing was lost: park the base on the last row
// without flagging. Only a further commit overwrites a recorded row, and that is what sets the flag.
FORCE_INLINE void pf_commit_row() {
#if FD_BENCH_PF_FETCH_WAYPOINTS
    pf_mark(kPfFetchReadWait, g_pf_fetch_read_wait);
    pf_mark(kPfFetchPtrOps, g_pf_fetch_ptr_ops);
    g_pf_fetch_read_wait = 0;
    g_pf_fetch_ptr_ops = 0;
#endif
    ++g_pf_rows;
    staging()[kPfCountersOff + kCtrRowCount] = g_pf_rows;
    if (g_pf_rows < kRowsCapacity) {
        g_pf_row_base += kPfRowWords;
    } else if (g_pf_rows > kRowsCapacity) {
        staging()[kPfCountersOff + kCtrDroppedRow] = 1;
    }
}

// Per-command accumulators for process_relay_linear_cmd. Reset at entry so a preceding command type that
// went through the same shared helpers contributes nothing, stored once at exit.
FORCE_INLINE void pf_linear_accum_reset() {
    g_pf_pub_acq = 0;
    g_pf_dram_read = 0;
    g_pf_publish = 0;
    g_pf_read_latency = 0;
}

// Records the prime read's latency at the first barrier to retire it; later barriers are no-ops.
FORCE_INLINE void pf_read_latency_capture(uint32_t now) {
    if (g_pf_read_latency == 0) {
        g_pf_read_latency = now - g_pf_read_issue;
    }
}

FORCE_INLINE void pf_linear_accum_store() {
    pf_mark(kPfPubAcqTotal, g_pf_pub_acq);
    pf_mark(kPfDramReadExposed, g_pf_dram_read);
    pf_mark(kPfPublish, g_pf_publish);
    pf_mark(kPfReadLatency, g_pf_read_latency);
}

// Called from the end-of-window marker command, NOT from terminate: the FD kernels are persistent and the
// host reads the log long before teardown. Flushes only the rows actually committed so far -- the marker
// recurs (once per host command batch), and flushing the full 32 KB capacity every time would be pure waste.
FORCE_INLINE void pf_publish_and_flag() {
#if defined(ARCH_QUASAR)
    flush_l2_cache_range(kStagingBase + kPfStagingOff * 4, (g_pf_row_base + kPfRowWords - kPfStagingOff) * 4);
    flush_l2_cache_range(kStagingBase + kPfCountersOff * 4, 64);
#endif
    control()[kPfDoneFlagOff] = kDoneMagic;
}

inline uint32_t g_dp_row_base = kDpStagingOff;
inline uint32_t g_dp_rows = 0;

FORCE_INLINE void dp_mark(uint32_t word, uint32_t t) { staging()[g_dp_row_base + word] = t; }

// See pf_commit_row() for why the flag is set at > capacity, not at == capacity.
FORCE_INLINE void dp_commit_row() {
    ++g_dp_rows;
    staging()[kDpCountersOff + kCtrRowCount] = g_dp_rows;
    if (g_dp_rows < kRowsCapacity) {
        g_dp_row_base += kDpRowWords;
    } else if (g_dp_rows > kRowsCapacity) {
        staging()[kDpCountersOff + kCtrDroppedRow] = 1;
    }
}

// `chunk` is a LOCAL in the calling handler, not a static -- it is only live within one command.
FORCE_INLINE void dp_chunk_mark(uint32_t chunk, uint32_t off, uint32_t t) {
    if (chunk < kDpChunkSlots) {
        dp_mark(kDpChunkBase + chunk * kDpChunkStride + off, t);
    } else {
        staging()[kDpCountersOff + kCtrDroppedChunk] = 1;
    }
}

// See pf_publish_and_flag() for why this hangs off the marker command and flushes only committed rows.
FORCE_INLINE void dp_publish_and_flag() {
#if defined(ARCH_QUASAR)
    flush_l2_cache_range(kStagingBase + kDpStagingOff * 4, (g_dp_row_base + kDpRowWords - kDpStagingOff) * 4);
    flush_l2_cache_range(kStagingBase + kDpCountersOff * 4, 64);
#endif
    control()[kDpDoneFlagOff] = kDoneMagic;
}

}  // namespace fd_copy_bench

#endif
