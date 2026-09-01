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
// The prefetcher counter block is TWO 64-byte lines: one HPM set per command population.
constexpr uint32_t kPfCountersOff = 0x8080 / 4;
constexpr uint32_t kPfCountersBytes = 0x80;
constexpr uint32_t kDpStagingOff = 0x8100 / 4;
constexpr uint32_t kDpCountersOff = 0xC100 / 4;
constexpr uint32_t kTotalBytes = 0xC140;
constexpr uint32_t kDoneMagic = 0x464C5348;  // 'FLSH'

// Counter block indices (relative to kPfCountersOff / kDpCountersOff)
constexpr uint32_t kCtrRowCount = 0;
constexpr uint32_t kCtrDroppedRow = 1;
constexpr uint32_t kCtrDroppedChunk = 2;
constexpr uint32_t kCtrUnexpectedCmd = 3;
// Absolute HPM counter values, sampled once per window. Divide by kCtrRowCount for a per-command average;
// difference consecutive windows for a per-batch rate. Zero unless FD_BENCH_PF_HPM.
constexpr uint32_t kCtrHpm0 = 4, kCtrHpm1 = 5, kCtrHpm2 = 6, kCtrHpm3 = 7;
// Denominators for the four above, accumulated over the SAME bracket so the ratios are self-consistent:
// number of process_cmd() calls, and cycles spent in them. Any HPM cycle counter exceeding kCtrHpmCycles
// is impossible and means the bracket is wrong.
constexpr uint32_t kCtrHpmCmds = 8, kCtrHpmCycles = 9;
// Instructions retired over the same bracket, for IPC = kCtrHpmInstret / kCtrHpmCycles.
constexpr uint32_t kCtrHpmInstret = 10;
// Second line: the same set again for RELAY_INLINE_NOFLUSH, which carries the dispatch command downstream
// and is what ALL of real program dispatch runs on. Commands that are neither population (marker,
// terminate) are accumulated into neither, so both sets stay pure.
constexpr uint32_t kCtrInl0 = 16, kCtrInl1 = 17, kCtrInl2 = 18, kCtrInl3 = 19;
constexpr uint32_t kCtrInlCmds = 20, kCtrInlCycles = 21, kCtrInlInstret = 22;
// Third set: fetch_q_get_cmds, which runs once per loop iteration ahead of every process_cmd. With all three
// published, fetch + inline + payload should account for the period; whatever is left is loop overhead.
constexpr uint32_t kCtrFet0 = 23, kCtrFet1 = 24, kCtrFet2 = 25, kCtrFet3 = 26;
constexpr uint32_t kCtrFetCmds = 27, kCtrFetCycles = 28, kCtrFetInstret = 29;
// A fetch that finds its queue empty spins on the host and costs ~1e6 cycles, so a mean over all samples is
// meaningless. Such calls are counted apart from the steady-state set rather than averaged into it.
constexpr uint32_t kCtrFetBlkCmds = 30, kCtrFetBlkCycles = 31;
constexpr uint32_t kFetchBlockedCycles = 2048;
// All sets are flushed together as kPfCountersBytes.
static_assert(kCtrFetBlkCycles * 4 < kPfCountersBytes, "counter block overflows its two cache lines");

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
// All 16 words are in use. Words 2/4/6 are the reusable slots -- probes retired once their question is
// answered, then re-added when a change could plausibly move them. Retired here: fetch_traversals (exactly
// 2.0 every run on both arches).
// Words 2 and 4 held fetch_read_wait / fetch_ptr_ops. Retired 2026-08-25, question answered twice:
// read_wait 15.3-16.1 (vs 14 on Tensix), ptr_ops 9.0-15.0 (vs 12), together ~3% of prefetch_external and
// flat across payload on both core types -- command fetch is neither memory- nor uncached-bound.
//
// Reused 2026-08-25 to size the UNCACHED-LOAD hypothesis. Quasar and BH execute near-identical instruction
// counts in prefetch_external (336.8 vs 334.1) yet Quasar spends 1047.9 cyc against 616.3 -- the whole
// 432 cyc deficit is stall. Load-free IPC is equal between the arches, so the stall must be memory, and the
// only memory on this path that BH does not also pay is Quasar's uncached alias. These two brackets pin the
// per-load cost at the two sites that use it, which decides between the candidate fixes (invalidate-then-
// read-cached vs coalescing loads) -- they are within ~50 cyc of each other, so the number matters.
//
// Each bracket encloses ONE load, so in an FD_BENCH_COUNT_INSTRET run the accumulator reads out the
// EXECUTION COUNT (~1 instruction per entry), and cycles/count is the per-load cost. Probe overhead
// (~8.9 cyc/probe in situ) is comparable to the load being measured -- calibrate with FD_BENCH_NULL_PROBE
// before trusting the absolute value.
// Word 2 held kPfFetchQRead. Retired 2026-08-25, question answered: 10.5-12.6 cyc over 8 instructions,
// flat across payload and ~1% of prefetch_external -- the fetch-queue pointer deref is not a hotspot.
//
// Word 2 also held kPfProcCmd (whole process_cmd() call). Retired 2026-08-26, question answered: that
// bracket wraps BOTH handlers, so it is not a sub-part of the prefetch_external residual at all -- it
// measured 1982.8 against a 1141.6 bucket. Never sum it with header/relay_cmd.
//
// Cost of pf_commit_row() itself: the staging row is exactly one 64 B line, written once and never
// re-read, so committing it takes a cold fill (~72 cyc on Quasar, ~0 on BH's write-through cache). It is
// called just after kPfProcessExit, i.e. OUTSIDE relay_cmd, so all of it lands in the prefetch_external
// residual and inflates it on Quasar only. This bracket makes that term subtractable instead of hidden.
// Reported one row late (like entry_flush): the mark for row i carries row i-1's cost, because the value
// cannot be stored by the very call being measured.
constexpr uint32_t kPfCommitRow = 2;
constexpr uint32_t kPfCmdRead = 4;  // uncached cmd->base.cmd_id discriminant read (DM-NoC-written cmddat_q)
// Whole fetch_q_get_cmds() call, accumulated over both commands that feed one row: the inflight window,
// trid rotation, wrap handling, NoC read programming, the ptr stores, and the outer while(true) traversals.
// Word 6 previously held kPfFetchHostWait -- the host-stall spin (WAYPOINT "HQW"). Retired 2026-08-21:
// measured 0.0 at every payload on Quasar dispatch-engine, i.e. the branch never executes and the
// prefetcher is never host-bound in steady state. Re-add if the fetchq depth or host write rate changes.
constexpr uint32_t kPfFetchTotal = 6;
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
// Gates the per-command accumulators: kPfFetchTotal, kPfCommitRow, and the single-load bracket kPfCmdRead.
// Independent of FD_BENCH_COUNT_INSTRET; all three accumulate in whatever unit bench_cycle() currently
// returns, so one cycles run plus one instret run gives cost and execution count per site.
#ifndef FD_BENCH_PF_FETCH_WAYPOINTS
#define FD_BENCH_PF_FETCH_WAYPOINTS 0
#endif

// Rocket HPM counters (mhpmcounter3..6 at 0xB03..0xB06, selected by mhpmevent3..6 at 0x323..0x326).
// Selector = event set in bits[7:0], one-hot event mask in bits[N:8]. The events are chosen by the csrw
// values in cq_prefetch.cpp's kernel_main_hd.
//
// Bracketed around process_cmd() only, NOT the whole loop: fetch_q_get_cmds() blocks waiting on the host,
// so a whole-loop bracket charges idle time to the commands. A free-running read at window end has the
// same defect and produced counter values larger than the entire period.
#ifndef FD_BENCH_PF_HPM
#define FD_BENCH_PF_HPM 0
#endif
// #if FD_BENCH_PF_HPM && !(defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM))
// #error "FD_BENCH_PF_HPM requires ARCH_QUASAR && COMPILE_FOR_DM"
// #endif

// Selects which prefetcher command carries instrumentation. Exactly one per build: a Quasar period moves
// by up to 8% on code placement alone, and a probe in a handler that never runs still perturbs the one that
// does, so compiling both sets in would make the two benchmarks' numbers incomparable with each other and
// with every earlier build. The shared RELAY_INLINE_NOFLUSH header handler is always instrumented.
#define FD_BENCH_CMD_RELAY_LINEAR 0
#define FD_BENCH_CMD_RELAY_PAGED_PACKED 0
#ifndef FD_BENCH_CMD
#define FD_BENCH_CMD FD_BENCH_CMD_RELAY_PAGED_PACKED
#endif

#define FD_BENCH_PF_LINEAR (FD_BENCH_PF_TIMELINE && (FD_BENCH_CMD == FD_BENCH_CMD_RELAY_LINEAR))
#define FD_BENCH_PF_LINEAR_CHUNKS (FD_BENCH_PF_CHUNK_WAYPOINTS && (FD_BENCH_CMD == FD_BENCH_CMD_RELAY_LINEAR))
#define FD_BENCH_PF_PAGED_PACKED (FD_BENCH_PF_TIMELINE && (FD_BENCH_CMD == FD_BENCH_CMD_RELAY_PAGED_PACKED))

#if FD_BENCH_CMD != FD_BENCH_CMD_RELAY_LINEAR && FD_BENCH_CMD != FD_BENCH_CMD_RELAY_PAGED_PACKED
#error "FD_BENCH_CMD must be FD_BENCH_CMD_RELAY_LINEAR or FD_BENCH_CMD_RELAY_PAGED_PACKED"
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
// Cost of the PREVIOUS row's pf_commit_row(); assigned at the call site after that call returns, so it
// is never reset here -- each row overwrites it.
inline uint32_t g_pf_commit_row = 0;
inline uint32_t g_pf_fetch_total = 0;
// cmd_read is scoped to process_cmd, which runs after the fetch that feeds it, but it is reset alongside
// the fetch accumulators so both commands feeding one row are summed the same way.
inline uint32_t g_pf_cmd_read = 0;

FORCE_INLINE void pf_mark(uint32_t word, uint32_t t) { staging()[g_pf_row_base + word] = t; }

#if FD_BENCH_PF_HPM
inline uint32_t g_pf_hpm[4] = {0, 0, 0, 0};
inline uint32_t g_pf_hpm_snap[4] = {0, 0, 0, 0};
inline uint32_t g_pf_hpm_cycles = 0;
inline uint32_t g_pf_hpm_cycle_snap = 0;
inline uint32_t g_pf_hpm_instret = 0;
inline uint32_t g_pf_hpm_instret_snap = 0;
inline uint32_t g_pf_hpm_cmds = 0;
inline uint32_t g_pf_inl[4] = {0, 0, 0, 0};
inline uint32_t g_pf_inl_cycles = 0;
inline uint32_t g_pf_inl_instret = 0;
inline uint32_t g_pf_inl_cmds = 0;
inline uint32_t g_pf_fet[4] = {0, 0, 0, 0};
inline uint32_t g_pf_fet_cycles = 0;
inline uint32_t g_pf_fet_instret = 0;
inline uint32_t g_pf_fet_cmds = 0;
inline uint32_t g_pf_fet_blk_cycles = 0;
inline uint32_t g_pf_fet_blk_cmds = 0;

// Which population a bracketed region belongs to. kBucketSkip samples are discarded so marker and terminate
// commands cannot distort either command average.
enum BenchBucket : uint32_t { kBucketPayload = 0, kBucketInline = 1, kBucketSkip = 2, kBucketFetch = 3 };

// Cycles and instret are both sampled last on entry and last on exit, so each delta carries the same fixed
// probe overhead and their ratio stays close to the bracket's true IPC.
FORCE_INLINE void pf_hpm_enter() {
    asm volatile("csrr %0, 0xB03" : "=r"(g_pf_hpm_snap[0]));
    asm volatile("csrr %0, 0xB04" : "=r"(g_pf_hpm_snap[1]));
    asm volatile("csrr %0, 0xB05" : "=r"(g_pf_hpm_snap[2]));
    asm volatile("csrr %0, 0xB06" : "=r"(g_pf_hpm_snap[3]));
    g_pf_hpm_cycle_snap = bench_cycle_raw();
    g_pf_hpm_instret_snap = bench_instret_raw();
}

// `bucket` routes the sample. The caller must read the command id INSIDE the bracket, or the header line's
// compulsory miss moves out of the measured window.
FORCE_INLINE void pf_hpm_accum(uint32_t bucket) {
    uint32_t d[4];
    uint32_t v;
    asm volatile("csrr %0, 0xB03" : "=r"(v));
    d[0] = v - g_pf_hpm_snap[0];
    asm volatile("csrr %0, 0xB04" : "=r"(v));
    d[1] = v - g_pf_hpm_snap[1];
    asm volatile("csrr %0, 0xB05" : "=r"(v));
    d[2] = v - g_pf_hpm_snap[2];
    asm volatile("csrr %0, 0xB06" : "=r"(v));
    d[3] = v - g_pf_hpm_snap[3];
    const uint32_t dcyc = bench_cycle_raw() - g_pf_hpm_cycle_snap;
    const uint32_t dins = bench_instret_raw() - g_pf_hpm_instret_snap;
    // Every bucket reads all six probes, so per-sample overhead is identical and the sets are comparable.
    if (bucket == kBucketPayload) {
        g_pf_hpm[0] += d[0];
        g_pf_hpm[1] += d[1];
        g_pf_hpm[2] += d[2];
        g_pf_hpm[3] += d[3];
        g_pf_hpm_cycles += dcyc;
        g_pf_hpm_instret += dins;
        ++g_pf_hpm_cmds;
    } else if (bucket == kBucketInline) {
        g_pf_inl[0] += d[0];
        g_pf_inl[1] += d[1];
        g_pf_inl[2] += d[2];
        g_pf_inl[3] += d[3];
        g_pf_inl_cycles += dcyc;
        g_pf_inl_instret += dins;
        ++g_pf_inl_cmds;
    } else if (bucket == kBucketFetch) {
        if (dcyc >= kFetchBlockedCycles) {
            g_pf_fet_blk_cycles += dcyc;
            ++g_pf_fet_blk_cmds;
            return;
        }
        g_pf_fet[0] += d[0];
        g_pf_fet[1] += d[1];
        g_pf_fet[2] += d[2];
        g_pf_fet[3] += d[3];
        g_pf_fet_cycles += dcyc;
        g_pf_fet_instret += dins;
        ++g_pf_fet_cmds;
    }
}
#endif

// At exactly kRowsCapacity rows the array is full but nothing was lost: park the base on the last row
// without flagging. Only a further commit overwrites a recorded row, and that is what sets the flag.
FORCE_INLINE void pf_commit_row() {
#if FD_BENCH_PF_FETCH_WAYPOINTS
    pf_mark(kPfCommitRow, g_pf_commit_row);
    pf_mark(kPfCmdRead, g_pf_cmd_read);
    pf_mark(kPfFetchTotal, g_pf_fetch_total);
    g_pf_cmd_read = 0;
    g_pf_fetch_total = 0;
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
#if FD_BENCH_PF_HPM
    staging()[kPfCountersOff + kCtrHpm0] = g_pf_hpm[0];
    staging()[kPfCountersOff + kCtrHpm1] = g_pf_hpm[1];
    staging()[kPfCountersOff + kCtrHpm2] = g_pf_hpm[2];
    staging()[kPfCountersOff + kCtrHpm3] = g_pf_hpm[3];
    staging()[kPfCountersOff + kCtrHpmCmds] = g_pf_hpm_cmds;
    staging()[kPfCountersOff + kCtrHpmCycles] = g_pf_hpm_cycles;
    staging()[kPfCountersOff + kCtrHpmInstret] = g_pf_hpm_instret;
    staging()[kPfCountersOff + kCtrInl0] = g_pf_inl[0];
    staging()[kPfCountersOff + kCtrInl1] = g_pf_inl[1];
    staging()[kPfCountersOff + kCtrInl2] = g_pf_inl[2];
    staging()[kPfCountersOff + kCtrInl3] = g_pf_inl[3];
    staging()[kPfCountersOff + kCtrInlCmds] = g_pf_inl_cmds;
    staging()[kPfCountersOff + kCtrInlCycles] = g_pf_inl_cycles;
    staging()[kPfCountersOff + kCtrInlInstret] = g_pf_inl_instret;
    staging()[kPfCountersOff + kCtrFet0] = g_pf_fet[0];
    staging()[kPfCountersOff + kCtrFet1] = g_pf_fet[1];
    staging()[kPfCountersOff + kCtrFet2] = g_pf_fet[2];
    staging()[kPfCountersOff + kCtrFet3] = g_pf_fet[3];
    staging()[kPfCountersOff + kCtrFetCmds] = g_pf_fet_cmds;
    staging()[kPfCountersOff + kCtrFetCycles] = g_pf_fet_cycles;
    staging()[kPfCountersOff + kCtrFetInstret] = g_pf_fet_instret;
    staging()[kPfCountersOff + kCtrFetBlkCmds] = g_pf_fet_blk_cmds;
    staging()[kPfCountersOff + kCtrFetBlkCycles] = g_pf_fet_blk_cycles;
#endif
#if defined(ARCH_QUASAR)
    flush_l2_cache_range(kStagingBase + kPfStagingOff * 4, (g_pf_row_base + kPfRowWords - kPfStagingOff) * 4);
    flush_l2_cache_range(kStagingBase + kPfCountersOff * 4, kPfCountersBytes);
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
