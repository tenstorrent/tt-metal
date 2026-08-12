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
constexpr uint32_t kPfRowWords = 32;
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

// Prefetcher row word indices
constexpr uint32_t kPfHeaderEnter = 0, kPfHeaderExit = 1, kPfFetchStart = 2, kPfFetchqSeen = 3, kPfReadIssue = 4,
                   kPfReadBarrierStart = 5, kPfReadRetire = 6, kPfFetchReturn = 7, kPfProcessEnter = 8,
                   kPfLinearEnter = 9, kPfLinearExit = 10, kPfProcessExit = 11;
constexpr uint32_t kPfChunkBase = 12, kPfChunkStride = 4, kPfChunkSlots = 4;
constexpr uint32_t kPfChunkReadStart = 0, kPfChunkReadEnd = 1, kPfChunkPubStart = 2, kPfChunkPubEnd = 3;

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
#if defined(ARCH_QUASAR)
constexpr uint32_t kStagingBase = kBaseQuasar;
constexpr uint32_t kControlBase = kBaseQuasar + MEM_L1_UNCACHED_BASE;
#else
constexpr uint32_t kStagingBase = kBaseTt1xx;
constexpr uint32_t kControlBase = kBaseTt1xx;
#endif

inline volatile uint32_t tt_l1_ptr* staging() { return reinterpret_cast<volatile uint32_t tt_l1_ptr*>(kStagingBase); }
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
#ifndef FD_BENCH_VALIDATE_STREAM
#define FD_BENCH_VALIDATE_STREAM 0
#endif

#if FD_BENCH_PF_CHUNK_WAYPOINTS && !FD_BENCH_PF_TIMELINE
#error "FD_BENCH_PF_CHUNK_WAYPOINTS requires FD_BENCH_PF_TIMELINE"
#endif
#if FD_BENCH_DP_CHUNK_WAYPOINTS && !FD_BENCH_DP_TIMELINE
#error "FD_BENCH_DP_CHUNK_WAYPOINTS requires FD_BENCH_DP_TIMELINE"
#endif

FORCE_INLINE uint32_t bench_cycle() {
#if FD_BENCH_NULL_PROBE
    return 0;
#else
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
#endif
}

// Word offset of the row currently being filled. Constant-initialized on purpose: a pointer initializer
// would need a namespace-scope reinterpret_cast, i.e. dynamic init, which may never run in a kernel.
inline uint32_t g_pf_row_base = kPfStagingOff;
inline uint32_t g_pf_rows = 0;

FORCE_INLINE void pf_mark(uint32_t word, uint32_t t) { staging()[g_pf_row_base + word] = t; }

FORCE_INLINE void pf_commit_row() {
    ++g_pf_rows;
    staging()[kPfCountersOff + kCtrRowCount] = g_pf_rows;
    if (g_pf_rows < kRowsCapacity) {
        g_pf_row_base += kPfRowWords;
    } else {
        staging()[kPfCountersOff + kCtrDroppedRow] = 1;
    }
}

// `chunk` is a LOCAL in the calling handler, not a static -- it is only live within one command.
FORCE_INLINE void pf_chunk_mark(uint32_t chunk, uint32_t off, uint32_t t) {
    if (chunk < kPfChunkSlots) {
        pf_mark(kPfChunkBase + chunk * kPfChunkStride + off, t);
    } else {
        staging()[kPfCountersOff + kCtrDroppedChunk] = 1;
    }
}

FORCE_INLINE void pf_publish_and_flag() {
#if defined(ARCH_QUASAR)
    flush_l2_cache_range(kStagingBase + kPfStagingOff * 4, kRowsCapacity * kPfRowWords * 4);
    flush_l2_cache_range(kStagingBase + kPfCountersOff * 4, 64);
#endif
    control()[kPfDoneFlagOff] = kDoneMagic;
}

inline uint32_t g_dp_row_base = kDpStagingOff;
inline uint32_t g_dp_rows = 0;

FORCE_INLINE void dp_mark(uint32_t word, uint32_t t) { staging()[g_dp_row_base + word] = t; }

FORCE_INLINE void dp_commit_row() {
    ++g_dp_rows;
    staging()[kDpCountersOff + kCtrRowCount] = g_dp_rows;
    if (g_dp_rows < kRowsCapacity) {
        g_dp_row_base += kDpRowWords;
    } else {
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

FORCE_INLINE void dp_publish_and_flag() {
#if defined(ARCH_QUASAR)
    flush_l2_cache_range(kStagingBase + kDpStagingOff * 4, kRowsCapacity * kDpRowWords * 4);
    flush_l2_cache_range(kStagingBase + kDpCountersOff * 4, 64);
#endif
    control()[kDpDoneFlagOff] = kDoneMagic;
}

}  // namespace fd_copy_bench

#endif
