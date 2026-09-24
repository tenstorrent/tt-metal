// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Profiler instrumentation for CB and semaphore sync operations.
//
// Each wait records a timed zone with a key (CB id or semaphore address).
// Each signal records a timestamp with the same key.
// This lets tooling match waits to signals.
//
// Instrumented in:
//   dataflow_api.h, noc_semaphore.h (BRISC/NCRISC)
//   llk_io_pack.h, llk_io_unpack.h, experimental/semaphore.h (TRISC)
//
// Enable with: TT_METAL_STREAMING_PROFILER=1 TT_METAL_DEVICE_PROFILER_SYNC_EVENTS=1

#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && defined(PROFILE_SYNC_EVENTS) && defined(PROFILE_STREAMING)

#include "tools/profiler/kernel_profiler.hpp"

// Records a timed zone for a blocking wait, with the key embedded inside.
#define SYNC_WAIT(name, key) \
    DeviceZoneScopedN(name); \
    DeviceTimestampedData(name "-KEY", (key))

// Records a timestamp for a signal event.
#define SYNC_SIGNAL(name, key) DeviceTimestampedData(name, (key))

// Tags a NoC address with the NoC that carried it: bit 60 is the NoC index, bit 61 marks the
// tag as present. A NoC address occupies bits 0-59 only (a 36-bit L1 offset plus four 6-bit
// coordinate fields), so the top nibble is free.
//
// Needed because a multicast address alone is ambiguous. get_safe_multicast_noc_addr passes
// the rectangle's corners as (end, start) on noc 1 and (start, end) on noc 0, so a noc-1
// rectangle arrives with both axes reversed -- indistinguishable from a noc-0 rectangle that
// wraps the torus on both axes. A noc-1 rectangle that wraps on one axis is likewise
// indistinguishable from a plain noc-0 wrap. No decoder can separate these after the fact,
// so the sender records which NoC it used.
//
// Wrap EVERY NoC address passed to SYNC_SIGNAL in this: if some emitters tag and others do
// not, an absent tag stops meaning "capture from an older build" and consumers lose their
// fallback. Consumers must strip both bits BEFORE reading any field -- unicast is detected
// as "nothing set above bit 47", which a tag bit would break.
#define SYNC_SIGNAL_NOC_ADDR(name, addr, noc) \
    SYNC_SIGNAL(name, ((uint64_t)(addr)) | (((uint64_t)((noc) & 1)) << 60) | (1ull << 61))

#else

// When profiling is off, these do nothing but still compile-check the arguments.
#define SYNC_WAIT(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL_NOC_ADDR(name, addr, noc) (void(sizeof(addr) + sizeof(noc)))

#endif
