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
//   llk_io_pack.h, llk_io_unpack.h (TRISC)
//
// Enable with: TT_METAL_STREAMING_PROFILER=1 TT_METAL_STREAMING_PROFILER_SYNC_EVENTS=1

#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && defined(PROFILE_SYNC_EVENTS) && defined(PROFILE_STREAMING)

#include "tools/profiler/kernel_profiler.hpp"

// Records a timed zone for a blocking wait, with the key embedded inside.
#define SYNC_WAIT(name, key) \
    DeviceZoneScopedN(name); \
    DeviceTimestampedData(name "-KEY", (key))

// Records a timestamp for a signal event.
#define SYNC_SIGNAL(name, key) DeviceTimestampedData(name, (key))

// Include NoC in upper bits to help decode multicast noc addresses
#define SYNC_SIGNAL_NOC_ADDR(name, addr, noc) \
    SYNC_SIGNAL(name, ((uint64_t)(addr)) | (((uint64_t)((noc) & 1)) << 62) | (1ull << 63))

#else

// When profiling is off, these do nothing but still compile-check the arguments.
#define SYNC_WAIT(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL_NOC_ADDR(name, addr, noc) (void(sizeof(addr) + sizeof(noc)))

#endif
