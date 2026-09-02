// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Synchronization-event instrumentation for the critical-path tool. Every blocking primitive gets a zone
// spanning the wait with one payload marker inside carrying the join key (CB id or semaphore address), and
// every releasing signal gets a point marker with the same key, so waits and signals pair across cores.
// Non-blocking waits are recorded too, as near-zero zones.
// Hooks sit at the common bottom of each processor class: dataflow_api.h's cb_* on BRISC/NCRISC and
// llk_io_{pack,unpack}.h's llk_* on TRISC. cb_api.h is not hooked: on TRISC it passes through to llk_*, and
// a hook there would double-count unmigrated kernels.
// Opt-in: TT_METAL_DEVICE_PROFILER_SYNC_EVENTS=1 injects PROFILE_SYNC_EVENTS into every JIT build (part of
// the cache key). The DISPATCH_KERNEL exclusion matters as in kernel_profiler.hpp: a marker written into an
// undrained ring wedges the next capture.

#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && defined(PROFILE_SYNC_EVENTS)

// The macros expand where the hook site is parsed, so this header includes the profiler header itself;
// compute kernels parse their api chain before any profiler include.
#include "tools/profiler/kernel_profiler.hpp"

// A blocking wait: an RAII zone in the caller's scope, so the hook site braces it around the spin. The key
// marker is emitted at zone open, so it falls inside the zone's span and the host attaches the two by
// containment; the "-KEY" suffix gives it its own structural id.
#define SYNC_WAIT(name, key) \
    DeviceZoneScopedN(name); \
    DeviceTimestampedData(name "-KEY", (key))

// A releasing signal: a point marker, since there is no interval to span.
#define SYNC_SIGNAL(name, key) DeviceTimestampedData(name, (key))

#else

// Null macros: sizeof is unevaluated, so arguments stay type-checked with zero codegen.

#define SYNC_WAIT(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL(name, key) (void(sizeof(key)))

#endif
