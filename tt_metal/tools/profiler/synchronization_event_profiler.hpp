// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Synchronization-event instrumentation for the critical-path tool: which core was blocked, on what,
// and for how long.
//
// Every blocking primitive gets a zone spanning the wait, with one payload marker inside it carrying
// the join key (a CB id, or a semaphore address); every releasing signal gets a point marker with the
// same key. The key is the join: pairing "who waited on X" with "who signalled X" reconstructs the
// cross-core dependency chain. Waits that did not block are recorded too, as near-zero-duration
// zones, so "did this block" stays a question the host can ask at any threshold.
//
// Hooks go at the common bottom of each processor class, never at both layers of a call chain: the
// legacy CB API and the DataflowBuffer API (internal/tt-1xx/dataflow_buffer.inl) both end up in
// dataflow_api.h's cb_* on BRISC/NCRISC and in llk_io_{pack,unpack}.h's llk_* on TRISC. cb_api.h is
// deliberately not hooked; on TRISC it is a pass-through to llk_*, so a hook there would nest a
// second zone on every legacy compute call, silently double-counting unmigrated kernels.
//
// Opt-in, off by default: TT_METAL_DEVICE_PROFILER_SYNC_EVENTS=1 injects PROFILE_SYNC_EVENTS into
// every JIT build (jit_build/build.cpp), so it is part of the JIT cache key. Needing that third gate
// on top of the profiler's own PROFILE_KERNEL and !DISPATCH_KERNEL is why these are macros rather
// than DeviceZoneScopedN / DeviceTimestampedData written straight at the hook sites.
//
// The DISPATCH_KERNEL exclusion is load-bearing, same as kernel_profiler.hpp's: dispatch cores have
// no relay, and a marker written into an undrained ring wedges the next capture.

#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && defined(PROFILE_SYNC_EVENTS)

// The macros expand where the hook site is parsed, so the profiler header has to be in first.
// Including it here rather than trusting each hook file's include order is what makes the hooks safe
// from compute kernels, whose api chain is parsed long before the kernel source includes any profiler
// header. The dependency runs one way: hook site -> this header -> kernel_profiler.hpp.
#include "tools/profiler/kernel_profiler.hpp"

// A blocking wait. Declares an RAII zone in the caller's scope, so the hook site must brace it around
// the spin.
//
// The key marker is emitted at zone open, so its timestamp falls inside the zone's span and the host
// attaches the two by containment on the lane. A zone ships at its close, so on the wire the key
// arrives before the zone containing it, and a consumer holds the key per lane until the zone shows
// up. The "-KEY" suffix keeps the two from sharing a name under two structural ids.
#define SYNC_WAIT(name, key) \
    DeviceZoneScopedN(name); \
    DeviceTimestampedData(name "-KEY", (key))

// A releasing signal: a point marker, since there is no interval to span.
#define SYNC_SIGNAL(name, key) DeviceTimestampedData(name, (key))

#else

// Null macros: the hook sites call these unconditionally, so they must exist in every build. sizeof
// is unevaluated, so arguments stay parsed and type-checked with zero codegen, which keeps this
// opt-in path from rotting between the rare builds that enable it.

#define SYNC_WAIT(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL(name, key) (void(sizeof(key)))

#endif
