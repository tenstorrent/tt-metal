// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Synchronization-event instrumentation for the critical-path tool: point markers around the
// device's blocking primitives (CB waits, semaphore waits) and at the signals that release them
// (CB pushes, semaphore sets), each carrying the CB id or semaphore address as payload. The
// payload is the join key: a host tool pairs "who was waiting on X" with "who signaled X" to
// reconstruct the cross-core dependency chain.
//
// Every event is an ordinary DeviceTimestampedData marker -- a named structural id resolved from
// the ELF like any other marker, NOT a hardcoded numeric id (the pre-port version used raw ids
// 1000-1006, which the zone-csv consumer now reconstructs from the names for the classic file
// format). WAIT-START/WAIT-END stay a pair of point events rather than one zone because a zone
// cannot carry the payload, and the payload is the point.
//
// OPT-IN, off by default: TT_METAL_DEVICE_PROFILER_SYNC_EVENTS=1 injects PROFILE_SYNC_EVENTS
// into every JIT build (jit_build/build.cpp). These macros sit inside cb_wait_front /
// cb_push_back / semaphore spins -- per-tile hot paths -- so an always-on version would tax every
// profiled run with two extra packets per wait; disabled they compile to nothing.
//
// The DISPATCH_KERNEL exclusion is load-bearing, same as kernel_profiler.hpp's: dispatch cores
// have no drainer, and a marker written into an undrained ring wedges the NEXT capture.

#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && defined(PROFILE_SYNC_EVENTS)

// The RECORD_* macros expand to DeviceTimestampedData, and expansion happens where the HOOK SITE is
// parsed -- an inline function body in cb_api.h / dataflow_api.h / the semaphore headers -- so the
// profiler header has to be in first. Pulling it in HERE rather than trusting each hook file's
// include order is what makes the hooks safe from compute kernels, whose api chain parses cb_api.h
// long before the kernel source includes any profiler header.
//
// Dependency runs ONE WAY on purpose: hook site -> this header -> kernel_profiler.hpp. The profiler
// header knows nothing about this tool; either backend's DeviceTimestampedData works, since that is
// the only thing used below.
#include "tools/profiler/kernel_profiler.hpp"

//////////////////////////////
// CB events
//////////////////////////////

#define RECORD_CB_PUSH_BACK(cb_id) DeviceTimestampedData("SYNC-CB-PUSH", (cb_id))

#define RECORD_CB_WAIT_FRONT_START(cb_id) DeviceTimestampedData("SYNC-CB-WAIT-START", (cb_id))

#define RECORD_CB_WAIT_FRONT_END(cb_id) DeviceTimestampedData("SYNC-CB-WAIT-END", (cb_id))

//////////////////////////////
// Semaphore events
//////////////////////////////

#define RECORD_SEMAPHORE_SET(semaphore_address) DeviceTimestampedData("SYNC-SEM-SET", (semaphore_address))

#define RECORD_SEMAPHORE_SET_REMOTE(semaphore_address) DeviceTimestampedData("SYNC-SEM-SET-REMOTE", (semaphore_address))

#define RECORD_SEMAPHORE_WAIT_START(semaphore_address) DeviceTimestampedData("SYNC-SEM-WAIT-START", (semaphore_address))

#define RECORD_SEMAPHORE_WAIT_END(semaphore_address) DeviceTimestampedData("SYNC-SEM-WAIT-END", (semaphore_address))

#else

// Null macros: the hook sites (cb_api.h, dataflow_api.h, noc_semaphore.h, experimental
// semaphore.h) call these unconditionally, so they must exist in every build -- non-profiled,
// dispatch, and profiled-without-the-opt-in alike -- with zero codegen.

#define RECORD_CB_PUSH_BACK(cb_id) (void(sizeof(cb_id)))
#define RECORD_CB_WAIT_FRONT_START(cb_id) (void(sizeof(cb_id)))
#define RECORD_CB_WAIT_FRONT_END(cb_id) (void(sizeof(cb_id)))

#define RECORD_SEMAPHORE_SET(semaphore_address) (void(sizeof(semaphore_address)))
#define RECORD_SEMAPHORE_SET_REMOTE(semaphore_address) (void(sizeof(semaphore_address)))
#define RECORD_SEMAPHORE_WAIT_START(semaphore_address) (void(sizeof(semaphore_address)))
#define RECORD_SEMAPHORE_WAIT_END(semaphore_address) (void(sizeof(semaphore_address)))

#endif
