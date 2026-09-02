// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Synchronization-event instrumentation for the critical-path tool.
//
// The question the tool answers is "which core was blocked, on what, and for how long". Every
// blocking primitive gets a ZONE spanning the wait, with one payload marker inside it carrying the
// join key (the CB id, or the semaphore address); every releasing signal gets a point marker with
// the same key. The key is the join: a host tool pairs "who was waiting on X" with "who signalled X"
// to reconstruct the cross-core dependency chain.
//
// WHY A ZONE AND A MARKER, NOT A MARKER PAIR. The previous shape bracketed each wait with
// WAIT-START / WAIT-END point markers, because a zone cannot carry a payload and the payload is the
// point. But that puts the burden of pairing two halves on whoever reads the data -- and nothing in
// the tree does it, on device, in the receiver, or in the CSV, so the stall duration was never
// actually computed anywhere. A zone arrives at the host as ONE record carrying (start, duration):
// no pairing to get wrong, no unpaired half when a capture is cut mid-wait, and a real span in Tracy
// instead of two dots the reader joins by eye. The key rides a marker INSIDE the zone, which costs
// one packet and leaves the zone doing what a zone is good at.
//
// EVERYTHING IS RECORDED, including waits that did not block. A wait that was already satisfied
// shows up as a near-zero-duration zone rather than as nothing at all, so "did this block" stays a
// question the host can ask (and re-ask, with a different threshold) instead of one the device
// answered destructively. Cheaper schemes exist -- emit only when the spin actually went round more
// than once -- and are deliberately deferred: they throw data away, and it is worth knowing what the
// full stream costs before optimising it.
//
// WHERE THE HOOKS GO. At the COMMON BOTTOM of each processor class, never at both layers of a call
// chain. Both the legacy CB API and the DataflowBuffer API (api/dataflow/dataflow_buffer.h) reach
// the same primitive in the end -- see internal/tt-1xx/dataflow_buffer.inl:
//
//     DM (BRISC/NCRISC)          TRISC
//     dataflow_api.h::cb_*       llk_io_{pack,unpack}.h::llk_*
//
// so hooking those two rows covers both APIs exactly once. cb_api.h is deliberately NOT hooked: on
// TRISC it is a pass-through to the llk_* functions, so a hook there would emit a second, nested
// zone for every legacy compute call while DFB calls emitted one -- silently double-counting the
// kernels that have not migrated.
//
// OPT-IN, off by default: TT_METAL_DEVICE_PROFILER_SYNC_EVENTS=1 injects PROFILE_SYNC_EVENTS into
// every JIT build (jit_build/build.cpp), so it is part of the JIT cache key. That flag is the entire
// reason these macros exist rather than calls to DeviceZoneScopedN / DeviceTimestampedData written
// straight at the hook sites: the profiler's own macros are gated on PROFILE_KERNEL and
// !DISPATCH_KERNEL, and this tool needs a third gate on top. Doing it with #if at each site would
// work, but then the disabled build would not COMPILE the hook expressions at all, and an opt-in
// path nothing routinely builds would rot silently as the primitives around it are refactored. The
// (void(sizeof(x))) form below keeps every argument parsed and type-checked in every build while
// emitting nothing.
//
// The DISPATCH_KERNEL exclusion is load-bearing, same as kernel_profiler.hpp's: dispatch cores have
// no drainer, and a marker written into an undrained ring wedges the NEXT capture.

#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && defined(PROFILE_SYNC_EVENTS)

// The macros expand where the HOOK SITE is parsed -- an inline function body in dataflow_api.h, the
// llk_io headers, or the semaphore headers -- so the profiler header has to be in first. Pulling it
// in HERE rather than trusting each hook file's include order is what makes the hooks safe from
// compute kernels, whose api chain is parsed long before the kernel source includes any profiler
// header. Only dataflow_api.h includes kernel_profiler.hpp on its own; the rest reach it through
// this line. The dependency runs ONE WAY: hook site -> this header -> kernel_profiler.hpp, which
// knows nothing about this tool.
#include "tools/profiler/kernel_profiler.hpp"

// A blocking wait. Declares an RAII zone in the CALLER'S scope, so the hook site must brace it
// around the spin:
//
//     WAYPOINT("CWFW");
//     {
//         SYNC_WAIT("SYNC-CB-WAIT", operand);
//         do { ... } while (...);
//     }
//     WAYPOINT("CWFD");
//
// The loop itself is untouched -- no peel, no restructure, nothing lifted into a macro argument.
//
// The key marker is emitted at zone OPEN, so its timestamp falls inside the zone's span and the host
// can attach the two by containment on the lane. Note the direction that implies: a zone ships at
// its CLOSE, so on the wire the KEY ARRIVES BEFORE THE ZONE THAT CONTAINS IT, and a consumer holds
// the key per lane until the zone shows up. The "-KEY" suffix keeps the two from sharing a name
// under two different structural ids.
#define SYNC_WAIT(name, key) \
    DeviceZoneScopedN(name); \
    DeviceTimestampedData(name "-KEY", (key))

// A releasing signal: a point marker, since there is no interval to span. Unconditional and
// per-call, exactly as the waits are.
#define SYNC_SIGNAL(name, key) DeviceTimestampedData(name, (key))

#else

// Null macros: the hook sites call these unconditionally, so they must exist in every build --
// non-profiled, dispatch, and profiled-without-the-opt-in alike -- with zero codegen. sizeof is
// unevaluated, so the argument is still parsed and type-checked but never emitted, which is what
// keeps this opt-in path from rotting between the rare builds that enable it.

#define SYNC_WAIT(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL(name, key) (void(sizeof(key)))

#endif
