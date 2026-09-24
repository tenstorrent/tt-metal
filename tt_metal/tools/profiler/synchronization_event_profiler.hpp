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

// Tags a NoC address with the NoC that carried it: bit 62 is the NoC index, bit 63 marks the
// tag as present. The address itself is untouched, and the tagged value never reaches
// hardware -- it exists only as the profiler payload, while the bare addr goes to the NoC.
//
// Recorded because the NoC index is not recoverable from the address. The two NoCs traverse
// the grid in opposite directions, so the same destination is reached by different routes and
// (on Blackhole, per NOC_0_{X,Y}_PHYS_COORD) mirrored physical coordinates. A consumer
// matching a remote signal to the core that waited on it has to know which NoC carried it.
//
// Bits 62 and 63 specifically: bit 60 is NOT free. Blackhole sets it on every PCIe address
// (NOC_XY_PCIE_ENCODING) and Quasar uses it for INVALID_MULTICAST_DESCRIPTOR, so a tag there
// would both alias those flags and be destroyed by stripping them. Address construction never
// sets bits 61-63: a multicast descriptor is 60 bits (36-bit local address + four 6-bit
// coordinates) and a unicast address is 48.
//
// Wrap EVERY NoC address passed to SYNC_SIGNAL in this: if some emitters tag and others do
// not, an absent tag stops meaning "capture from an older build" and consumers lose their
// fallback. Consumers must strip both bits BEFORE reading any field -- unicast is detected
// as "nothing set above bit 47", which a tag bit would break.
#define SYNC_SIGNAL_NOC_ADDR(name, addr, noc) \
    SYNC_SIGNAL(name, ((uint64_t)(addr)) | (((uint64_t)((noc) & 1)) << 62) | (1ull << 63))

#else

// When profiling is off, these do nothing but still compile-check the arguments.
#define SYNC_WAIT(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL_NOC_ADDR(name, addr, noc) (void(sizeof(addr) + sizeof(noc)))

#endif
