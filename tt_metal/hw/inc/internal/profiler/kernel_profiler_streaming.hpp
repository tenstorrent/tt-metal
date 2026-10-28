// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The macros firmware wraps every launch in, around device observability's ring (internal/debug/device_publish_ring.h),
// and the streaming profiler's device markers: zones, events and timestamped data, each a DevicePublish record
// (api/debug/device_publish.h).
//
// SELECTION: this header is reached ONLY through api/debug/kernel_profiler.hpp, which includes it when the JIT build
// defines DEVICE_OBSERVABILITY (TT_METAL_DEVICE_OBSERVABILITY=1, which TT_METAL_STREAMING_PROFILER=1 implies,
// jit_build/build.cpp). The markers compile in only under PROFILE_STREAMING (TT_METAL_STREAMING_PROFILER=1). Without
// DEVICE_OBSERVABILITY kernel_profiler.hpp is the DRAM profiler's own text, untouched. The two are never compiled into
// the same process: llrt/rtoptions.cpp TT_FATALs when TT_METAL_DEVICE_PROFILER is set too. Never include this file
// directly.

#pragma once

#if !defined(DEVICE_OBSERVABILITY)
#error "kernel_profiler_streaming.hpp is selected by kernel_profiler.hpp; do not include it directly."
#endif

#include <algorithm>
#include <climits>

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_AERISC) || defined(COMPILE_FOR_DM)
#include "risc_common.h"
#include "internal/dataflow/dataflow_api_addrgen.h"
#include "api/tensor/tensor_accessor.h"
#else
#include "ckernel.h"
#endif

#include "hostdev/streaming_profiler_common.h"
#include "internal/risc_attribs.h"

#include "hostdev/dev_msgs.h"

#include "internal/ethernet/erisc.h"

// Dispatch kernels and the relay kernel have no markers and no ring (internal/debug/device_publish_encoding.h).
#if defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && !defined(DEVICE_OBSERVABILITY_RELAY_KERNEL)

#include "api/debug/device_publish.h"

#include "noc_event_profiler.hpp"
#include "perf_counters.hpp"
#include "internal/profiler/synchronization_event_profiler.hpp"

#define DeviceValidateProfiler(condition) ::device_observability::set_launch_valid(condition);

#define DeviceZoneScopedMainN(name) ::device_observability::LaunchScope launch_scope;

#define DeviceZoneSetCounter(counter) ::device_observability::set_host_counter(counter);

// Shared firmware calls these unconditionally; they are real on the DRAM backend (trace replay) and empty
// here.
#define DeviceProfilerInit()
#define DeviceTraceOnlyProfilerInit()
#define DeviceIncrementTraceCount()

// NOT SUPPORTED on this backend, kept as null macros so the shared call sites compile: sum zones
// (llk_io_{pack,unpack}.h "CB-COMPUTE-*" accumulators) are a DRAM-profiler feature; the streaming wire ships
// every zone whole instead of summing on device.
#define DeviceZoneScopedSumN1(name) (void(name))
#define DeviceZoneScopedSumN2(name) (void(name))

#else

// Zero codegen when profiling is off.
#define DeviceValidateProfiler(condition) (void(sizeof(condition)))

#define DeviceZoneScopedMainN(name) (void(name))

#define DeviceZoneScopedSumN1(name) (void(name))

#define DeviceZoneScopedSumN2(name) (void(name))

#define DeviceTraceOnlyProfilerInit()

#define DeviceZoneSetCounter(counter) (void(sizeof(counter)))

#define DeviceProfilerInit()

#define DeviceIncrementTraceCount()

// null macros when noc tracing is disabled
#define RECORD_NOC_EVENT_WITH_ADDR(type, local_addr, noc_addr, num_bytes, vc, posted, noc)
#define RECORD_NOC_EVENT_WITH_ID(type, local_addr, noc_id, addrgen, offset, num_bytes, vc, posted, noc)
#define RECORD_NOC_EVENT(type, posted, noc)
#define NOC_TRACE_QUICK_PUSH_IF_LINKED(cmd_buf, linked)

// null macros when noc debugging is disabled
#define RECORD_SCOPED_LOCK_EVENT(event_type, locked_address_base, num_bytes)

// null macros when perf counters are disabled
#define StartPerfCounters()
#define StopPerfCounters()
#define RecordPerfCounters()

#define SYNC_WAIT(name, key) (void(sizeof(key)))
#define SYNC_WAIT_ZONE(name)
#define SYNC_WAIT_KEY(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL(name, key) (void(sizeof(key)))
#define SYNC_SIGNAL_NOC_ADDR(name, addr, noc) (void(sizeof(addr) + sizeof(noc)))

#endif

// The streaming profiler's markers, on kernels and firmware that have the ring.
#if defined(PROFILE_STREAMING) && defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && \
    !defined(DEVICE_OBSERVABILITY_RELAY_KERNEL)

#include "hostdevcommon/streaming_profiler_types.hpp"

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_ERISC)
// Global scope: a block-scope extern inside the namespace would look for kernel_profiler::__stack_base and
// fail to link.
extern uint32_t __stack_base[];
#endif

namespace kernel_profiler {

using namespace ::tt::tt_metal::experimental::streaming_profiler;

// Watcher-off net for a globals-heavy kernel whose loader-guaranteed stack floor is only MEM_*_STACK_MIN_SIZE
// (192-256 B): plant one word at __stack_base when the kernel zone opens, check it at close, publish a
// STACK-OVERFLOW event if frame data overwrote it. The pattern is the watcher's stack_usage_pattern on purpose, so an
// intact canary reads as painted-and-unused to measure_stack_usage(). Compiled out for firmware
// (__stack_base is a kernel-link symbol) and active ERISC (stack guarded by -Werror=stack-usage).
#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_ERISC)
constexpr uint32_t STACK_CANARY_PATTERN = 0xBABABABA;  // == watcher stack_usage_pattern
struct stackCanaryScope {
    FORCE_INLINE stackCanaryScope() { ::__stack_base[0] = STACK_CANARY_PATTERN; }
    FORCE_INLINE ~stackCanaryScope() {
        if (__builtin_expect(::__stack_base[0] != STACK_CANARY_PATTERN, 0)) {
            DevicePublish<EventMetadata{"STACK-OVERFLOW"_s}, device_observability::kPublishBatched>();
        }
    }
};
#else
struct stackCanaryScope {};  // FW builds and active ERISC: no kernel stack floor to watch
#endif

// A zone: when the scope it is declared in closes, it calls on_close, the zone's DevicePublish, with Interval{the time
// the scope opened}. Its open state is only that 8-byte start, since more would be register pressure across the user
// code inside the zone, and a globals-maxed kernel on the 192-256 B loader stack floor already gets tight around 10-20
// open zones.
template <typename OnClose>
struct TimedScope {
    FORCE_INLINE explicit TimedScope(OnClose on_close) :
        on_close(on_close), start(device_observability::Timestamp::now()) {}
    FORCE_INLINE ~TimedScope() {
        static_assert(
            sizeof(TimedScope) == sizeof(device_observability::Timestamp), "a zone holds only its start time");
        on_close(device_observability::Interval{start});
    }
    [[no_unique_address]] OnClose on_close;
    device_observability::Timestamp start;
};
// A zone gated on a bool read once at entry: false reads no clock and publishes nothing.
template <typename OnClose>
struct TimedScopeIf {
    FORCE_INLINE TimedScopeIf(OnClose on_close, bool active) : on_close(on_close), active(active) {
        if (active) {
            start = device_observability::Timestamp::now();
        }
    }
    FORCE_INLINE ~TimedScopeIf() {
        if (active) {
            on_close(device_observability::Interval{start});
        }
    }
    [[no_unique_address]] OnClose on_close;
    bool active;
    device_observability::Timestamp start;
};
}  // namespace kernel_profiler

// The markers publish batched: a kernel writes them often, and its launch's end publishes the rest.
// A zone's record, published from the line the zone is declared on: Interval{start}, whose end is the publish time.
#define DEVICE_ZONE_PUBLISH(name)                                \
    DevicePublish<                                               \
        ::kernel_profiler::ZoneMetadata{TT_RECORD_STRING(name)}, \
        ::device_observability::kPublishBatched,                 \
        ::device_observability::detail::kPublishTimeIn<&::device_observability::Interval::end>>

#define DeviceZoneScopedN(name) ::kernel_profiler::TimedScope zone(DEVICE_ZONE_PUBLISH(name));

#define DeviceZoneScopedNIf(name, active) ::kernel_profiler::TimedScopeIf zone(DEVICE_ZONE_PUBLISH(name), active);

#define DeviceTimestampedData(name, data)                                   \
    DevicePublish<                                                          \
        ::kernel_profiler::TimestampedDataMetadata{TT_RECORD_STRING(name)}, \
        ::device_observability::kPublishBatched>(static_cast<uint64_t>(data))

#define DeviceRecordEvent(name) \
    DevicePublish<::kernel_profiler::EventMetadata{TT_RECORD_STRING(name)}, ::device_observability::kPublishBatched>()

// The canary is declared second so its check runs first at scope exit, inside the still-open kernel zone.
#define DeviceZoneScopedMainChildN(name) \
    DeviceZoneScopedN(name);             \
    kernel_profiler::stackCanaryScope zone_stack_canary = kernel_profiler::stackCanaryScope();

#else

#define DeviceZoneScopedMainChildN(name) (void(name))

#define DeviceZoneScopedN(name) (void(name))

#define DeviceZoneScopedNIf(name, active) (void(name), void(sizeof(active)))

#define DeviceRecordEvent(event_id) (void(sizeof(event_id)))

#define DeviceTimestampedData(data_id, data) (void(sizeof(data_id) + sizeof(data)))

#endif
