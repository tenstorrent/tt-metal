// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
//
// SELECTION: this header is reached ONLY through api/debug/kernel_profiler.hpp, which includes it when
// the JIT build defines PROFILE_STREAMING (TT_METAL_STREAMING_PROFILER=1, jit_build/build.cpp). Without that
// define kernel_profiler.hpp is the DRAM profiler's own text, untouched. The two are never compiled into the
// same process: llrt/rtoptions.cpp TT_FATALs when both env vars are set. Never include this file directly.

#pragma once

#if defined(ARCH_QUASAR)
#error "The streaming profiler has no Quasar producer; TT_METAL_STREAMING_PROFILER is not supported on Quasar."
#endif
#if !defined(PROFILE_STREAMING)
#error \
    "kernel_profiler_streaming.hpp is selected by kernel_profiler.hpp under PROFILE_STREAMING; do not include it directly."
#endif

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_ERISC) || \
    defined(COMPILE_FOR_IDLE_ERISC) || defined(COMPILE_FOR_AERISC) || defined(COMPILE_FOR_DM)
#include "risc_common.h"
#include "internal/dataflow/dataflow_api_addrgen.h"
#include "api/tensor/tensor_accessor.h"
#else
#include "ckernel.h"
#endif

#include "internal/risc_attribs.h"

#include "hostdev/dev_msgs.h"

#include "internal/ethernet/erisc.h"


#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_ERISC)
// Global scope: a block-scope extern inside the namespace would look for kernel_profiler::__stack_base and
// fail to link.
extern uint32_t __stack_base[];
#endif

namespace kernel_profiler {

}

// Watcher-off net for a globals-heavy kernel whose loader-guaranteed stack floor is only MEM_*_STACK_MIN_SIZE
// (192-256 B): plant one word at __stack_base when the kernel zone opens, check it at close, emit a named
// intact canary reads as painted-and-unused to measure_stack_usage(). Compiled out for firmware
// (__stack_base is a kernel-link symbol) and active ERISC (stack guarded by -Werror=stack-usage).
#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_ERISC)
constexpr uint32_t STACK_CANARY_PATTERN = 0xBABABABA;  // == watcher stack_usage_pattern
struct stackCanaryScope {
    FORCE_INLINE stackCanaryScope() { ::__stack_base[0] = STACK_CANARY_PATTERN; }
    FORCE_INLINE ~stackCanaryScope() {
        if (__builtin_expect(::__stack_base[0] != STACK_CANARY_PATTERN, 0)) {
        }
    }
};
#else
struct stackCanaryScope {};  // FW builds and active ERISC: no kernel stack floor to watch
#endif

}  // namespace kernel_profiler

#include "noc_event_profiler.hpp"
#include "perf_counters.hpp"
#include "internal/profiler/synchronization_event_profiler.hpp"





// The canary is declared second so its check runs first at scope exit, inside the still-open kernel zone.
#define DeviceZoneScopedMainChildN(name) \
    DeviceZoneScopedN(name);             \
    kernel_profiler::stackCanaryScope zone_stack_canary = kernel_profiler::stackCanaryScope();

#define DeviceTraceOnlyProfilerInit()
#define DeviceIncrementTraceCount()
#define DeviceZoneScopedSumN1(name) (void(name))
#define DeviceZoneScopedSumN2(name) (void(name))

#else

// Zero codegen when profiling is off.
#define DeviceValidateProfiler(condition) (void(sizeof(condition)))

#define DeviceZoneScopedMainN(name) (void(name))

#define DeviceZoneScopedMainChildN(name) (void(name))

#define DeviceZoneScopedN(name) (void(name))

#define DeviceZoneScopedNIf(name, active) (void(name), void(sizeof(active)))

#define DeviceZoneScopedSumN1(name) (void(name))

#define DeviceZoneScopedSumN2(name) (void(name))

#define DeviceRecordEvent(event_id) (void(sizeof(event_id)))

#define DeviceTraceOnlyProfilerInit()

#define DeviceZoneSetCounter(counter) (void(sizeof(counter)))

#define DeviceTimestampedData(data_id, data) (void(sizeof(data_id) + sizeof(data)))

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
