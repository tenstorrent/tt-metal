// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(DEVICE_DEBUG_DUMP) && defined(PROFILE_KERNEL) && !defined(DISPATCH_KERNEL) && \
    (defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC))

#include "noc_debugging_metadata.hpp"
#include "internal/risc_attribs.h"
#include "kernel_profiler.hpp"
#include "hostdev/profiler_common.h"

namespace noc_debugging_profiler {

template <NocDebuggingEventMetadata::NocDebugEventType event_type>
FORCE_INLINE void recordScopedLockEvent(uint32_t locked_address_base, uint32_t num_bytes) {
    NocDebuggingEventMetadata ev_md;
    ev_md.setEventType(event_type);
    ev_md.setLockedRegion(locked_address_base, num_bytes);

    kernel_profiler::flush_to_dram_if_full<kernel_profiler::DoingDispatch::DISPATCH>();
    kernel_profiler::
        timeStampedData<kernel_profiler::NOC_DEBUGGING_STATIC_ID, kernel_profiler::DoingDispatch::DISPATCH>(
            ev_md.asU64());
}

}  // namespace noc_debugging_profiler

#define RECORD_SCOPED_LOCK_EVENT(event_type, locked_address_base, num_bytes) \
    noc_debugging_profiler::recordScopedLockEvent<event_type>((locked_address_base), (num_bytes))

// Unregister every DFB extent this RISC declared.
#define RECORD_DFB_REGION_CLEAR() \
    noc_debugging_profiler::recordScopedLockEvent<NocDebuggingEventMetadata::NocDebugEventType::DFB_REGION_CLEAR>(0, 0)

#else

#define RECORD_SCOPED_LOCK_EVENT(event_type, locked_address_base, num_bytes)
#define RECORD_DFB_REGION_CLEAR()

#endif
