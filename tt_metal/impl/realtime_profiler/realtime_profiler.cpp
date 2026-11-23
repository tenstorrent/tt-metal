// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/realtime_profiler.hpp>

#include <utility>

#include "realtime_profiler_service.hpp"

namespace tt::tt_metal::experimental {

ProgramRealtimeProfilerCallbackHandle RegisterProgramRealtimeProfilerCallback(
    ProgramRealtimeProfilerCallback callback) {
    return realtime_profiler_service().register_consumer(std::move(callback));
}

void UnregisterProgramRealtimeProfilerCallback(ProgramRealtimeProfilerCallbackHandle handle) {
    realtime_profiler_service().unregister_consumer(handle);
}

bool IsProgramRealtimeProfilerActive() { return realtime_profiler_service().is_active(); }

}  // namespace tt::tt_metal::experimental
