// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/realtime_profiler.hpp>

#include <utility>

#include "tt_metal/impl/dispatch/data_collection.hpp"

// Public experimental API — delegates to the internal tt:: functions.
namespace tt::tt_metal::experimental {

ProgramRealtimeProfilerCallbackHandle RegisterProgramRealtimeProfilerCallback(
    ProgramRealtimeProfilerCallback callback) {
    return tt::RegisterProgramRealtimeProfilerCallback(std::move(callback));
}

void UnregisterProgramRealtimeProfilerCallback(ProgramRealtimeProfilerCallbackHandle handle) {
    tt::UnregisterProgramRealtimeProfilerCallback(handle);
}

bool IsProgramRealtimeProfilerActive() { return tt::IsProgramRealtimeProfilerActive(); }

}  // namespace tt::tt_metal::experimental
