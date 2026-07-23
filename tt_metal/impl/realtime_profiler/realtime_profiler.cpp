// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/realtime_profiler.hpp>

#include <memory>
#include <utility>

#include <tt_stl/assert.hpp>

#include "impl/context/metal_context.hpp"
#include "realtime_profiler_consumer.hpp"
#include "realtime_profiler_service.hpp"

namespace tt::tt_metal::experimental {

ProgramRealtimeProfilerCallbackHandle RegisterProgramRealtimeProfilerCallback(
    ProgramRealtimeProfilerCallback callback) {
    auto& service = MetalContext::instance().realtime_profiler_service();
    TT_FATAL(service, "Real-time profiler service unavailable; no initialized MetalContext");
    return service->register_consumer(std::make_unique<RealtimeProfilerConsumer>(std::move(callback)));
}

void UnregisterProgramRealtimeProfilerCallback(ProgramRealtimeProfilerCallbackHandle handle) {
    auto& service = MetalContext::instance().realtime_profiler_service();
    if (!service) {
        return;
    }
    service->unregister_consumer(handle);
}

bool IsProgramRealtimeProfilerActive() {
    const auto& service = MetalContext::instance().realtime_profiler_service();
    return service && service->is_active();
}

}  // namespace tt::tt_metal::experimental
