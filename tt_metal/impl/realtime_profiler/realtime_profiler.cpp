// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/realtime_profiler.hpp>

#include <memory>
#include <utility>

#include "impl/context/metal_context.hpp"
#include "realtime_profiler_builtins.hpp"
#include "realtime_profiler_consumer.hpp"
#include "realtime_profiler_service.hpp"
#include "realtime_profiler_tracy_consumer.hpp"

namespace tt::tt_metal::experimental {

ProgramRealtimeProfilerCallbackHandle RegisterProgramRealtimeProfilerCallback(
    ProgramRealtimeProfilerCallback callback) {
    return MetalContext::instance().realtime_profiler_service()->register_consumer(
        std::make_unique<RealtimeProfilerConsumer>(std::move(callback)));
}

void UnregisterProgramRealtimeProfilerCallback(ProgramRealtimeProfilerCallbackHandle handle) {
    MetalContext::instance().realtime_profiler_service()->unregister_consumer(handle);
}

bool IsProgramRealtimeProfilerActive() { return MetalContext::instance().realtime_profiler_service()->is_active(); }

}  // namespace tt::tt_metal::experimental

namespace tt::tt_metal {

void register_builtin_realtime_profiler_consumers(RealtimeProfilerService& service, ContextId context_id) {
#if defined(TRACY_ENABLE)
    service.register_consumer(std::make_unique<RealtimeProfilerTracyConsumer>(context_id));
#else
    (void)service;
    (void)context_id;
#endif
}

}  // namespace tt::tt_metal
