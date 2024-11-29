#pragma once

#include <cstdint>
#include "lightmetal_capture_context.hpp"
#include "command_generated.h"
#include "tt_metal/common/logger.hpp"

// KCM - Temporary hack for bringup.
#define ENABLE_TRACING 1

#ifdef ENABLE_TRACING
    #define TRACE_FUNCTION_CALL(capture_func, ...) \
        do { \
            if (LightMetalCaptureContext::getInstance().isTracing()) { \
                capture_func(__VA_ARGS__); \
            } \
        } while (0)
#else
    #define TRACE_FUNCTION_CALL(capture_func, ...) do { } while (0)
#endif

// Generic helper to build command and add to vector of cmds (CQ)
inline void captureCommand(tt::target::CommandType cmd_type, ::flatbuffers::Offset<void> fb_offset) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    // FIXME - Handle device_id.
    ctx.getCmdsVector().push_back(tt::target::CreateCommand(ctx.getBuilder(), cmd_type ,fb_offset));
}

inline void captureReplayTrace(Device *device, uint8_t cq_id, uint32_t tid, bool blocking) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;
    log_info(tt::LogMetalTrace, "captureReplayTrace: cq_id: {}, tid: {}, blocking: {}", cq_id, tid, blocking);
    auto cmd_variant = tt::target::CreateReplayTraceCommand(ctx.getBuilder(), cq_id, tid, blocking);
    captureCommand(tt::target::CommandType::ReplayTraceCommand, cmd_variant.Union());
}

inline void captureEnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;
    log_info(tt::LogMetalTrace, "captureEnqueueTrace: cq_id: {}, trace_id: {}, blocking: {}", cq.id(), trace_id, blocking);
    auto cmd_variant = tt::target::CreateEnqueueTraceCommand(ctx.getBuilder(), cq.id(), trace_id, blocking);
    captureCommand(tt::target::CommandType::EnqueueTraceCommand, cmd_variant.Union());
}

inline void captureLoadTrace(Device *device, const uint8_t cq_id, const uint32_t tid) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    if (!ctx.isTracing()) return;
    log_info(tt::LogMetalTrace, "{}: cq_id: {}, tid: {}", __FUNCTION__, cq_id, tid);
    auto cmd_variant = tt::target::CreateLoadTraceCommand(ctx.getBuilder(), tid, cq_id);
    captureCommand(tt::target::CommandType::LoadTraceCommand, cmd_variant.Union());
}
