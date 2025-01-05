// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "lightmetal_capture.hpp"
#include "command_generated.h"
#include <tt-metalium/logger.hpp>

// FIXME (kmabee) - Temp hack, remove before merge and integrate as cmake define.
#define TT_ENABLE_LIGHT_METAL_TRACE 1

#if defined(TT_ENABLE_LIGHT_METAL_TRACE) && (TT_ENABLE_LIGHT_METAL_TRACE == 1)
#define TRACE_FUNCTION_CALL(capture_func, ...)             \
    do {                                                   \
        if (LightMetalCaptureContext::Get().IsTracing()) { \
            capture_func(__VA_ARGS__);                     \
        }                                                  \
    } while (0)
#else
#define TRACE_FUNCTION_CALL(capture_func, ...) \
    do {                                       \
    } while (0)
#endif

// Generic helper to build command and add to vector of cmds (CQ)
inline void CaptureCommand(tt::tt_metal::flatbuffer::CommandType cmd_type, ::flatbuffers::Offset<void> fb_offset) {
    auto& ctx = LightMetalCaptureContext::Get();
    ctx.GetCmdsVector().push_back(tt::tt_metal::flatbuffer::CreateCommand(ctx.GetBuilder(), cmd_type, fb_offset));
}

inline void CaptureReplayTrace(IDevice* device, uint8_t cq_id, uint32_t tid, bool blocking) {
    auto& ctx = LightMetalCaptureContext::Get();
    log_debug(tt::LogMetalTrace, "{}: cq_id: {}, tid: {}, blocking: {}", __FUNCTION__, cq_id, tid, blocking);
    auto cmd = tt::tt_metal::flatbuffer::CreateReplayTraceCommand(ctx.GetBuilder(), cq_id, tid, blocking);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::ReplayTraceCommand, cmd.Union());
}

inline void CaptureEnqueueTrace(CommandQueue& cq, uint32_t tid, bool blocking) {
    auto& ctx = LightMetalCaptureContext::Get();
    log_debug(tt::LogMetalTrace, "{}: cq_id: {}, tid: {}, blocking: {}", __FUNCTION__, cq.id(), tid, blocking);
    auto cmd = tt::tt_metal::flatbuffer::CreateEnqueueTraceCommand(ctx.GetBuilder(), cq.id(), tid, blocking);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::EnqueueTraceCommand, cmd.Union());
}

inline void CaptureLoadTrace(IDevice* device, const uint8_t cq_id, const uint32_t tid) {
    auto& ctx = LightMetalCaptureContext::Get();
    log_debug(tt::LogMetalTrace, "{}: cq_id: {}, tid: {}", __FUNCTION__, cq_id, tid);
    auto cmd = tt::tt_metal::flatbuffer::CreateLoadTraceCommand(ctx.GetBuilder(), tid, cq_id);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::LoadTraceCommand, cmd.Union());
}
