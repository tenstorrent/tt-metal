// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt_metal_profiler.hpp>
#include <tt_metal/impl/profiler/profiler_state_manager.hpp>

#if defined(TRACY_ENABLE)

#define TracyTTMetalBeginMeshTrace(device_ids, trace_id)                                                        \
    for (auto device_id : (device_ids)) {                                                                       \
        std::string trace_message = fmt::format("`TT_METAL_TRACE_BEGIN: {}, {}`", device_id, trace_id);         \
        tt::tt_metal::MetalContext::instance().profiler_state_manager()->mark_trace_begin(device_id, trace_id); \
        TracyMessage(trace_message.c_str(), trace_message.size());                                              \
    }

#define TracyTTMetalEndMeshTrace(device_ids, trace_id)                                                        \
    for (auto device_id : (device_ids)) {                                                                     \
        std::string trace_message = fmt::format("`TT_METAL_TRACE_END: {}, {}`", device_id, trace_id);         \
        tt::tt_metal::MetalContext::instance().profiler_state_manager()->mark_trace_end(device_id, trace_id); \
        TracyMessage(trace_message.c_str(), trace_message.size());                                            \
    }

#define TracyTTMetalReplayMeshTrace(device_ids, trace_id)                                                        \
    for (auto device_id : (device_ids)) {                                                                        \
        std::string trace_message = fmt::format("`TT_METAL_TRACE_REPLAY: {}, {}`", device_id, trace_id);         \
        tt::tt_metal::MetalContext::instance().profiler_state_manager()->mark_trace_replay(device_id, trace_id); \
        TracyMessage(trace_message.c_str(), trace_message.size());                                               \
    }

#define TracyTTMetalReleaseMeshTrace(device_ids, trace_id)                                                \
    for (auto device_id : (device_ids)) {                                                                 \
        std::string trace_message = fmt::format("`TT_METAL_TRACE_RELEASE: {}, {}`", device_id, trace_id); \
        TracyMessage(trace_message.c_str(), trace_message.size());                                        \
    }

#define TracyTTMetalEnqueueProgramTrace(device_id, trace_id, program_runtime_id)                              \
    std::string trace_message =                                                                               \
        fmt::format("`TT_METAL_TRACE_ENQUEUE_PROGRAM: {}, {}, {}`", device_id, trace_id, program_runtime_id); \
    tt::tt_metal::MetalContext::instance().profiler_state_manager()->add_runtime_id_to_trace(                 \
        device_id, trace_id, program_runtime_id);                                                             \
    TracyMessage(trace_message.c_str(), trace_message.size());

#else

#define TracyTTMetalBeginMeshTrace(device_ids, trace_id)
#define TracyTTMetalEndMeshTrace(device_ids, trace_id)
#define TracyTTMetalReplayMeshTrace(device_ids, trace_id)
#define TracyTTMetalReleaseMeshTrace(device_ids, trace_id)
#define TracyTTMetalEnqueueProgramTrace(device_id, trace_id, program_runtime_id)

#endif
