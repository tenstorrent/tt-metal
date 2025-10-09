// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt_metal_profiler.hpp>
#include "impl/context/metal_context.hpp"

#if defined(TRACY_ENABLE)

#define TracyTTMetalTraceTrackingEnabled() \
    tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_trace_tracking()

#define TracyTTMetalBeginTrace(device_id, trace_id)                                                     \
    if (TracyTTMetalTraceTrackingEnabled()) {                                                           \
        std::string trace_message = fmt::format("`TT_METAL_TRACE_BEGIN: {}, {}`", device_id, trace_id); \
        TracyMessage(trace_message.c_str(), trace_message.size());                                      \
    }

#define TracyTTMetalEndTrace(device_id, trace_id)                                                     \
    if (TracyTTMetalTraceTrackingEnabled()) {                                                         \
        std::string trace_message = fmt::format("`TT_METAL_TRACE_END: {}, {}`", device_id, trace_id); \
        TracyMessage(trace_message.c_str(), trace_message.size());                                    \
    }

#define TracyTTMetalReplayTrace(device_id, trace_id)                                                     \
    if (TracyTTMetalTraceTrackingEnabled()) {                                                            \
        std::string trace_message = fmt::format("`TT_METAL_TRACE_REPLAY: {}, {}`", device_id, trace_id); \
        TracyMessage(trace_message.c_str(), trace_message.size());                                       \
    }

#define TracyTTMetalReleaseTrace(device_id, trace_id)                                                     \
    if (TracyTTMetalTraceTrackingEnabled()) {                                                             \
        std::string trace_message = fmt::format("`TT_METAL_TRACE_RELEASE: {}, {}`", device_id, trace_id); \
        TracyMessage(trace_message.c_str(), trace_message.size());                                        \
    }

#define TracyTTMetalBeginMeshTrace(device_ids, trace_id)                                                    \
    if (TracyTTMetalTraceTrackingEnabled()) {                                                               \
        for (auto device_id : (device_ids)) {                                                               \
            std::string trace_message = fmt::format("`TT_METAL_TRACE_BEGIN: {}, {}`", device_id, trace_id); \
            TracyMessage(trace_message.c_str(), trace_message.size());                                      \
        }                                                                                                   \
    }

#define TracyTTMetalEndMeshTrace(device_ids, trace_id)                                                    \
    if (TracyTTMetalTraceTrackingEnabled()) {                                                             \
        for (auto device_id : (device_ids)) {                                                             \
            std::string trace_message = fmt::format("`TT_METAL_TRACE_END: {}, {}`", device_id, trace_id); \
            TracyMessage(trace_message.c_str(), trace_message.size());                                    \
        }                                                                                                 \
    }

#define TracyTTMetalReplayMeshTrace(device_ids, trace_id)                                                    \
    if (TracyTTMetalTraceTrackingEnabled()) {                                                                \
        for (auto device_id : (device_ids)) {                                                                \
            std::string trace_message = fmt::format("`TT_METAL_TRACE_REPLAY: {}, {}`", device_id, trace_id); \
            TracyMessage(trace_message.c_str(), trace_message.size());                                       \
        }                                                                                                    \
    }

#define TracyTTMetalReleaseMeshTrace(device_ids, trace_id)                                                    \
    if (TracyTTMetalTraceTrackingEnabled()) {                                                                 \
        for (auto device_id : (device_ids)) {                                                                 \
            std::string trace_message = fmt::format("`TT_METAL_TRACE_RELEASE: {}, {}`", device_id, trace_id); \
            TracyMessage(trace_message.c_str(), trace_message.size());                                        \
        }                                                                                                     \
    }

#else

#define TracyTTMetalBeginTrace(device_id, trace_id)
#define TracyTTMetalEndTrace(device_id, trace_id)
#define TracyTTMetalReplayTrace(device_id, trace_id)
#define TracyTTMetalReleaseTrace(device_id, trace_id)

#define TracyTTMetalBeginMeshTrace(device_ids, trace_id)
#define TracyTTMetalEndMeshTrace(device_ids, trace_id)
#define TracyTTMetalReplayMeshTrace(device_ids, trace_id)
#define TracyTTMetalReleaseMeshTrace(device_ids, trace_id)

#endif
