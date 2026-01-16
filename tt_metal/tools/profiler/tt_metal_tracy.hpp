// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <tt_metal_profiler.hpp>
#include "impl/context/metal_context.hpp"
#include "distributed/mesh_device_impl.hpp"
#include <tt_metal/impl/profiler/profiler_state.hpp>
#include <tt_metal/impl/profiler/profiler_state_manager.hpp>

#if defined(TRACY_ENABLE)

#define TracyTTMetalTraceTrackingEnabled() \
    tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_trace_tracking()

#define TracyTTMetalBeginMeshTrace(device_ids, trace_id)                                                            \
    if (tt::tt_metal::getDeviceProfilerState()) {                                                                   \
        for (auto device_id : (device_ids)) {                                                                       \
            tt::tt_metal::MetalContext::instance().profiler_state_manager()->mark_trace_begin(device_id, trace_id); \
            if (TracyTTMetalTraceTrackingEnabled()) {                                                               \
                std::string trace_message = fmt::format("`TT_METAL_TRACE_BEGIN: {}, {}`", device_id, trace_id);     \
                TracyMessage(trace_message.c_str(), trace_message.size());                                          \
            }                                                                                                       \
        }                                                                                                           \
    }

#define TracyTTMetalEndMeshTrace(device_ids, trace_id)                                                            \
    if (tt::tt_metal::getDeviceProfilerState()) {                                                                 \
        for (auto device_id : (device_ids)) {                                                                     \
            tt::tt_metal::MetalContext::instance().profiler_state_manager()->mark_trace_end(device_id, trace_id); \
            std::string trace_message = fmt::format("`TT_METAL_TRACE_END: {}, {}`", device_id, trace_id);         \
            TracyMessage(trace_message.c_str(), trace_message.size());                                            \
        }                                                                                                         \
    }

#define TracyTTMetalReplayMeshTrace(device_ids, trace_id)                                                            \
    if (tt::tt_metal::getDeviceProfilerState()) {                                                                    \
        for (auto device_id : (device_ids)) {                                                                        \
            tt::tt_metal::MetalContext::instance().profiler_state_manager()->mark_trace_replay(device_id, trace_id); \
            if (TracyTTMetalTraceTrackingEnabled()) {                                                                \
                std::string trace_message = fmt::format("`TT_METAL_TRACE_REPLAY: {}, {}`", device_id, trace_id);     \
                TracyMessage(trace_message.c_str(), trace_message.size());                                           \
            }                                                                                                        \
        }                                                                                                            \
    }

#define TracyTTMetalReleaseMeshTrace(device_ids, trace_id)                                                        \
    if (tt::tt_metal::getDeviceProfilerState()) {                                                                 \
        for (auto device_id : (device_ids)) {                                                                     \
            if (TracyTTMetalTraceTrackingEnabled()) {                                                             \
                std::string trace_message = fmt::format("`TT_METAL_TRACE_RELEASE: {}, {}`", device_id, trace_id); \
                TracyMessage(trace_message.c_str(), trace_message.size());                                        \
            }                                                                                                     \
        }                                                                                                         \
    }

#define TracyTTMetalEnqueueMeshWorkloadTrace(mesh_device, mesh_workload, trace_id)                            \
    if (tt::tt_metal::getDeviceProfilerState()) {                                                             \
        for (auto& [device_range, program] : (mesh_workload).get_programs()) {                                \
            if ((trace_id).has_value()) {                                                                     \
                for_each_local((mesh_device), device_range, [&](const auto& coord) {                          \
                    auto device = (mesh_device)->impl().get_device(coord);                                    \
                    tt::tt_metal::MetalContext::instance().profiler_state_manager()->add_runtime_id_to_trace( \
                        device->id(), *((trace_id).value()), program.get_runtime_id());                       \
                    if (TracyTTMetalTraceTrackingEnabled()) {                                                 \
                        std::string trace_message = fmt::format(                                              \
                            "`TT_METAL_TRACE_ENQUEUE_PROGRAM: {}, {}, {}`",                                   \
                            device->id(),                                                                     \
                            *((trace_id).value()),                                                            \
                            program.get_runtime_id());                                                        \
                        TracyMessage(trace_message.c_str(), trace_message.size());                            \
                    }                                                                                         \
                });                                                                                           \
            }                                                                                                 \
        }                                                                                                     \
    }

#else

#define TracyTTMetalBeginMeshTrace(device_ids, trace_id)
#define TracyTTMetalEndMeshTrace(device_ids, trace_id)
#define TracyTTMetalReplayMeshTrace(device_ids, trace_id)
#define TracyTTMetalReleaseMeshTrace(device_ids, trace_id)
#define TracyTTMetalEnqueueMeshWorkloadTrace(mesh_device, mesh_workload, trace_id)

#endif
