// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tracy_debug_zones.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "distributed/mesh_device_impl.hpp"
#include <tt_metal/impl/profiler/profiler_state.hpp>
#include <tt_metal/impl/profiler/profiler_state_manager.hpp>

#if defined(TRACY_ENABLE)

// These macros act on the env/context of the MeshDeviceImpl they are given, so each mesh device consults its own
// profiler state rather than the default context's.
#define TracyTTMetalTraceTrackingEnabled(mesh_impl) \
    (mesh_impl).metal_env().get_rtoptions().get_profiler_trace_tracking()

#define TracyTTMetalBeginMeshTrace(mesh_impl, device_ids, trace_id)                                             \
    if (tt::tt_metal::getDeviceProfilerState((mesh_impl).metal_env())) {                                        \
        for (auto device_id : (device_ids)) {                                                                   \
            (mesh_impl).metal_context().profiler_state_manager()->mark_trace_begin(device_id, trace_id);        \
            if (TracyTTMetalTraceTrackingEnabled(mesh_impl)) {                                                  \
                std::string trace_message = fmt::format("`TT_METAL_TRACE_BEGIN: {}, {}`", device_id, trace_id); \
                TracyMessage(trace_message.c_str(), trace_message.size());                                      \
            }                                                                                                   \
        }                                                                                                       \
    }

#define TracyTTMetalEndMeshTrace(mesh_impl, device_ids, trace_id)                                         \
    if (tt::tt_metal::getDeviceProfilerState((mesh_impl).metal_env())) {                                  \
        for (auto device_id : (device_ids)) {                                                             \
            (mesh_impl).metal_context().profiler_state_manager()->mark_trace_end(device_id, trace_id);    \
            std::string trace_message = fmt::format("`TT_METAL_TRACE_END: {}, {}`", device_id, trace_id); \
            TracyMessage(trace_message.c_str(), trace_message.size());                                    \
        }                                                                                                 \
    }

#define TracyTTMetalReplayMeshTrace(mesh_impl, device_ids, trace_id)                                             \
    if (tt::tt_metal::getDeviceProfilerState((mesh_impl).metal_env())) {                                         \
        for (auto device_id : (device_ids)) {                                                                    \
            (mesh_impl).metal_context().profiler_state_manager()->mark_trace_replay(device_id, trace_id);        \
            if (TracyTTMetalTraceTrackingEnabled(mesh_impl)) {                                                   \
                std::string trace_message = fmt::format("`TT_METAL_TRACE_REPLAY: {}, {}`", device_id, trace_id); \
                TracyMessage(trace_message.c_str(), trace_message.size());                                       \
            }                                                                                                    \
        }                                                                                                        \
    }

#define TracyTTMetalReleaseMeshTrace(mesh_impl, device_ids, trace_id)                                             \
    if (tt::tt_metal::getDeviceProfilerState((mesh_impl).metal_env())) {                                          \
        for (auto device_id : (device_ids)) {                                                                     \
            if (TracyTTMetalTraceTrackingEnabled(mesh_impl)) {                                                    \
                std::string trace_message = fmt::format("`TT_METAL_TRACE_RELEASE: {}, {}`", device_id, trace_id); \
                TracyMessage(trace_message.c_str(), trace_message.size());                                        \
            }                                                                                                     \
        }                                                                                                         \
    }

#define TracyTTMetalEnqueueMeshWorkloadTrace(mesh_device, mesh_workload, trace_id)                                \
    if (tt::tt_metal::getDeviceProfilerState((mesh_device)->impl().metal_env())) {                                \
        for (auto& [device_range, program] : (mesh_workload).get_programs()) {                                    \
            if ((trace_id).has_value()) {                                                                         \
                for_each_local((mesh_device), device_range, [&](const auto& coord) {                              \
                    auto device = (mesh_device)->impl().get_device(coord);                                        \
                    (mesh_device)                                                                                 \
                        ->impl()                                                                                  \
                        .metal_context()                                                                          \
                        .profiler_state_manager()                                                                 \
                        ->add_runtime_id_to_trace(device->id(), *((trace_id).value()), program.get_runtime_id()); \
                    if (TracyTTMetalTraceTrackingEnabled((mesh_device)->impl())) {                                \
                        std::string trace_message = fmt::format(                                                  \
                            "`TT_METAL_TRACE_ENQUEUE_PROGRAM: {}, {}, {}`",                                       \
                            device->id(),                                                                         \
                            *((trace_id).value()),                                                                \
                            program.get_runtime_id());                                                            \
                        TracyMessage(trace_message.c_str(), trace_message.size());                                \
                    }                                                                                             \
                });                                                                                               \
            }                                                                                                     \
        }                                                                                                         \
    }

#else

#define TracyTTMetalBeginMeshTrace(mesh_impl, device_ids, trace_id)
#define TracyTTMetalEndMeshTrace(mesh_impl, device_ids, trace_id)
#define TracyTTMetalReplayMeshTrace(mesh_impl, device_ids, trace_id)
#define TracyTTMetalReleaseMeshTrace(mesh_impl, device_ids, trace_id)
#define TracyTTMetalEnqueueMeshWorkloadTrace(mesh_device, mesh_workload, trace_id)

#endif
