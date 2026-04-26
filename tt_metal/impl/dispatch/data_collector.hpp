// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>
#include <optional>
#include <fstream>
#include "data_collection.hpp"

namespace tt::tt_metal {

// Class to hold dispatch write data for the DataCollector
class DispatchData {
public:
    DispatchData(data_collector_t type) : type(type) {}

    void Update(uint32_t transaction_size, std::optional<HalProcessorIdentifier> processor);

    void Merge(const DispatchData& other);

    void DumpStats(std::ofstream& outfile) const;

private:
    // processor -> transaction size -> count
    std::map<std::optional<HalProcessorIdentifier>, std::map<uint32_t, uint32_t>> data;
    data_collector_t type;
};

// Class to manage & dump dispatch data for each program
class DataCollector {
public:
    DataCollector() = default;
    ~DataCollector() = default;

    void RecordData(
        uint64_t program_id,
        data_collector_t type,
        uint32_t transaction_size,
        std::optional<tt_metal::HalProcessorIdentifier> processor);
    void RecordKernelGroup(
        tt_metal::detail::ProgramImpl& program,
        tt_metal::HalProgrammableCoreType core_type,
        const tt_metal::KernelGroup& kernel_group);
    void RecordProgramRun(uint64_t program_id);
    // Record the mapping from runtime_id to kernel source paths for a program.
    // Should be called at dispatch time when runtime_id is guaranteed to be set.
    // Only records the mapping once per runtime_id.
    void RecordKernelSourceMap(tt_metal::detail::ProgramImpl& program);
    // Look up the kernel source paths for a given runtime_id.
    // Returns a comma-separated string of kernel source paths, or empty string if not found.
    std::string GetKernelSourcesForRuntimeId(uint64_t runtime_id) const;
    // Look up the kernel source paths for a given runtime_id as a vector.
    std::vector<std::string> GetKernelSourcesVecForRuntimeId(uint64_t runtime_id) const;
    // Register a callback to be invoked when real-time profiler data arrives.
    // Returns a handle that can be used to unregister the callback.
    tt::ProgramRealtimeProfilerCallbackHandle RegisterProgramRealtimeProfilerCallback(
        tt::ProgramRealtimeProfilerCallback callback);
    // Unregister a previously registered callback by its handle.
    void UnregisterProgramRealtimeProfilerCallback(tt::ProgramRealtimeProfilerCallbackHandle handle);
    // Invoke all registered callbacks with the given record.
    void InvokeProgramRealtimeProfilerCallbacks(const tt::ProgramRealtimeRecord& record);

    // Real-time profiler liveness tracking.
    // MeshDevice calls NotifyRealtimeProfilerActivated(chip_id) after it successfully
    // finishes the init+sync handshake for a device, and NotifyRealtimeProfilerDeactivated
    // at close. IsRealtimeProfilerActive() returns true while at least one chip is
    // active, letting callers (e.g. tests) tell the difference between "RT profiler
    // produced no records yet" and "RT profiler isn't running on this configuration at
    // all" (the latter happens under ETH dispatch, where the profiler is tensix-only
    // and silently bows out).
    void NotifyRealtimeProfilerActivated(uint32_t chip_id);
    void NotifyRealtimeProfilerDeactivated(uint32_t chip_id);
    bool IsRealtimeProfilerActive() const;

    void DumpData();

private:
    struct KernelData {
        int watcher_kernel_id;
        HalProcessorClassType processor_class;
    };
    struct KernelGroupData {
        std::vector<KernelData> kernels;
        CoreRangeSet core_ranges;
    };
    std::map<uint64_t, std::vector<DispatchData>> program_id_to_dispatch_data;
    std::map<uint64_t, std::map<HalProgrammableCoreType, std::vector<KernelGroupData>>> program_id_to_kernel_groups;
    std::map<uint64_t, int> program_id_to_call_count;
    // runtime_id -> list of kernel source paths for that program.
    // Guarded by runtime_id_to_kernel_sources_mutex_ because RecordKernelSourceMap is
    // called from the main (dispatch) thread while GetKernelSources*ForRuntimeId is
    // called from the RealtimeProfiler receiver thread.
    std::map<uint64_t, std::vector<std::string>> runtime_id_to_kernel_sources;
    mutable std::mutex runtime_id_to_kernel_sources_mutex_;
    // Registered real-time profiler callbacks (called from receiver thread).
    // mutable because IsRealtimeProfilerActive() is a logically-const query that still
    // needs to lock the mutex to safely read realtime_profiler_active_chips_.
    mutable std::mutex program_realtime_profiler_callbacks_mutex_;
    std::vector<std::pair<tt::ProgramRealtimeProfilerCallbackHandle, tt::ProgramRealtimeProfilerCallback>>
        program_realtime_profiler_callbacks_;
    tt::ProgramRealtimeProfilerCallbackHandle next_callback_handle_{0};

    // Set of chip_ids whose RT profiler is currently live. Guarded by the same mutex as
    // the callback list because both are accessed from the receiver thread and the
    // open/close paths.
    std::unordered_set<uint32_t> realtime_profiler_active_chips_;
};

}  // namespace tt::tt_metal
