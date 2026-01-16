// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
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
    // Register a callback to be invoked when program perf telemetry data arrives.
    // Returns a handle that can be used to unregister the callback.
    tt::ProgramPerfCallbackHandle RegisterProgramPerfCallback(tt::ProgramPerfCallback callback);
    // Unregister a previously registered callback by its handle.
    void UnregisterProgramPerfCallback(tt::ProgramPerfCallbackHandle handle);
    // Invoke all registered callbacks with the given record.
    void InvokeProgramPerfCallbacks(const tt::ProgramPerfRecord& record);
    void DumpData();
    void DumpKernelSourceMap();

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
    // runtime_id -> list of kernel source paths for that program
    std::map<uint64_t, std::vector<std::string>> runtime_id_to_kernel_sources;
    // Registered program perf callbacks (called from telemetry receiver thread)
    std::mutex program_perf_callbacks_mutex_;
    std::vector<std::pair<tt::ProgramPerfCallbackHandle, tt::ProgramPerfCallback>> program_perf_callbacks_;
    tt::ProgramPerfCallbackHandle next_callback_handle_{0};
};

}  // namespace tt::tt_metal
