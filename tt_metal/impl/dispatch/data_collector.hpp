// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>
#include <fstream>
#include <tt-metalium/sub_device_types.hpp>
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
    void RecordProgramMetadata(tt_metal::detail::ProgramImpl& program);
    void RecordProgramSubDevice(
        tt::ChipId device_id,
        uint64_t sub_device_manager_id,
        uint64_t runtime_id,
        SubDeviceId sub_device_id,
        uint32_t num_available_worker_cores = 0);
    std::optional<tt::ProgramSubDeviceInfo> GetProgramSubDevice(tt::ChipId device_id, uint64_t runtime_id) const;
    // Look up kernel source paths by runtime_id; empty span if the runtime_id is unknown.
    // The returned span is valid until MetalContext teardown or reinitialization.
    std::span<const std::string_view> GetKernelSourcesForRuntimeId(uint16_t runtime_id) const noexcept {
        const auto* sources = runtime_id_to_kernel_sources_[runtime_id].load(std::memory_order_acquire);
        return sources != nullptr ? std::span<const std::string_view>(*sources) : std::span<const std::string_view>{};
    }
    // Register a callback to be invoked when real-time profiler data arrives.
    // Returns a handle that can be used to unregister the callback.
    tt::ProgramRealtimeProfilerCallbackHandle RegisterProgramRealtimeProfilerCallback(
        tt::ProgramRealtimeProfilerCallback callback);
    // Unregister a previously registered callback by its handle.
    void UnregisterProgramRealtimeProfilerCallback(tt::ProgramRealtimeProfilerCallbackHandle handle);
    void AttachRealtimeProfilerCallbackListener(tt::RealtimeProfilerCallbackListener* listener);
    void DetachRealtimeProfilerCallbackListener(tt::RealtimeProfilerCallbackListener* listener);

    // Real-time profiler liveness tracking. MeshDevice notifies activation after a
    // successful init+sync handshake and deactivation at close; IsRealtimeProfilerActive()
    // returns true while at least one chip is active.
    void NotifyRealtimeProfilerActivated(uint32_t chip_id);
    void NotifyRealtimeProfilerDeactivated(uint32_t chip_id);
    bool IsRealtimeProfilerActive() const;

    void DumpData();

private:
    static constexpr size_t kRuntimeIdSlots = 1u << 16;

    struct RealtimeCallbackRegistration {
        tt::ProgramRealtimeProfilerCallbackHandle handle;
        tt::ProgramRealtimeProfilerCallback callback;
    };

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
    // Kernel source bookkeeping for the real-time profiler. Guarded because the dispatch thread writes and
    // the real-time profiler receiver thread reads.
    mutable std::mutex kernel_source_mutex_;
    std::unordered_set<std::string> unique_kernel_sources_;
    std::unordered_map<uint64_t, std::vector<std::string_view>> program_id_to_kernel_sources_;
    std::array<std::atomic<const std::vector<std::string_view>*>, kRuntimeIdSlots> runtime_id_to_kernel_sources_{};
    std::map<std::pair<tt::ChipId, uint64_t>, tt::ProgramSubDeviceInfo> runtime_id_to_sub_device;
    mutable std::mutex runtime_id_to_sub_device_mutex_;
    mutable std::mutex program_realtime_profiler_callbacks_mutex_;
    std::vector<RealtimeCallbackRegistration> program_realtime_profiler_callbacks_;
    std::vector<tt::RealtimeProfilerCallbackListener*> realtime_callback_listeners_;
    tt::ProgramRealtimeProfilerCallbackHandle next_callback_handle_{0};

    mutable std::mutex realtime_profiler_active_chips_mutex_;
    std::unordered_set<uint32_t> realtime_profiler_active_chips_;
};

}  // namespace tt::tt_metal
