// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
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

class MetalEnvImpl;

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

// What the collector records about a program when it is dispatched, for a profiler to attach to the program's
// records.
struct ProgramMetadata {
    std::vector<std::string_view> kernel_sources;
    auto operator<=>(const ProgramMetadata&) const = default;
};

// Class to manage & dump dispatch data for each program
class DataCollector {
public:
    explicit DataCollector(MetalEnvImpl& env);
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
    // The metadata of the program last dispatched with this runtime id; empty if there was none.
    const ProgramMetadata& GetProgramMetadata(uint16_t runtime_id) const noexcept {
        static const ProgramMetadata none;
        const ProgramMetadata* metadata = runtime_id_to_metadata_[runtime_id].load(std::memory_order_acquire);
        return metadata != nullptr ? *metadata : none;
    }
    // A reader of the per-program metadata recorded during dispatch (RecordProgramMetadata /
    // RecordProgramSubDevice) holds a count here for as long as it reads. The device profiler holds one
    // whenever it is enabled.
    void AddProgramMetadataReader() noexcept { program_metadata_readers_.fetch_add(1, std::memory_order_relaxed); }
    void RemoveProgramMetadataReader() noexcept { program_metadata_readers_.fetch_sub(1, std::memory_order_relaxed); }

    // True while something reads the per-program metadata. Callers on the dispatch path check this
    // instead of recording unconditionally: the recording costs a mutex and a keyed insert per
    // program per device. Lock-free so the check itself stays off the critical path.
    bool IsProgramMetadataRecordingEnabled() const noexcept {
        return program_metadata_readers_.load(std::memory_order_relaxed) != 0;
    }

    void DumpData();

private:
    static constexpr size_t kRuntimeIdSlots = 1u << 16;

    struct KernelData {
        int watcher_kernel_id;
        HalProcessorClassType processor_class;
    };
    struct KernelGroupData {
        std::vector<KernelData> kernels;
        CoreRangeSet core_ranges;
    };

    MetalEnvImpl& env_;
    std::map<uint64_t, std::vector<DispatchData>> program_id_to_dispatch_data;
    std::map<uint64_t, std::map<HalProgrammableCoreType, std::vector<KernelGroupData>>> program_id_to_kernel_groups;
    std::map<uint64_t, int> program_id_to_call_count;
    // Lives for the whole process, so the spans handed to RT-profiler callbacks carry no lifetime caveat.
    struct ProgramMetadataStore {
        // Taken only the first time a program is recorded. Profiler threads read lock-free through
        // runtime_id_to_metadata_.
        std::mutex mutex;
        std::unordered_set<std::string> kernel_source_paths;
        // One entry per distinct list of kernel sources; programs with the same kernels share it.
        std::set<ProgramMetadata> metadata;
    };
    static ProgramMetadataStore& program_metadata_store();
    std::array<std::atomic<const ProgramMetadata*>, kRuntimeIdSlots> runtime_id_to_metadata_{};
    std::map<std::pair<tt::ChipId, uint64_t>, tt::ProgramSubDeviceInfo> runtime_id_to_sub_device;
    mutable std::mutex runtime_id_to_sub_device_mutex_;
    std::atomic<uint32_t> program_metadata_readers_;
};

}  // namespace tt::tt_metal
