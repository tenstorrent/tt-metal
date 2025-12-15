// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
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
};

}  // namespace tt::tt_metal
