// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "data_collection.hpp"

#include <core_coord.hpp>
#include <kernel.hpp>
#include <enchantum/enchantum.hpp>
#include <enchantum/generators.hpp>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "assert.hpp"
#include "dev_msgs.h"
#include "tt-metalium/program.hpp"
#include <umd/device/tt_core_coordinates.h>
#include "impl/context/metal_context.hpp"

using namespace tt;
using namespace tt::tt_metal;

using tt::tt_metal::detail::ProgramImpl;
namespace {

// Class to track stats for DispatchData
class DispatchStats {
public:
    uint32_t max_transaction_size = 0;
    uint32_t min_transaction_size = UINT32_MAX;
    uint32_t num_writes = 0;
    uint64_t total_write_size = 0;

    void Update(
        uint32_t max_transaction_size, uint32_t min_transaction_size, uint32_t num_writes, uint64_t total_write_size) {
        this->max_transaction_size = std::max(this->max_transaction_size, max_transaction_size);
        this->min_transaction_size = std::min(this->min_transaction_size, min_transaction_size);
        this->num_writes += num_writes;
        this->total_write_size += total_write_size;
    }
    void Update(uint32_t transaction_size, uint32_t transaction_count) {
        Update(transaction_size, transaction_size, transaction_count, transaction_count * transaction_size);
    }
    void Update(DispatchStats& other) {
        Update(other.max_transaction_size, other.min_transaction_size, other.num_writes, other.total_write_size);
    }

    void Dump(std::ofstream& outfile, const std::map<uint32_t, uint32_t>& raw_data) const {
        outfile << fmt::format("\t\tmax_transaction_size = {}\n", max_transaction_size);
        outfile << fmt::format("\t\tmin_transaction_size = {}\n", min_transaction_size);
        outfile << fmt::format("\t\tnum_writes           = {}\n", num_writes);
        outfile << fmt::format("\t\ttotal_write_size     = {}\n", total_write_size);
        outfile << "\t\ttransaction_counts   = [";
        for (auto& size_and_count : raw_data) {
            outfile << size_and_count.first << ":" << size_and_count.second << " ";
        }
        outfile << "]\n";
    }
};

// Class to hold dispatch write data for the DataCollector
class DispatchData {
public:
    DispatchData(data_collector_t type) : type(type) {}

    void Update(uint32_t transaction_size, std::optional<HalProcessorIdentifier> processor) {
        data[processor][transaction_size]++;
    }

    void Merge(const DispatchData& other) {
        for (auto& [processor, processor_data] : other.data) {
            for (auto& [size, count] : processor_data) {
                this->data[processor][size] += count;
            }
        }
    }

    void DumpStats(std::ofstream& outfile) const {
        // Only dump if this has data
        if (data.size() == 0) {
            return;
        }
        outfile << fmt::format("\t{} stats:\n", type);

        // Track stats for all RISCS, as well as per RISC
        DispatchStats total_stats;
        std::map<uint32_t, uint32_t> total_data;
        for (auto& [processor, processor_data] : data) {
            // Go through all data and update stats
            DispatchStats processor_stats;
            for (auto& [size, count] : processor_data) {
                processor_stats.Update(size, count);
                total_data[size] += count;
            }
            total_stats.Update(processor_stats);

            // Only for binaries, print for each RISC type
            if (type == DISPATCH_DATA_BINARY) {
                TT_ASSERT(processor != std::nullopt);
                outfile << "\t  " << *processor << " binary data:\n";
                processor_stats.Dump(outfile, processor_data);
            }
        }

        // For types other than binaries, just print once
        if (type == DISPATCH_DATA_BINARY) {
            outfile << "\t  Overall binaries data:\n";
        }
        total_stats.Dump(outfile, total_data);
    }

private:
    // processor -> transaction size -> count
    std::map<std::optional<HalProcessorIdentifier>, std::map<uint32_t, uint32_t>> data;
    data_collector_t type;
};

// Class to manage & dump dispatch data for each program
class DataCollector {
public:
    // Single instance of the data collector
    static DataCollector* inst;

    DataCollector() {
        TT_ASSERT(inst == nullptr);
        inst = this;
    };
    ~DataCollector() { inst = nullptr; };

    void RecordData(
        uint64_t program_id,
        data_collector_t type,
        uint32_t transaction_size,
        std::optional<HalProcessorIdentifier> processor);
    void RecordKernelGroups(ProgramImpl& program, CoreType core_type, std::vector<KernelGroup>& kernel_groups);
    void RecordProgramRun(uint64_t program_id);
    void DumpData();

private:
    std::map<uint64_t, std::vector<DispatchData>> program_id_to_dispatch_data;
    std::map<uint64_t, std::map<CoreType, std::vector<std::pair<kernel_id_array_t, CoreRangeSet>>>>
        program_id_to_kernel_groups;
    std::map<uint64_t, int> program_id_to_call_count;
};

void DataCollector::RecordData(
    uint64_t program_id,
    data_collector_t type,
    uint32_t transaction_size,
    std::optional<HalProcessorIdentifier> processor) {
    auto& dispatch_data = program_id_to_dispatch_data[program_id];
    if (dispatch_data.empty()) {
        // If no existing data for this program, initialize starting values.
        dispatch_data.reserve(enchantum::count<data_collector_t>);
        for (auto idx : enchantum::values_generator<data_collector_t>) {
            dispatch_data.emplace_back(idx);
        }
    }
    dispatch_data.at(type).Update(transaction_size, processor);
}

void DataCollector::RecordKernelGroups(
    ProgramImpl& program, CoreType core_type, std::vector<KernelGroup>& kernel_groups) {
    uint64_t program_id = program.get_id();
    // Make a copy of relevant info, since user may destroy program before we dump.
    for (KernelGroup& kernel_group : kernel_groups) {
        kernel_id_array_t watcher_kernel_ids;
        for (int idx = 0; idx < kernel_group.kernel_ids.size(); idx++) {
            if (kernel_group.kernel_ids[idx]) {
                watcher_kernel_ids[idx] = program.get_kernel(*kernel_group.kernel_ids[idx])->get_watcher_kernel_id();
            }
        }
        program_id_to_kernel_groups[program_id][core_type].push_back({watcher_kernel_ids, kernel_group.core_ranges});
    }
}

void DataCollector::RecordProgramRun(uint64_t program_id) {
    program_id_to_call_count[program_id]++;
}

std::string DispatchClassToString(enum dispatch_core_processor_classes proc_class, CoreType core_type) {
    switch (core_type) {
        case CoreType::WORKER:
            switch (proc_class) {
                case DISPATCH_CLASS_TENSIX_DM0: return "brisc:";
                case DISPATCH_CLASS_TENSIX_DM1: return "ncrisc:";
                case DISPATCH_CLASS_TENSIX_COMPUTE: return "trisc:";
                default: return "";
            }
        case CoreType::ETH:
            if (proc_class == DISPATCH_CLASS_ETH_DM0) {
                return "erisc:";
            } else {
                return "";
            }
        default: TT_THROW("Incompatible core type: {}", enchantum::to_string(core_type));
    }
    return "";
}

void DataCollector::DumpData() {
    std::ofstream outfile = std::ofstream("dispatch_data.txt");

    // Extra DispatchData objects to collect data across programs
    std::vector<DispatchData> cross_program_data;
    cross_program_data.reserve(enchantum::count<data_collector_t>);
    for (auto idx : enchantum::values_generator<data_collector_t>) {
        cross_program_data.emplace_back(idx);
    }

    // Go through all programs, and dump relevant data
    for (const auto& [program_id, data] : program_id_to_dispatch_data) {
        outfile << fmt::format("Program {}: Ran {} time(s).\n", program_id, program_id_to_call_count[program_id]);

        // Dump kernel ids for each kernel group in this program
        for (const auto& [core_type, kernel_groups] : program_id_to_kernel_groups[program_id]) {
            outfile << fmt::format("\t{} Kernel Groups: {}\n", core_type, kernel_groups.size());
            for (const auto& [ids, ranges] : kernel_groups) {
                // Dump kernel ids in this group
                outfile << "\t\t{";
                for (int i = 0; i < DISPATCH_CLASS_MAX; i++) {
                    outfile << DispatchClassToString(static_cast<enum dispatch_core_processor_classes>(i), core_type);
                    if (ids[i]) {
                        outfile << *ids[i];
                    }
                    outfile << " ";
                }
                outfile << "} on cores ";

                // Dump the cores this kernel group contains
                outfile << ranges.str() << "\n";
            }
        }

        // Dump dispatch write stats
        for (auto type : enchantum::values_generator<data_collector_t>) {
            const DispatchData& type_data = data.at(type);
            cross_program_data[type].Merge(type_data);
            type_data.DumpStats(outfile);
        }
    }

    // Dump cross-program stats
    outfile << "Cross-Program Data:\n";
    for (const auto& type_data : cross_program_data) {
        type_data.DumpStats(outfile);
    }
    outfile.close();
}

DataCollector* DataCollector::inst = nullptr;

void DumpDispatchDataAndClose() {
    DataCollector::inst->DumpData();
    delete DataCollector::inst;
}

// Helper function to init the data collector if it isn't already up.
void InitDataCollector() {
    if (DataCollector::inst == nullptr) {
        new DataCollector();
        std::atexit(DumpDispatchDataAndClose);
    }
}

}  // namespace

namespace tt {

void RecordDispatchData(
    uint64_t program_id,
    data_collector_t type,
    uint32_t transaction_size,
    std::optional<HalProcessorIdentifier> processor) {
    // Do nothing if we're not enabling data collection.
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_data_collection_enabled()) {
        return;
    }

    InitDataCollector();
    DataCollector::inst->RecordData(program_id, type, transaction_size, processor);
}

void RecordKernelGroups(ProgramImpl& program, CoreType core_type, std::vector<KernelGroup>& kernel_groups) {
    // Do nothing if we're not enabling data collection.
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_data_collection_enabled()) {
        return;
    }

    InitDataCollector();
    DataCollector::inst->RecordKernelGroups(program, core_type, kernel_groups);
}

void RecordProgramRun(uint64_t program_id) {
    // Do nothing if we're not enabling data collection.
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_data_collection_enabled()) {
        return;
    }

    InitDataCollector();
    DataCollector::inst->RecordProgramRun(program_id);
}

}  // namespace tt
