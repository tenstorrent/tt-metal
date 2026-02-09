// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "data_collector.hpp"
#include <enchantum/enchantum.hpp>
#include <enchantum/generators.hpp>
#include <enchantum/iostream.hpp>
#include <filesystem>
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "tt-metalium/program.hpp"

using tt::tt_metal::detail::ProgramImpl;
namespace tt::tt_metal {

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
        for (const auto& size_and_count : raw_data) {
            outfile << size_and_count.first << ":" << size_and_count.second << " ";
        }
        outfile << "]\n";
    }
};

void DispatchData::Update(uint32_t transaction_size, std::optional<HalProcessorIdentifier> processor) {
    data[processor][transaction_size]++;
}

void DispatchData::Merge(const DispatchData& other) {
    for (const auto& [processor, processor_data] : other.data) {
        for (const auto& [size, count] : processor_data) {
            this->data[processor][size] += count;
        }
    }
}

void DispatchData::DumpStats(std::ofstream& outfile) const {
    // Only dump if this has data
    if (data.empty()) {
        return;
    }
    outfile << fmt::format("\t{} stats:\n", type);

    // Track stats for all RISCS, as well as per RISC
    DispatchStats total_stats;
    std::map<uint32_t, uint32_t> total_data;
    for (const auto& [processor, processor_data] : data) {
        // Go through all data and update stats
        DispatchStats processor_stats;
        for (const auto& [size, count] : processor_data) {
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

// Class to hold dispatch write data for the DataCollector
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

void DataCollector::RecordKernelGroup(
    ProgramImpl& program, HalProgrammableCoreType core_type, const KernelGroup& kernel_group) {
    uint64_t program_id = program.get_id();
    // Make a copy of relevant info, since user may destroy program before we dump.
    std::vector<KernelData> kernel_data;
    kernel_data.reserve(kernel_group.kernel_ids.size());
    for (auto kernel_id : kernel_group.kernel_ids) {
        auto kernel = program.get_kernel(kernel_id);
        kernel_data.push_back({kernel->get_watcher_kernel_id(), kernel->get_kernel_processor_class()});
    }
    program_id_to_kernel_groups[program_id][core_type].push_back({std::move(kernel_data), kernel_group.core_ranges});
}

void DataCollector::RecordProgramRun(uint64_t program_id) { program_id_to_call_count[program_id]++; }

void DataCollector::RecordKernelSourceMap(ProgramImpl& program) {
    uint64_t runtime_id = program.get_runtime_id();
    // Only record once per runtime_id
    if (runtime_id_to_kernel_sources.count(runtime_id)) {
        return;
    }
    auto& hal = MetalContext::instance().hal();
    std::vector<std::string> sources;
    for (uint32_t i = 0; i < hal.get_programmable_core_type_count(); i++) {
        for (const auto& [handle, kernel] : program.get_kernels(i)) {
            sources.push_back(kernel->kernel_source().source_);
        }
    }
    runtime_id_to_kernel_sources[runtime_id] = std::move(sources);
}

std::string DataCollector::GetKernelSourcesForRuntimeId(uint64_t runtime_id) const {
    auto it = runtime_id_to_kernel_sources.find(runtime_id);
    if (it == runtime_id_to_kernel_sources.end()) {
        return "";
    }
    std::string result;
    for (size_t i = 0; i < it->second.size(); i++) {
        if (i > 0) {
            result += ",\n";
        }
        result += it->second[i];
    }
    return result;
}

std::vector<std::string> DataCollector::GetKernelSourcesVecForRuntimeId(uint64_t runtime_id) const {
    auto it = runtime_id_to_kernel_sources.find(runtime_id);
    if (it == runtime_id_to_kernel_sources.end()) {
        return {};
    }
    return it->second;
}

tt::ProgramRealtimeProfilerCallbackHandle DataCollector::RegisterProgramRealtimeProfilerCallback(
    tt::ProgramRealtimeProfilerCallback callback) {
    std::lock_guard<std::mutex> lock(program_realtime_profiler_callbacks_mutex_);
    auto handle = next_callback_handle_++;
    program_realtime_profiler_callbacks_.emplace_back(handle, std::move(callback));
    return handle;
}

void DataCollector::UnregisterProgramRealtimeProfilerCallback(tt::ProgramRealtimeProfilerCallbackHandle handle) {
    std::lock_guard<std::mutex> lock(program_realtime_profiler_callbacks_mutex_);
    program_realtime_profiler_callbacks_.erase(
        std::remove_if(
            program_realtime_profiler_callbacks_.begin(),
            program_realtime_profiler_callbacks_.end(),
            [handle](const auto& entry) { return entry.first == handle; }),
        program_realtime_profiler_callbacks_.end());
}

void DataCollector::InvokeProgramRealtimeProfilerCallbacks(const tt::ProgramRealtimeRecord& record) {
    std::lock_guard<std::mutex> lock(program_realtime_profiler_callbacks_mutex_);
    for (const auto& [handle, callback] : program_realtime_profiler_callbacks_) {
        callback(record);
    }
}

void DataCollector::DumpData() {
    if (program_id_to_dispatch_data.empty() && program_id_to_kernel_groups.empty() &&
        program_id_to_call_count.empty()) {
        return;
    }
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
            for (const auto& [kernels, ranges] : kernel_groups) {
                // Dump kernel ids in this group
                outfile << "\t\t{";
                for (const auto& kernel : kernels) {
                    using enchantum::iostream_operators::operator<<;
                    outfile << core_type << "_" << kernel.processor_class << ":" << kernel.watcher_kernel_id << " ";
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
    log_info(tt::LogMetal, "Dispatch data dumped to {}", std::filesystem::absolute("dispatch_data.txt").string());
}

}  // namespace tt::tt_metal
