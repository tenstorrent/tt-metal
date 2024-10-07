// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "data_collection.hpp"
#include "llrt/rtoptions.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/common/core_coord.h"

#include "magic_enum.hpp"

using namespace tt;

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
    void Update(DispatchStats &other) {
        Update(other.max_transaction_size, other.min_transaction_size, other.num_writes, other.total_write_size);
    }

    void Dump(std::ofstream &outfile,  map<uint32_t, uint32_t> &raw_data) {
        outfile << fmt::format("\t\tmax_transaction_size = {}\n", max_transaction_size);
        outfile << fmt::format("\t\tmin_transaction_size = {}\n", min_transaction_size);
        outfile << fmt::format("\t\tnum_writes           = {}\n", num_writes);
        outfile << fmt::format("\t\ttotal_write_size     = {}\n", total_write_size);
        outfile << "\t\ttransaction_counts   = [";
        for (auto &size_and_count : raw_data) {
            outfile << size_and_count.first << ":" << size_and_count.second << " ";
        }
        outfile << "]\n";
    }
};

// Class to hold dispatch write data for the DataCollector
class DispatchData {
public:
    DispatchData(data_collector_t type): type(type) {}
    DispatchData(int type_int) : DispatchData(static_cast<data_collector_t>(type_int)) {}

    void Update(uint32_t transaction_size, RISCV riscv) {
        data[riscv][transaction_size]++;
    }

    void Merge(const DispatchData &other) {
        for (auto &riscv_and_data : other.data) {
            for (auto &size_and_count : riscv_and_data.second) {
                this->data[riscv_and_data.first][size_and_count.first] += size_and_count.second;
            }
        }
    }

    void DumpStats(std::ofstream &outfile) {
        // Only dump if this has data
        if (data.size() == 0)
            return;
        outfile << fmt::format("\t{} stats:\n", type);

        // Track stats for all RISCS, as well as per RISC
        DispatchStats total_stats;
        map<uint32_t, uint32_t> total_data;
        for (auto &riscv_and_data : data) {
            // Go through all data and update stats
            DispatchStats riscv_stats;
            for (auto &size_and_count : riscv_and_data.second) {
                riscv_stats.Update(size_and_count.first, size_and_count.second);
                total_data[size_and_count.first] += size_and_count.second;
            }
            total_stats.Update(riscv_stats);

            // Only for binaries, print for each RISC type
            if (type == DISPATCH_DATA_BINARY) {
                outfile << "\t  " << riscv_and_data.first << " binary data:\n";
                riscv_stats.Dump(outfile, riscv_and_data.second);
            }
        }

        // For types other than binaries, just print once
        if (type == DISPATCH_DATA_BINARY)
            outfile << "\t  Overall binaries data:\n";
        total_stats.Dump(outfile, total_data);
    }

private:
    map<RISCV, map<uint32_t, uint32_t>> data; // RISCV -> transaction size -> count
    data_collector_t type;
};

// Class to manage & dump dispatch data for each program
class DataCollector {
public:
    // Single instance of the data collector
    static DataCollector *inst;

    DataCollector() {
        TT_ASSERT(inst == nullptr);
        inst = this;
    };
    ~DataCollector() {
        inst = nullptr;
    };

    void RecordData(Program &program, data_collector_t type, uint32_t transaction_size, RISCV riscv);
    void RecordKernelGroups(Program &program, CoreType core_type, vector<KernelGroup> &kernel_groups);
    void RecordProgramRun(Program &program);
    void DumpData();

private:
    map<uint64_t, vector<DispatchData>> program_id_to_dispatch_data;
    map<uint64_t, map<CoreType, vector<pair<kernel_id_array_t, CoreRangeSet>>>> program_id_to_kernel_groups;
    map<uint64_t, int> program_id_to_call_count;
};

void DataCollector::RecordData(Program &program, data_collector_t type, uint32_t transaction_size, RISCV riscv) {
    uint64_t program_id = program.get_id();
    if (program_id_to_dispatch_data.count(program_id) == 0) {
        // If no existing data for this program, initialize starting values.
        program_id_to_dispatch_data[program_id] = vector<DispatchData>();
        for (int idx = 0; idx < DISPATCH_DATA_COUNT; idx++) {
            data_collector_t curr_type = static_cast<data_collector_t>(idx);
            DispatchData data(curr_type);
            program_id_to_dispatch_data[program_id].push_back(data);
        }
    }

    program_id_to_dispatch_data[program_id].at(type).Update(transaction_size, riscv);
}

void DataCollector::RecordKernelGroups(Program &program, CoreType core_type, vector<KernelGroup> &kernel_groups) {
    uint64_t program_id = program.get_id();
    // Make a copy of relevant info, since user may destroy program before we dump.
    for (KernelGroup &kernel_group : kernel_groups) {
        kernel_id_array_t watcher_kernel_ids;
        for (int idx = 0; idx < kernel_group.kernel_ids.size(); idx++) {
            if (kernel_group.kernel_ids[idx]) {
                watcher_kernel_ids[idx] = program.get_kernel(*kernel_group.kernel_ids[idx])->get_watcher_kernel_id();
            }
        }
        program_id_to_kernel_groups[program_id][core_type].push_back({watcher_kernel_ids, kernel_group.core_ranges});
    }
}

void DataCollector::RecordProgramRun(Program &program) {
    uint64_t program_id = program.get_id();
    program_id_to_call_count[program_id]++;
}

string DispatchClassToString(enum dispatch_core_processor_classes proc_class, CoreType core_type) {
    switch (core_type) {
        case CoreType::WORKER:
            switch (proc_class) {
                case DISPATCH_CLASS_TENSIX_DM0:
                    return "brisc:";
                case DISPATCH_CLASS_TENSIX_DM1:
                    return "ncrisc:";
                case DISPATCH_CLASS_TENSIX_COMPUTE:
                    return "trisc:";
                default:
                    return "";
            }
        case CoreType::ETH:
            if (proc_class == DISPATCH_CLASS_ETH_DM0)
                return "erisc:";
            else
                return "";
        default:
            TT_THROW("Incompatible core type: {}", magic_enum::enum_name(core_type));
    }
    return "";
}

void DataCollector::DumpData() {
    std::ofstream outfile = std::ofstream("dispatch_data.txt");

    // Extra DispatchData objects to collect data across programs
    vector<DispatchData *> cross_program_data;
    for (int idx = 0; idx < DISPATCH_DATA_COUNT; idx++) {
        cross_program_data.push_back(new DispatchData(idx));
    }

    // Go through all programs, and dump relevant data
    for (auto &id_and_data : program_id_to_dispatch_data) {
        uint64_t program_id = id_and_data.first;
        outfile << fmt::format("Program {}: Ran {} time(s).\n", program_id, program_id_to_call_count[program_id]);

        // Dump kernel ids for each kernel group in this program
        for (auto &core_type_and_kernel_groups : program_id_to_kernel_groups[program_id]) {
            CoreType core_type = core_type_and_kernel_groups.first;
            vector<pair<kernel_id_array_t, CoreRangeSet>> &kernel_groups = core_type_and_kernel_groups.second;
            outfile << fmt::format("\t{} Kernel Groups: {}\n", core_type, kernel_groups.size());
            for (auto &ids_and_ranges : kernel_groups) {
                // Dump kernel ids in this group
                outfile << "\t\t{";
                for (int i = 0; i < DISPATCH_CLASS_MAX; i++) {
                    outfile << DispatchClassToString(static_cast<enum dispatch_core_processor_classes>(i), core_type);
                    if (ids_and_ranges.first[i]) {
                        outfile << *ids_and_ranges.first[i];
                    }
                    outfile << " ";
                }
                outfile << "} on cores ";

                // Dump the cores this kernel group contains
                outfile << ids_and_ranges.second.str() << "\n";
            }
        }

        // Dump dispatch write stats
        for (int type_int = 0; type_int != DISPATCH_DATA_COUNT; type_int++) {
            DispatchData &data = id_and_data.second.at(type_int);
            cross_program_data[type_int]->Merge(data);
            data.DumpStats(outfile);
        }
    }

    // Dump cross-program stats
    outfile << "Cross-Program Data:\n";
    for (int type_int = 0; type_int != DISPATCH_DATA_COUNT; type_int++) {
        cross_program_data[type_int]->DumpStats(outfile);
        delete cross_program_data[type_int];
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

} // end anon namespae

namespace tt {

void RecordDispatchData(Program &program, data_collector_t type, uint32_t transaction_size, RISCV riscv) {
    // Do nothing if we're not enabling data collection.
    if (!tt::llrt::OptionsG.get_dispatch_data_collection_enabled())
        return;

    InitDataCollector();
    DataCollector::inst->RecordData(program, type, transaction_size, riscv);
}

void RecordKernelGroups(Program &program, CoreType core_type, vector<KernelGroup> &kernel_groups) {
    // Do nothing if we're not enabling data collection.
    if (!tt::llrt::OptionsG.get_dispatch_data_collection_enabled())
        return;

    InitDataCollector();
    DataCollector::inst->RecordKernelGroups(program, core_type, kernel_groups);
}

void RecordProgramRun(Program &program) {
    // Do nothing if we're not enabling data collection.
    if (!tt::llrt::OptionsG.get_dispatch_data_collection_enabled())
        return;

    InitDataCollector();
    DataCollector::inst->RecordProgramRun(program);
}

} // end namepsace tt
