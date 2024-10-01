// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include "core_coord.h"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "noc/noc_parameters.h"

extern bool debug_g;
extern bool use_coherent_data_g;
extern uint32_t dispatch_buffer_page_size_g;
extern uint32_t min_xfer_size_bytes_g;
extern uint32_t max_xfer_size_bytes_g;
extern bool send_to_all_g;
extern bool perf_test_g;
extern uint32_t hugepage_issue_buffer_size_g;

struct one_core_data_t {
    CoreType core_type;
    CoreCoord logical_core;
    CoreCoord phys_core;
    int bank_id;
    int bank_offset;
    vector<bool> valid;
    vector<uint32_t> data;
};

class DeviceData {
 private:
    bool banked;  // TODO banked and unbanked tests still don't play nicely together
    int amt_written;
    // 10 is a hack...bigger than any core_type
    uint64_t base_data_addr[10];
    uint64_t base_result_data_addr[10];
    std::unordered_map<CoreCoord, std::unordered_map<uint32_t, one_core_data_t>> all_data;
    CoreCoord host_core;

    // Validate a single core's worth of results vs expected
    bool validate_one_core(Device *device, std::unordered_set<CoreCoord> &validated_cores,
                           const one_core_data_t& one_core_data,
                           const uint32_t start_index, uint32_t result_addr);
    bool validate_host(std::unordered_set<CoreCoord> &validated_cores,
                       const one_core_data_t& one_core_data);

    void prepopulate_dram(Device *device, uint32_t size_words);

 public:
    DeviceData(Device *device, CoreRange workers,
               uint32_t l1_data_addr, uint32_t dram_data_addr, void * pcie_data_addr,
               bool is_banked, uint32_t dram_data_size_words);

    // Add expected data to a core
    void push_one(CoreCoord core, int bank, uint32_t datum);
    void push_one(CoreCoord core, uint32_t datum);
    void push_range(const CoreRange& cores, uint32_t datum, bool is_mcast);

    // Add invalid data
    void pad(CoreCoord core, int bank, uint32_t alignment);

    // Some tests write to the same address across multiple cores
    // This takes those core types and pads any that are "behind" with invalid data
    void relevel(CoreType core_type);
    void relevel(CoreRange range);

    // Clear data between tests
    void reset();
    uint32_t get_base_result_addr(CoreType core_type);
    uint32_t get_result_data_addr(CoreCoord core, int bank_id = 0);

    bool validate(Device *device);
    void overflow_check(Device *device);

    int size() { return amt_written; }
    int size(CoreCoord core, int bank_id = 0) { return this->all_data[core][bank_id].data.size(); }

    std::unordered_map<CoreCoord, std::unordered_map<uint32_t, one_core_data_t>>& get_data() { return this->all_data; }

    CoreType get_core_type(CoreCoord core) { return this->all_data[core][0].core_type; }
    uint32_t size_at(CoreCoord core, int bank_id);
    uint32_t at(CoreCoord core, int bank_id, uint32_t addr);
    CoreCoord get_host_core() { return this->host_core; }
    bool core_and_bank_present(CoreCoord core, uint32_t bank);
};

DeviceData::DeviceData(Device *device,
                       CoreRange workers,
                       uint32_t l1_data_addr, uint32_t dram_data_addr, void * pcie_data_addr,
                       bool is_banked,
                       uint32_t dram_data_size_words) {

    this->base_data_addr[static_cast<int>(CoreType::WORKER)] = l1_data_addr;
    this->base_data_addr[static_cast<int>(CoreType::PCIE)] = (uint64_t)pcie_data_addr;
    this->base_data_addr[static_cast<int>(CoreType::DRAM)] = dram_data_addr;
    this->base_result_data_addr[static_cast<int>(CoreType::WORKER)] = l1_data_addr;
    this->base_result_data_addr[static_cast<int>(CoreType::PCIE)] = (uint64_t)pcie_data_addr;
    this->base_result_data_addr[static_cast<int>(CoreType::DRAM)] = dram_data_addr;

    this->banked = is_banked;
    this->amt_written = 0;

    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device->id());
    const std::vector<CoreCoord>& pcie_cores = soc_d.get_pcie_cores();
    for (CoreCoord core : pcie_cores) {
        // TODO: make this all work w/ phys coords
        // this is really annoying
        // the PCIE phys core conflicts w/ worker logical cores
        // so we hack the physical core for tracking, blech
        // no simple way to handle this, need to use phys cores
        core = {100,100};
        this->all_data[core][0] = one_core_data_t();
        this->all_data[core][0].logical_core = core;
        this->all_data[core][0].phys_core = core;
        this->all_data[core][0].core_type = CoreType::PCIE;
        this->all_data[core][0].bank_id = 20;
        this->all_data[core][0].bank_offset = 0;
        this->host_core = core;
    }

    // Always populate DRAM
    auto num_banks = device->num_banks(BufferType::DRAM);
    for (int bank_id = 0; bank_id < num_banks; bank_id++) {
        auto dram_channel = device->dram_channel_from_bank_id(bank_id);
        CoreCoord phys_core = device->dram_core_from_dram_channel(dram_channel);
        int32_t bank_offset = device->bank_offset(BufferType::DRAM, bank_id);
        this->all_data[phys_core][bank_id] = one_core_data_t();
        this->all_data[phys_core][bank_id].logical_core = phys_core;
        this->all_data[phys_core][bank_id].phys_core = phys_core;
        this->all_data[phys_core][bank_id].core_type = CoreType::DRAM;
        this->all_data[phys_core][bank_id].bank_id = bank_id;
        this->all_data[phys_core][bank_id].bank_offset = bank_offset;
    }

    // TODO: make banked L1 tests play nicely w/ non-banked L1 tests
    if (is_banked) {
        num_banks = device->num_banks(BufferType::L1);
        for (int bank_id = 0; bank_id < num_banks; bank_id++) {
            CoreCoord core = device->logical_core_from_bank_id(bank_id);
            CoreCoord phys_core = device->worker_core_from_logical_core(core);
            int32_t bank_offset = device->bank_offset(BufferType::L1, bank_id);
            this->all_data[core][bank_id] = one_core_data_t();
            this->all_data[core][bank_id].logical_core = core;
            this->all_data[core][bank_id].phys_core = phys_core;
            this->all_data[core][bank_id].core_type = CoreType::WORKER;
            this->all_data[core][bank_id].bank_id = bank_id;
            this->all_data[core][bank_id].bank_offset = bank_offset;
        }
    } else {
        for (uint32_t y = workers.start_coord.y; y <= workers.end_coord.y; y++) {
            for (uint32_t x = workers.start_coord.x; x <= workers.end_coord.x; x++) {
                CoreCoord core = {x, y};
                CoreCoord phys_core = device->worker_core_from_logical_core(core);
                this->all_data[core][0] = one_core_data_t();
                this->all_data[core][0].logical_core = core;
                this->all_data[core][0].phys_core = phys_core;
                this->all_data[core][0].core_type = CoreType::WORKER;
                this->all_data[core][0].bank_id = 0;
                this->all_data[core][0].bank_offset = 0;
            }
        }
    }

    prepopulate_dram(device, dram_data_size_words);
}

// Populate interleaved DRAM with data for later readback.  Can we extended to L1 if needed.
void DeviceData::prepopulate_dram(Device *device, uint32_t size_words) {

    uint32_t num_dram_banks = device->num_banks(BufferType::DRAM);

    for (int bank_id = 0; bank_id < num_dram_banks; bank_id++) {
        auto offset = device->bank_offset(BufferType::DRAM, bank_id);
        auto dram_channel = device->dram_channel_from_bank_id(bank_id);
        auto bank_core = device->dram_core_from_dram_channel(dram_channel);
        one_core_data_t& data = this->all_data[bank_core][bank_id];

        // Generate random or coherent data per bank of specific size.
        for (uint32_t i = 0; i < size_words; i++) {
            uint32_t datum = (use_coherent_data_g) ? (((bank_id & 0xFF) << 24) | i) : std::rand();

            // Note: don't bump amt_written
            data.data.push_back(datum);
            data.valid.push_back(true);

            if (i < 10) {
                log_debug(tt::LogTest, "{} - bank_id: {:2d} core: {} offset: 0x{:08x} using i: {:2d} datum: 0x{:08x}",
                    __FUNCTION__, bank_id, bank_core.str(), offset, i, datum);
            }
        }

        // Write to device once per bank (appropriate core and offset)
        tt::Cluster::instance().write_core(static_cast<const void*>(&data.data[0]),
            data.data.size() * sizeof(uint32_t), tt_cxy_pair(device->id(), bank_core),
            this->base_data_addr[static_cast<int>(CoreType::DRAM)] + offset);;

        this->base_result_data_addr[static_cast<int>(CoreType::DRAM)] =
            this->base_data_addr[static_cast<int>(CoreType::DRAM)] + data.data.size() * sizeof(uint32_t);
    }
}

bool DeviceData::core_and_bank_present(CoreCoord core, uint32_t bank) {
    if (this->all_data.find(core) != this->all_data.end()) {
        std::unordered_map<uint32_t, one_core_data_t>& core_data = this->all_data.find(core)->second;
        if (core_data.find(bank) != core_data.end()) {
            return true;
        }
    }
    return false;
}

void DeviceData::push_one(CoreCoord core, int bank, uint32_t datum) {
    if (core_and_bank_present(core, bank)) {
        this->amt_written++;
        this->all_data[core][bank].data.push_back(datum);
        this->all_data[core][bank].valid.push_back(true);
    }
}

void DeviceData::push_one(CoreCoord core, uint32_t datum) {
    if (core_and_bank_present(core, 0)) {
        this->amt_written++;
        this->all_data[core][0].data.push_back(datum);
        this->all_data[core][0].valid.push_back(true);
    }
}

void DeviceData::push_range(const CoreRange& cores, uint32_t datum, bool is_mcast) {

    bool counted = false;
    for (auto y = cores.start_coord.y; y <= cores.end_coord.y; y++) {
        for (auto x = cores.start_coord.x; x <= cores.end_coord.x; x++) {
            CoreCoord core = {x, y};
            if (core_and_bank_present(core, 0)) {
                if (not counted || not is_mcast) {
                    this->amt_written++;
                    counted = true;
                }

                TT_ASSERT(this->all_data.find(core) != this->all_data.end());
                this->all_data[core][0].data.push_back(datum);
                this->all_data[core][0].valid.push_back(true);
            }
        }
    }
}

inline uint32_t padded_size(uint32_t size, uint32_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

void DeviceData::pad(CoreCoord core, int bank, uint32_t alignment) {
    if (core_and_bank_present(core, bank)) {
        uint32_t padded = padded_size(this->all_data[core][bank].data.size(), alignment / sizeof(uint32_t));
        this->all_data[core][bank].data.resize(padded);
        this->all_data[core][bank].valid.resize(padded); // pushes false
    }
}

// Some tests write to the same address across multiple cores
// This takes cores that match core_type and pads any that are "behind" with invalid data
void DeviceData::relevel(CoreType core_type) {
    size_t max = 0;
    for (auto & [coord, bank_device_data] : this->all_data) {
        for (auto & [bank, one_core_data] : bank_device_data) {
            if (one_core_data.core_type == core_type && one_core_data.data.size() > max) {
                max = one_core_data.data.size();
            }
        }
    }
    for (auto & [coord, bank_device_data] : this->all_data) {
        for (auto & [bank, one_core_data] : bank_device_data) {
            if (one_core_data.core_type == core_type) {
                one_core_data.data.resize(max);
                one_core_data.valid.resize(max); // fills with false
            }
        }
    }
}

void DeviceData::relevel(CoreRange range) {
    size_t max = 0;

    constexpr uint32_t bank = 0;
    for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; y++) {
        for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; x++) {
            CoreCoord core = {x, y};
            if (this->all_data[core][bank].data.size() > max) {
                max = this->all_data[core][bank].data.size();
            }
        }
    }

    for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; y++) {
        for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; x++) {
            CoreCoord core = {x, y};
            this->all_data[core][bank].data.resize(max);
            this->all_data[core][bank].valid.resize(max);
        }
    }
}

// Result expected results
void DeviceData::reset() {
    this->amt_written = 0;
    for (auto& [coord, bank_device_data] : this->all_data) {
        for (auto & [bank, one_core_data] : bank_device_data) {
            CoreType core_type = one_core_data.core_type;
            uint32_t default_size_bytes = this->base_result_data_addr[static_cast<int>(core_type)] - this->base_data_addr[static_cast<int>(core_type)];
            one_core_data.valid.resize(default_size_bytes / sizeof(uint32_t));
            one_core_data.data.resize(default_size_bytes / sizeof(uint32_t));
        }
    }
}

uint32_t DeviceData::get_base_result_addr(CoreType core_type) {
    return this->base_result_data_addr[static_cast<int>(core_type)];
}

uint32_t DeviceData::get_result_data_addr(CoreCoord core, int bank_id) {
    uint32_t base_addr = this->base_result_data_addr[static_cast<int>(this->all_data[core][bank_id].core_type)];
    return base_addr + this->all_data[core][bank_id].data.size() * sizeof(uint32_t);
}

uint32_t DeviceData::size_at(CoreCoord core, int bank_id) {
    return this->all_data[core][bank_id].data.size();
}

uint32_t DeviceData::at(CoreCoord core, int bank_id, uint32_t offset) {
    return this->all_data[core][bank_id].data[offset];
}

inline bool DeviceData::validate_one_core(Device *device,
                                          std::unordered_set<CoreCoord> &validated_cores,
                                          const one_core_data_t& one_core_data,
                                          const uint32_t start_index,
                                          uint32_t result_addr) {
    int fail_count = 0;
    const std::vector<uint32_t>& dev_data = one_core_data.data;
    const vector<bool>& dev_valid = one_core_data.valid;
    const CoreCoord logical_core = one_core_data.logical_core;
    const CoreCoord phys_core = one_core_data.phys_core;
    const CoreType core_type = one_core_data.core_type;
    const int bank_id = one_core_data.bank_id;
    const int bank_offset = one_core_data.bank_offset;
    uint32_t size_bytes = (dev_data.size() - start_index) * sizeof(uint32_t);

    if (size_bytes == 0)  return false;

    string core_string;
    if (core_type == CoreType::WORKER) {
        core_string = "L1";
    } else if (core_type == CoreType::DRAM) {
        core_string = "DRAM";
    } else if (core_type == CoreType::PCIE) {
        core_string = "PCIE";
    } else {
        tt::log_fatal("Logical core: {} physical core {} core type {}", logical_core, phys_core, core_type);
        TT_ASSERT(false, "Core type not found");
    }

    // Read results from device and compare to expected for this core.
    result_addr += bank_offset;
    vector<uint32_t> results = tt::llrt::read_hex_vec_from_core(device->id(), phys_core, result_addr, size_bytes);

    log_info(tt::LogTest, "Validating {} bytes from {} bank {} log_core {}: phys_core: {} at addr: 0x{:x}",
             size_bytes, core_string, bank_id, logical_core.str(), phys_core.str(), result_addr);

    for (int i = 0; i <  size_bytes / sizeof(uint32_t); i++) {
        int index = start_index + i;
        if (!dev_valid[index]) continue;
        validated_cores.insert(phys_core);

        if (results[i] != dev_data[index]) {
            if (!fail_count) {
                log_fatal(tt::LogTest, "Data mismatch - First 20 failures for logical_core: {} (physical: {})", logical_core.str(), phys_core.str());
            }
            log_fatal(tt::LogTest, "[{:02d}] (Fail) Expected: 0x{:08x} Observed: 0x{:08x}", i, (unsigned int)dev_data[index], (unsigned int)results[i]);
            if (fail_count++ > 20) {
                break;
            }
        } else {
            log_debug(tt::LogTest, "[{:02d}] (Pass) Expected: 0x{:08x} Observed: 0x{:08x}", i, (unsigned int)dev_data[index], (unsigned int)results[i]);
        }
    }

    return fail_count;
}

bool DeviceData::validate_host(std::unordered_set<CoreCoord> &validated_cores,
                               const one_core_data_t& host_data) {

    uint32_t size_bytes = host_data.data.size() * sizeof(uint32_t);
    log_info(tt::LogTest, "Validating {} bytes from hugepage", size_bytes);

    bool failed = false;

    static int host_data_index = 0;
    uint32_t *results = (uint32_t *)this->base_data_addr[static_cast<int>(CoreType::PCIE)];

    int fail_count = 0;
    bool done = false;
    for (int data_index = 0; data_index < host_data.data.size(); data_index++) {
        validated_cores.insert(this->host_core);
        if (host_data.data[data_index] != results[host_data_index] && fail_count < 20) {
            if (!failed) {
                log_fatal(tt::LogTest, "Data mismatch - First 20 host data failures: [idx] expected->read");
            }

            log_fatal(tt::LogTest, "  [{:02d}] 0x{:08x}->0x{:08x}", host_data_index, (unsigned int)host_data.data[data_index], (unsigned int)results[host_data_index]);

            failed = true;
            fail_count++;
        }

        host_data_index++;

        if (host_data_index * sizeof(uint32_t) > hugepage_issue_buffer_size_g) {
            TT_THROW("Host test hugepage data wrap not (yet) supported, reduce test size/iterations");
        }
    }

    return failed;
}

bool DeviceData::validate(Device *device) {

    bool failed = false;
    std::unordered_set<CoreCoord> validated_cores;

    for (const auto & [core, bank_device_data] : this->all_data) {
        for (auto & [bank, one_core_data] : bank_device_data) {
            if (one_core_data.data.size() == 0) continue;

            const uint32_t start_index = (this->base_result_data_addr[static_cast<int>(one_core_data.core_type)] -
                                          this->base_data_addr[static_cast<int>(one_core_data.core_type)]) / sizeof(uint32_t);
            uint32_t result_addr = this->base_result_data_addr[static_cast<int>(one_core_data.core_type)];
            if (one_core_data.phys_core == this->host_core) {
                failed |= validate_host(validated_cores, one_core_data);
            } else {
                failed |= validate_one_core(device, validated_cores, one_core_data, start_index, result_addr);
            }
        }
    }

    log_info(tt::LogTest, "Validated {} non-empty cores total.", validated_cores.size());

    return !failed;
}

void DeviceData::overflow_check(Device *device) {

    for (const auto & [core, bank_device_data] : this->all_data) {
        for (auto & [bank, one_core_data] : bank_device_data) {
            if (one_core_data.core_type == CoreType::WORKER) {
                TT_FATAL(one_core_data.data.size() * sizeof(uint32_t) + base_data_addr[static_cast<int>(CoreType::WORKER)] <= device->l1_size_per_core(),
                         "Test overflowed L1 memory");
            } else if (one_core_data.core_type == CoreType::PCIE) {
                TT_FATAL(one_core_data.data.size() * sizeof(uint32_t)  <= hugepage_issue_buffer_size_g,
                         "Test overflowed PCIE memory");
            } else if (one_core_data.core_type == CoreType::DRAM) {
                // TODO
            }
        }
    }
}

template<bool is_dram_variant,
         bool is_host_variant>
void configure_kernel_variant(
    Program& program,
    string path,
    std::vector<uint32_t> compile_args, // yes, copy
    CoreCoord my_core,
    CoreCoord phys_my_core,
    CoreCoord phys_upstream_core,
    CoreCoord phys_downstream_core,
    Device * device,
    NOC my_noc_index,
    NOC upstream_noc_index,
    NOC downstream_noc_index) {

    const auto& grid_size = device->grid_size();

    std::map<string, string> defines = {
        {"MY_NOC_X", std::to_string(NOC_0_X(my_noc_index, grid_size.x, phys_my_core.x))},
        {"MY_NOC_Y", std::to_string(NOC_0_Y(my_noc_index, grid_size.y, phys_my_core.y))},
        {"UPSTREAM_NOC_INDEX", std::to_string(upstream_noc_index)},
        {"UPSTREAM_NOC_X", std::to_string(NOC_0_X(upstream_noc_index, grid_size.x, phys_upstream_core.x))},
        {"UPSTREAM_NOC_Y", std::to_string(NOC_0_Y(upstream_noc_index, grid_size.y, phys_upstream_core.y))},
        {"DOWNSTREAM_NOC_X", std::to_string(NOC_0_X(downstream_noc_index, grid_size.x, phys_downstream_core.x))},
        {"DOWNSTREAM_NOC_Y", std::to_string(NOC_0_Y(downstream_noc_index, grid_size.y, phys_downstream_core.y))},
        {"FD_CORE_TYPE", std::to_string(0)}, // todo, support dispatch on eth
    };
    compile_args.push_back(is_dram_variant);
    compile_args.push_back(is_host_variant);
    tt::tt_metal::CreateKernel(
        program,
        path,
        {my_core},
        tt::tt_metal::DataMovementConfig {
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = my_noc_index,
            .compile_args = compile_args,
            .defines = defines
        }
    );
}

// Specific to this test. This test doesn't use Buffers, and for Storage cores in L1 that have 2 banks, they are intended
// to be allocated top-down and carry "negative" offsets via bank_to_l1_offset for cores that have 2 banks. This function
// will scan through all banks bank_to_l1_offset and return the minimum required buffer addr to avoid bank_to_l1_offset
// being applied and underflowing.  In GS this is basically 512B or half the L1 Bank size.
inline uint32_t get_min_required_buffer_addr(Device *device, bool is_dram){

    int32_t smallest_offset = std::numeric_limits<int32_t>::max();
    BufferType buffer_type = is_dram ? BufferType::DRAM : BufferType::L1;
    uint32_t num_banks = device->num_banks(buffer_type);

    for (int bank_id = 0; bank_id < num_banks; bank_id++) {
        int32_t offset = device->bank_offset(buffer_type, bank_id);
        smallest_offset = offset < smallest_offset ? offset : smallest_offset;
    }

    // If negative, flip it and this becomes the min required positive offset for a buffer in bank.
    uint32_t min_required_positive_offset = smallest_offset < 0 ? 0 - smallest_offset : 0;
    log_debug(tt::LogTest, "{} - smallest_offset: {} min_required_positive_offset: {}", __FUNCTION__, smallest_offset, min_required_positive_offset);

    return min_required_positive_offset;
}

inline void generate_random_payload(vector<uint32_t>& cmds,
                                    uint32_t length) {

    for (uint32_t i = 0; i < length; i++) {
        uint32_t datum = (use_coherent_data_g) ? i : std::rand();
        cmds.push_back(datum);
    }
}

inline void generate_random_payload(vector<uint32_t>& cmds,
                                    const CoreRange& workers,
                                    DeviceData& data,
                                    uint32_t length_words,
                                    CQDispatchCmd cmd,
                                    bool is_mcast = false,
                                    bool prepend_cmd = false) {

    static uint32_t coherent_count = 0;
    const uint32_t bank_id = 0; // No interleaved pages here.

    // Host data puts the command in the datastream...
    if (prepend_cmd) {
        uint32_t datum = *(uint32_t *)&cmd;
        data.push_range(workers, datum, is_mcast);
        datum = *(((uint32_t *)&cmd) + 1);
        data.push_range(workers, datum, is_mcast);
        datum = *(((uint32_t *)&cmd) + 2);
        data.push_range(workers, datum, is_mcast);
        datum = *(((uint32_t *)&cmd) + 3);
        data.push_range(workers, datum, is_mcast);
    }

    // Note: the dst address marches in unison regardless of whether or not a core is written to
    for (uint32_t i = 0; i < length_words; i++) {
        uint32_t datum = (use_coherent_data_g) ? coherent_count++ : std::rand();
        cmds.push_back(datum);
        data.push_range(workers, datum, is_mcast);
    }
}

// Generate a random payload for a paged write command. Note: Doesn't currently support using the base_addr here.
inline void generate_random_paged_payload(Device *device,
                                          CQDispatchCmd cmd,
                                          vector<uint32_t>& cmds,
                                          DeviceData& data,
                                          uint32_t start_page,
                                          bool is_dram) {

    static uint32_t coherent_count = 0x100; // Abitrary starting value, avoid 0x0 since matches with DRAM prefill.
    auto buf_type = is_dram ? BufferType::DRAM : BufferType::L1;
    uint32_t num_banks = device->num_banks(buf_type);
    uint32_t words_per_page = cmd.write_paged.page_size / sizeof(uint32_t);
    log_debug(tt::LogTest, "Starting {} w/ is_dram: {} start_page: {} words_per_page: {}", __FUNCTION__, is_dram, start_page, words_per_page);

    // Note: the dst address marches in unison regardless of whether or not a core is written to
    for (uint32_t page_id = start_page; page_id < start_page + cmd.write_paged.pages; page_id++) {

        constexpr uint32_t page_size_alignment_bytes = ALLOCATOR_ALIGNMENT;
        CoreCoord bank_core;
        uint32_t bank_id = page_id % num_banks;
        uint32_t bank_offset = align(cmd.write_paged.page_size, page_size_alignment_bytes) * (page_id / num_banks);

        if (is_dram) {
            auto dram_channel = device->dram_channel_from_bank_id(bank_id);
            bank_core = device->dram_core_from_dram_channel(dram_channel);
        } else {
            bank_core = device->logical_core_from_bank_id(bank_id);
        }

        // Generate data and add to cmd for sending to device, and device_data for correctness checking.
        for (uint32_t i = 0; i < words_per_page; i++) {
            uint32_t datum = (use_coherent_data_g) ? (((page_id & 0xFF) << 24) | coherent_count++) : std::rand();
            log_debug(tt::LogTest, "{} - Setting {} page_id: {} word: {} on core: {} (bank_id: {} bank_offset: {}) => datum: 0x{:x}",
                __FUNCTION__, is_dram ? "DRAM" : "L1", page_id, i, bank_core.str(), bank_id, bank_offset, datum);
            cmds.push_back(datum); // Push to device.
            data.push_one(bank_core, bank_id, datum);
        }

        data.pad(bank_core, bank_id, page_size_alignment_bytes);
    }
}

inline void generate_random_packed_payload(vector<uint32_t>& cmds,
                                           vector<CoreCoord>& worker_cores,
                                           DeviceData& data,
                                           uint32_t size_words,
                                           bool repeat = false) {

    static uint32_t coherent_count = 0;
    const uint32_t bank_id = 0; // No interleaved pages here.

    bool first_core = true;
    vector<uint32_t>results;
    CoreCoord first_worker = worker_cores[0];
    for (uint32_t i = 0; i < size_words; i++) {
        uint32_t datum = (use_coherent_data_g) ? ((first_worker.x << 16) | (first_worker.y << 24) | coherent_count++) : std::rand();
        results.push_back(datum);
    }
    for (CoreCoord core : worker_cores) {
        for (uint32_t i = 0; i < size_words; i++) {
            data.push_one(core, bank_id, results[i]);
            if (!repeat || first_core) {
                cmds.push_back(results[i]);
            }
        }

        cmds.resize(padded_size(cmds.size(), hal.get_alignment(HalMemType::L1) / sizeof(uint32_t)));
        data.pad(core, bank_id, hal.get_alignment(HalMemType::L1));
        first_core = false;
    }
}

inline void generate_random_packed_large_payload(vector<uint32_t>& generated_data,
                                                 CoreRange range,
                                                 DeviceData& data,
                                                 uint32_t size_words) {

    static uint32_t coherent_count = 0;
    const uint32_t bank_id = 0; // No interleaved pages here.

    bool first_core = true;
    CoreCoord first_worker = range.start_coord;
    uint32_t data_base = generated_data.size();
    for (uint32_t i = 0; i < size_words; i++) {
        uint32_t datum = (use_coherent_data_g) ? ((first_worker.x << 16) | (first_worker.y << 24) | coherent_count++) : std::rand();
        generated_data.push_back(datum);
    }
    generated_data.resize(padded_size(generated_data.size(), hal.get_alignment(HalMemType::L1) / sizeof(uint32_t)));

    for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; y++) {
        for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; x++) {
            CoreCoord core = {x, y};
            for (uint32_t i = 0; i < size_words; i++) {
                data.push_one(core, bank_id, generated_data[data_base + i]);
            }
            data.pad(core, bank_id, hal.get_alignment(HalMemType::L1));
        }
    }
}

inline void add_bare_dispatcher_cmd(vector<uint32_t>& cmds,
                                    CQDispatchCmd cmd) {
    static_assert(sizeof(CQDispatchCmd) % sizeof(uint32_t) == 0, "CQDispatchCmd size must be a multiple of uint32_t size");
    const size_t num_uint32s = sizeof(CQDispatchCmd) / sizeof(uint32_t);
    uint32_t buf[num_uint32s];

    memcpy(buf, &cmd, sizeof(cmd));
    for (size_t i = 0; i < num_uint32s; i++) {
        cmds.push_back(buf[i]);
    }
}

inline size_t debug_prologue(vector<uint32_t>& cmds) {
    size_t prior = cmds.size();

    if (debug_g) {
        CQDispatchCmd debug_cmd;
        memset(&debug_cmd, 0, sizeof(CQDispatchCmd));

        debug_cmd.base.cmd_id = CQ_DISPATCH_CMD_DEBUG;
        // compiler compains w/o these filled in later fields
        debug_cmd.debug.key = 0;
        debug_cmd.debug.checksum = 0;
        debug_cmd.debug.size = 0;
        debug_cmd.debug.stride = 0;
        add_bare_dispatcher_cmd(cmds, debug_cmd);
    }

    return prior;
}

inline void debug_epilogue(vector<uint32_t>& cmds,
                           size_t prior_end) {
    if (debug_g) {
        // Doing a checksum on the full command length is problematic in the kernel
        // as it requires the debug code to pull all the pages in before the actual
        // command is processed.  So, limit this to doing a checksum on the first page
        // (which is disappointing).  Any other value requires the checksum code to handle
        // buffer wrap which then messes up the routines w/ the embedded insn - not worth it
        CQDispatchCmd* debug_cmd_ptr;
        debug_cmd_ptr = (CQDispatchCmd *)&cmds[prior_end];
        uint32_t full_size = (cmds.size() - prior_end) * sizeof(uint32_t) - sizeof(CQDispatchCmd);
        uint32_t max_size = dispatch_buffer_page_size_g - sizeof(CQDispatchCmd);
        uint32_t size = (full_size > max_size) ? max_size : full_size;
        debug_cmd_ptr->debug.size = size;
        debug_cmd_ptr->debug.stride = sizeof(CQDispatchCmd);
        uint32_t checksum = 0;
        uint32_t start = prior_end + sizeof(CQDispatchCmd) / sizeof(uint32_t);
        for (uint32_t i = start; i < start + size / sizeof(uint32_t); i++) {
            checksum += cmds[i];
        }
        debug_cmd_ptr->debug.checksum = checksum;
    }
}

inline void add_dispatcher_cmd(vector<uint32_t>& cmds,
                               CQDispatchCmd cmd,
                               uint32_t length) {

    size_t prior_end = debug_prologue(cmds);

    add_bare_dispatcher_cmd(cmds, cmd);
    uint32_t length_words = length / sizeof(uint32_t);
    generate_random_payload(cmds, length_words);

    debug_epilogue(cmds, prior_end);
}

inline void add_dispatcher_cmd(vector<uint32_t>& cmds,
                               const CoreRange& workers,
                               DeviceData& device_data,
                               CQDispatchCmd cmd,
                               uint32_t length,
                               bool is_mcast = false,
                               bool prepend_cmd = false) {

    size_t prior_end = debug_prologue(cmds);

    add_bare_dispatcher_cmd(cmds, cmd);
    uint32_t length_words = length / sizeof(uint32_t);
    generate_random_payload(cmds, workers, device_data, length_words, cmd, is_mcast, prepend_cmd);

    debug_epilogue(cmds, prior_end);
}

inline void add_dispatcher_paged_cmd(Device *device,
                                     vector<uint32_t>& cmds,
                                     DeviceData& device_data,
                                     CQDispatchCmd cmd,
                                     uint32_t start_page,
                                     bool is_dram) {

    size_t prior_end = debug_prologue(cmds);
    add_bare_dispatcher_cmd(cmds, cmd);
    generate_random_paged_payload(device, cmd, cmds, device_data, start_page, is_dram);
    debug_epilogue(cmds, prior_end);
}

inline void add_dispatcher_packed_cmd(Device *device,
                                      vector<uint32_t>& cmds,
                                      vector<CoreCoord>& worker_cores,
                                      DeviceData& device_data,
                                      CQDispatchCmd cmd,
                                      uint32_t size_words,
                                      bool repeat = false) {

    size_t prior_end = debug_prologue(cmds);

    add_bare_dispatcher_cmd(cmds, cmd);
    for (CoreCoord core : worker_cores) {
        CoreCoord phys_worker_core = device->worker_core_from_logical_core(core);
        cmds.push_back(NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y));
    }
    cmds.resize(padded_size(cmds.size(), hal.get_alignment(HalMemType::L1)/sizeof(uint32_t)));

    generate_random_packed_payload(cmds, worker_cores, device_data, size_words, repeat);

    debug_epilogue(cmds, prior_end);
}

// bare: doesn't generate random payload data, for use w/ eg, dram reads
inline void gen_bare_dispatcher_unicast_write_cmd(Device *device,
                                                  vector<uint32_t>& cmds,
                                                  CoreCoord worker_core,
                                                  DeviceData& device_data,
                                                  uint32_t length) {

    CQDispatchCmd cmd;
    memset(&cmd, 0, sizeof(CQDispatchCmd));

    CoreCoord phys_worker_core = device->worker_core_from_logical_core(worker_core);
    const uint32_t bank_id = 0; // No interleaved pages here.

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
    cmd.write_linear.noc_xy_addr = NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y);
    cmd.write_linear.addr = device_data.get_result_data_addr(worker_core, bank_id);
    cmd.write_linear.length = length;
    cmd.write_linear.num_mcast_dests = 0;

    TT_FATAL((cmd.write_linear.addr & (hal.get_alignment(HalMemType::L1) - 1)) == 0, "Error");

    add_bare_dispatcher_cmd(cmds, cmd);
}

inline void gen_dispatcher_unicast_write_cmd(Device *device,
                                             vector<uint32_t>& cmds,
                                             CoreCoord worker_core,
                                             DeviceData& device_data,
                                             uint32_t length) {

    CQDispatchCmd cmd;
    memset(&cmd, 0, sizeof(CQDispatchCmd));

    CoreCoord phys_worker_core = device->worker_core_from_logical_core(worker_core);
    const uint32_t bank_id = 0; // No interleaved pages here.

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
    cmd.write_linear.noc_xy_addr = NOC_XY_ENCODING(phys_worker_core.x, phys_worker_core.y);
    cmd.write_linear.addr = device_data.get_result_data_addr(worker_core, bank_id);
    cmd.write_linear.length = length;
    cmd.write_linear.num_mcast_dests = 0;

    add_dispatcher_cmd(cmds, worker_core, device_data, cmd, length);
}

inline void gen_dispatcher_multicast_write_cmd(Device *device,
                                             vector<uint32_t>& cmds,
                                             CoreRange worker_core_range,
                                             DeviceData& device_data,
                                             uint32_t length) {

    // Pad w/ blank data until all workers are at the same address
    // TODO Hmm, ideally only need to relevel the core range
    device_data.relevel(CoreType::WORKER);

    CQDispatchCmd cmd;
    memset(&cmd, 0, sizeof(CQDispatchCmd));

    CoreCoord physical_start = device->physical_core_from_logical_core(worker_core_range.start_coord, CoreType::WORKER);
    CoreCoord physical_end = device->physical_core_from_logical_core(worker_core_range.end_coord, CoreType::WORKER);
    const uint32_t bank_id = 0; // No interleaved pages here.

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR;
    cmd.write_linear.noc_xy_addr = NOC_MULTICAST_ENCODING(physical_start.x, physical_start.y, physical_end.x, physical_end.y);
    cmd.write_linear.addr = device_data.get_result_data_addr(worker_core_range.start_coord);
    cmd.write_linear.length = length;
    cmd.write_linear.num_mcast_dests = worker_core_range.size();

    add_dispatcher_cmd(cmds, worker_core_range, device_data, cmd, length, true);
}

inline void gen_dispatcher_paged_write_cmd(Device *device,
                                             vector<uint32_t>& cmds,
                                             DeviceData& device_data,
                                             bool is_dram,
                                             uint32_t start_page,
                                             uint32_t page_size,
                                             uint32_t pages) {

    constexpr uint32_t page_size_alignment_bytes = ALLOCATOR_ALIGNMENT;
    uint32_t num_banks = device->num_banks(is_dram ? BufferType::DRAM : BufferType::L1);
    CoreType core_type = is_dram ? CoreType::DRAM : CoreType::WORKER;

    // Not safe to mix paged L1 and paged DRAM writes currently in this test since same book-keeping.
    static uint32_t prev_is_dram = -1;
    TT_ASSERT(prev_is_dram == -1 || prev_is_dram == is_dram, "Mixing paged L1 and paged DRAM writes not supported in this test.");
    prev_is_dram = is_dram;

    // Assumption embedded in this function (seems reasonable, true with a single buffer) that paged size will never change.
    static uint32_t prev_page_size = -1;
    TT_ASSERT(prev_page_size == -1 || prev_page_size == page_size, "Page size changed between calls to gen_dispatcher_paged_write_cmd - not supported.");
    prev_page_size = page_size;

    // For the CMD generation, start_page is 8 bits, so much wrap around, and increase base_addr instead based on page size,
    // which assumes page size never changed between calls to this function (checked above).
    uint32_t bank_offset = align(page_size, page_size_alignment_bytes) * (start_page / num_banks);
    // TODO: make this take the latest address, change callers to not manage this
    uint32_t base_addr = device_data.get_base_result_addr(core_type) + bank_offset;
    uint16_t start_page_cmd = start_page % num_banks;

    CQDispatchCmd cmd;
    memset(&cmd, 0, sizeof(CQDispatchCmd));
    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_PAGED;
    cmd.write_paged.is_dram = is_dram;
    cmd.write_paged.start_page = start_page_cmd;
    cmd.write_paged.base_addr = base_addr;
    cmd.write_paged.page_size = page_size;
    cmd.write_paged.pages = pages;

    log_debug(tt::LogTest, "Adding CQ_DISPATCH_CMD_WRITE_PAGED - is_dram: {} start_page: {} start_page_cmd: {} base_addr: 0x{:x} bank_offset: 0x{:x} page_size: {} pages: {})",
        is_dram, start_page, start_page_cmd, base_addr, bank_offset, page_size, pages);

    add_dispatcher_paged_cmd(device, cmds, device_data, cmd, start_page, is_dram);
}


inline void gen_dispatcher_packed_write_cmd(Device *device,
                                            vector<uint32_t>& cmds,
                                            vector<CoreCoord>& worker_cores,
                                            DeviceData& device_data,
                                            uint32_t size_words,
                                            bool repeat = false) {

    // Pad w/ blank data until all workers are at the same address
    device_data.relevel(CoreType::WORKER);

    CQDispatchCmd cmd;
    memset(&cmd, 0, sizeof(CQDispatchCmd));

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_PACKED;
    cmd.write_packed.flags = repeat ? CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NO_STRIDE : CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_NONE;
    cmd.write_packed.count = worker_cores.size();
    cmd.write_packed.addr = device_data.get_result_data_addr(worker_cores[0]);
    cmd.write_packed.size = size_words * sizeof(uint32_t);

    uint32_t sub_cmds_size = padded_size(worker_cores.size() * sizeof(CQDispatchWritePackedUnicastSubCmd), sizeof(CQDispatchCmd));
    TT_FATAL(repeat == false || size_words * sizeof(uint32_t) + sizeof(CQDispatchCmd) + sub_cmds_size <= dispatch_buffer_page_size_g, "Error");

    add_dispatcher_packed_cmd(device, cmds, worker_cores, device_data, cmd, size_words, repeat);
}

inline void gen_rnd_dispatcher_packed_write_cmd(Device *device,
                                                vector<uint32_t>& cmds,
                                                DeviceData& device_data) {

    // Note: this cmd doesn't clamp to a max size which means it can overflow L1 buffer
    // However, this cmd doesn't send much data and the L1 buffer is < L1 limit, so...

    uint32_t xfer_size_words = (std::rand() % (dispatch_buffer_page_size_g / sizeof(uint32_t))) + 1;
    uint32_t xfer_size_bytes = xfer_size_words * sizeof(uint32_t);
    if (perf_test_g) {
        TT_ASSERT(max_xfer_size_bytes_g <= dispatch_buffer_page_size_g);
        if (xfer_size_bytes > max_xfer_size_bytes_g) xfer_size_bytes = max_xfer_size_bytes_g;
        if (xfer_size_bytes < min_xfer_size_bytes_g) xfer_size_bytes = min_xfer_size_bytes_g;
    }

    vector<CoreCoord> gets_data;
    while (gets_data.size() == 0) {
        for (auto & [core, one_worker] : device_data.get_data()) {
            if (device_data.core_and_bank_present(core, 0) &&
                one_worker[0].core_type == CoreType::WORKER) {
                if (send_to_all_g || std::rand() % 2) {
                    gets_data.push_back(core);
                }
            }
        }
    }

    bool repeat = std::rand() % 2;
    if (repeat) {
        // TODO fix this if/when we add mcast
        uint32_t sub_cmds_size = padded_size(gets_data.size() * sizeof(uint32_t), hal.get_alignment(HalMemType::L1));
        if (xfer_size_bytes + sizeof (CQDispatchCmd) + sub_cmds_size > dispatch_buffer_page_size_g) {
            static bool warned = false;
            if (!warned) {
                log_warning(tt::LogTest, "Clamping packed_write cmd w/ stride=0 size to fit a dispatch page.  Adjust max/min xfer sizes for reliable perf data");
                warned = true;
            }
            xfer_size_bytes = dispatch_buffer_page_size_g - sizeof (CQDispatchCmd) - sub_cmds_size;
        }
    }

    gen_dispatcher_packed_write_cmd(device, cmds, gets_data, device_data,
                                    xfer_size_bytes / sizeof(uint32_t), repeat);
}

inline bool gen_rnd_dispatcher_packed_write_large_cmd(Device *device,
                                                      CoreRange workers,
                                                      vector<uint32_t>& cmds,
                                                      DeviceData& device_data,
                                                      uint32_t space_available) {

    int ntransactions = perf_test_g ? (CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS / 2) :
        ((std:: rand() % CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_MAX_SUB_CMDS) + 1);

    vector<uint32_t> sizes;
    for (int i = 0; i < ntransactions; i++) {
        constexpr uint32_t max_pages = 4;
        uint32_t xfer_size_16b = (std::rand() % (dispatch_buffer_page_size_g * max_pages / hal.get_alignment(HalMemType::L1))) + 1;
        uint32_t xfer_size_words = xfer_size_16b * 4;
        uint32_t xfer_size_bytes = xfer_size_words * sizeof(uint32_t);
        if (perf_test_g) {
            TT_ASSERT(max_xfer_size_bytes_g <= dispatch_buffer_page_size_g);
            if (xfer_size_bytes > max_xfer_size_bytes_g) xfer_size_bytes = max_xfer_size_bytes_g;
            if (xfer_size_bytes < min_xfer_size_bytes_g) xfer_size_bytes = min_xfer_size_bytes_g;
        }

        if (xfer_size_bytes > space_available) {
            if (ntransactions == 0) {
                return true;
            }
            ntransactions = i;
            break;
        }

        sizes.push_back(xfer_size_bytes);
        space_available -= xfer_size_bytes;
    }

    CQDispatchCmd cmd;
    memset(&cmd, 0, sizeof(CQDispatchCmd));
    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_PACKED_LARGE;
    cmd.write_packed_large.count = ntransactions;
    cmd.write_packed_large.alignment = hal.get_alignment(HalMemType::L1);
    add_bare_dispatcher_cmd(cmds, cmd);

    vector<uint32_t> data;
    for (int i = 0; i < ntransactions; i++) {
        uint32_t xfer_size_bytes = sizes[i];

        CoreRange range = workers;
        if (!perf_test_g) {
            // Not random, but gives some variation
            uint32_t span = workers.end_coord.x - workers.start_coord.x + 1;
            range.end_coord.x = std::rand() % span + range.start_coord.x;
            span = workers.end_coord.y - workers.start_coord.y + 1;
            range.end_coord.y = std::rand() % span + range.start_coord.y;
        }

        device_data.relevel(range);

        CQDispatchWritePackedLargeSubCmd sub_cmd;
        CoreCoord physical_start = device->physical_core_from_logical_core(range.start_coord, CoreType::WORKER);
        CoreCoord physical_end = device->physical_core_from_logical_core(range.end_coord, CoreType::WORKER);
        sub_cmd.noc_xy_addr = NOC_MULTICAST_ENCODING(physical_start.x, physical_start.y, physical_end.x, physical_end.y);
        sub_cmd.addr = device_data.get_result_data_addr(range.start_coord);
        sub_cmd.length = xfer_size_bytes;
        sub_cmd.num_mcast_dests = (range.end_coord.x - range.start_coord.x + 1) * (range.end_coord.y - range.start_coord.y + 1);
        sub_cmd.flags = CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_FLAG_UNLINK;

        for (uint32_t i = 0; i < sizeof(CQDispatchWritePackedLargeSubCmd) / sizeof(uint32_t); i++) {
            cmds.push_back(((uint32_t *)&sub_cmd)[i]);
        }

        generate_random_packed_large_payload(data, range, device_data, xfer_size_bytes / sizeof(uint32_t));
    }
    cmds.resize(padded_size(cmds.size(), hal.get_alignment(HalMemType::L1) / sizeof(uint32_t)));

    for (uint32_t datum : data) {
        cmds.push_back(datum);
    }

    return false;
}

inline void gen_dispatcher_host_write_cmd(vector<uint32_t>& cmds,
                                          DeviceData& device_data,
                                          uint32_t length) {

    CQDispatchCmd cmd;
    memset(&cmd, 0, sizeof(CQDispatchCmd));

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST;
    // Include cmd in transfer
    cmd.write_linear_host.length = length + sizeof(CQDispatchCmd);

    add_dispatcher_cmd(cmds, device_data.get_host_core(), device_data, cmd, length, false, true);
}

inline void gen_bare_dispatcher_host_write_cmd(vector<uint32_t>& cmds, uint32_t length) {

    CQDispatchCmd cmd;
    memset(&cmd, 0, sizeof(CQDispatchCmd));

    cmd.base.cmd_id = CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST;
    // Include cmd in transfer
    cmd.write_linear_host.length = length + sizeof(CQDispatchCmd);

    add_bare_dispatcher_cmd(cmds, cmd);
}

inline void gen_dispatcher_set_write_offset_cmd(vector<uint32_t>& cmds, uint32_t wo0, uint32_t wo1 = 0, uint32_t wo2 = 0) {

    CQDispatchCmd cmd;
    memset(&cmd, 0, sizeof(CQDispatchCmd));

    cmd.base.cmd_id = CQ_DISPATCH_CMD_SET_WRITE_OFFSET;
    cmd.set_write_offset.offset0 = wo0;
    cmd.set_write_offset.offset1 = wo1;
    cmd.set_write_offset.offset2 = wo2;
    uint32_t payload_length = 0;
    add_dispatcher_cmd(cmds, cmd, payload_length);
}

inline void gen_dispatcher_terminate_cmd(vector<uint32_t>& cmds) {

    CQDispatchCmd cmd;
    memset(&cmd, 0, sizeof(CQDispatchCmd));

    cmd.base.cmd_id = CQ_DISPATCH_CMD_TERMINATE;
    uint32_t payload_length = 0;
    add_dispatcher_cmd(cmds, cmd, payload_length);
}
