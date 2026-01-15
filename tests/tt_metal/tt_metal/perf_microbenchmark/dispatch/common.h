// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <random>
#include <tt-metalium/core_coord.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/allocator.hpp>

#include "tt_metal.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"

#include "llrt.hpp"
#include <tt-metalium/tt_align.hpp>

#include "llrt/hal.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include <variant>
#include <llrt/tt_cluster.hpp>

#include "tt_metal/distributed/fd_mesh_command_queue.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "dispatch/device_command_calculator.hpp"
#include "command_queue_fixture.hpp"
#include "tests/tt_metal/tt_metal/common/mesh_dispatch_fixture.hpp"
#include "tt_metal/impl/dispatch/system_memory_manager.hpp"
#include <impl/dispatch/dispatch_mem_map.hpp>
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::tt_dispatch_tests::Common {

constexpr uint32_t DRAM_DATA_SIZE_BYTES = 16 * 1024 * 1024;
constexpr uint32_t DRAM_DATA_SIZE_WORDS = DRAM_DATA_SIZE_BYTES / sizeof(uint32_t);

struct DispatchTestConfig {
    bool use_coherent_data = false;
    uint32_t dispatch_buffer_page_size = 1u << tt::tt_metal::DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE;
    uint32_t min_xfer_size_bytes = 16;
    uint32_t max_xfer_size_bytes = 4096;
    bool send_to_all = true;
    bool perf_test = false;
    uint32_t hugepage_issue_buffer_size = 256 * 1024 * 1024;
};

struct one_core_data_t {
    tt::CoreType core_type{tt::CoreType::COUNT};
    CoreCoord logical_core;
    CoreCoord phys_core;
    int bank_id{};
    int bank_offset{};
    std::vector<bool> valid;
    std::vector<uint32_t> data;
};

class DeviceData {
private:
    int amt_written{0};
    // 10 is a hack...bigger than any core_type
    uint64_t base_data_addr[static_cast<size_t>(tt::CoreType::COUNT)]{};
    uint64_t base_result_data_addr[static_cast<size_t>(tt::CoreType::COUNT)]{};
    std::unordered_map<CoreCoord, std::unordered_map<uint32_t, one_core_data_t>> all_data;
    CoreCoord host_core;
    size_t host_data_index = 0;

    // Test Config
    bool use_coherent_data_;
    uint32_t hugepage_issue_buffer_size_;

    // Validate a single core's worth of results vs expected
    bool validate_one_core(
        distributed::MeshDevice::IDevice* device,
        std::unordered_set<CoreCoord>& validated_cores,
        const one_core_data_t& one_core_data,
        uint32_t start_index,
        uint32_t result_addr);
    bool validate_host(std::unordered_set<CoreCoord>& validated_cores, const one_core_data_t& host_data);

    void prepopulate_dram(distributed::MeshDevice::IDevice* device, uint32_t size_words);

public:
    DeviceData(
        distributed::MeshDevice::IDevice* device,
        CoreRange workers,
        uint32_t l1_data_addr,
        uint32_t dram_data_addr,
        void* pcie_data_addr,
        bool is_banked,
        uint32_t dram_data_size_words,
        const DispatchTestConfig& cfg);

    // Add expected data to a core
    void push_one(CoreCoord core, int bank, uint32_t datum);
    void push_one(CoreCoord core, uint32_t datum);
    void push_range(const CoreRange& cores, uint32_t datum, bool is_mcast);

    // Add invalid data
    void pad(CoreCoord core, int bank, uint32_t alignment);

    // Some tests write to the same address across multiple cores
    // This takes those core types and pads any that are "behind" with invalid data
    void relevel(tt::CoreType core_type);
    void relevel(CoreRange range);

    // Clear data between tests
    void reset();
    uint32_t get_base_result_addr(tt::CoreType core_type);
    uint32_t get_result_data_addr(CoreCoord core, int bank_id = 0);

    bool validate(distributed::MeshDevice::IDevice* device);
    void overflow_check(distributed::MeshDevice::IDevice* device);

    int size() const { return amt_written; }
    int size(CoreCoord core, int bank_id = 0) { return this->all_data[core][bank_id].data.size(); }

    std::unordered_map<CoreCoord, std::unordered_map<uint32_t, one_core_data_t>>& get_data() { return this->all_data; }

    tt::CoreType get_core_type(CoreCoord core) { return this->all_data[core][0].core_type; }
    uint32_t size_at(CoreCoord core, int bank_id);
    uint32_t at(CoreCoord core, int bank_id, uint32_t offset);
    CoreCoord get_host_core() { return this->host_core; }
    bool core_and_bank_present(CoreCoord core, uint32_t bank);
};

inline DeviceData::DeviceData(
    distributed::MeshDevice::IDevice* device,
    CoreRange workers,
    uint32_t l1_data_addr,
    uint32_t dram_data_addr,
    void* pcie_data_addr,
    bool is_banked,
    uint32_t dram_data_size_words,
    const DispatchTestConfig& cfg) :
    use_coherent_data_(cfg.use_coherent_data),
    hugepage_issue_buffer_size_(cfg.hugepage_issue_buffer_size) {
    this->base_data_addr[static_cast<int>(tt::CoreType::WORKER)] = l1_data_addr;
    this->base_data_addr[static_cast<int>(tt::CoreType::PCIE)] = (uint64_t)pcie_data_addr;
    this->base_data_addr[static_cast<int>(tt::CoreType::DRAM)] = dram_data_addr;
    this->base_result_data_addr[static_cast<int>(tt::CoreType::WORKER)] = l1_data_addr;
    this->base_result_data_addr[static_cast<int>(tt::CoreType::PCIE)] = (uint64_t)pcie_data_addr;
    this->base_result_data_addr[static_cast<int>(tt::CoreType::DRAM)] = dram_data_addr;

    // TODO: make this all work w/ phys coords
    // this is really annoying
    // the PCIE phys core conflicts w/ worker logical cores
    // so we hack the physical core for tracking, blech
    // no simple way to handle this, need to use phys cores
    CoreCoord core = {100, 100};
    this->all_data[core][0] = one_core_data_t();
    this->all_data[core][0].logical_core = core;
    this->all_data[core][0].phys_core = core;
    this->all_data[core][0].core_type = tt::CoreType::PCIE;
    this->all_data[core][0].bank_id = 20;
    this->all_data[core][0].bank_offset = 0;
    this->host_core = core;

    // Always populate DRAM
    auto num_banks = device->allocator()->get_num_banks(BufferType::DRAM);
    for (int bank_id = 0; bank_id < num_banks; bank_id++) {
        auto dram_channel = device->allocator_impl()->get_dram_channel_from_bank_id(bank_id);
        CoreCoord phys_core = device->logical_core_from_dram_channel(dram_channel);
        int32_t bank_offset = device->allocator()->get_bank_offset(BufferType::DRAM, bank_id);
        this->all_data[phys_core][bank_id] = one_core_data_t();
        this->all_data[phys_core][bank_id].logical_core = phys_core;
        this->all_data[phys_core][bank_id].phys_core = phys_core;
        this->all_data[phys_core][bank_id].core_type = tt::CoreType::DRAM;
        this->all_data[phys_core][bank_id].bank_id = bank_id;
        this->all_data[phys_core][bank_id].bank_offset = bank_offset;
    }

    // TODO: make banked L1 tests play nicely w/ non-banked L1 tests
    if (is_banked) {
        num_banks = device->allocator()->get_num_banks(BufferType::L1);
        for (int bank_id = 0; bank_id < num_banks; bank_id++) {
            CoreCoord core = device->allocator()->get_logical_core_from_bank_id(bank_id);
            CoreCoord phys_core = device->worker_core_from_logical_core(core);
            int32_t bank_offset = device->allocator()->get_bank_offset(BufferType::L1, bank_id);
            this->all_data[core][bank_id] = one_core_data_t();
            this->all_data[core][bank_id].logical_core = core;
            this->all_data[core][bank_id].phys_core = phys_core;
            this->all_data[core][bank_id].core_type = tt::CoreType::WORKER;
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
                this->all_data[core][0].core_type = tt::CoreType::WORKER;
                this->all_data[core][0].bank_id = 0;
                this->all_data[core][0].bank_offset = 0;
            }
        }
    }

    prepopulate_dram(device, dram_data_size_words);
}

// Populate interleaved DRAM with data for later readback.  Can we extended to L1 if needed.
inline void DeviceData::prepopulate_dram(distributed::MeshDevice::IDevice* device, uint32_t size_words) {
    uint32_t num_dram_banks = device->allocator()->get_num_banks(BufferType::DRAM);

    for (int bank_id = 0; bank_id < num_dram_banks; bank_id++) {
        [[maybe_unused]] auto offset = device->allocator()->get_bank_offset(BufferType::DRAM, bank_id);
        auto dram_channel = device->allocator_impl()->get_dram_channel_from_bank_id(bank_id);
        auto bank_core = device->logical_core_from_dram_channel(dram_channel);
        one_core_data_t& data = this->all_data[bank_core][bank_id];

        // Generate random or coherent data per bank of specific size.
        for (uint32_t i = 0; i < size_words; i++) {
            uint32_t datum = (use_coherent_data_) ? (((bank_id & 0xFF) << 24) | i) : std::rand();

            // Note: don't bump amt_written
            data.data.push_back(datum);
            data.valid.push_back(true);

            if (i < 10) {
                log_debug(
                    tt::LogTest,
                    "{} - bank_id: {:2d} core: {} offset: 0x{:08x} using i: {:2d} datum: 0x{:08x}",
                    __FUNCTION__,
                    bank_id,
                    bank_core.str(),
                    offset,
                    i,
                    datum);
            }
        }

        // Write to device once per bank (appropriate core and offset)
        tt::tt_metal::detail::WriteToDeviceDRAMChannel(
            device, bank_id, this->base_data_addr[static_cast<int>(tt::CoreType::DRAM)], data.data);

        this->base_result_data_addr[static_cast<int>(tt::CoreType::DRAM)] =
            this->base_data_addr[static_cast<int>(tt::CoreType::DRAM)] + data.data.size() * sizeof(uint32_t);
    }
}

inline bool DeviceData::core_and_bank_present(CoreCoord core, uint32_t bank) {
    if (this->all_data.contains(core)) {
        std::unordered_map<uint32_t, one_core_data_t>& core_data = this->all_data.find(core)->second;
        if (core_data.contains(bank)) {
            return true;
        }
    }
    return false;
}

inline void DeviceData::push_one(CoreCoord core, int bank, uint32_t datum) {
    if (core_and_bank_present(core, bank)) {
        this->amt_written++;
        this->all_data[core][bank].data.push_back(datum);
        this->all_data[core][bank].valid.push_back(true);
    }
}

inline void DeviceData::push_one(CoreCoord core, uint32_t datum) {
    if (core_and_bank_present(core, 0)) {
        this->amt_written++;
        this->all_data[core][0].data.push_back(datum);
        this->all_data[core][0].valid.push_back(true);
    }
}

inline void DeviceData::push_range(const CoreRange& cores, uint32_t datum, bool is_mcast) {
    bool counted = false;
    for (auto y = cores.start_coord.y; y <= cores.end_coord.y; y++) {
        for (auto x = cores.start_coord.x; x <= cores.end_coord.x; x++) {
            CoreCoord core = {x, y};
            if (core_and_bank_present(core, 0)) {
                if (not counted || not is_mcast) {
                    this->amt_written++;
                    counted = true;
                }

                TT_FATAL(this->all_data.contains(core), "Core {} not found in all_data", core);
                this->all_data[core][0].data.push_back(datum);
                this->all_data[core][0].valid.push_back(true);
            }
        }
    }
}

inline uint32_t padded_size(uint32_t size, uint32_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

inline void DeviceData::pad(CoreCoord core, int bank, uint32_t alignment) {
    if (core_and_bank_present(core, bank)) {
        uint32_t padded = padded_size(this->all_data[core][bank].data.size(), alignment / sizeof(uint32_t));
        this->all_data[core][bank].data.resize(padded);
        this->all_data[core][bank].valid.resize(padded);  // pushes false
    }
}

// Some tests write to the same address across multiple cores
// This takes cores that match core_type and pads any that are "behind" with invalid data
inline void DeviceData::relevel(tt::CoreType core_type) {
    size_t max = 0;
    for (auto& [coord, bank_device_data] : this->all_data) {
        for (auto& [bank, one_core_data] : bank_device_data) {
            if (one_core_data.core_type == core_type && one_core_data.data.size() > max) {
                max = one_core_data.data.size();
            }
        }
    }
    for (auto& [coord, bank_device_data] : this->all_data) {
        for (auto& [bank, one_core_data] : bank_device_data) {
            if (one_core_data.core_type == core_type) {
                one_core_data.data.resize(max);
                one_core_data.valid.resize(max);  // fills with false
            }
        }
    }
}

inline void DeviceData::relevel(CoreRange range) {
    size_t max = 0;

    constexpr uint32_t bank = 0;
    for (uint32_t y = range.start_coord.y; y <= range.end_coord.y; y++) {
        for (uint32_t x = range.start_coord.x; x <= range.end_coord.x; x++) {
            CoreCoord core = {x, y};
            max = std::max(this->all_data[core][bank].data.size(), max);
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
inline void DeviceData::reset() {
    this->amt_written = 0;
    host_data_index = 0;
    for (auto& [coord, bank_device_data] : this->all_data) {
        for (auto& [bank, one_core_data] : bank_device_data) {
            tt::CoreType core_type = one_core_data.core_type;
            uint32_t default_size_bytes = this->base_result_data_addr[static_cast<int>(core_type)] -
                                          this->base_data_addr[static_cast<int>(core_type)];
            one_core_data.valid.resize(default_size_bytes / sizeof(uint32_t));
            one_core_data.data.resize(default_size_bytes / sizeof(uint32_t));
        }
    }
}

inline uint32_t DeviceData::get_base_result_addr(tt::CoreType core_type) {
    return this->base_result_data_addr[static_cast<int>(core_type)];
}

inline uint32_t DeviceData::get_result_data_addr(CoreCoord core, int bank_id) {
    uint32_t base_addr = this->base_result_data_addr[static_cast<int>(this->all_data[core][bank_id].core_type)];
    return base_addr + (this->all_data[core][bank_id].data.size() * sizeof(uint32_t));
}

inline uint32_t DeviceData::size_at(CoreCoord core, int bank_id) { return this->all_data[core][bank_id].data.size(); }

inline uint32_t DeviceData::at(CoreCoord core, int bank_id, uint32_t offset) {
    return this->all_data[core][bank_id].data[offset];
}

inline bool DeviceData::validate_one_core(
    distributed::MeshDevice::IDevice* device,
    std::unordered_set<CoreCoord>& validated_cores,
    const one_core_data_t& one_core_data,
    const uint32_t start_index,
    uint32_t result_addr) {
    int fail_count = 0;
    const std::vector<uint32_t>& dev_data = one_core_data.data;
    const std::vector<bool>& dev_valid = one_core_data.valid;
    const CoreCoord logical_core = one_core_data.logical_core;
    const CoreCoord phys_core = one_core_data.phys_core;
    const tt::CoreType core_type = one_core_data.core_type;
    const int bank_id = one_core_data.bank_id;
    const int bank_offset = one_core_data.bank_offset;
    uint32_t size_bytes = (dev_data.size() - start_index) * sizeof(uint32_t);

    if (size_bytes == 0) {
        return false;
    }

    std::string core_string;
    if (core_type == tt::CoreType::WORKER) {
        core_string = "L1";
    } else if (core_type == tt::CoreType::DRAM) {
        core_string = "DRAM";
    } else if (core_type == tt::CoreType::PCIE) {
        core_string = "PCIE";
    } else {
        log_fatal(tt::LogTest, "Logical core: {} physical core {} core type {}", logical_core, phys_core, core_type);
        TT_FATAL(false, "Core type not found");
    }

    // Read results from device and compare to expected for this core.
    std::vector<uint32_t> results;
    if (core_type == tt::CoreType::DRAM) {
        tt::tt_metal::detail::ReadFromDeviceDRAMChannel(device, bank_id, result_addr, size_bytes, results);
    } else {
        result_addr += bank_offset;
        results = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            device->id(), phys_core, result_addr, size_bytes);
    }

    log_info(
        tt::LogTest,
        "Validating {} bytes from {} bank {} log_core {}: phys_core: {} at addr: 0x{:x}",
        size_bytes,
        core_string,
        bank_id,
        logical_core.str(),
        phys_core.str(),
        result_addr);

    for (int i = 0; i < size_bytes / sizeof(uint32_t); i++) {
        int index = start_index + i;
        if (!dev_valid[index]) {
            continue;
        }
        validated_cores.insert(phys_core);

        if (results[i] != dev_data[index]) {
            if (!fail_count) {
                log_fatal(
                    tt::LogTest,
                    "Data mismatch - First 20 failures for logical_core: {} (physical: {})",
                    logical_core.str(),
                    phys_core.str());
            }
            log_fatal(
                tt::LogTest,
                "[{:02d}] (Fail) Expected: 0x{:08x} Observed: 0x{:08x}",
                i,
                (unsigned int)dev_data[index],
                (unsigned int)results[i]);
            if (fail_count++ > 20) {
                break;
            }
        } else {
            log_debug(
                tt::LogTest,
                "[{:02d}] (Pass) Expected: 0x{:08x} Observed: 0x{:08x}",
                i,
                (unsigned int)dev_data[index],
                (unsigned int)results[i]);
        }
    }

    return fail_count;
}

inline bool DeviceData::validate_host(
    std::unordered_set<CoreCoord>& validated_cores, const one_core_data_t& host_data) {
    uint32_t size_bytes = host_data.data.size() * sizeof(uint32_t);
    log_info(tt::LogTest, "Validating {} bytes from hugepage", size_bytes);

    bool failed = false;

    uint32_t* results = (uint32_t*)this->base_data_addr[static_cast<int>(tt::CoreType::PCIE)];

    int fail_count = 0;
    for (unsigned int val : host_data.data) {
        validated_cores.insert(this->host_core);
        if (val != results[host_data_index] && fail_count < 20) {
            if (!failed) {
                log_fatal(tt::LogTest, "Data mismatch - First 20 host data failures: [idx] expected->read");
            }

            log_fatal(
                tt::LogTest,
                "  [{:02d}] 0x{:08x}->0x{:08x}",
                host_data_index,
                (unsigned int)val,
                (unsigned int)results[host_data_index]);

            failed = true;
            fail_count++;
        }

        host_data_index++;

        if (host_data_index * sizeof(uint32_t) > hugepage_issue_buffer_size_) {
            TT_THROW("Host test hugepage data wrap not (yet) supported, reduce test size/iterations");
        }
    }

    return failed;
}

inline bool DeviceData::validate(distributed::MeshDevice::IDevice* device) {
    bool failed = false;
    std::unordered_set<CoreCoord> validated_cores;

    for (const auto& [core, bank_device_data] : this->all_data) {
        for (const auto& [bank, one_core_data] : bank_device_data) {
            if (one_core_data.data.empty()) {
                continue;
            }

            const uint32_t start_index = (this->base_result_data_addr[static_cast<int>(one_core_data.core_type)] -
                                          this->base_data_addr[static_cast<int>(one_core_data.core_type)]) /
                                         sizeof(uint32_t);
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

inline void DeviceData::overflow_check(distributed::MeshDevice::IDevice* device) {
    for (const auto& [core, bank_device_data] : this->all_data) {
        for (const auto& [bank, one_core_data] : bank_device_data) {
            if (one_core_data.core_type == tt::CoreType::WORKER) {
                TT_FATAL(
                    one_core_data.data.size() * sizeof(uint32_t) +
                            base_data_addr[static_cast<int>(tt::CoreType::WORKER)] <=
                        device->l1_size_per_core(),
                    "Test overflowed L1 memory");
            } else if (one_core_data.core_type == tt::CoreType::PCIE) {
                TT_FATAL(
                    one_core_data.data.size() * sizeof(uint32_t) <= hugepage_issue_buffer_size_,
                    "Test overflowed PCIE memory");
            } else if (one_core_data.core_type == tt::CoreType::DRAM) {
                // TODO
            }
        }
    }
}

// Forward declare the accessor
// This accessor class provides test access to private members
// of FDMeshCommandQueue
class FDMeshCQTestAccessor {
public:
    static tt_metal::SystemMemoryManager& sysmem(tt_metal::distributed::FDMeshCommandQueue& cq) {
        return cq.reference_sysmem_manager();
    }
};

namespace DeviceDataUpdater {

// Update DeviceData for linear write
// Mirrors a dispatcher linear-write transaction into the DeviceData expectation model
// Takes provided payload and pushes exactly those values into destination worker_range
inline void update_linear_write(
    const std::vector<uint32_t>& payload, DeviceData& device_data, const CoreRange& worker_range, bool is_mcast) {
    // Update expected device_data
    if (is_mcast) {
        for (const uint32_t datum : payload) {
            device_data.push_range(worker_range, datum, true);
        }
    } else {
        for (const uint32_t datum : payload) {
            device_data.push_one(worker_range.start_coord, 0, datum);
        }
    }
    // Relevel for next multicast command
    if (is_mcast) {
        device_data.relevel(tt::CoreType::WORKER);
    }
}

// Update DeviceData for paged write
// Tracks page-wise writes so validate() can check DRAM/L1 bank contents after the test runs
inline void update_paged_write(
    const std::vector<uint32_t>& payload,
    DeviceData& device_data,
    const CoreCoord& bank_core,
    uint32_t bank_id,
    uint32_t page_alignment) {
    for (const uint32_t datum : payload) {
        device_data.push_one(bank_core, bank_id, datum);
    }
    device_data.pad(bank_core, bank_id, page_alignment);
}

// Update DeviceData for packed write
// Applies packed write payloads to every selected worker
inline void update_packed_write(
    const std::vector<uint32_t>& payload,
    DeviceData& device_data,
    const std::vector<CoreCoord>& worker_cores,
    uint32_t l1_alignment) {
    // Update expected device_data for all cores
    for (const auto& core : worker_cores) {
        for (const uint32_t datum : payload) {
            device_data.push_one(core, 0, datum);
        }
        device_data.pad(core, 0, l1_alignment);
    }

    // Re-relevel for next command
    device_data.relevel(tt::CoreType::WORKER);
}
}  // namespace DeviceDataUpdater

// Host-side helpers used by tests to emit the same CQ commands
// that dispatcher code emits. This namespace replicates the production code's command generation logic
// for testing purposes.
namespace CommandBuilder {

// Emits a single linear write, optionally multicast, with inline data
template <bool flush_prefetch, bool inline_data>
HostMemDeviceCommand build_linear_write_command(
    const std::vector<uint32_t>& payload,
    const CoreRange& worker_range,
    bool is_mcast,
    uint32_t noc_xy,
    uint32_t addr,
    uint32_t xfer_size_bytes) {
    // Calculate the command size using DeviceCommandCalculator
    // Pre-calculate the exact size to allocate correct amount of memory in HostMemDeviceCommand buffer
    DeviceCommandCalculator cmd_calc;
    cmd_calc.add_dispatch_write_linear<flush_prefetch, inline_data>(xfer_size_bytes);
    const uint32_t command_size_bytes = cmd_calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(command_size_bytes);

    // Add the dispatch write linear command
    cmd.add_dispatch_write_linear<flush_prefetch, inline_data>(
        is_mcast ? worker_range.size() : 0,  // num_mcast_dests
        noc_xy,                              // NOC coordinates
        addr,                                // destination address
        xfer_size_bytes,                     // data size
        payload.data()                       // payload data
    );

    return cmd;
}

// Emits a paged write (DRAM or L1) chunk
// payload is already stitched together for all pages in the chunk
template <bool inline_data>
HostMemDeviceCommand build_paged_write_command(
    const std::vector<uint32_t>& payload,
    uint32_t base_addr,
    uint32_t page_size_bytes,
    uint32_t pages_in_chunk,
    uint16_t start_page_cmd,
    bool is_dram) {
    // Calculate the command size
    DeviceCommandCalculator cmd_calc;
    cmd_calc.add_dispatch_write_paged<inline_data>(page_size_bytes, pages_in_chunk);
    const uint32_t command_size_bytes = cmd_calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(command_size_bytes);

    // Add the dispatch write paged command
    cmd.add_dispatch_write_paged<inline_data>(
        true,                           // flush_prefetch (inline data)
        static_cast<uint8_t>(is_dram),  // is_dram
        start_page_cmd,                 // start_page
        base_addr,                      // base_addr
        page_size_bytes,                // page_size
        pages_in_chunk,                 // pages
        payload.data()                  // payload for this chunk
    );

    return cmd;
}

// Serializes a packed-unicast command including sub-command table
// and optional replicated payloads when stride is enabled
inline HostMemDeviceCommand build_packed_write_command(
    const std::vector<uint32_t>& payload,
    const std::vector<CQDispatchWritePackedUnicastSubCmd>& sub_cmds,
    uint32_t common_addr,
    uint32_t l1_alignment,
    uint32_t packed_write_max_unicast_sub_cmds,
    bool no_stride) {
    const uint32_t num_sub_cmds = static_cast<uint32_t>(sub_cmds.size());
    const uint32_t sub_cmds_bytes = tt::align(num_sub_cmds * sizeof(CQDispatchWritePackedUnicastSubCmd), l1_alignment);
    uint32_t num_data_copies = no_stride ? 1u : static_cast<uint32_t>(num_sub_cmds);

    // Pre-calculate all sizes needed
    const uint32_t payload_size_bytes = payload.size() * sizeof(uint32_t);
    const uint32_t data_bytes = num_data_copies * tt::align(payload_size_bytes, l1_alignment);
    const uint32_t payload_bytes = tt::align(sizeof(CQDispatchCmd) + sub_cmds_bytes, l1_alignment) + data_bytes;

    // Calculate the command size
    DeviceCommandCalculator cmd_calc;
    cmd_calc.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        num_sub_cmds,        // num_sub_cmds
        payload_size_bytes,  // packed_data_sizeB
        packed_write_max_unicast_sub_cmds,
        no_stride  // no_stride
    );
    const uint32_t command_size_bytes = cmd_calc.write_offset_bytes();

    // Create the HostMemDeviceCommand with pre-calculated size
    HostMemDeviceCommand cmd(command_size_bytes);

    // Build data_collection pointing to the payload
    std::vector<std::pair<const void*, uint32_t>> data_collection;
    const void* payload_data = payload.data();

    if (no_stride) {
        data_collection.emplace_back(payload_data, payload_size_bytes);
    } else {
        data_collection.resize(num_sub_cmds, {payload_data, payload_size_bytes});
    }

    // Add the dispatch write packed command
    cmd.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        0,                                          // type
        num_sub_cmds,                               // num_sub_cmds
        common_addr,                                // common_addr
        static_cast<uint16_t>(payload_size_bytes),  // packed_data_sizeB
        payload_bytes,                              // payload_sizeB
        sub_cmds,                                   // sub_cmds
        data_collection,                            // data_collection
        packed_write_max_unicast_sub_cmds,          // packed_write_max_unicast_sub_cmds
        0,                                          // offset_idx
        no_stride);                                 // no_stride

    return cmd;
}

}  // namespace CommandBuilder

// DispatchPayloadGenerator is used to generate payloads for the tests
class DispatchPayloadGenerator {
public:
    struct Config {
        bool use_coherent_data = false;
        uint32_t coherent_start_val = COHERENT_DATA_START_VALUE;
        uint32_t seed = 0;

        // Perf test configuration
        bool perf_test = false;
        uint32_t min_xfer_size_bytes = 0;
        uint32_t max_xfer_size_bytes = 0;
    };

    DispatchPayloadGenerator(const Config& cfg) : config_(cfg), coherent_count_(cfg.coherent_start_val) {
        if (config_.seed == 0) {
            std::random_device rd;
            rng_.seed(rd());
        } else {
            rng_.seed(config_.seed);
        }
    }

    // Getter to log the seed used
    uint32_t get_seed() const { return config_.seed; }

    // Helper for random number generation in a range [min, max]
    template <typename T>
    T get_rand(T min, T max) {
        static_assert(std::is_integral_v<T>, "T must be an integral type");
        std::uniform_int_distribution<T> dist(min, max);
        return dist(rng_);
    }

    // Helper for generating a random boolean (replaces std::rand() % 2)
    bool get_rand_bool() { return (bool_dist(rng_) != 0); }

    // Generates either deterministic (coherent) or random 32-bit words
    // for the requested byte count
    // In coherent mode, the counter is incremented so validation knows
    // the exact pattern
    std::vector<uint32_t> generate_payload(uint32_t xfer_size_bytes) {
        const uint32_t size_words = xfer_size_bytes / sizeof(uint32_t);
        std::vector<uint32_t> payload;
        payload.reserve(size_words);

        for (uint32_t i = 0; i < size_words; ++i) {
            const uint32_t datum = config_.use_coherent_data ? coherent_count_++ : uint32_dist(rng_);
            payload.push_back(datum);
        }

        return payload;
    }

    // Generate payload with page id
    // Pass page_id to use for coherent data generation
    std::vector<uint32_t> generate_payload_with_page_id(uint32_t page_size_words, uint32_t page_id) {
        std::vector<uint32_t> payload;
        payload.reserve(page_size_words);

        for (uint32_t i = 0; i < page_size_words; ++i) {
            const uint32_t datum = config_.use_coherent_data
                                       ? (((page_id & 0xFF) << 24) | (coherent_count_++ & 0xFFFFFF))
                                       : uint32_dist(rng_);
            payload.push_back(datum);
        }

        return payload;
    }

    // Helper to generate payload data for a given core
    // Pass core_id to use for coherent data generation
    std::vector<uint32_t> generate_payload_with_core(
        const CoreCoord& core_id,  // Pass the core to use
        uint32_t xfer_size_bytes) {
        const uint32_t size_words = xfer_size_bytes / sizeof(uint32_t);
        std::vector<uint32_t> payload;
        payload.reserve(size_words);

        for (uint32_t i = 0; i < size_words; ++i) {
            const uint32_t datum =
                config_.use_coherent_data
                    ? (((core_id.x & 0xFF) << 16) | ((core_id.y & 0xFF) << 24) | (coherent_count_++ & 0xFFFF))
                    : uint32_dist(rng_);
            payload.push_back(datum);
        }

        return payload;
    }

    // Chooses a payload size in 16B units, respecting perf mode clamps and remaining budget
    uint32_t get_random_size(uint32_t max_allowed, uint32_t bytes_per_unit, uint32_t remaining_bytes) {
        // Generate random transfer size
        std::uniform_int_distribution<uint32_t> dist(1, max_allowed);
        uint32_t xfer_size_16B = dist(rng_);
        uint32_t xfer_size_bytes = xfer_size_16B * bytes_per_unit;  // Convert 16B units to bytes

        // Clamp to remaining bytes
        xfer_size_bytes = std::min(xfer_size_bytes, remaining_bytes);

        // Apply perf_test_ constraints if enabled
        if (config_.perf_test) {
            xfer_size_bytes = std::clamp(xfer_size_bytes, config_.min_xfer_size_bytes, config_.max_xfer_size_bytes);
        }

        return xfer_size_bytes;
    }

private:
    // Start offset to avoid 0x0 which matches DRAM prefill
    static constexpr uint32_t COHERENT_DATA_START_VALUE = 0x100;
    Config config_{};
    uint32_t coherent_count_ = COHERENT_DATA_START_VALUE;

    // Random number generation
    std::mt19937 rng_;
    // Distributions for random number generation
    std::uniform_int_distribution<int> bool_dist{0, 1};
    std::uniform_int_distribution<uint32_t> uint32_dist{
        std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max()};
};

namespace PackedWriteUtils {
// Build subcmds once - reused for all commands
inline std::vector<CQDispatchWritePackedUnicastSubCmd> build_sub_cmds(
    distributed::MeshDevice::IDevice* device,
    const std::vector<CoreCoord>& worker_cores,
    tt::tt_metal::NOC downstream_noc) {
    std::vector<CQDispatchWritePackedUnicastSubCmd> sub_cmds;
    sub_cmds.reserve(worker_cores.size());
    for (const auto& core : worker_cores) {
        const CoreCoord virtual_core = device->virtual_core_from_logical_core(core, CoreType::WORKER);
        CQDispatchWritePackedUnicastSubCmd sub_cmd{};
        sub_cmd.noc_xy_addr = device->get_noc_unicast_encoding(downstream_noc, virtual_core);
        sub_cmds.push_back(sub_cmd);
    }
    return sub_cmds;
}

// Clamp xfer_size to fit within max_fetch_bytes_
inline uint32_t clamp_to_max_fetch(
    uint32_t max_fetch_bytes,
    uint32_t xfer_size_bytes,
    uint32_t num_sub_cmds,
    uint32_t packed_write_max_unicast_sub_cmds,
    bool no_stride,
    uint32_t l1_alignment) {
    // Calculate the command size
    DeviceCommandCalculator cmd_calc;
    cmd_calc.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
        num_sub_cmds,     // num_sub_cmds
        xfer_size_bytes,  // packed_data_sizeB
        packed_write_max_unicast_sub_cmds,
        no_stride  // no_stride
    );
    uint32_t command_size_bytes = cmd_calc.write_offset_bytes();

    // If the command size is less than max_fetch_bytes_, return the transfer size
    if (command_size_bytes <= max_fetch_bytes) {
        return xfer_size_bytes;
    }

    // Else, linearly decrement by alignment until it fits
    uint32_t result = xfer_size_bytes;
    while (result > 0 && command_size_bytes > max_fetch_bytes) {
        result -= l1_alignment;
        cmd_calc.add_dispatch_write_packed<CQDispatchWritePackedUnicastSubCmd>(
            num_sub_cmds,  // num_sub_cmds
            result,        // packed_data_sizeB
            packed_write_max_unicast_sub_cmds,
            no_stride  // no_stride
        );
        command_size_bytes = cmd_calc.write_offset_bytes();
    }

    return result;
}
}  // namespace PackedWriteUtils

// BaseTestFixture forms the basis for prefetch and dispatcher tests.
// Inherits from GenericMeshDeviceFixture which determines the mesh device type automatically
class BaseTestFixture : public tt_metal::GenericMeshDeviceFixture {
protected:
    // DispatchPayloadGenerator for generating payloads
    std::unique_ptr<DispatchPayloadGenerator> payload_generator_;

    // Common constants
    static constexpr CoreCoord default_worker_start = {0, 1};
    static constexpr uint32_t bytes_per_16B_unit = 16;  // conversion factor to convert 16-byte "chunks" to bytes
    static constexpr uint32_t wait_completion_timeout = 10000;  // wait in milliseconds

    // Common setup for all dispatch tests
    // Provides shared wiring for mesh device access,
    // and command-buffer helpers so derived fixtures
    // only implement workload-specific planning
    tt_metal::distributed::FDMeshCommandQueue* fdcq_ = nullptr;
    tt_metal::SystemMemoryManager* mgr_ = nullptr;
    distributed::MeshDevice::IDevice* device_ = nullptr;

    // HW properties
    uint32_t host_alignment_ = 0;
    uint32_t max_fetch_bytes_ = 0;

    // Knobs
    uint32_t dispatch_buffer_page_size_ = 0;
    bool send_to_all_ = false;

    // Test Config defaults
    DispatchTestConfig cfg_;

    void SetUp() override {
        if (!validate_dispatch_mode()) {
            GTEST_SKIP();
        }
        tt_metal::GenericMeshDeviceFixture::SetUp();

        // Setup Config
        DispatchPayloadGenerator::Config pgcfg;
        pgcfg.use_coherent_data = cfg_.use_coherent_data;
        pgcfg.perf_test = cfg_.perf_test;
        pgcfg.min_xfer_size_bytes = cfg_.min_xfer_size_bytes;
        pgcfg.max_xfer_size_bytes = cfg_.max_xfer_size_bytes;

        // Handle Seeding
        std::random_device rd;
        pgcfg.seed = rd();

        // Initialize Generator
        payload_generator_ = std::make_unique<DispatchPayloadGenerator>(pgcfg);
        log_info(tt::LogTest, "Random seed set to {}", pgcfg.seed);

        // These are used for test logic (loops, alignment, etc.) rather than generation
        dispatch_buffer_page_size_ = cfg_.dispatch_buffer_page_size;
        send_to_all_ = cfg_.send_to_all;

        // Initialize common pointers
        auto& mcq = mesh_device_->mesh_command_queue();
        fdcq_ = &dynamic_cast<distributed::FDMeshCommandQueue&>(mcq);
        // mgr_ = &FDMeshCQTestAccessor::sysmem(*fdcq_);
        device_ = mesh_device_->get_devices()[0];
        mgr_ = &device_->sysmem_manager();  // Use Chip 0's SystemMemoryManager

        // Initialize common HW properties
        host_alignment_ = tt_metal::MetalContext::instance().hal().get_alignment(tt_metal::HalMemType::HOST);
        max_fetch_bytes_ = tt_metal::MetalContext::instance().dispatch_mem_map().max_prefetch_command_size();
    }

    bool validate_dispatch_mode() {
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            log_info(tt::LogTest, "This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
            return false;
        }
        return true;
    }

    // Helper function that polls completion queue until expected data is written into by dispatcher
    // Without this, we can fail validation as there can a be an occasional race condition
    // TODO: Alternatively, could we use tt_driver_atomics::mfence before validation?
    void wait_for_completion_queue_bytes(uint32_t total_expected_cq_payload, uint32_t timeout_ms = 0) {
        std::atomic<bool> exit_condition{false};
        const auto start = std::chrono::steady_clock::now();
        uint32_t avail = 0;
        while (avail < total_expected_cq_payload) {
            const uint32_t completion_queue_write_ptr_and_toggle =
                mgr_->completion_queue_wait_front(fdcq_->id(), exit_condition);
            const uint32_t completion_q_write_ptr = (completion_queue_write_ptr_and_toggle & 0x7fffffff) << 4;
            const uint32_t completion_q_write_toggle = completion_queue_write_ptr_and_toggle >> (31);
            const uint32_t completion_q_read_ptr = mgr_->get_completion_queue_read_ptr(fdcq_->id());
            const uint32_t completion_q_read_toggle = mgr_->get_completion_queue_read_toggle(fdcq_->id());
            const uint32_t limit = mgr_->get_completion_queue_limit(fdcq_->id());  // offset of end, in bytes

            if (completion_q_write_toggle == completion_q_read_toggle) {
                avail = (completion_q_write_ptr > completion_q_read_ptr)
                            ? completion_q_write_ptr - completion_q_read_ptr
                            : 0u;
            } else {
                avail = (limit - completion_q_read_ptr) + completion_q_write_ptr;
            }

            const auto elapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            if (elapsed > timeout_ms) {
                exit_condition.store(true);
                TT_FATAL(
                    false,
                    "CQ wait timed out after {} ms (needed {} bytes, had {})",
                    elapsed,
                    total_expected_cq_payload,
                    avail);
            }
        }

        log_info(
            LogTest, "written in completion queue {} B vs expected amount {} B: ", avail, total_expected_cq_payload);
    }

    // Helper function to report performance
    void report_performance(
        DeviceData& device_data,
        size_t num_cores_to_log,
        std::chrono::duration<double> elapsed,
        uint32_t num_iterations) {
        const float total_words = static_cast<float>(device_data.size()) * num_iterations;
        const float bw_gbps = total_words * sizeof(uint32_t) / (elapsed.count() * 1024.0 * 1024.0 * 1024.0);

        log_info(
            LogTest,
            "BW: {:.3f} GB/s (total_words: {:.0f}, size: {:.2f} MB, iterations: {}, cores: {})",
            bw_gbps,
            total_words,
            total_words * sizeof(uint32_t) / (1024.0 * 1024.0),
            num_iterations,
            num_cores_to_log);
    }

    // Helper function to execute generated commands
    // Orchestrates the command buffer reservation, writing, and submission
    virtual void execute_generated_commands(
        const std::vector<HostMemDeviceCommand>& commands_per_iteration,
        DeviceData& device_data,
        size_t num_cores_to_log,
        uint32_t num_iterations,
        bool wait_for_completion = true,
        bool wait_for_host_writes = false) {
        // PHASE 2: Calculate total command buffer size
        uint64_t per_iter_total = 0;
        for (const auto& cmd : commands_per_iteration) {
            per_iter_total += cmd.size_bytes();
        }

        const uint64_t total_cmd_bytes = num_iterations * per_iter_total;
        log_info(tt::LogTest, "Total command bytes: {}", total_cmd_bytes);

        // PHASE 3: Reserve and write commands
        // Reserve a continuous block in the system memory issue queue for all commands across all iterations
        // This memory is mapped and visible to the device's prefetcher kernel
        void* cmd_buffer_base = mgr_->issue_queue_reserve(total_cmd_bytes, fdcq_->id());

        // Use DeviceCommand helper (HugepageDeviceCommand) to write to the issue queue memory
        // Two stage command construction:
        // 1. Staging (HostMemDeviceCommand):
        //    - commands_per_iteration: vector of HostMemDeviceCommand objects which holds a deep copy
        //      of each command header + payload assembled offline without holding issue queue space
        // 2. Writing (HugepageDeviceCommand):
        //    - wraps a pointer that points directly to the issue queue memory (cmd_buffer_base)
        //    - the loop below copies staged commands into the issue queue memory
        HugepageDeviceCommand dc(cmd_buffer_base, total_cmd_bytes);

        // Store the size of each command entry (per-chunk)
        std::vector<uint32_t> entry_sizes;

        // Calculate the total number of entries to reserve
        size_t total_num_entries = num_iterations * commands_per_iteration.size();
        entry_sizes.reserve(total_num_entries);

        // Write commands to the command buffer for all iterations
        for (uint32_t iter = 0; iter < num_iterations; ++iter) {
            for (const auto& cmd : commands_per_iteration) {
                // Add the command data to the command buffer
                dc.add_data(cmd.data(), cmd.size_bytes(), cmd.size_bytes());
                entry_sizes.push_back(cmd.size_bytes());
            }
        }

        // Add barrier wait command after all commands across all iterations
        // Helpful to ensure all commands are completed flush before terminating
        // Without this, there can be occasional timeouts in MetalContext::initialize_and_launch_firmware()
        // between test fixtures possibly because the previously issued commands
        // are not completed before next firmware launch
        DeviceCommandCalculator cmd_calc;
        cmd_calc.add_dispatch_wait();
        HostMemDeviceCommand cmd(cmd_calc.write_offset_bytes());
        cmd.add_dispatch_wait(CQ_DISPATCH_CMD_WAIT_FLAG_BARRIER, 0, 0, 0);
        dc.add_data(cmd.data(), cmd.size_bytes(), cmd.size_bytes());
        entry_sizes.push_back(cmd.size_bytes());

        // Verifies destination memory bounds
        device_data.overflow_check(device_);

        // PHASE 4: Submit and execute commands
        // Update host-side write pointer
        // Tells the SystemMemoryManager that valid data exists in the issue queue upto this point
        mgr_->issue_queue_push_back(dc.write_offset_bytes(), fdcq_->id());

        // Write the commands to the device-side fetch queue
        // This updates the read/write pointers in the Device's L1 memory, effectively
        // Signals to the prefetcher kernel that new commands are available to fetch
        const auto start = std::chrono::steady_clock::now();
        for (const uint32_t sz : entry_sizes) {
            mgr_->fetch_queue_reserve_back(fdcq_->id());
            mgr_->fetch_queue_write(sz, fdcq_->id());
        }

        // Wait for completion of the issued commands
        if (wait_for_completion) {
            distributed::Finish(mesh_device_->mesh_command_queue());
        } else if (wait_for_host_writes) {
            uint32_t total_expected_cq_payload = device_data.size() * sizeof(uint32_t);
            // For host writes, wait until expected data is written into completion queue by dispatcher
            wait_for_completion_queue_bytes(total_expected_cq_payload, wait_completion_timeout);
        }
        const auto end = std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed = end - start;
        log_info(tt::LogTest, "Ran in {:f} ms (for {} iterations)", elapsed.count() * 1000.0, num_iterations);

        // Validate results
        const bool pass = device_data.validate(device_);
        EXPECT_TRUE(pass) << "Dispatcher test failed validation";

        // Report performance
        if (pass) {
            report_performance(device_data, num_cores_to_log, elapsed, num_iterations);
        }
    }
};
}  // namespace tt::tt_metal::tt_dispatch_tests::Common
