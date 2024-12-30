// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <atomic>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt::tt_metal {
inline namespace v0 {

class Program;
class Device;

}  // namespace v0
namespace detail {
struct MemoryView;

/**
 * Enable generation of reports for memory allocation statistics.
 * Three reports are generated in .reports/tt_metal:
 *  - `prorgam_l1_usage_summary.csv` has a table with an entry for each program indicating the minimum largest free L1
 * block and size of largest L1 buffer that can be interleaved across available free L1 blocks
 *  - `program_memory_usage_summary.csv` for each program there is an entry indicating total allocatable, allocated,
 * free, and largest free block sizes for each DRAM and L1 bank
 *  - `program_detailed_memory_usage.csv` expands on the memory usage summary report by including each memory block
 * address, size, and allocation status
 *
 * Note: These reports are generated when program is being compiled so any DRAM or L1 buffer created after program
 * compilation will not be captured! To dump
 *
 * Return value: void
 *
 */
void EnableMemoryReports();

/**
 * Disable generation of memory allocation statistics reports.
 *
 * Return value: void
 *
 */
void DisableMemoryReports();

/**
 * Generates reports to dump device memory state. Three reports are generated:
 *  - `l1_usage_summary.csv` has a table with an entry for each program indicating the minimum largest free L1 block and
 * size of largest L1 buffer that can be interleaved across available free L1 blocks
 *  - `memory_usage_summary.csv` for each program there is an entry indicating total allocatable, allocated, free, and
 * largest free block sizes for each DRAM and L1 bank
 *  - `detailed_memory_usage.csv` expands on the memory usage summary report by including each memory block address,
 * size, and allocation status
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type            | Valid Range | Required |
 * |---------------|---------------------------------------------------|-----------------|--------------------------------------------------------|----------|
 * | device        | The device for which memory stats will be dumped. | const Device *  | | True     |
 * */
void DumpDeviceMemoryState(const Device* device, const std::string& prefix = "");

/**
 * Populates MemoryView for DRAM. Used when storing to disk is not an option.
 *
 * num_banks: total number of dram banks for given device
 * total_allocatable_per_bank_size_bytes: total allocatable size per bank of dram in bytes
 * total_allocated_per_bank_size_bytes: currently allocated size per bank of dram in bytes
 * total_free_per_bank_size_bytes: total free size per bank of dram in bytes
 * total_allocatable_size_bytes: total allocatable size of dram in bytes
 * total_allocated_size_bytes: currently allocated size of dram in bytes
 * total_free_size_bytes: total free size of dram in bytes
 * largest_contiguous_free_block_per_bank_size_bytes: largest contiguous free block of dram in bytes
 * blockTable: list of all blocks in dram (blockID, address, size, prevID, nextID, allocated)
 *
 * std::vector<std::unordered_map<std::string, std::string>>: list of all blocks in dram (blockID, address, size,
 * prevID, nextID, allocated)
 *
 * | Argument      | Description                                       | Type            | Valid Range | Required |
 * |---------------|---------------------------------------------------|-----------------|--------------------------------------------------------|----------|
 * | device        | The device for which memory stats will be dumped. | const Device *  | | True     |
 * */
MemoryView GetDramMemoryView(const Device* device);

/**
 * Populates MemoryView for L1. Used when storing to disk is not an option.
 *
 * num_banks: total number of dram banks for given device
 * total_allocatable_per_bank_size_bytes: total allocatable size per bank of dram in bytes
 * total_allocated_per_bank_size_bytes: currently allocated size per bank of dram in bytes
 * total_free_per_bank_size_bytes: total free size per bank of dram in bytes
 * total_allocatable_size_bytes: total allocatable size of dram in bytes
 * total_allocated_size_bytes: currently allocated size of dram in bytes
 * total_free_size_bytes: total free size of dram in bytes
 * largest_contiguous_free_block_per_bank_size_bytes: largest contiguous free block of dram in bytes
 * blockTable: list of all blocks in dram (blockID, address, size, prevID, nextID, allocated)
 *
 * | Argument      | Description                                       | Type            | Valid Range | Required |
 * |---------------|---------------------------------------------------|-----------------|--------------------------------------------------------|----------|
 * | device        | The device for which memory stats will be dumped. | const Device *  | | True     |
 * */
MemoryView GetL1MemoryView(const Device* device);

struct MemoryView {
    std::uint64_t num_banks;
    size_t total_allocatable_per_bank_size_bytes;
    size_t total_allocated_per_bank_size_bytes;
    size_t total_free_per_bank_size_bytes;
    size_t total_allocatable_size_bytes;  // total_allocatable_per_bank_size_bytes * num_banks
    size_t total_allocated_size_bytes;    // total_allocated_per_bank_size_bytes * num_banks
    size_t total_free_size_bytes;         // total_free_per_bank_size_bytes * num_banks
    size_t largest_contiguous_free_block_per_bank_size_bytes;
    std::vector<std::unordered_map<std::string, std::string>> blockTable;
};

class MemoryReporter {
public:
    MemoryReporter& operator=(const MemoryReporter&) = delete;
    MemoryReporter& operator=(MemoryReporter&& other) noexcept = delete;
    MemoryReporter(const MemoryReporter&) = delete;
    MemoryReporter(MemoryReporter&& other) noexcept = delete;

    void flush_program_memory_usage(uint64_t program_id, const Device* device);

    void dump_memory_usage_state(const Device* device, const std::string& prefix = "") const;
    MemoryView get_dram_memory_view(const Device* device) const;
    MemoryView get_l1_memory_view(const Device* device) const;

    static void toggle(bool state);
    static MemoryReporter& inst();
    static bool enabled();

private:
    MemoryReporter() {};
    ~MemoryReporter();
    void init_reports();
    static std::atomic<bool> is_enabled_;
    std::ofstream program_l1_usage_summary_report_;
    std::ofstream program_memory_usage_summary_report_;
    std::ofstream program_detailed_memory_usage_report_;
};

}  // namespace detail
}  // namespace tt::tt_metal
