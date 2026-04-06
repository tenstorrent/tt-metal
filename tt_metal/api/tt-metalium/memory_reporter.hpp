// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tt::tt_metal {
enum class BufferType;
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

/*
MemoryBlockTable is a list of memory blocks in the following format:
[{"blockID": "0", "address": "0", "size": "0", "prevID": "0", "nextID": "0", "allocated": true}]
address: bytes
size: bytes
*/
using MemoryBlockTable = std::vector<std::unordered_map<std::string, std::string>>;

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
 * | device        | The device for which memory stats will be dumped. | const IDevice*  | | True     |
 * */
void DumpDeviceMemoryState(const IDevice* device, const std::string& prefix = "");

/**
 * Populates MemoryView for BufferType [dram, l1, l1 small, trace]. Used when storing to disk is not an option.
 *
 * num_banks: total number of BufferType banks for given device
 * total_bytes_per_bank: total allocatable size per bank of BufferType in bytes
 * total_bytes_allocated_per_bank: currently allocated size per bank of BufferType in bytes
 * total_bytes_free_per_bank: total free size per bank of BufferType in bytes
 * largest_contiguous_bytes_free_per_bank: largest contiguous free block of BufferType in bytes
 * block_table: list of all blocks in BufferType (blockID, address, size, prevID, nextID, allocated)
 *
 * | Argument      | Description                                       | Type            | Valid Range | Required |
 * |---------------|---------------------------------------------------|-----------------|--------------------------------------------------------|----------|
 * | device        | The device for which memory stats will be dumped. | const IDevice *  | | True     |
 * | buffer_type   | The type of buffer to populate the memory view.   | const BufferType& | | True     |
 * */
MemoryView GetMemoryView(const IDevice* device, const BufferType& buffer_type);

struct MemoryView {
    std::uint64_t num_banks = 0;
    size_t total_bytes_per_bank = 0;
    size_t total_bytes_allocated_per_bank = 0;
    size_t total_bytes_free_per_bank = 0;
    size_t largest_contiguous_bytes_free_per_bank = 0;
    MemoryBlockTable block_table;
};

}  // namespace detail
}  // namespace tt::tt_metal
