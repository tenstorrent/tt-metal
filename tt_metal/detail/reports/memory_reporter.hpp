// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <atomic>
#include <fstream>
#include <string>
namespace tt::tt_metal {
inline namespace v0 {

class Program;
class Device;

}  // namespace v0
namespace detail {

/**
 * Enable generation of reports for memory allocation statistics.
 * Three reports are generated in .reports/tt_metal:
 *  - `prorgam_l1_usage_summary.csv` has a table with an entry for each program indicating the minimum largest free L1 block and size of largest L1 buffer that can be interleaved across available free L1 blocks
 *  - `program_memory_usage_summary.csv` for each program there is an entry indicating total allocatable, allocated, free, and largest free block sizes for each DRAM and L1 bank
 *  - `program_detailed_memory_usage.csv` expands on the memory usage summary report by including each memory block address, size, and allocation status
 *
 * Note: These reports are generated when program is being compiled so any DRAM or L1 buffer created after program compilation will not be captured! To dump
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
 *  - `l1_usage_summary.csv` has a table with an entry for each program indicating the minimum largest free L1 block and size of largest L1 buffer that can be interleaved across available free L1 blocks
 *  - `memory_usage_summary.csv` for each program there is an entry indicating total allocatable, allocated, free, and largest free block sizes for each DRAM and L1 bank
 *  - `detailed_memory_usage.csv` expands on the memory usage summary report by including each memory block address, size, and allocation status
 *
 * Return value: void
 *
 * | Argument      | Description                                       | Type            | Valid Range                                            | Required |
 * |---------------|---------------------------------------------------|-----------------|--------------------------------------------------------|----------|
 * | device        | The device for which memory stats will be dumped. | const Device *  |                                                        | True     |
 * */
void DumpDeviceMemoryState(const Device *device, std::string prefix="");

class MemoryReporter {
   public:
    MemoryReporter& operator=(const MemoryReporter&) = delete;
    MemoryReporter& operator=(MemoryReporter&& other) noexcept = delete;
    MemoryReporter(const MemoryReporter&) = delete;
    MemoryReporter(MemoryReporter&& other) noexcept = delete;

    void flush_program_memory_usage(const Program &program, const Device *device);

    void dump_memory_usage_state(const Device *device, std::string prefix="") const;

    static void toggle(bool state);
    static MemoryReporter& inst();
    static bool enabled();
   private:
    MemoryReporter(){};
    ~MemoryReporter();
    void init_reports();
    static std::atomic<bool> is_enabled_;
    std::ofstream program_l1_usage_summary_report_;
    std::ofstream program_memory_usage_summary_report_;
    std::ofstream program_detailed_memory_usage_report_;
};

}  // namespace detail
}  // namespace tt::tt_metal
