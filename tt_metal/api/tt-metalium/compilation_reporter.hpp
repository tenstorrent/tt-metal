// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <mutex>

#include "kernel.hpp"
#include "program_impl.hpp"

using std::mutex;
using std::unique_lock;

namespace tt::tt_metal {

namespace detail {

/**
 * Enable generation of reports for compilation statistics.
 * Two reports are generated in .reports/tt_metal:
 *  - `compile_program_summary.csv` has a table with an entry for each program that indicates number of Compute and Data
 * movement CreateKernel API calls, and number of kernel compilation cache hits and misses
 *  - `compile_program.csv` expands on the summary report by including a per program table with an entry for each
 * kernel, indicating the cores the kernel is placed on, kernel attributes and whether there was a cache hit or miss
 * when compiling the kernel.
 *
 * Return value: void
 *
 */
void EnableCompilationReports();

/**
 * Disable generation compilation statistics reports.
 *
 * Return value: void
 *
 */
void DisableCompilationReports();

class CompilationReporter {
public:
    CompilationReporter& operator=(const CompilationReporter&) = delete;
    CompilationReporter& operator=(CompilationReporter&& other) noexcept = delete;
    CompilationReporter(const CompilationReporter&) = delete;
    CompilationReporter(CompilationReporter&& other) noexcept = delete;

    void add_kernel_compile_stats(
        uint64_t program_id, const std::shared_ptr<Kernel>& kernel, bool cache_hit, size_t kernel_hash);

    void flush_program_entry(
        uint64_t program_id,
        size_t num_kernels,
        std::function<std::shared_ptr<Kernel>(size_t)> get_kernel,
        bool persistent_compilation_cache_enabled);
    static CompilationReporter& inst();
    static void toggle(bool state);
    static bool enabled();

private:
    CompilationReporter();

    ~CompilationReporter();

    void init_reports();

    struct cache_counters {
        int misses = 0;
        int hits = 0;
    };

    static std::atomic<bool> enable_compile_reports_;
    std::mutex mutex_;
    size_t total_num_compile_programs_ = 0;
    std::unordered_map<uint64_t, cache_counters> program_id_to_cache_hit_counter_;
    std::unordered_map<uint64_t, std::vector<string>> program_id_to_kernel_stats_;
    std::ofstream detailed_report_;
    std::ofstream summary_report_;
};

}  // namespace detail

}  // namespace tt::tt_metal
