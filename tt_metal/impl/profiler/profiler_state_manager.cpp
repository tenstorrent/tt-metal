// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <thread>
#include <vector>
#include "profiler_state_manager.hpp"
#include <tt_stl/assert.hpp>
#include "hostdevcommon/profiler_common.h"
#include "context/metal_context.hpp"
#include "math.hpp"

namespace tt {

namespace tt_metal {

constexpr static uint32_t DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT = 1000;
constexpr static uint32_t DEFAULT_PROFILER_L1_PROGRAM_MIN_OPTIONAL_MARKER_COUNT = 2;

uint32_t get_profiler_dram_bank_size_per_risc_bytes(llrt::RunTimeOptions& rtoptions) {
    std::optional<uint32_t> profiler_program_support_count = rtoptions.get_profiler_program_support_count();
    const bool do_profiler_sum = rtoptions.get_profiler_sum();

    if (!profiler_program_support_count.has_value()) {
        profiler_program_support_count = DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT;
    }

    const uint32_t profiler_l1_program_min_optional_marker_count =
        do_profiler_sum ? DEFAULT_PROFILER_L1_PROGRAM_MIN_OPTIONAL_MARKER_COUNT : 0;
    const uint32_t dram_bank_size_per_risc_bytes_single_program =
        kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE *
        (kernel_profiler::PROFILER_L1_PROGRAM_ID_COUNT + kernel_profiler::PROFILER_L1_GUARANTEED_MARKER_COUNT +
         profiler_l1_program_min_optional_marker_count) *
        sizeof(uint32_t);

    if (profiler_program_support_count <=
        ((kernel_profiler::PROFILER_L1_BUFFER_SIZE) / dram_bank_size_per_risc_bytes_single_program)) {
        const uint32_t old_profiler_program_support_count = profiler_program_support_count.value();
        profiler_program_support_count =
            div_up(kernel_profiler::PROFILER_L1_BUFFER_SIZE, dram_bank_size_per_risc_bytes_single_program);
        log_warning(
            tt::LogMetal,
            "Profiler program support count must be >= {}. Increasing program support count from {} to {}.",
            profiler_program_support_count.value(),
            old_profiler_program_support_count,
            profiler_program_support_count.value());
    }

    const uint32_t dram_bank_size_per_risc_bytes =
        dram_bank_size_per_risc_bytes_single_program * profiler_program_support_count.value();

    rtoptions.set_profiler_program_support_count(profiler_program_support_count.value());

    TT_ASSERT(dram_bank_size_per_risc_bytes > kernel_profiler::PROFILER_L1_BUFFER_SIZE);
    return dram_bank_size_per_risc_bytes;
}

uint32_t get_profiler_dram_bank_size_per_risc_bytes() {
    llrt::RunTimeOptions& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    return get_profiler_dram_bank_size_per_risc_bytes(rtoptions);
}

ProfilerStateManager::ProfilerStateManager() : do_sync_on_close(true) {}

void ProfilerStateManager::cleanup_device_profilers() {
    std::vector<std::thread> threads(this->device_profiler_map.size());

    uint32_t i = 0;
    for (auto it = this->device_profiler_map.begin(); it != this->device_profiler_map.end(); ++it) {
        threads[i] = std::thread([it]() {
            DeviceProfiler& profiler = it->second;
            profiler.dumpDeviceResults();
            profiler.destroyTracyContexts();
        });
        i++;
    }

    for (auto& thread : threads) {
        thread.join();
    }

    this->device_profiler_map.clear();
}

uint32_t ProfilerStateManager::calculate_optimal_num_threads_for_device_profiler_thread_pool() const {
    const uint32_t num_threads_available = std::thread::hardware_concurrency();

    if (num_threads_available == 0 || this->device_profiler_map.size() > num_threads_available) {
        // If hardware_concurrency() is unable to determine the number of threads supported by the CPU, or the
        // number of device profilers is greater than the max number of threads, return 2
        return 2;
    } else {
        // Otherwise, return min(8, number of threads available / number of device profilers)
        // Empirically, 8 threads per device profiler seems to result in optimal performance
        return std::min(8U, static_cast<uint32_t>(num_threads_available / this->device_profiler_map.size()));
    }
}

void ProfilerStateManager::mark_trace_begin(ChipId device_id, uint32_t trace_id) {
    TT_ASSERT(this->device_profiler_map.find(device_id) != this->device_profiler_map.end());
    DeviceProfiler& device_profiler = this->device_profiler_map.at(device_id);
    device_profiler.markTraceBegin(trace_id);
}

void ProfilerStateManager::mark_trace_end(ChipId device_id, uint32_t trace_id) {
    TT_ASSERT(this->device_profiler_map.find(device_id) != this->device_profiler_map.end());
    DeviceProfiler& device_profiler = this->device_profiler_map.at(device_id);
    device_profiler.markTraceEnd(trace_id);
}

void ProfilerStateManager::mark_trace_replay(ChipId device_id, uint32_t trace_id) {
    TT_ASSERT(this->device_profiler_map.find(device_id) != this->device_profiler_map.end());
    DeviceProfiler& device_profiler = this->device_profiler_map.at(device_id);
    device_profiler.markTraceReplay(trace_id);
}

void ProfilerStateManager::add_runtime_id_to_trace(ChipId device_id, uint32_t trace_id, uint32_t runtime_id) {
    TT_ASSERT(this->device_profiler_map.find(device_id) != this->device_profiler_map.end());
    DeviceProfiler& device_profiler = this->device_profiler_map.at(device_id);
    device_profiler.addRuntimeIdToTrace(trace_id, runtime_id);
}

}  // namespace tt_metal

}  // namespace tt
