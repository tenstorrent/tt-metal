// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>
#include "profiler_state_manager.hpp"
#include <tt_stl/assert.hpp>
#include <impl/debug/noc_debugging.hpp>
#include "hostdev/profiler_common.h"
#include "context/metal_context.hpp"
#include "impl/context/metal_env_impl.hpp"
#include "math.hpp"
#include "tt_cluster.hpp"
#include <tt-metalium/device.hpp>

namespace tt::tt_metal {

constexpr static uint32_t DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT = 1000;
constexpr static uint32_t DEFAULT_PROFILER_L1_PROGRAM_MIN_OPTIONAL_MARKER_COUNT = 2;

namespace {

// Convert a wall-time margin into device ticks for the NOC-debug watermark.
uint64_t noc_debug_margin_to_ticks(const std::vector<IDevice*>& devices, std::chrono::milliseconds margin) {
    uint32_t aiclk_mhz = 0;
    for (auto* device : devices) {
        aiclk_mhz = std::max(
            aiclk_mhz, static_cast<uint32_t>(MetalContext::instance().get_cluster().get_device_aiclk(device->id())));
    }
    if (aiclk_mhz == 0) {
        aiclk_mhz = 1000;  // conservative nominal clock if the frequency is unavailable
    }
    // aiclk_mhz ticks per microsecond -> aiclk_mhz * 1000 ticks per millisecond.
    return static_cast<uint64_t>(margin.count()) * static_cast<uint64_t>(aiclk_mhz) * 1'000ULL;
}

}  // namespace

uint32_t get_profiler_dram_bank_size_per_risc_bytes(llrt::RunTimeOptions& rtoptions) {
    std::optional<uint32_t> profiler_program_support_count = rtoptions.get_profiler_program_support_count();
    const bool do_profiler_sum = rtoptions.get_profiler_sum();
    const bool debug_dump_enabled = rtoptions.get_experimental_noc_debug_dump_enabled();

    if (!profiler_program_support_count.has_value()) {
        profiler_program_support_count = DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT;
        if (debug_dump_enabled) {
            profiler_program_support_count = profiler_program_support_count.value() / 2;
            log_info(
                tt::LogMetal,
                "Device Debug Dump enabled: reducing profiler program support count to {} to maintain same DRAM usage",
                profiler_program_support_count.value());
        }
    }

    const uint32_t profiler_l1_program_min_optional_marker_count =
        do_profiler_sum ? DEFAULT_PROFILER_L1_PROGRAM_MIN_OPTIONAL_MARKER_COUNT : 0;
    uint32_t dram_bank_size_per_risc_bytes_single_program =
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

uint32_t get_profiler_dram_bank_size_for_hal_allocation(llrt::RunTimeOptions& rtoptions) {
    const uint32_t per_buffer_size = get_profiler_dram_bank_size_per_risc_bytes(rtoptions);
    const bool debug_dump_enabled = rtoptions.get_experimental_noc_debug_dump_enabled();

    // There are 2 DRAM buffers per risc when debug dump is enabled.
    // The size of each buffer returned by get_profiler_dram_bank_size_per_risc_bytes is half to maintain the same
    // total profiler size.
    if (debug_dump_enabled) {
        return per_buffer_size * 2;
    }
    return per_buffer_size;
}

ProfilerStateManager::ProfilerStateManager(MetalEnvImpl& env) : env_(env), do_sync_on_close(true) {}

void ProfilerStateManager::cleanup_device_profilers() {
    // This thread only exists when debug dump is enabled
    if (this->debug_dump_thread.joinable()) {
        this->stop_debug_dump_thread = true;
        this->stop_debug_dump_thread_cv.notify_all();
        this->debug_dump_thread.join();
    }
    std::vector<std::thread> threads(this->device_profiler_map.size());

    uint32_t i = 0;
    // NOLINTNEXTLINE(modernize-loop-convert)
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
    std::lock_guard<std::recursive_mutex> lock{this->device_profiler_map_mutex};
    const uint32_t num_threads_available = std::thread::hardware_concurrency();

    if (num_threads_available == 0 || this->device_profiler_map.size() > num_threads_available) {
        // If hardware_concurrency() is unable to determine the number of threads supported by the CPU, or the
        // number of device profilers is greater than the max number of threads, return 2
        return 2;
    }  // Otherwise, return min(8, number of threads available / number of device profilers)
    // Empirically, 8 threads per device profiler seems to result in optimal performance
    return std::min(8U, static_cast<uint32_t>(num_threads_available / this->device_profiler_map.size()));
}

void ProfilerStateManager::mark_trace_begin(ChipId device_id, uint32_t trace_id) {
    TT_ASSERT(this->device_profiler_map.contains(device_id));
    std::lock_guard<std::recursive_mutex> lock{this->device_profiler_map_mutex};
    DeviceProfiler& device_profiler = this->device_profiler_map.at(device_id);
    device_profiler.markTraceBegin(trace_id);
}

void ProfilerStateManager::mark_trace_end(ChipId device_id, uint32_t trace_id) {
    TT_ASSERT(this->device_profiler_map.contains(device_id));
    std::lock_guard<std::recursive_mutex> lock{this->device_profiler_map_mutex};
    DeviceProfiler& device_profiler = this->device_profiler_map.at(device_id);
    device_profiler.markTraceEnd(trace_id);
}

void ProfilerStateManager::mark_trace_replay(ChipId device_id, uint32_t trace_id) {
    TT_ASSERT(this->device_profiler_map.contains(device_id));
    std::lock_guard<std::recursive_mutex> lock{this->device_profiler_map_mutex};
    DeviceProfiler& device_profiler = this->device_profiler_map.at(device_id);
    device_profiler.markTraceReplay(trace_id);
}

void ProfilerStateManager::add_runtime_id_to_trace(ChipId device_id, uint32_t trace_id, uint32_t runtime_id) {
    TT_ASSERT(this->device_profiler_map.contains(device_id));
    std::lock_guard<std::recursive_mutex> lock{this->device_profiler_map_mutex};
    DeviceProfiler& device_profiler = this->device_profiler_map.at(device_id);
    device_profiler.addRuntimeIdToTrace(trace_id, runtime_id);
}

void ProfilerStateManager::signal_debug_dump_read() {
    if (this->debug_dump_thread.joinable()) {
        std::unique_lock<std::mutex> lock{this->debug_dump_thread_mutex};
        this->force_read_complete = false;
        this->force_read_debug_dump = true;
        this->stop_debug_dump_thread_cv.notify_all();
        this->force_read_complete_cv.wait(lock, [&] { return this->force_read_complete.load(); });
    }
}

void ProfilerStateManager::start_debug_dump_thread(
    std::vector<IDevice*> active_devices, std::unordered_map<ChipId, std::vector<CoreCoord>> virtual_cores_map) {
    TT_ASSERT(!this->debug_dump_thread.joinable());
    // Reset stop flag in case it was set by a previous cleanup_device_profilers() call
    this->stop_debug_dump_thread = false;
    // Faster polling to unblock cores quickly at the expense of more NoC PCIe traffic.
    // Tunable via TT_METAL_NOC_DEBUG_POLL_INTERVAL_MS.
    const auto interval = this->env_.get_rtoptions().get_noc_debug_poll_interval();
    const auto full_read_interval = this->env_.get_rtoptions().get_noc_debug_full_read_interval();
    const auto watermark_margin = this->env_.get_rtoptions().get_noc_debug_watermark_margin();

    TT_FATAL(
        watermark_margin > interval,
        "TT_METAL_NOC_DEBUG_WATERMARK_MARGIN_MS ({}) must be greater than TT_METAL_NOC_DEBUG_POLL_INTERVAL_MS ({}); "
        "the margin has to cover the record-to-host latency bounded by the poll interval.",
        watermark_margin.count(),
        interval.count());

    constexpr uint64_t max_margin_ticks = uint64_t(1) << (kernel_profiler::PROFILER_MARKER_TS_BITS - 1);
    TT_FATAL(
        noc_debug_margin_to_ticks(active_devices, watermark_margin) < max_margin_ticks,
        "TT_METAL_NOC_DEBUG_WATERMARK_MARGIN_MS ({}) is too large: it exceeds half the {}-bit device timestamp wrap "
        "window, which would silently disable the mid-run watermark.",
        watermark_margin.count(),
        kernel_profiler::PROFILER_MARKER_TS_BITS);

    // Convert the full-read period into the number of idle polls the thread counts, rounding up so the configured
    // period is an upper bound and any non-zero period still yields at least one poll of waiting.
    const uint32_t full_read_cycles =
        full_read_interval.count() == 0
            ? 0
            : static_cast<uint32_t>((full_read_interval + interval - std::chrono::milliseconds(1)) / interval);

    this->debug_dump_thread = std::thread([this,
                                           active_devices = std::move(active_devices),
                                           virtual_cores_map = std::move(virtual_cores_map),
                                           interval = interval,
                                           full_read_cycles = full_read_cycles,
                                           watermark_margin = watermark_margin]() {
        uint32_t idle_cycles = 0;
        while (true) {
            {
                std::lock_guard<std::recursive_mutex> lock{this->device_profiler_map_mutex};
                for (auto* device : active_devices) {
                    auto profiler_it = this->device_profiler_map.find(device->id());
                    TT_ASSERT(this->device_profiler_map.contains(device->id()));
                    DeviceProfiler& profiler = profiler_it->second;
                    // Only process stalled buffers during periodic polling
                    profiler.pollDebugDumpResults(device, virtual_cores_map.at(device->id()), /*is_final_poll=*/false);
                }
            }

            std::unique_lock<std::mutex> lock{this->debug_dump_thread_mutex};
            if (this->stop_debug_dump_thread_cv.wait_for(lock, interval, [&] {
                    return this->stop_debug_dump_thread.load() || this->force_read_debug_dump.load();
                })) {
                bool was_force_read = this->force_read_debug_dump.exchange(false);
                bool is_stopping = this->stop_debug_dump_thread.load();

                idle_cycles = 0;
                for (auto* device : active_devices) {
                    {
                        auto profiler_it = this->device_profiler_map.find(device->id());
                        TT_ASSERT(profiler_it != this->device_profiler_map.end());
                        DeviceProfiler& profiler = profiler_it->second;
                        profiler.pollDebugDumpResults(
                            device, virtual_cores_map.at(device->id()), /*is_final_poll=*/true);
                    }
                    constexpr auto state = ProfilerReadState::LAST_FD_READ;
                    detail::ReadDeviceProfilerResultsInternal(
                        device->get_mesh_device().get(),
                        device,
                        virtual_cores_map.at(device->id()),
                        state,
                        {},
                        was_force_read);

                    auto profiler_it = this->device_profiler_map.find(device->id());
                    TT_ASSERT(profiler_it != this->device_profiler_map.end());
                    DeviceProfiler& profiler = profiler_it->second;
                    if (this->env_.get_rtoptions().get_profiler_trace_only() || was_force_read) {
                        profiler.processResults(
                            device,
                            virtual_cores_map.at(device->id()),
                            state,
                            ProfilerDataBufferSource::DRAM_AND_L1,
                            {});
                    } else {
                        profiler.processResults(
                            device, virtual_cores_map.at(device->id()), state, ProfilerDataBufferSource::DRAM, {});
                    }
                    if (was_force_read && !profiler.device_markers_per_core_risc_map.empty()) {
                        profiler.dumpDeviceResults(/*is_mid_run_dump=*/true);
                    }
                    // cleanup_device_profilers() handles the final dump
                }

                if (was_force_read) {
                    this->force_read_complete = true;
                    this->force_read_complete_cv.notify_all();
                }

                if (is_stopping) {
                    break;
                }
            } else if (full_read_cycles > 0 && ++idle_cycles >= full_read_cycles) {
                idle_cycles = 0;
                std::lock_guard<std::recursive_mutex> map_lock{this->device_profiler_map_mutex};
                for (auto* device : active_devices) {
                    auto profiler_it = this->device_profiler_map.find(device->id());
                    TT_ASSERT(profiler_it != this->device_profiler_map.end());
                    DeviceProfiler& profiler = profiler_it->second;
                    profiler.pollDebugDumpResults(device, virtual_cores_map.at(device->id()), /*is_final_poll=*/true);
                    detail::ReadDeviceProfilerResultsInternal(
                        device->get_mesh_device().get(),
                        device,
                        virtual_cores_map.at(device->id()),
                        ProfilerReadState::LAST_FD_READ,
                        {},
                        /*include_l1=*/true);
                    profiler.processResults(
                        device,
                        virtual_cores_map.at(device->id()),
                        ProfilerReadState::LAST_FD_READ,
                        ProfilerDataBufferSource::DRAM_AND_L1,
                        {});
                }
                if (auto& noc_debug_state = MetalContext::instance().noc_debug_state();
                    noc_debug_state && !active_devices.empty()) {
                    // Recomputed each pass rather than hoisted, because the aiclk can change at runtime (DVFS).
                    const uint64_t margin_ticks = noc_debug_margin_to_ticks(active_devices, watermark_margin);
                    noc_debug_state->process_accumulated_events_up_to(margin_ticks);
                    noc_debug_state->report_new_issues();
                }
                // Discharge the marker set. This is what keeps host memory bounded on a long run: dumpDeviceResults
                // consumes the markers and its last statement clears device_markers_per_core_risc_map, which
                // otherwise grows for the whole run (it is only cleared at device close). Safe to clear here because
                // the set's other job -- deduplicating the repeated parses of one undrained buffer, which is what
                // gates pushing NOC-debug events.
                for (auto* device : active_devices) {
                    auto profiler_it = this->device_profiler_map.find(device->id());
                    TT_ASSERT(profiler_it != this->device_profiler_map.end());
                    DeviceProfiler& profiler = profiler_it->second;
                    if (!profiler.device_markers_per_core_risc_map.empty()) {
                        profiler.dumpDeviceResults(/*is_mid_run_dump=*/true);
                    }
                }
            }
        }
    });
}

}  // namespace tt::tt_metal
