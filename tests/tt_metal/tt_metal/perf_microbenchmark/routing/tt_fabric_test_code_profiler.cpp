// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_code_profiler.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <ranges>

#include "impl/context/metal_context.hpp"

namespace {
// Calculate code profiling buffer address (right after telemetry buffer)
uint32_t get_code_profiling_buffer_addr() {
    uint32_t addr = ::tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    // Add telemetry buffer size (32 bytes) if telemetry is enabled or on Blackhole
    // This mirrors the logic in FabricEriscDatamoverConfig constructor
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    if (rtoptions.get_enable_fabric_bw_telemetry() ||
        tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE) {
        addr += 32;  // telemetry buffer size
    }
    return addr;
}
}  // namespace

CodeProfiler::CodeProfiler(EthCoreBufferReadback& eth_readback) : eth_readback_(eth_readback) {}

void CodeProfiler::set_enabled(bool enabled) { enabled_ = enabled; }

bool CodeProfiler::is_enabled() const { return enabled_; }

void CodeProfiler::clear_code_profiling_buffers() {
    entries_.clear();
    auto& ctx = tt::tt_metal::MetalContext::instance();

    // Check if any code profiling is enabled
    auto& rtoptions = ctx.rtoptions();
    if (!rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd()) {
        return;  // No profiling enabled, nothing to clear
    }

    // Get code profiling buffer address and size
    uint32_t code_profiling_addr = get_code_profiling_buffer_addr();
    constexpr size_t code_profiling_buffer_size =
        get_max_code_profiling_timer_types() * sizeof(CodeProfilingTimerResult);

    eth_readback_.clear_buffer(code_profiling_addr, code_profiling_buffer_size);
}

void CodeProfiler::read_code_profiling_results() {
    entries_.clear();
    auto& ctx = tt::tt_metal::MetalContext::instance();

    // Check if any code profiling is enabled
    auto& rtoptions = ctx.rtoptions();
    if (!rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd()) {
        return;  // No profiling enabled, nothing to read
    }

    // Get code profiling buffer address and size
    uint32_t code_profiling_addr = get_code_profiling_buffer_addr();
    constexpr size_t code_profiling_buffer_size =
        get_max_code_profiling_timer_types() * sizeof(CodeProfilingTimerResult);

    // Read buffer data from all active ethernet cores (including intermediate forwarding hops)
    auto results = eth_readback_.read_buffer(code_profiling_addr, code_profiling_buffer_size, true);

    // Process results for each enabled timer type
    std::vector<CodeProfilingTimerType> enabled_timers;
    if (rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd()) {
        enabled_timers.push_back(CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD);
    }

    for (const auto& location : results) {
        const auto& core_data = location.buffer_data;

        // Process each enabled timer type
        for (const auto& timer_type : enabled_timers) {
            // Calculate offset for this timer type
            uint32_t timer_bit_position = std::countr_zero(static_cast<uint32_t>(timer_type));
            size_t offset = timer_bit_position * sizeof(CodeProfilingTimerResult);

            // Extract CodeProfilingTimerResult from buffer
            CodeProfilingTimerResult result{};
            if (offset + sizeof(CodeProfilingTimerResult) <= core_data.size() * sizeof(uint32_t)) {
                // Safe to read the result
                const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(core_data.data()) + offset;
                if (reinterpret_cast<uintptr_t>(data_ptr) % alignof(CodeProfilingTimerResult) == 0) {
                    result = *reinterpret_cast<const CodeProfilingTimerResult*>(data_ptr);
                } else {
                    // Fall back to memcpy approach
                    std::array<std::byte, sizeof(CodeProfilingTimerResult)> staging_buf{};
                    memcpy(staging_buf.data(), data_ptr, sizeof(CodeProfilingTimerResult));
                    result = std::bit_cast<CodeProfilingTimerResult>(staging_buf);
                }
            }

            // Only add entry if timer fired (num_instances > 0)
            if (result.num_instances > 0) {
                double avg_cycles_per_instance =
                    static_cast<double>(result.total_cycles) / static_cast<double>(result.num_instances);
                entries_.push_back(
                    {location.coord,
                     location.eth_channel,
                     timer_type,
                     result.total_cycles,
                     result.num_instances,
                     avg_cycles_per_instance});
            }
        }
    }
}

void CodeProfiler::report_code_profiling_results() const {
    if (entries_.empty()) {
        log_info(tt::LogTest, "Code Profiling Results: No data collected");
        return;
    }

    log_info(tt::LogTest, "Code Profiling Results:");

    auto get_timer_type_name = [](CodeProfilingTimerType timer_type) -> std::string {
        switch (timer_type) {
            case CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD: return "RECEIVER_CHANNEL_FORWARD";
            default: return "UNKNOWN";
        }
    };

    for (const auto& entry : entries_) {
        log_info(
            tt::LogTest,
            "  Device {} Core {}: {} - Total Cycles: {}, Instances: {}, avg_cycles/Instance: {:.2f}",
            entry.coord,
            entry.eth_channel,
            get_timer_type_name(entry.timer_type),
            entry.total_cycles,
            entry.num_instances,
            entry.avg_cycles_per_instance);
    }
}

void CodeProfiler::reset() {
    entries_.clear();
    clear_code_profiling_buffers();
}

const std::vector<CodeProfilingEntry>& CodeProfiler::get_entries() const { return entries_; }
