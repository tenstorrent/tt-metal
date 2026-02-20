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
    if (!rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd() &&
        !rtoptions.get_enable_fabric_code_profiling_speedy_path()) {
        return;  // No profiling enabled, nothing to clear
    }

    // Get code profiling buffer address and size
    uint32_t code_profiling_addr = get_code_profiling_buffer_addr();
    constexpr size_t num_timers = get_max_code_profiling_timer_types();
    constexpr size_t code_profiling_buffer_size = num_timers * sizeof(CodeProfilingTimerResult);

    // Build properly initialized buffer: total_cycles=0, num_instances=0,
    // min_cycles=UINT64_MAX, max_cycles=0
    std::vector<uint8_t> init_buf(code_profiling_buffer_size, 0);
    auto* results = reinterpret_cast<CodeProfilingTimerResult*>(init_buf.data());
    for (size_t i = 0; i < num_timers; i++) {
        results[i].total_cycles = 0;
        results[i].num_instances = 0;
        results[i].min_cycles = UINT64_MAX;
        results[i].max_cycles = 0;
    }

    eth_readback_.write_buffer(code_profiling_addr, init_buf);
}

void CodeProfiler::read_code_profiling_results() {
    entries_.clear();
    auto& ctx = tt::tt_metal::MetalContext::instance();

    // Check if any code profiling is enabled
    auto& rtoptions = ctx.rtoptions();
    if (!rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd() &&
        !rtoptions.get_enable_fabric_code_profiling_speedy_path()) {
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
    if (rtoptions.get_enable_fabric_code_profiling_speedy_path()) {
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_SENDER_FULL);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_SENDER_SEND_DATA);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_SENDER_CHECK_COMPLETIONS);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_SENDER_CREDITS_UPSTREAM);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_RECEIVER_FULL);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_RECEIVER_FORWARD);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_RECEIVER_FLUSH);
        // Sender send_next_data sub-timers
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_SENDER_SEND_ETH);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_SENDER_SEND_ADV);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_SENDER_SEND_NOTIFY);
        // Receiver forward sub-timers
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_RECEIVER_FWD_HDR);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_RECEIVER_FWD_NOC);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_RECEIVER_FWD_BOOK);
        // Spin iteration counters (sender)
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_SENDER_ETH_TXQ_SPIN_1);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_SENDER_ETH_TXQ_SPIN_2);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_SENDER_NOC_FLUSH_SPIN);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_SENDER_NOC_CMD_BUF_SPIN);
        // Spin iteration counters (receiver)
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_RECEIVER_NOC_CMD_BUF_SPIN);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_RECEIVER_FLUSH_ETH_TXQ_SPIN);
        // Receiver flush sub-timers
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_RECEIVER_FLUSH_TRID);
        enabled_timers.push_back(CodeProfilingTimerType::SPEEDY_RECEIVER_FLUSH_SEND);
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
                     avg_cycles_per_instance,
                     result.min_cycles,
                     result.max_cycles});
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
            case CodeProfilingTimerType::SPEEDY_SENDER_FULL: return "SPEEDY_SENDER_FULL";
            case CodeProfilingTimerType::SPEEDY_SENDER_SEND_DATA: return "SPEEDY_SENDER_SEND_DATA";
            case CodeProfilingTimerType::SPEEDY_SENDER_CHECK_COMPLETIONS: return "SPEEDY_SENDER_CHECK_COMPLETIONS";
            case CodeProfilingTimerType::SPEEDY_SENDER_CREDITS_UPSTREAM: return "SPEEDY_SENDER_CREDITS_UPSTREAM";
            case CodeProfilingTimerType::SPEEDY_RECEIVER_FULL: return "SPEEDY_RECEIVER_FULL";
            case CodeProfilingTimerType::SPEEDY_RECEIVER_FORWARD: return "SPEEDY_RECEIVER_FORWARD";
            case CodeProfilingTimerType::SPEEDY_RECEIVER_FLUSH: return "SPEEDY_RECEIVER_FLUSH";
            case CodeProfilingTimerType::SPEEDY_SENDER_SEND_ETH: return "SPEEDY_SENDER_SEND_ETH";
            case CodeProfilingTimerType::SPEEDY_SENDER_SEND_ADV: return "SPEEDY_SENDER_SEND_ADV";
            case CodeProfilingTimerType::SPEEDY_SENDER_SEND_NOTIFY: return "SPEEDY_SENDER_SEND_NOTIFY";
            case CodeProfilingTimerType::SPEEDY_RECEIVER_FWD_HDR: return "SPEEDY_RECEIVER_FWD_HDR";
            case CodeProfilingTimerType::SPEEDY_RECEIVER_FWD_NOC: return "SPEEDY_RECEIVER_FWD_NOC";
            case CodeProfilingTimerType::SPEEDY_RECEIVER_FWD_BOOK: return "SPEEDY_RECEIVER_FWD_BOOK";
            // Spin iteration counters (values = iterations, not cycles)
            case CodeProfilingTimerType::SPEEDY_SENDER_ETH_TXQ_SPIN_1: return "SPEEDY_SENDER_ETH_TXQ_SPIN_1";
            case CodeProfilingTimerType::SPEEDY_SENDER_ETH_TXQ_SPIN_2: return "SPEEDY_SENDER_ETH_TXQ_SPIN_2";
            case CodeProfilingTimerType::SPEEDY_SENDER_NOC_FLUSH_SPIN: return "SPEEDY_SENDER_NOC_FLUSH_SPIN";
            case CodeProfilingTimerType::SPEEDY_SENDER_NOC_CMD_BUF_SPIN: return "SPEEDY_SENDER_NOC_CMD_BUF_SPIN";
            case CodeProfilingTimerType::SPEEDY_RECEIVER_NOC_CMD_BUF_SPIN: return "SPEEDY_RECEIVER_NOC_CMD_BUF_SPIN";
            case CodeProfilingTimerType::SPEEDY_RECEIVER_FLUSH_ETH_TXQ_SPIN:
                return "SPEEDY_RECEIVER_FLUSH_ETH_TXQ_SPIN";
            // Receiver flush sub-timers
            case CodeProfilingTimerType::SPEEDY_RECEIVER_FLUSH_TRID: return "SPEEDY_RECEIVER_FLUSH_TRID";
            case CodeProfilingTimerType::SPEEDY_RECEIVER_FLUSH_SEND: return "SPEEDY_RECEIVER_FLUSH_SEND";
            default: return "UNKNOWN";
        }
    };

    for (const auto& entry : entries_) {
        log_info(
            tt::LogTest,
            "  Device {} Core {}: {} - Total Cycles: {}, Instances: {}, avg/Instance: {:.2f}, min: {}, max: {}",
            entry.coord,
            entry.eth_channel,
            get_timer_type_name(entry.timer_type),
            entry.total_cycles,
            entry.num_instances,
            entry.avg_cycles_per_instance,
            entry.min_cycles,
            entry.max_cycles);
    }
}

void CodeProfiler::reset() {
    entries_.clear();
    clear_code_profiling_buffers();
}

const std::vector<CodeProfilingEntry>& CodeProfiler::get_entries() const { return entries_; }
