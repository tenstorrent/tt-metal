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

CodeProfiler::CodeProfiler(EthCoreBufferReadback& eth_readback) : eth_readback_(eth_readback) {
    // Setup code profiling CSV path
    std::ostringstream code_profiling_results_oss;
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    code_profiling_results_oss << "code_profiling_results_" << arch_name << ".csv";

    std::filesystem::path output_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        std::string(OUTPUT_DIR);

    code_profiling_csv_file_path_ = output_path / code_profiling_results_oss.str();
}

void CodeProfiler::set_enabled(bool enabled) { enabled_ = enabled; }

bool CodeProfiler::is_enabled() const { return enabled_; }

void CodeProfiler::clear_code_profiling_buffers() {
    entries_.clear();
    auto& ctx = tt::tt_metal::MetalContext::instance();

    // Check if any code profiling is enabled
    auto& rtoptions = ctx.rtoptions();
    if (!rtoptions.fabric_code_profiling_enabled()) {
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
    if (!rtoptions.fabric_code_profiling_enabled()) {
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
    if (rtoptions.fabric_code_profiling_enabled()) {
        CodeProfilingTimerType code_profiling_timer_type =
            convert_to_code_profiling_timer_type(rtoptions.get_fabric_code_profiling_timer_str());
        TT_FATAL(
            code_profiling_timer_type != CodeProfilingTimerType::NONE,
            "Invalid code profiling timer string: {}",
            rtoptions.get_fabric_code_profiling_timer_str());
        enabled_timers.push_back(code_profiling_timer_type);
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

void CodeProfiler::initialize_code_profiling_results_csv_file() {
    std::filesystem::path output_path = code_profiling_csv_file_path_.parent_path();

    if (!std::filesystem::exists(output_path)) {
        std::filesystem::create_directories(output_path);
    }

    // Create detailed CSV file with header
    std::ofstream code_profiling_csv_stream(
        code_profiling_csv_file_path_, std::ios::out | std::ios::trunc);  // Truncate file
    if (!code_profiling_csv_stream.is_open()) {
        log_error(tt::LogTest, "Failed to create code profiling CSV file: {}", code_profiling_csv_file_path_.string());
        return;
    }

    // Write detailed header
    code_profiling_csv_stream << "test_name,ftype,ntype,topology,num_links,packet_size,iteration_number,";
    code_profiling_csv_stream
        << "device_coord,core,code_profiling_timer_type,total_cycles,num_instances,avg_cycles_per_instance";
    code_profiling_csv_stream << "\n";

    log_info(tt::LogTest, "Initialized code profiling CSV file: {}", code_profiling_csv_file_path_.string());

    code_profiling_csv_stream.close();
}

std::string CodeProfiler::convert_coord_to_string(const MeshCoordinate& coord) {
    return "[" + std::to_string(coord[0]) + "," + std::to_string(coord[1]) + "]";
}

void CodeProfiler::dump_code_profiling_results_to_csv(const TestConfig& config) {
    // Extract representative ftype, ntype, packet_size from first sender's first pattern
    const TrafficPatternConfig& first_pattern = fetch_first_traffic_pattern(config);
    std::string ftype_str = fetch_pattern_ftype(first_pattern);
    std::string ntype_str = fetch_pattern_ntype(first_pattern);
    uint32_t packet_size = fetch_pattern_packet_size(first_pattern);

    // Open CSV file in append mode
    std::ofstream code_profiling_csv_stream(code_profiling_csv_file_path_, std::ios::out | std::ios::app);
    if (!code_profiling_csv_stream.is_open()) {
        log_error(
            tt::LogTest,
            "Failed to open Code Profiling CSV file for appending: {}",
            code_profiling_csv_file_path_.string());
        return;
    }

    // Write data rows (header already written in initialize_code_profiling_results_csv_file)
    for (const auto& entry : entries_) {
        std::string coord_str = convert_coord_to_string(entry.coord);
        code_profiling_csv_stream << config.name << "," << ftype_str << "," << ntype_str << ","
                                  << enchantum::to_string(config.fabric_setup.topology) << ","
                                  << config.fabric_setup.num_links << "," << packet_size << ","
                                  << config.iteration_number << ",";
        code_profiling_csv_stream << "\"" << coord_str << "\"," << entry.eth_channel << ","
                                  << convert_code_profiling_timer_type_to_str(entry.timer_type) << ","
                                  << entry.total_cycles << "," << entry.num_instances << "," << std::fixed
                                  << std::setprecision(6) << entry.avg_cycles_per_instance << "\n";
    }

    code_profiling_csv_stream.close();

    log_info(tt::LogTest, "Code Profiling results appended to CSV file: {}", code_profiling_csv_file_path_.string());
}

void CodeProfiler::reset() {
    entries_.clear();
    clear_code_profiling_buffers();
}

const std::vector<CodeProfilingEntry>& CodeProfiler::get_entries() const { return entries_; }
