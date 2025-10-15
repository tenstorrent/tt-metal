// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_context.hpp"

#include <unordered_map>
#include "impl/context/metal_context.hpp"

static double calc_bw_bytes_per_cycle(uint32_t total_words, uint64_t cycles) {
    constexpr uint32_t bytes_per_eth_word = 16;
    return (total_words * bytes_per_eth_word) / static_cast<double>(cycles);
}

// Calculate code profiling buffer address (right after telemetry buffer)
static uint32_t get_code_profiling_buffer_addr() {
    uint32_t addr = ::tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    // Add telemetry buffer size (32 bytes) if telemetry is enabled or on Blackhole
    // This mirrors the logic in FabricEriscDatamoverConfig constructor
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    if (rtoptions.get_enable_fabric_telemetry() ||
        tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE) {
        addr += 32; // telemetry buffer size
    }
    return addr;
}

void TestContext::read_telemetry() {
    telemetry_entries_.clear();

    // Get telemetry buffer address and size
    const auto telemetry_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    const size_t telemetry_buffer_size = sizeof(LowResolutionBandwidthTelemetryResult);

    // Read buffer data from all active ethernet cores
    auto results = get_eth_readback().read_buffer(telemetry_addr, telemetry_buffer_size);

    // Process telemetry results
    auto& ctx = tt::tt_metal::MetalContext::instance();
    auto& cluster = ctx.get_cluster();
    auto& control_plane = ctx.get_control_plane();

    for (const auto& [coord, test_device] : test_devices_) {
        auto device_id = test_device.get_node_id();
        auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(device_id);
        auto& soc_desc = cluster.get_soc_desc(physical_chip_id);
        auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(physical_chip_id);
        auto freq_mhz = get_device_frequency_mhz(device_id);
        double freq_ghz = double(freq_mhz) / 1000.0;

        for (const auto& [direction, link_indices] : test_device.get_used_fabric_connections()) {
            const auto& eth_cores =
                control_plane.get_active_fabric_eth_channels_in_direction(fabric_node_id, direction);
            for (const auto& link_index : link_indices) {
                const tt::tt_fabric::chan_id_t eth_channel = eth_cores.at(link_index);
                const CoreCoord& eth_core = soc_desc.get_eth_core_for_channel(eth_channel, CoordSystem::LOGICAL);
                const auto& core_data = results.at(fabric_node_id).at(eth_core);

                LowResolutionBandwidthTelemetryResult tel{};
                if (reinterpret_cast<uintptr_t>(core_data.data()) % alignof(LowResolutionBandwidthTelemetryResult) ==
                    0) {
                    constexpr size_t NUM_ELEMENTS =
                        tt::align(sizeof(LowResolutionBandwidthTelemetryResult), sizeof(uint32_t)) / sizeof(uint32_t);
                    const std::array<uint32_t, NUM_ELEMENTS>& data_array =
                        *reinterpret_cast<const std::array<uint32_t, NUM_ELEMENTS>*>(core_data.data());
                    tel = std::bit_cast<LowResolutionBandwidthTelemetryResult>(data_array);
                } else {
                    // Fall back to memcpy approach
                    std::array<std::byte, sizeof(LowResolutionBandwidthTelemetryResult)> staging_buf{};
                    memcpy(staging_buf.data(), core_data.data(), sizeof(LowResolutionBandwidthTelemetryResult));
                    tel = std::bit_cast<LowResolutionBandwidthTelemetryResult>(staging_buf);
                }

                uint64_t cycles = tel.duration.full;
                double bytes_per_cycle = calc_bw_bytes_per_cycle(tel.num_words_sent, cycles);
                double bw_GB_s = bytes_per_cycle * double(freq_ghz);
                double time_s = static_cast<double>(cycles) / (freq_mhz * 1e6);
                double pps = static_cast<double>(tel.num_packets_sent) / time_s;
                log_info(
                    tt::LogTest,
                    "Telemetry from {} core {}: BW (GB/s)={:.6f}, pps={:.6f}, cycles={:d}, eth_words_sent={:d}, "
                    "packets_sent={:d}",
                    coord,
                    eth_core.str(),
                    bw_GB_s,
                    pps,
                    cycles,
                    tel.num_words_sent,
                    tel.num_packets_sent);
                auto [connected_physical_id, connected_eth_core] =
                    cluster.get_connected_ethernet_core({physical_chip_id, eth_core});
                auto connected_device_id =
                    control_plane.get_fabric_node_id_from_physical_chip_id(connected_physical_id);
                ::tt::tt_metal::distributed::MeshCoordinate connected_coord =
                    fixture_->get_device_coord(connected_device_id);
                uint32_t connected_eth_channel =
                    cluster.get_soc_desc(connected_physical_id).logical_eth_core_to_chan_map.at(connected_eth_core);
                telemetry_entries_.push_back(
                    {coord, eth_channel, bw_GB_s, pps, connected_coord, connected_eth_channel});
            }
        }
    }
}

void TestContext::clear_telemetry() {
    telemetry_entries_.clear();

    // Get telemetry buffer address and size
    const auto telemetry_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    const size_t telemetry_buffer_size = sizeof(LowResolutionBandwidthTelemetryResult);

    get_eth_readback().clear_buffer(telemetry_addr, telemetry_buffer_size);
}

void TestContext::clear_code_profiling_buffers() {
    code_profiling_entries_.clear();
    auto& ctx = tt::tt_metal::MetalContext::instance();

    // Check if any code profiling is enabled
    auto& rtoptions = ctx.rtoptions();
    if (!rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd()) {
        return; // No profiling enabled, nothing to clear
    }

    // Get code profiling buffer address and size
    uint32_t code_profiling_addr = get_code_profiling_buffer_addr();
    constexpr size_t code_profiling_buffer_size = get_max_code_profiling_timer_types() * sizeof(CodeProfilingTimerResult);

    get_eth_readback().clear_buffer(code_profiling_addr, code_profiling_buffer_size);
}

void TestContext::read_code_profiling_results() {
    code_profiling_entries_.clear();
    auto& ctx = tt::tt_metal::MetalContext::instance();

    // Check if any code profiling is enabled
    auto& rtoptions = ctx.rtoptions();
    if (!rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd()) {
        return; // No profiling enabled, nothing to read
    }

    // Get code profiling buffer address and size
    uint32_t code_profiling_addr = get_code_profiling_buffer_addr();
    constexpr size_t code_profiling_buffer_size = get_max_code_profiling_timer_types() * sizeof(CodeProfilingTimerResult);

    // Read buffer data from all active ethernet cores
    auto results = get_eth_readback().read_buffer(code_profiling_addr, code_profiling_buffer_size);

    // Process results for each enabled timer type
    std::vector<CodeProfilingTimerType> enabled_timers;
    if (rtoptions.get_enable_fabric_code_profiling_rx_ch_fwd()) {
        enabled_timers.push_back(CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD);
    }

    auto& cluster = ctx.get_cluster();
    auto& control_plane = ctx.get_control_plane();

    for (const auto& [coord, test_device] : test_devices_) {
        auto device_id = test_device.get_node_id();
        auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(device_id);
        auto& soc_desc = cluster.get_soc_desc(physical_chip_id);
        auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(physical_chip_id);

        for (const auto& [direction, link_indices] : test_device.get_used_fabric_connections()) {
            const auto& eth_cores =
                control_plane.get_active_fabric_eth_channels_in_direction(fabric_node_id, direction);
            for (const auto& link_index : link_indices) {
                const tt::tt_fabric::chan_id_t eth_channel = eth_cores.at(link_index);
                const CoreCoord& eth_core = soc_desc.get_eth_core_for_channel(eth_channel, CoordSystem::LOGICAL);
                const auto& core_data = results.at(fabric_node_id).at(eth_core);

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
                        double avg_cycles_per_instance = static_cast<double>(result.total_cycles) / static_cast<double>(result.num_instances);
                        code_profiling_entries_.push_back(
                            {coord,
                             eth_channel,
                             timer_type,
                             result.total_cycles,
                             result.num_instances,
                             avg_cycles_per_instance});
                    }
                }
            }
        }
    }
}

void TestContext::report_code_profiling_results() {
    if (code_profiling_entries_.empty()) {
        log_info(tt::LogTest, "Code Profiling Results: No data collected");
        return;
    }

    log_info(tt::LogTest, "Code Profiling Results:");

    // Helper function to get timer type name
    auto get_timer_type_name = [](CodeProfilingTimerType timer_type) -> std::string {
        switch (timer_type) {
            case CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD:
                return "RECEIVER_CHANNEL_FORWARD";
            default:
                return "UNKNOWN";
        }
    };

    for (const auto& entry : code_profiling_entries_) {
        log_info(
            tt::LogTest,
            "  Device {} Core {}: {} - Total Cycles: {}, Instances: {}, Avg Cycles/Instance: {:.2f}",
            entry.coord,
            entry.eth_channel,
            get_timer_type_name(entry.timer_type),
            entry.total_cycles,
            entry.num_instances,
            entry.avg_cycles_per_instance);
    }
}

void TestContext::process_telemetry_for_golden() {
    std::sort(
        telemetry_entries_.begin(), telemetry_entries_.end(), [](const TelemetryEntry& a, const TelemetryEntry& b) {
            return a.bw_gbps > b.bw_gbps;
        });

    if (telemetry_entries_.empty()) {
        measured_bw_min_ = std::numeric_limits<double>::max();
        measured_bw_avg_ = 0.0;
        measured_bw_max_ = 0.0;
        return;
    }
    measured_bw_min_ = std::numeric_limits<double>::max();
    // Doesn't account for traffic flows, simply averages the raw BW values, which may or may not make sense
    // This could be thought of as the system/algorithmic BW;
    measured_bw_avg_ = 0.0;
    measured_bw_max_ = 0.0;

    auto max_entry = telemetry_entries_.front();
    auto max_coord = max_entry.coord;
    auto max_connected = max_entry.connected_coord;

    double minimal_accepted_BW_GB_s = 0.08;
    for (const auto& entry : telemetry_entries_) {
        if (entry.bw_gbps < minimal_accepted_BW_GB_s) {
            continue;
        }

        measured_bw_min_ = std::min(measured_bw_min_, entry.bw_gbps);
        measured_bw_max_ = std::max(measured_bw_max_, entry.bw_gbps);
        measured_bw_avg_ += entry.bw_gbps;
    }
    measured_bw_avg_ /= telemetry_entries_.size();

    auto is_on_edge = [&](const TelemetryEntry& e) {
        return (e.coord == max_coord && e.connected_coord == max_connected) ||
               (e.coord == max_connected && e.connected_coord == max_coord);
    };

    std::map<int, std::vector<double>> plane_bws;
    for (const auto& entry : telemetry_entries_) {
        if (is_on_edge(entry)) {
            int plane = entry.eth_channel / 4;
            plane_bws[plane].push_back(entry.bw_gbps);
        }
    }

    double min_of_mins = std::numeric_limits<double>::max();
    for (const auto& [plane, bws] : plane_bws) {
        if (!bws.empty()) {
            double min_bw = *std::min_element(bws.begin(), bws.end());
            min_of_mins = std::min(min_of_mins, min_bw);
        }
    }

    log_info(
        tt::LogTest,
        "Measured BW (min/avg/max): {:.6f} GB/s, {:.6f} GB/s, {:.6f} GB/s",
        measured_bw_min_,
        measured_bw_avg_,
        measured_bw_max_);
}

void TestContext::dump_raw_telemetry_csv(const TestConfig& config) {
    std::filesystem::path raw_telemetry_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        (output_dir + "/" + config.name + "_raw_telemetry.csv");

    if (!std::filesystem::exists(raw_telemetry_path)) {
        std::ofstream header_stream(raw_telemetry_path, std::ios::out | std::ios::trunc);
        if (header_stream.is_open()) {
            header_stream << "coord,eth_channel,bw_gbps,pps,connected_coord,connected_eth_channel\n";
            header_stream.close();
        }
    }

    std::ofstream data_stream(raw_telemetry_path, std::ios::out | std::ios::app);
    if (data_stream.is_open()) {
        for (const auto& entry : telemetry_entries_) {
            data_stream << "\"" << entry.coord[0] << "," << entry.coord[1] << "\"," << entry.eth_channel << ","
                        << entry.bw_gbps << "," << entry.pps << "," << entry.connected_coord << ","
                        << entry.connected_eth_channel << "\n";
        }
        data_stream.close();
    }
    log_info(tt::LogTest, "Dumped raw telemetry to: {}", raw_telemetry_path.string());
}

// Converts vector of num_devices to a string representation eg. <2, 4> -> "[2, 4]"
std::string TestContext::convert_num_devices_to_string(const std::vector<uint32_t>& num_devices) {
    std::string num_devices_str = "[";
    for (size_t i = 0; i < num_devices.size(); ++i) {
        if (i > 0) {
            num_devices_str += ",";
        }
        num_devices_str += std::to_string(num_devices[i]);
    }
    num_devices_str += "]";
    return num_devices_str;
}

std::vector<GoldenCsvEntry>::iterator TestContext::fetch_corresponding_golden_entry(
    const BandwidthResultSummary& test_result) {
    std::string num_devices_str = convert_num_devices_to_string(test_result.num_devices);
    auto golden_it =
        std::find_if(golden_csv_entries_.begin(), golden_csv_entries_.end(), [&](const GoldenCsvEntry& golden) {
            return golden.test_name == test_result.test_name && golden.ftype == test_result.ftype &&
                   golden.ntype == test_result.ntype && golden.topology == test_result.topology &&
                   golden.num_devices == num_devices_str && golden.num_links == test_result.num_links &&
                   golden.packet_size == test_result.packet_size;
        });
    return golden_it;
}

ComparisonResult TestContext::create_comparison_result(const BandwidthResultSummary& test_result) {
    std::string num_devices_str = convert_num_devices_to_string(test_result.num_devices);
    ComparisonResult comp_result;
    comp_result.test_name = test_result.test_name;
    comp_result.ftype = test_result.ftype;
    comp_result.ntype = test_result.ntype;
    comp_result.topology = test_result.topology;
    comp_result.num_devices = num_devices_str;
    comp_result.num_links = test_result.num_links;
    comp_result.packet_size = test_result.packet_size;
    comp_result.num_iterations = test_result.num_iterations;
    return comp_result;
}

// Creates common CSV format string for any failure case
std::string TestContext::generate_failed_test_format_string(const BandwidthResultSummary& test_result, double test_result_avg_bandwidth, double difference_percent, double acceptable_tolerance) {
    std::ostringstream tolerance_stream;
    tolerance_stream << std::fixed << std::setprecision(1) << acceptable_tolerance;
    // Because statistics order may change, we need to find the index of average cycles and packets per second
    double test_result_avg_cycles = -1;
    auto cycles_stat_location = std::find(stat_order_.begin(), stat_order_.end(), BandwidthStatistics::CyclesMean);
    if (cycles_stat_location == stat_order_.end()) {
        log_warning(tt::LogTest, "Average cycles statistic not found, omitting it in failure report");
    } else {
        int cycles_stat_index = std::distance(stat_order_.begin(), cycles_stat_location);
        test_result_avg_cycles = test_result.statistics_vector[cycles_stat_index];
    }
    double test_result_avg_packets_per_second = -1;
    auto packets_per_second_stat_location =
        std::find(stat_order_.begin(), stat_order_.end(), BandwidthStatistics::PacketsPerSecondMean);
    if (packets_per_second_stat_location == stat_order_.end()) {
        log_warning(tt::LogTest, "Average packets per second statistic not found, omitting it in failure report");
    } else {
        int packets_per_second_stat_index = std::distance(stat_order_.begin(), packets_per_second_stat_location);
        test_result_avg_packets_per_second = test_result.statistics_vector[packets_per_second_stat_index];
    }
    std::string num_devices_str = convert_num_devices_to_string(test_result.num_devices);
    std::string csv_format_string =
        test_result.test_name + ","
        + test_result.ftype + ","
        + test_result.ntype + ","
        + test_result.topology
        + ",\"" + num_devices_str + "\","
        + std::to_string(test_result.num_links) + ","
        + std::to_string(test_result.packet_size) + ","
        + std::to_string(test_result.num_iterations) + ","
        + std::to_string(test_result_avg_cycles) + ","
        + std::to_string(test_result_avg_bandwidth) + ","
        + std::to_string(test_result_avg_packets_per_second) + ","
        + std::to_string(difference_percent) + ","
        + tolerance_stream.str();
    return csv_format_string;
}

void TestContext::set_comparison_statistics_csv_file_path() {
    // Bandwidth summary CSV file is generated separately from Bandwidth CSV because we need to wait for all multirun
    // tests to complete Generate detailed CSV filename
    std::ostringstream comparison_statistics_oss;
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    comparison_statistics_oss << "bandwidth_comparison_statistics_" << arch_name << ".csv";
    // Output directory already set in initialize_bandwidth_results_csv_file()
    std::filesystem::path output_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / output_dir;
    comparison_statistics_csv_file_path_ = output_path / comparison_statistics_oss.str();
}
