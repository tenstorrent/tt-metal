// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_context.hpp"

#include <unordered_map>

void TestContext::wait_for_programs_with_progress() {
    if (!progress_config_.enabled) {
        fixture_->wait_for_programs();
        return;
    }

    // Create progress monitor (but don't start polling thread yet)
    TestProgressMonitor monitor(this, progress_config_);

    // Poll and check for completion in this thread
    log_info(
        tt::LogTest,
        "Progress monitoring started (poll interval: {}s, hung threshold: {}s)",
        progress_config_.poll_interval_seconds,
        progress_config_.hung_threshold_seconds);

    monitor.poll_until_complete();

    log_info(tt::LogTest, "Progress monitoring complete");

    // Now call wait_for_programs() to ensure proper cleanup
    log_info(tt::LogTest, "Waiting for programs to complete...");
    fixture_->wait_for_programs();
}

static double calc_bw_bytes_per_cycle(uint32_t total_words, uint64_t cycles) {
    constexpr uint32_t bytes_per_eth_word = 16;
    return (total_words * bytes_per_eth_word) / static_cast<double>(cycles);
}

void TestContext::read_telemetry() {
    telemetry_entries_.clear();
    auto& ctx = tt::tt_metal::MetalContext::instance();
    auto& cluster = ctx.get_cluster();
    auto& control_plane = ctx.get_control_plane();

    const auto telemetry_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();

    std::unordered_map<FabricNodeId, std::unordered_map<CoreCoord, std::vector<uint32_t>>> results;
    auto results_num_elements = tt::align(sizeof(LowResolutionBandwidthTelemetryResult), sizeof(uint32_t));
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
                results[fabric_node_id][eth_core] = std::vector<uint32_t>(results_num_elements, 0);
            }
        }
    }

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

                TT_FATAL(
                    cluster.is_ethernet_link_up(physical_chip_id, eth_core),
                    "Ethernet link is not up for {}",
                    eth_core);

                std::vector<CoreCoord> cores = {eth_core};
                fixture_->read_buffer_from_ethernet_cores(
                    coord,
                    cores,
                    telemetry_addr,
                    sizeof(LowResolutionBandwidthTelemetryResult),
                    false,
                    results[fabric_node_id]);
            }
        }
    }

    fixture_->barrier_reads();
    for (const auto& [coord, test_device] : test_devices_) {
        auto device_id = test_device.get_node_id();
        auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(device_id);
        auto& soc_desc = cluster.get_soc_desc(physical_chip_id);
        auto active_eth_cores = control_plane.get_active_ethernet_cores(physical_chip_id);
        auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(physical_chip_id);
        auto freq_mhz = get_device_frequency_mhz(device_id);
        double freq_ghz = double(freq_mhz) / 1000.0;
        // Wait for reads to complete
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
    auto& ctx = tt::tt_metal::MetalContext::instance();
    auto& cluster = ctx.get_cluster();
    auto& control_plane = ctx.get_control_plane();

    // Also need to write 0s to the telemetry address
    std::vector<uint8_t> zero_vec(sizeof(LowResolutionBandwidthTelemetryResult), 0);
    for (const auto& [coord, test_device] : test_devices_) {
        auto device_id = test_device.get_node_id();
        auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(device_id);
        auto active_eth_cores = control_plane.get_active_ethernet_cores(physical_chip_id);
        for (const auto& eth_core : active_eth_cores) {
            if (!cluster.is_ethernet_link_up(physical_chip_id, eth_core)) {
                continue;
            }

            std::vector<CoreCoord> cores = {eth_core};
            fixture_->write_buffer_to_ethernet_cores(coord, cores, telemetry_addr, zero_vec);
        }
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

ComparisonResult TestContext::create_comparison_result(
    const BandwidthResultSummary& test_result, double test_result_avg_bandwidth) {
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
    comp_result.current_bandwidth_GB_s = test_result_avg_bandwidth;
    return comp_result;
}

// Creates common CSV format string for any failure case
std::string TestContext::generate_failed_test_format_string(
    const BandwidthResultSummary& test_result,
    double test_result_avg_bandwidth,
    double difference_percent,
    double acceptable_tolerance) {
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
        test_result.test_name + "," + test_result.ftype + "," + test_result.ntype + "," + test_result.topology + ",\"" +
        num_devices_str + "\"," + std::to_string(test_result.num_links) + "," +
        std::to_string(test_result.packet_size) + "," + std::to_string(test_result.num_iterations) + "," +
        std::to_string(test_result_avg_cycles) + "," + std::to_string(test_result_avg_bandwidth) + "," +
        std::to_string(test_result_avg_packets_per_second) + "," + std::to_string(difference_percent) + "," +
        tolerance_stream.str();
    return csv_format_string;
}
