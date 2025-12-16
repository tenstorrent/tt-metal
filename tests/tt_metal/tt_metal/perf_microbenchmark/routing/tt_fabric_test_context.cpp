// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_context.hpp"

#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

void TestContext::add_sync_traffic_to_devices(const TestConfig& config) {
    for (const auto& sync_config : config.sync_configs) {
        // currently initializing our sync configs to be on senders local to the current host
        const auto& sync_sender = sync_config.sender_config;
        if (fixture_->is_local_fabric_node_id(sync_sender.device)) {
            CoreCoord sync_core = sync_sender.core.value();
            const auto& device_coord = this->fixture_->get_device_coord(sync_sender.device);

            // Track global sync core for this device
            device_global_sync_cores_[sync_sender.device] = sync_core;

            // Process each already-split sync pattern for this device
            for (const auto& sync_pattern : sync_sender.patterns) {
                // Convert sync pattern to TestTrafficSenderConfig format
                const auto& dest = sync_pattern.destination.value();

                TrafficParameters sync_traffic_parameters = {
                    .chip_send_type = sync_pattern.ftype.value(),
                    .noc_send_type = sync_pattern.ntype.value(),
                    .payload_size_bytes = sync_pattern.size.value(),
                    .num_packets = sync_pattern.num_packets.value(),
                    .atomic_inc_val = sync_pattern.atomic_inc_val,
                    .mcast_start_hops = sync_pattern.mcast_start_hops,
                    .seed = config.seed,
                    .is_2D_routing_enabled = fixture_->is_2D_routing_enabled(),
                    .mesh_shape = this->fixture_->get_mesh_shape(),
                    .topology = this->fixture_->get_topology()};

                // For sync patterns, we use a dummy destination core and fixed sync address
                // The actual sync will be handled by atomic operations
                CoreCoord dummy_dst_core = {0, 0};  // Sync doesn't need specific dst core
                uint32_t sync_address = this->sender_memory_map_.get_global_sync_address();  // Hard-coded sync address
                uint32_t dst_noc_encoding =
                    this->fixture_->get_worker_noc_encoding(sync_core);  // populate the master coord

                TestTrafficSenderConfig sync_traffic_sender_config = {
                    .parameters = sync_traffic_parameters,
                    .src_node_id = sync_sender.device,
                    .dst_logical_core = dummy_dst_core,
                    .target_address = sync_address,
                    .atomic_inc_address = sync_address,
                    .dst_noc_encoding = dst_noc_encoding,
                    .link_id = sync_sender.link_id};  // Derive from SenderConfig (always 0 for sync)

                // Determine destination node IDs
                auto single_direction_hops = dest.hops.value();
                sync_traffic_sender_config.hops = single_direction_hops;
                // for 2d mcast case
                sync_traffic_sender_config.dst_node_ids = this->fixture_->get_dst_node_ids_from_hops(
                    sync_sender.device, single_direction_hops, sync_traffic_parameters.chip_send_type);
                // for 2d, we need to specify the mcast start node id
                if (fixture_->is_2D_routing_enabled() &&
                    sync_traffic_parameters.chip_send_type == ChipSendType::CHIP_MULTICAST) {
                    sync_traffic_sender_config.mcast_start_node_id =
                        fixture_->get_mcast_start_node_id(sync_sender.device, single_direction_hops);
                } else {
                    sync_traffic_sender_config.mcast_start_node_id = std::nullopt;
                }

                // Add sync config to the master sender on this device
                TestTrafficSyncConfig sync_traffic_sync_config = {
                    .sync_val = sync_config.sync_val, .sender_config = std::move(sync_traffic_sender_config)};

                this->test_devices_.at(device_coord)
                    .add_sender_sync_config(sync_core, std::move(sync_traffic_sync_config));
            }
        }
    }
}

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
    log_info(tt::LogTest, "Progress monitoring complete, waiting for programs to finish...");

    // Now call wait_for_programs() to ensure proper cleanup
    fixture_->wait_for_programs();
}

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
    if (rtoptions.get_enable_fabric_bw_telemetry() ||
        tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::BLACKHOLE) {
        addr += 32;  // telemetry buffer size
    }
    return addr;
}

void TestContext::read_telemetry() {
    telemetry_entries_.clear();

    // Get telemetry buffer address and size
    const auto telemetry_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    const size_t telemetry_buffer_size = sizeof(LowResolutionBandwidthTelemetryResult);

    // Read buffer data from all active ethernet cores (including intermediate forwarding hops)
    auto results = get_eth_readback().read_buffer(telemetry_addr, telemetry_buffer_size, false);

    // Process telemetry results
    auto& ctx = tt::tt_metal::MetalContext::instance();
    auto& cluster = ctx.get_cluster();
    auto& control_plane = ctx.get_control_plane();

    for (const auto& result : results) {
        const auto& core_data = result.buffer_data;
        auto device_id = result.fabric_node_id;
        auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(device_id);
        auto freq_mhz = get_device_frequency_mhz(device_id);
        double freq_ghz = double(freq_mhz) / 1000.0;

        LowResolutionBandwidthTelemetryResult tel{};
        if (reinterpret_cast<uintptr_t>(core_data.data()) % alignof(LowResolutionBandwidthTelemetryResult) == 0) {
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
            result.coord,
            result.eth_core.str(),
            bw_GB_s,
            pps,
            cycles,
            tel.num_words_sent,
            tel.num_packets_sent);
        auto [connected_physical_id, connected_eth_core] =
            cluster.get_connected_ethernet_core({physical_chip_id, result.eth_core});
        auto connected_device_id = control_plane.get_fabric_node_id_from_physical_chip_id(connected_physical_id);
        ::tt::tt_metal::distributed::MeshCoordinate connected_coord = fixture_->get_device_coord(connected_device_id);
        uint32_t connected_eth_channel =
            cluster.get_soc_desc(connected_physical_id).logical_eth_core_to_chan_map.at(connected_eth_core);
        telemetry_entries_.push_back(
            {result.coord, result.eth_channel, bw_GB_s, pps, connected_coord, connected_eth_channel});
    }
}

void TestContext::clear_telemetry() {
    telemetry_entries_.clear();

    // Get telemetry buffer address and size
    const auto telemetry_addr = tt::tt_metal::hal::get_erisc_l1_unreserved_base();
    const size_t telemetry_buffer_size = sizeof(LowResolutionBandwidthTelemetryResult);

    // Note: clear_buffer already clears all active ethernet cores regardless of registration
    get_eth_readback().clear_buffer(telemetry_addr, telemetry_buffer_size);
}

void TestContext::clear_code_profiling_buffers() {
    code_profiling_entries_.clear();
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

    get_eth_readback().clear_buffer(code_profiling_addr, code_profiling_buffer_size);
}

void TestContext::read_code_profiling_results() {
    code_profiling_entries_.clear();
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
    auto results = get_eth_readback().read_buffer(code_profiling_addr, code_profiling_buffer_size, true);

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
                code_profiling_entries_.push_back(
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

void TestContext::report_code_profiling_results() {
    if (code_profiling_entries_.empty()) {
        log_info(tt::LogTest, "Code Profiling Results: No data collected");
        return;
    }

    log_info(tt::LogTest, "Code Profiling Results:");

    // Helper function to get timer type name
    auto get_timer_type_name = [](CodeProfilingTimerType timer_type) -> std::string {
        switch (timer_type) {
            case CodeProfilingTimerType::RECEIVER_CHANNEL_FORWARD: return "RECEIVER_CHANNEL_FORWARD";
            default: return "UNKNOWN";
        }
    };

    for (const auto& entry : code_profiling_entries_) {
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

std::vector<GoldenLatencyEntry>::iterator TestContext::fetch_corresponding_golden_latency_entry(
    const LatencyResult& test_result) {
    auto golden_it = std::find_if(
        golden_latency_entries_.begin(), golden_latency_entries_.end(), [&](const GoldenLatencyEntry& golden) {
            return golden.test_name == test_result.test_name && golden.ftype == test_result.ftype &&
                   golden.ntype == test_result.ntype && golden.topology == test_result.topology &&
                   golden.num_devices == test_result.num_devices && golden.num_links == test_result.num_links &&
                   golden.payload_size == test_result.payload_size;
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

void TestContext::collect_latency_results() {
    log_info(tt::LogTest, "Collecting latency results from sender and responder devices");

    // Find sender and responder locations
    auto sender_location = get_latency_sender_location();
    auto responder_location = get_latency_receiver_location();

    // Get num_samples from sender config
    const auto& sender_configs = sender_location.device->get_senders().begin()->second.get_configs();
    uint32_t num_samples = sender_configs[0].first.parameters.num_packets;
    uint32_t result_buffer_size = num_samples * sizeof(uint32_t);

    // Read latency samples from sender device
    auto sender_result_data = fixture_->read_buffer_from_cores(
        sender_location.mesh_coord,
        {sender_location.core},
        sender_memory_map_.get_result_buffer_address(),
        result_buffer_size);

    // Read responder timestamps from responder device
    auto responder_result_data = fixture_->read_buffer_from_cores(
        responder_location.mesh_coord,
        {responder_location.core},
        sender_memory_map_.get_result_buffer_address(),
        result_buffer_size);

    log_info(tt::LogTest, "Collected {} latency samples from sender and responder", num_samples);
}

void TestContext::report_latency_results(const TestConfig& config) {
    log_info(tt::LogTest, "Reporting latency results for test: {}", config.parametrized_name);

    // Find sender and responder locations
    auto sender_location = get_latency_sender_location();
    auto responder_location = get_latency_receiver_location();

    const TestDevice* sender_device = sender_location.device;
    MeshCoordinate sender_coord = sender_location.mesh_coord;
    CoreCoord sender_core = sender_location.core;
    FabricNodeId sender_node_id = sender_location.node_id;

    MeshCoordinate responder_coord = responder_location.mesh_coord;
    CoreCoord responder_core = responder_location.core;
    FabricNodeId responder_node_id = responder_location.node_id;

    // Get latency parameters from sender config
    const auto& sender_configs = sender_device->get_senders().begin()->second.get_configs();
    const auto& sender_config = sender_configs[0].first;
    uint32_t num_samples = sender_config.parameters.num_packets;
    uint32_t payload_size = sender_config.parameters.payload_size_bytes;

    // Calculate number of hops between sender and responder
    uint32_t num_hops_to_responder = 0;
    auto hops_map = fixture_->get_hops_to_chip(sender_node_id, responder_node_id);
    for (const auto& [dir, hop_count] : hops_map) {
        num_hops_to_responder += hop_count;
    }
    TT_FATAL(num_hops_to_responder != 0, "Number of hops to responder is 0");
    // Multiply by 2 for round-trip (sender -> responder -> sender)
    num_hops_to_responder *= 2;

    uint32_t result_buffer_size = num_samples * sizeof(uint32_t);

    // Read latency samples from sender device
    auto sender_result_data = fixture_->read_buffer_from_cores(
        sender_coord, {sender_core}, sender_memory_map_.get_result_buffer_address(), result_buffer_size);
    const auto& sender_data = sender_result_data.at(sender_core);

    // Read responder elapsed times from responder device
    auto responder_result_data = fixture_->read_buffer_from_cores(
        responder_coord, {responder_core}, sender_memory_map_.get_result_buffer_address(), result_buffer_size);
    const auto& responder_data = responder_result_data.at(responder_core);

    // Parse elapsed times and compute latencies
    // Data is stored as uint32_t elapsed times (in cycles)
    std::vector<uint64_t> raw_latencies_cycles;
    std::vector<uint64_t> responder_times_cycles;
    std::vector<uint64_t> net_latencies_cycles;
    std::vector<uint64_t> per_hop_latency_cycles;

    raw_latencies_cycles.reserve(num_samples);
    responder_times_cycles.reserve(num_samples);
    net_latencies_cycles.reserve(num_samples);
    per_hop_latency_cycles.reserve(num_samples);

    for (uint32_t i = 0; i < num_samples; i++) {
        // Read elapsed times directly (already computed on device)
        uint64_t raw_latency = sender_data[i];
        uint64_t responder_time = responder_data[i];

        // Validate that responder time is reasonable
        TT_FATAL(raw_latency > 0, "Invalid sender latency (zero) for sample {}", i);
        TT_FATAL(responder_time > 0, "Invalid responder time (zero) for sample {}", i);

        // Check for clock synchronization issues between sender and responder devices
        // If responder time exceeds raw latency, this indicates unsynchronized clocks
        if (responder_time >= raw_latency) {
            log_warning(
                tt::LogTest,
                "Sample {}: Responder time ({} cycles) exceeds raw latency ({} cycles). "
                "This indicates clock drift/skew between devices. Sender and responder timestamps "
                "cannot be directly compared without clock synchronization.",
                i,
                responder_time,
                raw_latency);
            TT_FATAL(
                false,
                "Clock synchronization issue detected: responder processing time cannot exceed round-trip time. "
                "The sender device clock and responder device clock are not synchronized.");
        }

        uint64_t net_latency = raw_latency - responder_time;
        uint64_t per_hop_latency = net_latency / num_hops_to_responder;

        raw_latencies_cycles.push_back(raw_latency);
        responder_times_cycles.push_back(responder_time);
        net_latencies_cycles.push_back(net_latency);
        per_hop_latency_cycles.push_back(per_hop_latency);
    }

    if (raw_latencies_cycles.empty()) {
        log_warning(tt::LogTest, "No valid latency samples collected");
        return;
    }

    // Sort for percentile calculation
    std::sort(raw_latencies_cycles.begin(), raw_latencies_cycles.end());
    std::sort(responder_times_cycles.begin(), responder_times_cycles.end());
    std::sort(net_latencies_cycles.begin(), net_latencies_cycles.end());
    std::sort(per_hop_latency_cycles.begin(), per_hop_latency_cycles.end());

    // Get device frequency for conversion to ns
    uint32_t freq_mhz = get_device_frequency_mhz(sender_node_id);
    double freq_ghz = static_cast<double>(freq_mhz) / 1000.0;
    double ns_per_cycle = 1.0 / freq_ghz;

    // Helper lambda to calculate statistics
    auto calc_stats = [](const std::vector<uint64_t>& data) {
        struct Stats {
            uint64_t min, max, p50, p99;
            double avg;
        };
        Stats stats{};
        stats.min = data.front();
        stats.max = data.back();
        uint64_t sum = std::accumulate(data.begin(), data.end(), 0ULL);
        stats.avg = static_cast<double>(sum) / data.size();
        stats.p50 = data[data.size() / 2];
        stats.p99 = data[static_cast<size_t>(data.size() * 0.99)];
        return stats;
    };

    auto raw_stats = calc_stats(raw_latencies_cycles);
    auto responder_stats = calc_stats(responder_times_cycles);
    auto net_stats = calc_stats(net_latencies_cycles);
    auto per_hop_latency_stats = calc_stats(per_hop_latency_cycles);

    // Log results in table format
    log_info(tt::LogTest, "");
    log_info(tt::LogTest, "=== Latency Test Results for {} ===", config.parametrized_name);
    log_info(tt::LogTest, "Payload size: {} bytes | Num samples: {}", payload_size, raw_latencies_cycles.size());
    log_info(tt::LogTest, "");
    log_info(
        tt::LogTest, "Metric    Raw Latency (ns)    Responder Time (ns)    Net Latency (ns)    Per-Hop Latency (ns)");
    log_info(tt::LogTest, "------------------------------------------------------------------------");
    log_info(
        tt::LogTest,
        "Min       {:>15.2f}    {:>18.2f}    {:>15.2f}    {:>15.2f}",
        raw_stats.min * ns_per_cycle,
        responder_stats.min * ns_per_cycle,
        net_stats.min * ns_per_cycle,
        per_hop_latency_stats.min * ns_per_cycle);
    log_info(
        tt::LogTest,
        "Max       {:>15.2f}    {:>18.2f}    {:>15.2f}    {:>15.2f}",
        raw_stats.max * ns_per_cycle,
        responder_stats.max * ns_per_cycle,
        net_stats.max * ns_per_cycle,
        per_hop_latency_stats.max * ns_per_cycle);
    log_info(
        tt::LogTest,
        "Avg       {:>15.2f}    {:>18.2f}    {:>15.2f}    {:>15.2f}",
        raw_stats.avg * ns_per_cycle,
        responder_stats.avg * ns_per_cycle,
        net_stats.avg * ns_per_cycle,
        per_hop_latency_stats.avg * ns_per_cycle);
    log_info(
        tt::LogTest,
        "P50       {:>15.2f}    {:>18.2f}    {:>15.2f}    {:>15.2f}",
        raw_stats.p50 * ns_per_cycle,
        responder_stats.p50 * ns_per_cycle,
        net_stats.p50 * ns_per_cycle,
        per_hop_latency_stats.p50 * ns_per_cycle);
    log_info(
        tt::LogTest,
        "P99       {:>15.2f}    {:>18.2f}    {:>15.2f}    {:>15.2f}",
        raw_stats.p99 * ns_per_cycle,
        responder_stats.p99 * ns_per_cycle,
        net_stats.p99 * ns_per_cycle,
        per_hop_latency_stats.p99 * ns_per_cycle);
    log_info(tt::LogTest, "========================================================================");
    log_info(tt::LogTest, "");

    // Populate LatencyResult structure for CSV export
    LatencyResult latency_result;
    latency_result.test_name = config.name;

    // Extract ftype and ntype from first sender's first pattern
    const TrafficPatternConfig& first_pattern = fetch_first_traffic_pattern(config);
    latency_result.ftype = fetch_pattern_ftype(first_pattern);
    latency_result.ntype = fetch_pattern_ntype(first_pattern);

    latency_result.topology = enchantum::to_string(config.fabric_setup.topology);
    latency_result.num_devices = test_devices_.size();
    latency_result.num_links = config.fabric_setup.num_links;
    latency_result.num_samples = raw_latencies_cycles.size();
    latency_result.payload_size = payload_size;

    // Net latency statistics (most important)
    latency_result.net_min_ns = net_stats.min * ns_per_cycle;
    latency_result.net_max_ns = net_stats.max * ns_per_cycle;
    latency_result.net_avg_ns = net_stats.avg * ns_per_cycle;
    latency_result.net_p99_ns = net_stats.p99 * ns_per_cycle;

    // Responder processing time statistics
    latency_result.responder_min_ns = responder_stats.min * ns_per_cycle;
    latency_result.responder_max_ns = responder_stats.max * ns_per_cycle;
    latency_result.responder_avg_ns = responder_stats.avg * ns_per_cycle;
    latency_result.responder_p99_ns = responder_stats.p99 * ns_per_cycle;

    // Raw latency statistics
    latency_result.raw_min_ns = raw_stats.min * ns_per_cycle;
    latency_result.raw_max_ns = raw_stats.max * ns_per_cycle;
    latency_result.raw_avg_ns = raw_stats.avg * ns_per_cycle;
    latency_result.raw_p99_ns = raw_stats.p99 * ns_per_cycle;

    // Per-hop latency statistics
    latency_result.per_hop_min_ns = per_hop_latency_stats.min * ns_per_cycle;
    latency_result.per_hop_max_ns = per_hop_latency_stats.max * ns_per_cycle;
    latency_result.per_hop_avg_ns = per_hop_latency_stats.avg * ns_per_cycle;
    latency_result.per_hop_p99_ns = per_hop_latency_stats.p99 * ns_per_cycle;

    // Add to results vector
    latency_results_.push_back(latency_result);
}

// Setup latency test workers with latency-specific configurations
void TestContext::setup_latency_test_workers(TestConfig& config) {
    log_debug(tt::LogTest, "Latency test mode: manually populating sender and receiver workers");

    // Latency tests have exactly one sender with one pattern
    TT_FATAL(config.senders.size() == 1, "Latency test mode requires exactly one sender");
    TT_FATAL(config.senders[0].patterns.size() == 1, "Latency test mode requires exactly one pattern");

    const auto& sender = config.senders[0];
    const auto& pattern = sender.patterns[0];
    const auto& dest = pattern.destination.value();

    // Use default core if not specified (latency tests typically use a fixed core)
    CoreCoord sender_core = sender.core.value_or(CoreCoord{0, 0});
    CoreCoord receiver_core = dest.core.value_or(CoreCoord{0, 0});
    FabricNodeId sender_device_id = sender.device;
    FabricNodeId receiver_device_id = dest.device.value();

    // Create sender worker on sender device
    if (fixture_->is_local_fabric_node_id(sender_device_id)) {
        const auto& sender_coord = fixture_->get_device_coord(sender_device_id);
        auto& sender_test_device = test_devices_.at(sender_coord);

        // Create latency sender config with actual parameters
        TrafficParameters latency_traffic_params = {
            .chip_send_type = pattern.ftype.value(),
            .noc_send_type = pattern.ntype.value(),
            .payload_size_bytes = pattern.size.value(),
            .num_packets = pattern.num_packets.value(),
            .atomic_inc_val = pattern.atomic_inc_val,
            .mcast_start_hops = std::nullopt,
            .enable_flow_control = false,
            .seed = config.seed,
            .is_2D_routing_enabled = fixture_->is_2D_routing_enabled(),
            .mesh_shape = fixture_->get_mesh_shape(),
            .topology = fixture_->get_topology()};

        TestTrafficSenderConfig latency_sender_config = {
            .parameters = latency_traffic_params,
            .src_node_id = sender_device_id,
            .dst_node_ids = {receiver_device_id},
            .hops = std::nullopt,
            .mcast_start_node_id = std::nullopt,
            .dst_logical_core = receiver_core,
            .target_address = 0,
            .atomic_inc_address = std::nullopt,
            .dst_noc_encoding = fixture_->get_worker_noc_encoding(receiver_core),
            .payload_buffer_size = 0,
            .link_id = 0};

        sender_test_device.add_sender_traffic_config(sender_core, std::move(latency_sender_config));

        // Set latency sender kernel
        sender_test_device.set_sender_kernel_src(
            sender_core, "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_latency_sender.cpp");

        log_debug(
            tt::LogTest, "Created latency sender worker on device {} core {}", sender_device_id.chip_id, sender_core);
    }

    // Create receiver worker on receiver device
    if (fixture_->is_local_fabric_node_id(receiver_device_id)) {
        const auto& receiver_coord = fixture_->get_device_coord(receiver_device_id);
        auto& receiver_test_device = test_devices_.at(receiver_coord);

        // Create dummy receiver config just to populate the worker
        TestTrafficReceiverConfig dummy_receiver_config = {
            .parameters = TrafficParameters{},
            .sender_id = 0,
            .target_address = 0,
            .atomic_inc_address = std::nullopt,
            .payload_buffer_size = 0,
            .link_id = 0};

        receiver_test_device.add_receiver_traffic_config(receiver_core, dummy_receiver_config);

        // Set latency responder kernel
        receiver_test_device.set_receiver_kernel_src(
            receiver_core,
            "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_latency_responder.cpp");

        log_debug(
            tt::LogTest,
            "Created latency receiver worker on device {} core {}",
            receiver_device_id.chip_id,
            receiver_core);
    }
}

// Create latency kernels for a device based on its role (sender, responder, or neither)
void TestContext::create_latency_kernels_for_device(TestDevice& test_device) {
    // For latency tests, check if this device has any senders or receivers
    const auto& senders = test_device.get_senders();
    const auto& receivers = test_device.get_receivers();

    bool has_sender = !senders.empty();
    bool has_receiver = !receivers.empty();

    if (has_sender) {
        // This is the latency sender device
        // Get the sender core and config (there should be exactly one)
        TT_FATAL(senders.size() == 1, "Latency test should have exactly one sender per device");
        const auto& [sender_core, sender_worker] = *senders.begin();
        const auto& sender_configs = sender_worker.get_configs();
        TT_FATAL(!sender_configs.empty(), "Latency sender should have at least one config");

        // Extract parameters from the stored config
        const auto& sender_config = sender_configs[0].first;
        FabricNodeId responder_device_id = sender_config.dst_node_ids[0];

        // Find the responder device to get its virtual core coordinates
        TestDevice* responder_device = nullptr;
        for (auto& [responder_coord, responder_test_device] : test_devices_) {
            if (responder_test_device.get_node_id() == responder_device_id) {
                responder_device = &responder_test_device;
                break;
            }
        }
        TT_FATAL(
            responder_device != nullptr,
            "Could not find responder device with node_id {}",
            responder_device_id.chip_id);

        // Get responder's virtual core coordinates for correct NOC addressing
        CoreCoord responder_virtual_core =
            responder_device->get_device_info_provider()->get_virtual_core_from_logical_core(
                sender_config.dst_logical_core);

        // Create sender kernel - pass responder's actual coordinates
        test_device.create_latency_sender_kernel(
            sender_core,
            responder_device_id,
            sender_config.parameters.payload_size_bytes,
            sender_config.parameters.num_packets,
            sender_config.parameters.noc_send_type,
            responder_virtual_core);
    } else if (has_receiver) {
        // This is the latency responder device
        // Get the receiver core (there should be exactly one)
        TT_FATAL(receivers.size() == 1, "Latency test should have exactly one receiver per device");
        const auto& [receiver_core, receiver_worker] = *receivers.begin();

        // Find the sender device to query its parameters and buffer addresses
        auto sender_location = get_latency_sender_location();
        TestDevice* sender_device = sender_location.device;
        CoreCoord sender_core = sender_location.core;

        // Get sender's config to extract parameters
        const auto& sender_senders = sender_device->get_senders();
        const auto& sender_worker = sender_senders.at(sender_core);
        const auto& sender_configs = sender_worker.get_configs();
        const auto& sender_config = sender_configs[0].first;

        uint32_t payload_size = sender_config.parameters.payload_size_bytes;
        uint32_t num_samples = sender_config.parameters.num_packets;
        NocSendType noc_send_type = sender_config.parameters.noc_send_type;
        FabricNodeId sender_device_id = sender_config.src_node_id;

        // Get sender's actual buffer addresses (passing payload_size as parameter)
        uint32_t sender_send_buffer_address = sender_device->get_latency_send_buffer_address();
        uint32_t sender_receive_buffer_address = sender_device->get_latency_receive_buffer_address(payload_size);

        // Get sender's virtual core coordinates for correct NOC addressing
        CoreCoord sender_virtual_core =
            sender_device->get_device_info_provider()->get_virtual_core_from_logical_core(sender_core);

        // Create responder kernel - pass sender's actual addresses and coordinates
        test_device.create_latency_responder_kernel(
            receiver_core,
            sender_device_id,
            payload_size,
            num_samples,
            noc_send_type,
            sender_send_buffer_address,
            sender_receive_buffer_address,
            sender_virtual_core);
    } else {
        // For non-latency devices in a latency test, create normal kernels
        test_device.create_kernels();
    }
}

// Configures latency test mode - validates config and sets performance_test_mode flag
void TestContext::setup_latency_test_mode(const TestConfig& config) {
    TT_FATAL(
        config.performance_test_mode == PerformanceTestMode::LATENCY,
        "setup_latency_test_mode called when latency test mode is not enabled");

    this->set_performance_test_mode(PerformanceTestMode::LATENCY);

    // Validate that latency tests don't use multiple iterations
    TT_FATAL(
        config.iteration_number == 1 || config.iteration_number == 0,
        "Latency tests do not support multiple iterations. Use num_packets in the test config instead to "
        "collect multiple samples. Got {} iterations.",
        config.iteration_number);

    // Validate latency test structure
    TT_FATAL(config.senders.size() == 1, "Latency test mode requires exactly one sender");
    TT_FATAL(config.senders[0].patterns.size() == 1, "Latency test mode requires exactly one pattern");

    const auto& sender = config.senders[0];
    const auto& pattern = sender.patterns[0];
    const auto& dest = pattern.destination.value();

    log_info(
        tt::LogTest,
        "Latency test mode: sender={}, responder={}, payload={} bytes, samples={}",
        sender.device.chip_id,
        dest.device.value().chip_id,
        pattern.size.value(),
        pattern.num_packets.value());
}

void TestContext::compare_latency_results_with_golden() {
    if (golden_latency_entries_.empty()) {
        log_warning(tt::LogTest, "Skipping golden latency comparison - no golden file found");
        return;
    }
    if (latency_results_.size() != golden_latency_entries_.size()) {
        log_warning(
            tt::LogTest,
            "Number of latency results ({}) does not match number of golden entries ({})",
            latency_results_.size(),
            golden_latency_entries_.size());
    }

    for (const auto& test_result : latency_results_) {
        auto golden_it = fetch_corresponding_golden_latency_entry(test_result);

        // Create comparison result
        LatencyComparisonResult comp_result;
        comp_result.test_name = test_result.test_name;
        comp_result.ftype = test_result.ftype;
        comp_result.ntype = test_result.ntype;
        comp_result.topology = test_result.topology;
        comp_result.num_devices = test_result.num_devices;
        comp_result.num_links = test_result.num_links;
        comp_result.num_samples = test_result.num_samples;
        comp_result.payload_size = test_result.payload_size;
        comp_result.current_per_hop_avg_ns = test_result.per_hop_avg_ns;

        // Populate golden value and tolerance/status using common helper
        if (golden_it != golden_latency_entries_.end()) {
            comp_result.golden_per_hop_avg_ns = golden_it->per_hop_avg_ns;
        } else {
            comp_result.golden_per_hop_avg_ns = 0.0;
        }
        populate_comparison_tolerance_and_status(comp_result, golden_it, golden_latency_entries_.end());

        latency_comparison_results_.push_back(comp_result);

        // Only count as failure if golden entry exists and test failed
        // NO_GOLDEN status is just a warning, not a failure
        if (!comp_result.within_tolerance && comp_result.status != "NO_GOLDEN") {
            std::ostringstream oss;
            oss << comp_result.test_name << " [" << comp_result.status << "]";
            all_failed_latency_tests_.push_back(oss.str());
        }
    }

    // Initialize diff CSV file using common helper
    auto diff_csv_stream = init_diff_csv_file(
        latency_diff_csv_file_path_,
        "test_name,ftype,ntype,topology,num_devices,num_links,num_samples,payload_size,"
        "current_per_hop_avg_ns,golden_per_hop_avg_ns,difference_percent,status",
        "latency");

    if (diff_csv_stream.is_open()) {
        for (const auto& result : latency_comparison_results_) {
            diff_csv_stream << result.test_name << "," << result.ftype << "," << result.ntype << "," << result.topology
                            << "," << result.num_devices << "," << result.num_links << "," << result.num_samples << ","
                            << result.payload_size << "," << std::fixed << std::setprecision(2)
                            << result.current_per_hop_avg_ns << "," << result.golden_per_hop_avg_ns << ","
                            << result.difference_percent() << "," << result.status << "\n";
        }
        diff_csv_stream.close();
        log_info(
            tt::LogTest, "Latency comparison diff CSV results written to: {}", latency_diff_csv_file_path_.string());
    }
}

void TestContext::initialize_latency_results_csv_file() {
    // Create output directory
    std::filesystem::path tt_metal_home =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
    std::filesystem::path latency_results_path = tt_metal_home / output_dir;

    if (!std::filesystem::exists(latency_results_path)) {
        std::filesystem::create_directories(latency_results_path);
    }

    // Generate CSV filename (similar to bandwidth summary)
    // Note: The actual file will be created in generate_latency_results_csv() after golden is loaded,
    // similar to how bandwidth_summary_results_*.csv is created in generate_bandwidth_summary_csv()
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    std::ostringstream oss;
    oss << "latency_results_" << arch_name << ".csv";
    latency_csv_file_path_ = latency_results_path / oss.str();

    log_info(tt::LogTest, "Initialized latency CSV file path: {}", latency_csv_file_path_.string());
}

std::vector<std::string> TestContext::get_all_failed_tests() const {
    std::vector<std::string> combined;
    combined.insert(combined.end(), all_failed_bandwidth_tests_.begin(), all_failed_bandwidth_tests_.end());
    combined.insert(combined.end(), all_failed_latency_tests_.begin(), all_failed_latency_tests_.end());
    return combined;
}

void TestContext::generate_latency_results_csv() {
    // Open CSV file in write mode (truncate existing file, similar to bandwidth)
    std::ofstream csv_stream(latency_csv_file_path_, std::ios::out | std::ios::trunc);
    if (!csv_stream.is_open()) {
        log_error(tt::LogTest, "Failed to open latency CSV file for writing: {}", latency_csv_file_path_.string());
        return;
    }

    // Write header
    csv_stream << "test_name,ftype,ntype,topology,num_devices,num_links,num_samples,payload_size,"
                  "net_min_ns,net_max_ns,net_avg_ns,net_p99_ns,"
                  "responder_min_ns,responder_max_ns,responder_avg_ns,responder_p99_ns,"
                  "raw_min_ns,raw_max_ns,raw_avg_ns,raw_p99_ns,"
                  "per_hop_min_ns,per_hop_max_ns,per_hop_avg_ns,per_hop_p99_ns,tolerance_percent\n";

    // Write all results
    for (const auto& result : latency_results_) {
        csv_stream << result.test_name << "," << result.ftype << "," << result.ntype << "," << result.topology << ","
                   << result.num_devices << "," << result.num_links << "," << result.num_samples << ","
                   << result.payload_size << "," << std::fixed << std::setprecision(2) << result.net_min_ns << ","
                   << result.net_max_ns << "," << result.net_avg_ns << "," << result.net_p99_ns << ","
                   << result.responder_min_ns << "," << result.responder_max_ns << "," << result.responder_avg_ns << ","
                   << result.responder_p99_ns << "," << result.raw_min_ns << "," << result.raw_max_ns << ","
                   << result.raw_avg_ns << "," << result.raw_p99_ns << "," << result.per_hop_min_ns << ","
                   << result.per_hop_max_ns << "," << result.per_hop_avg_ns << "," << result.per_hop_p99_ns << ",";

        // Find the corresponding golden entry for tolerance (like bandwidth does)
        auto golden_it = fetch_corresponding_golden_latency_entry(result);
        if (golden_it == golden_latency_entries_.end()) {
            log_warning(
                tt::LogTest,
                "Golden latency entry not found for test {}, putting tolerance of 1.0 in CSV",
                result.test_name);
            csv_stream << 1.0;
        } else {
            csv_stream << golden_it->tolerance_percent;
        }
        csv_stream << "\n";
    }

    csv_stream.close();
    log_info(tt::LogTest, "Latency results written to CSV file: {}", latency_csv_file_path_.string());
}

std::string TestContext::get_golden_latency_csv_filename() {
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();

    // Convert cluster type enum to lowercase string
    std::string cluster_name = std::string(enchantum::to_string(cluster_type));
    std::transform(cluster_name.begin(), cluster_name.end(), cluster_name.begin(), ::tolower);

    std::string file_name = "golden_latency_" + arch_name + "_" + cluster_name + ".csv";
    return file_name;
}

bool TestContext::load_golden_latency_csv() {
    golden_latency_entries_.clear();

    std::string golden_filename = get_golden_latency_csv_filename();
    std::filesystem::path golden_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_metal/perf_microbenchmark/routing/golden" / golden_filename;

    if (!std::filesystem::exists(golden_path)) {
        log_warning(tt::LogTest, "Golden latency CSV file not found: {}", golden_path.string());
        return false;
    }

    std::ifstream golden_file(golden_path);
    if (!golden_file.is_open()) {
        log_error(tt::LogTest, "Failed to open golden latency CSV file: {}", golden_path.string());
        return false;
    }

    std::string line;
    bool is_header = true;
    while (std::getline(golden_file, line)) {
        if (is_header) {
            is_header = false;
            continue;  // Skip header
        }

        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        // Parse CSV line
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        // Expected format: test_name,ftype,ntype,topology,num_devices,num_links,num_samples,payload_size,
        //                  net_min_ns,net_max_ns,net_avg_ns,net_p99_ns,
        //                  responder_min_ns,responder_max_ns,responder_avg_ns,responder_p99_ns,
        //                  raw_min_ns,raw_max_ns,raw_avg_ns,raw_p99_ns,
        //                  per_hop_min_ns,per_hop_max_ns,per_hop_avg_ns,per_hop_p99_ns[,tolerance_percent]
        // Note: per_hop fields and tolerance_percent are optional for backward compatibility
        if (tokens.size() < 20) {
            log_error(
                tt::LogTest,
                "Invalid CSV format in golden latency file. Expected at least 20 fields, got {}",
                tokens.size());
            continue;
        }

        GoldenLatencyEntry entry;
        entry.test_name = tokens[0];
        entry.ftype = tokens[1];
        entry.ntype = tokens[2];
        entry.topology = tokens[3];
        entry.num_devices = std::stoul(tokens[4]);
        entry.num_links = std::stoul(tokens[5]);
        entry.num_samples = std::stoul(tokens[6]);
        entry.payload_size = std::stoul(tokens[7]);

        entry.net_min_ns = std::stod(tokens[8]);
        entry.net_max_ns = std::stod(tokens[9]);
        entry.net_avg_ns = std::stod(tokens[10]);
        entry.net_p99_ns = std::stod(tokens[11]);

        entry.responder_min_ns = std::stod(tokens[12]);
        entry.responder_max_ns = std::stod(tokens[13]);
        entry.responder_avg_ns = std::stod(tokens[14]);
        entry.responder_p99_ns = std::stod(tokens[15]);

        entry.raw_min_ns = std::stod(tokens[16]);
        entry.raw_max_ns = std::stod(tokens[17]);
        entry.raw_avg_ns = std::stod(tokens[18]);
        entry.raw_p99_ns = std::stod(tokens[19]);

        // Per-hop fields are optional for backward compatibility
        if (tokens.size() >= 24) {
            entry.per_hop_min_ns = std::stod(tokens[20]);
            entry.per_hop_max_ns = std::stod(tokens[21]);
            entry.per_hop_avg_ns = std::stod(tokens[22]);
            entry.per_hop_p99_ns = std::stod(tokens[23]);
        } else {
            // If per-hop fields are missing, set to 0
            entry.per_hop_min_ns = 0.0;
            entry.per_hop_max_ns = 0.0;
            entry.per_hop_avg_ns = 0.0;
            entry.per_hop_p99_ns = 0.0;
        }

        // Tolerance is optional for backward compatibility
        if (tokens.size() >= 25) {
            entry.tolerance_percent = std::stod(tokens[24]);
        } else if (tokens.size() >= 21 && tokens.size() < 24) {
            // Old format: tolerance is at position 20
            entry.tolerance_percent = std::stod(tokens[20]);
        } else {
            entry.tolerance_percent = 10.0;  // Default tolerance if not specified
        }
        golden_latency_entries_.push_back(entry);
    }

    golden_file.close();
    log_info(
        tt::LogTest, "Loaded {} golden latency entries from: {}", golden_latency_entries_.size(), golden_path.string());
    return true;
}

void TestContext::validate_against_golden() {
    // Handle both bandwidth and latency comparisons
    bool has_bandwidth_results = !comparison_results_.empty();
    bool has_latency_results = !latency_comparison_results_.empty();

    if (!has_bandwidth_results && !has_latency_results) {
        log_info(tt::LogTest, "No golden comparison performed (no golden file found)");
        return;
    }

    // Report bandwidth failures separately
    if (has_bandwidth_results) {
        if (!all_failed_bandwidth_tests_.empty()) {
            has_test_failures_ = true;
            log_error(tt::LogTest, "=== BANDWIDTH TEST FAILURES ===");
            log_error(
                tt::LogTest,
                "{} bandwidth test(s) failed golden comparison (using per-test tolerance):",
                all_failed_bandwidth_tests_.size());

            // Print detailed failure information
            for (const auto& result : comparison_results_) {
                if (!result.within_tolerance && result.status != "NO_GOLDEN") {
                    // Look up tolerance from golden entry by searching directly
                    double tolerance = 1.0;
                    for (const auto& golden : golden_csv_entries_) {
                        if (golden.test_name == result.test_name && golden.ftype == result.ftype &&
                            golden.ntype == result.ntype && golden.topology == result.topology &&
                            golden.num_links == result.num_links && golden.packet_size == result.packet_size) {
                            tolerance = golden.tolerance_percent;
                            break;
                        }
                    }

                    log_error(tt::LogTest, "  - {} [{}]:", result.test_name, result.status);
                    log_error(tt::LogTest, "      Expected: {:.6f} GB/s", result.golden_bandwidth_GB_s);
                    log_error(tt::LogTest, "      Actual:   {:.6f} GB/s", result.current_bandwidth_GB_s);
                    log_error(
                        tt::LogTest,
                        "      Diff:     {:.2f}% (tolerance: {:.2f}%)",
                        result.difference_percent(),
                        tolerance);
                }
            }
        } else {
            log_info(
                tt::LogTest,
                "All {} bandwidth tests passed golden comparison using per-test tolerance values",
                comparison_results_.size());
        }
    }

    // Report latency failures separately
    if (has_latency_results) {
        if (!all_failed_latency_tests_.empty()) {
            has_test_failures_ = true;
            log_error(tt::LogTest, "=== LATENCY TEST FAILURES ===");
            log_error(
                tt::LogTest,
                "{} latency test(s) failed golden comparison (using per-test tolerance):",
                all_failed_latency_tests_.size());

            // Print detailed failure information
            for (const auto& result : latency_comparison_results_) {
                if (!result.within_tolerance && result.status != "NO_GOLDEN") {
                    // Look up tolerance from golden entry by searching directly
                    double tolerance = 1.0;
                    for (const auto& golden : golden_latency_entries_) {
                        if (golden.test_name == result.test_name && golden.ftype == result.ftype &&
                            golden.ntype == result.ntype && golden.topology == result.topology &&
                            golden.num_devices == result.num_devices && golden.num_links == result.num_links &&
                            golden.payload_size == result.payload_size) {
                            tolerance = golden.tolerance_percent;
                            break;
                        }
                    }

                    log_error(tt::LogTest, "  - {} [{}]:", result.test_name, result.status);
                    log_error(
                        tt::LogTest,
                        "      Test config: {} {} {} {} devs {} links payload={}B",
                        result.ftype,
                        result.ntype,
                        result.topology,
                        result.num_devices,
                        result.num_links,
                        result.payload_size);
                    log_error(tt::LogTest, "      Expected per-hop: {:.2f} ns", result.golden_per_hop_avg_ns);
                    log_error(tt::LogTest, "      Actual per-hop:   {:.2f} ns", result.current_per_hop_avg_ns);
                    log_error(
                        tt::LogTest,
                        "      Diff:     {:.2f}% (tolerance: {:.2f}%)",
                        result.difference_percent(),
                        tolerance);
                }
            }
        } else {
            log_info(
                tt::LogTest,
                "All {} latency tests passed golden comparison using per-test tolerance values",
                latency_comparison_results_.size());
        }
    }
}
