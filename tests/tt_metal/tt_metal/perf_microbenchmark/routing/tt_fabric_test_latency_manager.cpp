// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_latency_manager.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"

LatencyTestManager::LatencyTestManager(TestFixture& fixture, SenderMemoryMap& sender_memory_map) :
    fixture_(fixture), sender_memory_map_(sender_memory_map) {}

void LatencyTestManager::create_latency_kernels_for_device(
    TestDevice& test_device, std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    const auto& senders = test_device.get_senders();
    const auto& receivers = test_device.get_receivers();

    bool has_sender = !senders.empty();
    bool has_receiver = !receivers.empty();

    if (has_sender) {
        TT_FATAL(senders.size() == 1, "Latency test should have exactly one sender per device");
        const auto& [sender_core, sender_worker] = *senders.begin();
        const auto& sender_configs = sender_worker.get_configs();
        TT_FATAL(!sender_configs.empty(), "Latency sender should have at least one config");

        const auto& sender_config = sender_configs[0].first;
        FabricNodeId responder_device_id = sender_config.dst_node_ids[0];

        TestDevice* responder_device = nullptr;
        for (auto& [responder_coord, responder_test_device] : test_devices) {
            if (responder_test_device.get_node_id() == responder_device_id) {
                responder_device = &responder_test_device;
                (void)responder_coord;
                break;
            }
        }
        TT_FATAL(
            responder_device != nullptr,
            "Could not find responder device with node_id {}",
            responder_device_id.chip_id);

        CoreCoord responder_virtual_core =
            responder_device->get_device_info_provider()->get_virtual_core_from_logical_core(
                sender_config.dst_logical_core);

        test_device.create_latency_sender_kernel(
            sender_core,
            responder_device_id,
            sender_config.parameters.payload_size_bytes,
            sender_config.parameters.num_packets,
            sender_config.parameters.noc_send_type,
            responder_virtual_core);
    } else if (has_receiver) {
        TT_FATAL(receivers.size() == 1, "Latency test should have exactly one receiver per device");
        const auto& [receiver_core, receiver_worker] = *receivers.begin();

        auto sender_location = get_latency_sender_location(test_devices);
        TestDevice* sender_device = sender_location.device;
        CoreCoord sender_core = sender_location.core;

        const auto& sender_senders = sender_device->get_senders();
        const auto& sender_worker = sender_senders.at(sender_core);
        const auto& sender_configs = sender_worker.get_configs();
        const auto& sender_config = sender_configs[0].first;

        uint32_t payload_size = sender_config.parameters.payload_size_bytes;
        uint32_t num_samples = sender_config.parameters.num_packets;
        NocSendType noc_send_type = sender_config.parameters.noc_send_type;
        FabricNodeId sender_device_id = sender_config.src_node_id;

        uint32_t sender_send_buffer_address = sender_device->get_latency_send_buffer_address();
        uint32_t sender_receive_buffer_address = sender_device->get_latency_receive_buffer_address(payload_size);

        CoreCoord sender_virtual_core =
            sender_device->get_device_info_provider()->get_virtual_core_from_logical_core(sender_core);

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
        test_device.create_kernels();
    }
}

std::ofstream LatencyTestManager::init_diff_csv_file(
    std::filesystem::path& diff_csv_path, const std::string& csv_header, const std::string& test_type) {
    std::filesystem::path output_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        std::string(OUTPUT_DIR);
    std::ostringstream diff_oss;
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    diff_oss << test_type << "_diff_" << arch_name << ".csv";
    diff_csv_path = output_path / diff_oss.str();

    std::ofstream diff_csv_stream(diff_csv_path, std::ios::out | std::ios::trunc);
    if (!diff_csv_stream.is_open()) {
        log_error(tt::LogTest, "Failed to create {} diff CSV file: {}", test_type, diff_csv_path.string());
    } else {
        diff_csv_stream << csv_header << "\n";
        log_info(tt::LogTest, "Initialized {} diff CSV file: {}", test_type, diff_csv_path.string());
    }
    return diff_csv_stream;
}

void LatencyTestManager::setup_latency_test_mode(const TestConfig& config) {
    TT_FATAL(
        config.performance_test_mode == PerformanceTestMode::LATENCY,
        "setup_latency_test_mode called when latency test mode is not enabled");

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

void LatencyTestManager::setup_latency_test_workers(
    TestConfig& config, std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
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
    if (fixture_.is_local_fabric_node_id(sender_device_id)) {
        const auto& sender_coord = fixture_.get_device_coord(sender_device_id);
        auto& sender_test_device = test_devices.at(sender_coord);

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
            .is_2D_routing_enabled = fixture_.is_2D_routing_enabled(),
            .mesh_shape = fixture_.get_mesh_shape(),
            .topology = fixture_.get_topology()};

        TestTrafficSenderConfig latency_sender_config = {
            .parameters = latency_traffic_params,
            .src_node_id = sender_device_id,
            .dst_node_ids = {receiver_device_id},
            .hops = std::nullopt,
            .mcast_start_node_id = std::nullopt,
            .dst_logical_core = receiver_core,
            .target_address = 0,
            .atomic_inc_address = std::nullopt,
            .dst_noc_encoding = fixture_.get_worker_noc_encoding(receiver_core),
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
    if (fixture_.is_local_fabric_node_id(receiver_device_id)) {
        const auto& receiver_coord = fixture_.get_device_coord(receiver_device_id);
        auto& receiver_test_device = test_devices.at(receiver_coord);

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

LatencyTestManager::LatencyWorkerLocation LatencyTestManager::get_latency_sender_location(
    std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    return find_latency_worker_device(
        test_devices, [](TestDevice& d) -> const auto& { return d.get_senders(); }, "sender");
}

LatencyTestManager::LatencyWorkerLocation LatencyTestManager::get_latency_receiver_location(
    std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    return find_latency_worker_device(
        test_devices, [](TestDevice& d) -> const auto& { return d.get_receivers(); }, "receiver");
}

void LatencyTestManager::collect_latency_results(std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    log_info(tt::LogTest, "Collecting latency results from sender and responder devices");

    // Find sender and responder locations
    auto sender_location = get_latency_sender_location(test_devices);
    auto responder_location = get_latency_receiver_location(test_devices);

    // Get num_samples from sender config
    const auto& sender_configs = sender_location.device->get_senders().begin()->second.get_configs();
    uint32_t num_samples = sender_configs[0].first.parameters.num_packets;
    uint32_t result_buffer_size = num_samples * sizeof(uint32_t);

    // Read latency samples from sender device
    fixture_.read_buffer_from_cores(
        sender_location.mesh_coord,
        {sender_location.core},
        sender_memory_map_.get_result_buffer_address(),
        result_buffer_size);

    // Read responder timestamps from responder device
    fixture_.read_buffer_from_cores(
        responder_location.mesh_coord,
        {responder_location.core},
        sender_memory_map_.get_result_buffer_address(),
        result_buffer_size);

    log_info(tt::LogTest, "Collected {} latency samples from sender and responder", num_samples);
}

void LatencyTestManager::report_latency_results(
    const TestConfig& config, std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    log_info(tt::LogTest, "Reporting latency results for test: {}", config.parametrized_name);

    // Find sender and responder locations
    auto sender_location = get_latency_sender_location(test_devices);
    auto responder_location = get_latency_receiver_location(test_devices);

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
    auto hops_map = fixture_.get_hops_to_chip(sender_node_id, responder_node_id);
    for (const auto& [dir, hop_count] : hops_map) {
        (void)dir;
        num_hops_to_responder += hop_count;
    }
    TT_FATAL(num_hops_to_responder != 0, "Number of hops to responder is 0");
    // Multiply by 2 for round-trip (sender -> responder -> sender)
    num_hops_to_responder *= 2;

    uint32_t result_buffer_size = num_samples * sizeof(uint32_t);

    // Read latency samples from sender device
    auto sender_result_data = fixture_.read_buffer_from_cores(
        sender_coord, {sender_core}, sender_memory_map_.get_result_buffer_address(), result_buffer_size);
    const auto& sender_data = sender_result_data.at(sender_core);

    // Read responder elapsed times from responder device
    auto responder_result_data = fixture_.read_buffer_from_cores(
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
    uint32_t freq_mhz = fixture_.get_device_frequency_mhz(sender_node_id);
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
    latency_result.num_devices = test_devices.size();
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

void LatencyTestManager::initialize_latency_results_csv_file() {
    // Create output directory
    std::filesystem::path tt_metal_home =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
    std::filesystem::path latency_results_path = tt_metal_home / std::string(OUTPUT_DIR);

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

void LatencyTestManager::generate_latency_results_csv() {
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
        auto golden_it = std::find_if(
            golden_latency_entries_.begin(), golden_latency_entries_.end(), [&](const GoldenLatencyEntry& golden) {
                return golden.test_name == result.test_name && golden.ftype == result.ftype &&
                       golden.ntype == result.ntype && golden.topology == result.topology &&
                       golden.num_devices == result.num_devices && golden.num_links == result.num_links &&
                       golden.payload_size == result.payload_size;
            });

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

std::string LatencyTestManager::get_golden_latency_csv_filename() {
    auto arch_name = tt::tt_metal::hal::get_arch_name();
    auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();

    // Convert cluster type enum to lowercase string
    std::string cluster_name = std::string(enchantum::to_string(cluster_type));
    std::transform(cluster_name.begin(), cluster_name.end(), cluster_name.begin(), ::tolower);

    std::string file_name = "golden_latency_" + arch_name + "_" + cluster_name + ".csv";
    return file_name;
}

bool LatencyTestManager::load_golden_latency_csv() {
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

void LatencyTestManager::compare_latency_results_with_golden() {
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
        auto golden_it = std::find_if(
            golden_latency_entries_.begin(), golden_latency_entries_.end(), [&](const GoldenLatencyEntry& golden) {
                return golden.test_name == test_result.test_name && golden.ftype == test_result.ftype &&
                       golden.ntype == test_result.ntype && golden.topology == test_result.topology &&
                       golden.num_devices == test_result.num_devices && golden.num_links == test_result.num_links &&
                       golden.payload_size == test_result.payload_size;
            });

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

void LatencyTestManager::validate_against_golden() {
    bool has_latency_results = !latency_comparison_results_.empty();

    // Report latency failures separately
    if (has_latency_results) {
        if (!all_failed_latency_tests_.empty()) {
            has_failures_ = true;
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

void LatencyTestManager::generate_latency_summary() {
    // Load golden latency CSV file
    load_golden_latency_csv();

    // Generate latency results CSV file with all results
    generate_latency_results_csv();

    // Compare latency results with golden CSV
    compare_latency_results_with_golden();

    // Validate latency results against golden
    validate_against_golden();
}

void LatencyTestManager::setup_ci_artifacts() {
    std::filesystem::path tt_metal_home =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
    std::filesystem::path ci_artifacts_path = tt_metal_home / std::string(CI_ARTIFACTS_DIR);
    if (!std::filesystem::exists(ci_artifacts_path)) {
        try {
            std::filesystem::create_directories(ci_artifacts_path);
        } catch (const std::filesystem::filesystem_error& e) {
            log_error(
                tt::LogTest, "Failed to create CI artifacts directory, skipping CI artifacts creation: {}", e.what());
            return;
        }
    }

    // Latency artifacts
    for (const std::filesystem::path& csv_filepath : {latency_csv_file_path_, latency_diff_csv_file_path_}) {
        if (csv_filepath.empty()) {
            continue;
        }
        try {
            std::filesystem::copy_file(
                csv_filepath,
                ci_artifacts_path / csv_filepath.filename(),
                std::filesystem::copy_options::overwrite_existing);
        } catch (const std::filesystem::filesystem_error& e) {
            log_debug(
                tt::LogTest,
                "Failed to copy CSV file {} to CI artifacts directory: {}",
                csv_filepath.filename().string(),
                e.what());
        }
    }
    log_trace(tt::LogTest, "Copied latency CSV files to CI artifacts directory: {}", ci_artifacts_path.string());
}

void LatencyTestManager::reset_state() {
    latency_comparison_results_.clear();
    all_failed_latency_tests_.clear();
    has_failures_ = false;
}
