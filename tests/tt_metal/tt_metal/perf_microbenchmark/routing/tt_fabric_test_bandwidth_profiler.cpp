// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_bandwidth_profiler.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <limits>
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/tt_fabric_test_constants.hpp"

BandwidthProfiler::BandwidthProfiler(
    IDeviceInfoProvider& device_info, IRouteManager& route_manager, TestFixture& fixture) :
    device_info_(device_info), route_manager_(route_manager), fixture_(fixture) {}

void BandwidthProfiler::profile_results(
    const TestConfig& config,
    const std::unordered_map<MeshCoordinate, TestDevice>& test_devices,
    const SenderMemoryMap& sender_memory_map) {
    // Clear per-test state
    latest_results_.clear();
    latest_result_ = BandwidthResult{};
    latest_summary_ = BandwidthResultSummary{};
    outgoing_traffic_.clear();
    device_direction_cycles_.clear();
    device_core_cycles_.clear();

    calculate_outgoing_traffics_through_device_boundaries(test_devices);
    read_performance_results(test_devices, sender_memory_map);
    convert_core_cycles_to_direction_cycles(test_devices);
    calculate_bandwidth(config, test_devices);
}

void BandwidthProfiler::set_telemetry_bandwidth(double min, double avg, double max) {
    telemetry_bw_min_ = min;
    telemetry_bw_avg_ = avg;
    telemetry_bw_max_ = max;
}

void BandwidthProfiler::reset() {
    latest_result_ = BandwidthResult{};
    latest_summary_ = BandwidthResultSummary{};
    latest_results_.clear();
    outgoing_traffic_.clear();
    device_direction_cycles_.clear();
    device_core_cycles_.clear();
    telemetry_bw_min_.reset();
    telemetry_bw_avg_.reset();
    telemetry_bw_max_.reset();
}

std::map<FabricNodeId, std::map<RoutingDirection, uint32_t>>
BandwidthProfiler::calculate_outgoing_traffics_through_device_boundaries(
    const std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    outgoing_traffic_.clear();  // Clear previous data

    log_debug(tt::LogTest, "Calculating outgoing traffic through device boundaries");

    // Process each test device and its sender configurations
    for (const auto& [device_coord, test_device] : test_devices) {
        const auto& src_node_id = test_device.get_node_id();

        // Process regular senders only (ignore sync senders)
        for (const auto& [core_coord, sender] : test_device.get_senders()) {
            (void)core_coord;
            for (const auto& [config, _] : sender.get_configs()) {
                uint32_t link_id = config.link_id;
                if (link_id == 0) {
                    trace_traffic_path(src_node_id, config);
                }
            }
        }
    }
    return outgoing_traffic_;
}

void BandwidthProfiler::trace_traffic_path(const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config) {
    // Use proper topology detection from fixture
    if (fixture_.get_topology() == Topology::Ring) {
        // Ring topology - use ring traversal logic with boundary turning
        trace_ring_traffic_path(src_node_id, config);
    } else {
        // Regular hop-based tracing for linear/mesh topologies
        trace_line_or_mesh_traffic_path(src_node_id, config);
    }
}

void BandwidthProfiler::trace_ring_traffic_path(
    const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config) {
    const auto& hops = config.hops;

    // Find the initial direction and total hops for ring traversal
    for (const auto& [initial_direction, hop_count] : *hops) {
        if (hop_count == 0) {
            continue;
        }

        // Use the appropriate ring traversal helper based on mesh type
        std::vector<std::pair<FabricNodeId, RoutingDirection>> ring_path;

        // Check if this is a wrap-around mesh
        bool is_wrap_around = fixture_.wrap_around_mesh(src_node_id);

        if (is_wrap_around) {
            // Use the existing wrap-around mesh logic
            ring_path = fixture_.trace_wrap_around_mesh_ring_path(src_node_id, initial_direction, hop_count);
        } else {
            // Use the new non wrap-around mesh logic
            ring_path = fixture_.trace_ring_path(src_node_id, initial_direction, hop_count);
        }

        // Count traffic at each device boundary
        // ring_path contains (destination_node, direction_to_reach_it)
        // We need to record traffic from the SOURCE node that sent in that direction
        FabricNodeId source_node = src_node_id;
        for (const auto& [destination_node, direction] : ring_path) {
            outgoing_traffic_[source_node][direction]++;
            source_node = destination_node;  // Move to next source for the next hop
        }
    }
}

void BandwidthProfiler::trace_line_or_mesh_traffic_path(
    const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config) {
    auto remaining_hops = config.hops.value();  // Make a copy to modify
    FabricNodeId current_node = src_node_id;

    // For mesh topology, use dimension-order routing
    // Continue until all hops are consumed
    while (true) {
        // Check if all remaining hops are 0
        bool all_hops_zero = true;
        for (const auto& [direction, hop_count] : remaining_hops) {
            if (hop_count > 0) {
                all_hops_zero = false;
                break;
            }
        }
        if (all_hops_zero) {
            break;  // No more hops to process
        }

        // Find the next direction to route in
        RoutingDirection next_direction = route_manager_.get_forwarding_direction(remaining_hops);

        // Check if we have any remaining hops in this direction
        if (!remaining_hops.contains(next_direction) || remaining_hops[next_direction] == 0) {
            // If no hops left in this direction, mark as 0 and continue
            remaining_hops[next_direction] = 0;
            continue;
        }

        // Consume one hop in the chosen direction
        remaining_hops[next_direction]--;

        // Log traffic from current node in the chosen direction
        outgoing_traffic_[current_node][next_direction]++;

        // Move to next node in this direction
        current_node = route_manager_.get_neighbor_node_id(current_node, next_direction);
    }
}

void BandwidthProfiler::read_performance_results(
    const std::unordered_map<MeshCoordinate, TestDevice>& test_devices, const SenderMemoryMap& sender_memory_map) {
    // Clear previous data
    device_core_cycles_.clear();

    log_debug(tt::LogTest, "Reading performance results from sender cores");

    // Fixed group size for concurrent reads
    constexpr uint32_t MAX_CONCURRENT_DEVICES = 16;

    // Prepare read operation tracking
    struct DeviceReadInfo {
        MeshCoordinate device_coord;
        FabricNodeId device_node_id;
        std::vector<CoreCoord> sender_cores;
        TestFixture::ReadBufferOperation read_op;
    };

    // Collect all devices that need reading
    std::vector<DeviceReadInfo> all_devices;
    for (const auto& [device_coord, test_device] : test_devices) {
        const auto& device_node_id = test_device.get_node_id();

        // Get sender cores (excluding sync cores)
        std::vector<CoreCoord> sender_cores;
        sender_cores.reserve(test_device.get_senders().size());
        for (const auto& [core, _] : test_device.get_senders()) {
            sender_cores.push_back(core);
        }

        if (!sender_cores.empty()) {
            all_devices.push_back({device_coord, device_node_id, sender_cores, {}});
        }
    }

    // Process devices in groups
    for (size_t group_start = 0; group_start < all_devices.size(); group_start += MAX_CONCURRENT_DEVICES) {
        size_t group_end = std::min(group_start + MAX_CONCURRENT_DEVICES, all_devices.size());

        log_debug(
            tt::LogTest, "Processing device group {}-{} of {}", group_start, group_end - 1, all_devices.size() - 1);

        // First loop: Initiate non-blocking reads for group
        for (size_t i = group_start; i < group_end; ++i) {
            auto& device = all_devices[i];
            device.read_op = fixture_.initiate_read_buffer_from_cores(
                device.device_coord,
                device.sender_cores,
                sender_memory_map.get_result_buffer_address(),
                sender_memory_map.get_result_buffer_size());
        }

        // Barrier to wait for all reads in this group to complete
        fixture_.barrier_reads();

        // Second loop: Process completed results
        for (size_t i = group_start; i < group_end; ++i) {
            auto& device = all_devices[i];
            auto data = fixture_.complete_read_buffer_from_cores(device.read_op);

            // Extract cycles from each core and store in map
            for (const auto& [core, core_data] : data) {
                // Cycles are stored as 64-bit value split across two 32-bit words
                uint32_t cycles_low = core_data[TT_FABRIC_CYCLES_INDEX];
                uint32_t cycles_high = core_data[TT_FABRIC_CYCLES_INDEX + 1];
                uint64_t total_cycles = static_cast<uint64_t>(cycles_high) << 32 | cycles_low;

                device_core_cycles_[device.device_node_id][core] = total_cycles;
            }
        }
    }
}

void BandwidthProfiler::convert_core_cycles_to_direction_cycles(
    const std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    // Clear previous data
    device_direction_cycles_.clear();

    log_debug(tt::LogTest, "Converting core cycles to direction cycles");

    for (const auto& [device_coord, test_device] : test_devices) {
        (void)device_coord;
        const auto& device_node_id = test_device.get_node_id();

        // Process each sender core
        for (const auto& [core, sender] : test_device.get_senders()) {
            // Get cycles for this core (if available)
            if (!device_core_cycles_.contains(device_node_id) || !device_core_cycles_[device_node_id].contains(core)) {
                continue;
            }

            uint64_t core_cycles = device_core_cycles_[device_node_id][core];

            // Get unique (direction, link_id) pairs this core sends traffic to
            std::set<std::pair<RoutingDirection, uint32_t>> core_direction_links;
            for (const auto& [config, _] : sender.get_configs()) {
                RoutingDirection direction = route_manager_.get_forwarding_direction(*config.hops);
                uint32_t link_id = config.link_id;
                core_direction_links.insert({direction, link_id});
            }

            // Add cycles to each (direction, link_id) pair this core sends to
            // Only one core per device should send in each (direction, link) combination
            for (const auto& [direction, link_id] : core_direction_links) {
                if (device_direction_cycles_[device_node_id][direction].contains(link_id)) {
                    TT_THROW(
                        "Multiple cores on device {} are sending traffic in direction {} on link {}. "
                        "Only one core per device should send in each (direction, link) combination.",
                        device_node_id.chip_id,
                        direction,
                        link_id);
                }
                device_direction_cycles_[device_node_id][direction][link_id] = core_cycles;
            }
        }
    }
}

void BandwidthProfiler::calculate_bandwidth(
    const TestConfig& config, const std::unordered_map<MeshCoordinate, TestDevice>& test_devices) {
    log_debug(tt::LogTest, "Calculating bandwidth (GB/s) by direction:");

    // Clear previous bandwidth results
    latest_results_.clear();

    uint64_t max_cycles = 0;
    uint32_t max_traffic_count = 0;
    // Calculate total bytes and packets sent in this direction from this device
    uint64_t total_bytes = 0;
    uint32_t total_packets = 0;
    uint32_t total_traffic_count = 0;
    uint32_t packet_size = 0;
    uint32_t num_packets = 0;
    uint32_t device_freq = std::numeric_limits<uint32_t>::max();
    std::set<uint32_t> num_devices_set;

    // Pre-compute topology information (moved outside inner loop)
    const auto mesh_shape = fixture_.get_mesh_shape();
    const auto topology = fixture_.get_topology();
    // Pre-compute sender config lookup cache to avoid O(n³) search in inner loop
    std::unordered_map<std::string, std::tuple<uint32_t, uint32_t, uint32_t>> config_cache;
    for (const auto& [device_coord, test_device] : test_devices) {
        (void)device_coord;
        const auto& device_id = test_device.get_node_id();
        for (const auto& [core, sender] : test_device.get_senders()) {
            (void)core;
            for (const auto& [sender_config, _] : sender.get_configs()) {
                RoutingDirection config_direction = route_manager_.get_forwarding_direction(sender_config.hops.value());
                uint32_t config_link_id = sender_config.link_id;

                // Create cache key: device_id + direction + link_id
                std::string cache_key = std::to_string(device_id.chip_id) + "_" +
                                        std::to_string(static_cast<int>(config_direction)) + "_" +
                                        std::to_string(config_link_id);

                config_cache[cache_key] = std::make_tuple(
                    sender_config.parameters.payload_size_bytes,
                    sender_config.parameters.num_packets,
                    sender_config.parameters.payload_size_bytes  // packet_size
                );
            }
        }
    }

    for (const auto& [device_id, direction_map] : device_direction_cycles_) {
        for (const auto& [direction, link_map] : direction_map) {
            // Calculate num_devices once per direction (moved outside link loop)
            uint32_t num_devices = 0;
            if (topology == Topology::Linear) {
                if (direction == RoutingDirection::N or direction == RoutingDirection::S) {
                    num_devices = mesh_shape[0];
                } else {
                    num_devices = mesh_shape[1];
                }
            } else if (topology == Topology::Ring) {
                num_devices = 2 * (mesh_shape[0] - 1 + mesh_shape[1] - 1);
            } else if (topology == Topology::Mesh) {
                num_devices = mesh_shape[0] * mesh_shape[1];
            }

            for (const auto& [link_id, cycles] : link_map) {
                if (cycles == 0) {
                    continue;  // Skip to avoid division by zero
                }

                // Get traffic count for this device and direction
                if (outgoing_traffic_.contains(device_id) && outgoing_traffic_[device_id].contains(direction)) {
                    total_traffic_count = outgoing_traffic_[device_id][direction];
                }

                // calculate the max for summary info
                max_cycles = std::max(max_cycles, cycles);
                max_traffic_count = std::max(max_traffic_count, total_traffic_count);

                // Use cache lookup instead of triply nested loop (O(1) vs O(n³))
                std::string cache_key = std::to_string(device_id.chip_id) + "_" +
                                        std::to_string(static_cast<int>(direction)) + "_" + std::to_string(link_id);

                TT_FATAL(
                    config_cache.contains(cache_key),
                    "Config not found in cache for device {} direction {} link {}",
                    device_id.chip_id,
                    static_cast<int>(direction),
                    link_id);
                auto [payload_size_bytes, num_packets_val, packet_size_val] = config_cache.at(cache_key);
                num_packets = num_packets_val;
                packet_size = packet_size_val;
                total_bytes = static_cast<uint64_t>(payload_size_bytes) * num_packets * total_traffic_count;
                total_packets = static_cast<uint64_t>(num_packets) * total_traffic_count;

                // Calculate bandwidth in Bytes/cycle and convert to GB/s
                const auto device_frequency_mhz = device_info_.get_device_frequency_mhz(device_id);
                uint32_t device_frequency_hz = device_frequency_mhz * 1e6;
                // use min frequency (in real scenario we will have the same freq)
                device_freq = std::min(device_freq, device_frequency_hz);
                const auto duration_seconds = static_cast<double>(cycles) / static_cast<double>(device_frequency_hz);

                double bandwidth_bytes_per_cycle = static_cast<double>(total_bytes) / static_cast<double>(cycles);
                double bandwidth_GB_s = (bandwidth_bytes_per_cycle * device_frequency_mhz) / 1e3;
                double packets_per_second = static_cast<double>(total_packets) / duration_seconds;

                // save all possible num devices
                num_devices_set.insert(num_devices);

                auto bw_result = BandwidthResult{
                    .num_devices = num_devices,
                    .device_id = device_id.chip_id,
                    .direction = direction,
                    .total_traffic_count = total_traffic_count,
                    .num_packets = num_packets,
                    .packet_size = packet_size,
                    .cycles = cycles,
                    .bandwidth_GB_s = bandwidth_GB_s,
                    .packets_per_second = packets_per_second};

                if (telemetry_bw_min_.has_value()) {
                    bw_result.telemetry_bw_GB_s_min = telemetry_bw_min_;
                    bw_result.telemetry_bw_GB_s_avg = telemetry_bw_avg_;
                    bw_result.telemetry_bw_GB_s_max = telemetry_bw_max_;
                }

                // Store result for CSV generation (using GB/s)
                latest_results_.emplace_back(bw_result);
            }
        }
    }

    // Pick last result as latest_result_ for compatibility with previous behavior
    if (!latest_results_.empty()) {
        latest_result_ = latest_results_.back();
    }

    // Calculate and store a summary of this test
    total_bytes = static_cast<uint64_t>(packet_size) * num_packets * max_traffic_count;
    double bandwidth_bytes_per_cycle = static_cast<double>(total_bytes) / static_cast<double>(max_cycles);
    double bandwidth_GB_s = (bandwidth_bytes_per_cycle * device_freq) / 1e9;

    // Calculate packets per second
    double duration_seconds = static_cast<double>(max_cycles) / static_cast<double>(device_freq);
    double packets_per_second = static_cast<double>(max_traffic_count * num_packets) / duration_seconds;

    // Build summary entry
    const TrafficPatternConfig& first_pattern = fetch_first_traffic_pattern(config);
    std::string ftype_str = fetch_pattern_ftype(first_pattern);
    std::string ntype_str = fetch_pattern_ntype(first_pattern);
    uint32_t num_packets_first_pattern = fetch_pattern_num_packets(first_pattern);
    uint32_t packet_size_first_pattern = fetch_pattern_packet_size(first_pattern);

    // Create a new entry that represents all iterations of the same test (per call we handle single iteration)
    // Get max_packet_size from FabricContext (already accounts for user override and topology)
    uint32_t max_pkt_size = tt::tt_metal::MetalContext::instance()
                                .get_control_plane()
                                .get_fabric_context()
                                .get_fabric_max_payload_size_bytes();

    latest_summary_ = BandwidthResultSummary{
        .test_name = config.name,
        .num_iterations = 1,
        .ftype = ftype_str,
        .ntype = ntype_str,
        .topology = std::string(enchantum::to_string(config.fabric_setup.topology)),
        .num_links = config.fabric_setup.num_links,
        .num_packets = num_packets_first_pattern,
        .num_devices = std::vector<uint32_t>(num_devices_set.begin(), num_devices_set.end()),
        .packet_size = packet_size_first_pattern,
        .cycles_vector = {static_cast<double>(max_cycles)},
        .bandwidth_vector_GB_s = {bandwidth_GB_s},
        .packets_per_second_vector = {packets_per_second},
        .max_packet_size = max_pkt_size,
    };
}
