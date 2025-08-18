// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <filesystem>
#include <memory>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <optional>
#include <set>
#include <sstream>
#include <enchantum/enchantum.hpp>

#include "tt_fabric_test_config.hpp"
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_device_setup.hpp"
#include "tt_fabric_test_traffic.hpp"
#include "tt_fabric_test_allocator.hpp"
#include "tt_fabric_test_memory_map.hpp"

// Constants
const std::string output_dir = "generated/fabric";
const std::string default_built_tests_dump_file = "built_tests.yaml";

using TestFixture = tt::tt_fabric::fabric_tests::TestFixture;
using TestDevice = tt::tt_fabric::fabric_tests::TestDevice;
using TestConfig = tt::tt_fabric::fabric_tests::TestConfig;
using TestFabricSetup = tt::tt_fabric::fabric_tests::TestFabricSetup;
using TrafficParameters = tt::tt_fabric::fabric_tests::TrafficParameters;
using TestTrafficConfig = tt::tt_fabric::fabric_tests::TestTrafficConfig;
using TestTrafficSenderConfig = tt::tt_fabric::fabric_tests::TestTrafficSenderConfig;
using TestTrafficReceiverConfig = tt::tt_fabric::fabric_tests::TestTrafficReceiverConfig;
using TestWorkerType = tt::tt_fabric::fabric_tests::TestWorkerType;

using ChipSendType = tt::tt_fabric::ChipSendType;
using NocSendType = tt::tt_fabric::NocSendType;
using FabricNodeId = tt::tt_fabric::FabricNodeId;
using RoutingDirection = tt::tt_fabric::RoutingDirection;

using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;

using TestConfigBuilder = tt::tt_fabric::fabric_tests::TestConfigBuilder;
using YamlConfigParser = tt::tt_fabric::fabric_tests::YamlConfigParser;
using CmdlineParser = tt::tt_fabric::fabric_tests::CmdlineParser;
using YamlTestConfigSerializer = tt::tt_fabric::fabric_tests::YamlTestConfigSerializer;
using ParsedTestConfig = tt::tt_fabric::fabric_tests::ParsedTestConfig;

using Topology = tt::tt_fabric::Topology;
using FabricConfig = tt::tt_fabric::FabricConfig;
using RoutingType = tt::tt_fabric::fabric_tests::RoutingType;

// Bandwidth measurement result structures
struct BandwidthResult {
    uint32_t num_devices;
    uint32_t device_id;
    RoutingDirection direction;
    uint32_t total_traffic_count;
    uint32_t num_packets;
    uint32_t packet_size;
    uint64_t cycles;
    double bandwidth_gb_s;
    double packets_per_second;
};

struct BandwidthResultSummary {
    std::vector<uint32_t> num_devices;
    uint32_t packet_size;
    uint64_t cycles;
    double bandwidth_gb_s;
    double packets_per_second;
};

// Golden CSV comparison structures
struct GoldenCsvEntry {
    std::string test_name;
    std::string ftype;
    std::string ntype;
    std::string topology;
    std::string num_devices;
    uint32_t num_links;
    uint32_t packet_size;
    uint64_t cycles;
    double bandwidth_gb_s;
    double packets_per_second;
    double tolerance_percent;  // Per-test tolerance percentage
};

struct ComparisonResult {
    std::string test_name;
    std::string ftype;
    std::string ntype;
    std::string topology;
    std::string num_devices;
    uint32_t num_links;
    uint32_t packet_size;
    double current_bandwidth_gb_s;
    double golden_bandwidth_gb_s;
    double difference_percent;
    bool within_tolerance;
    std::string status;
};

class TestContext {
public:
    void init(std::shared_ptr<TestFixture> fixture, const tt::tt_fabric::fabric_tests::AllocatorPolicies& policies) {
        fixture_ = std::move(fixture);
        allocation_policies_ = policies;

        // Initialize memory maps for all available devices
        initialize_memory_maps();

        // Create allocator with memory maps
        this->allocator_ = std::make_unique<tt::tt_fabric::fabric_tests::GlobalAllocator>(
            *this->fixture_, *this->fixture_, policies, sender_memory_map_, receiver_memory_map_);
    }

    uint32_t get_randomized_master_seed() const { return fixture_->get_randomized_master_seed(); }

    void setup_devices() {
        const auto& available_coords = this->fixture_->get_host_local_device_coordinates();
        for (const auto& coord : available_coords) {
            // Create TestDevice with access to memory maps
            test_devices_.emplace(
                coord, TestDevice(coord, this->fixture_, this->fixture_, &sender_memory_map_, &receiver_memory_map_));
        }
    }

    void reset_devices() {
        test_devices_.clear();
        device_global_sync_cores_.clear();
        device_local_sync_cores_.clear();
        this->allocator_->reset();
        reset_local_variables();
    }

    void process_traffic_config(TestConfig& config) {
        this->allocator_->allocate_resources(config);
        log_info(tt::LogTest, "Resource allocation complete");

        if (config.global_sync) {
            // set it only after the test_config is built since it needs set the sync value during expand the high-level
            // patterns.
            this->set_global_sync(config.global_sync);
            this->set_global_sync_val(config.global_sync_val);
            this->set_benchmark_mode(config.benchmark_mode);

            log_info(tt::LogTest, "Enabled sync, global sync value: {}, ", global_sync_val_);
            log_info(tt::LogTest, "Ubenchmark mode: {}, ", benchmark_mode_);

            for (const auto& sync_sender : config.global_sync_configs) {
                // currently initializing our sync configs to be on senders local to the current hos
                if (fixture_->is_local_fabric_node_id(sync_sender.device)) {
                    CoreCoord sync_core = sync_sender.core.value();
                    const auto& device_coord = this->fixture_->get_device_coord(sync_sender.device);

                    // Track global sync core for this device
                    device_global_sync_cores_[sync_sender.device] = sync_core;

                    // Process each already-split sync pattern for this device
                    for (const auto& sync_pattern : sync_sender.patterns) {
                        // Convert sync pattern to TestTrafficSenderConfig format
                        const auto& dest = sync_pattern.destination.value();

                        // Patterns are now already split into single-direction hops
                        auto single_direction_hops = dest.hops.value();

                        TrafficParameters sync_traffic_parameters = {
                            .chip_send_type = sync_pattern.ftype.value(),
                            .noc_send_type = sync_pattern.ntype.value(),
                            .payload_size_bytes = sync_pattern.size.value(),
                            .num_packets = sync_pattern.num_packets.value(),
                            .atomic_inc_val = sync_pattern.atomic_inc_val,
                            .atomic_inc_wrap = sync_pattern.atomic_inc_wrap,
                            .mcast_start_hops = sync_pattern.mcast_start_hops,
                            .seed = config.seed,
                            .is_2D_routing_enabled = fixture_->is_2D_routing_enabled(),
                            .is_dynamic_routing_enabled = fixture_->is_dynamic_routing_enabled(),
                            .mesh_shape = this->fixture_->get_mesh_shape(),
                        };

                        // For sync patterns, we use a dummy destination core and fixed sync address
                        // The actual sync will be handled by atomic operations
                        CoreCoord dummy_dst_core = {0, 0};  // Sync doesn't need specific dst core
                        uint32_t sync_address =
                            this->sender_memory_map_.get_global_sync_address();  // Hard-coded sync address
                        uint32_t dst_noc_encoding =
                            this->fixture_->get_worker_noc_encoding(sync_core);  // populate the master coord

                        // for 2d mcast case
                        auto dst_node_ids = this->fixture_->get_dst_node_ids_from_hops(
                            sync_sender.device, single_direction_hops, sync_traffic_parameters.chip_send_type);

                        // for 2d, we need to spcify the mcast start node id
                        std::optional<FabricNodeId> mcast_start_node_id = std::nullopt;
                        if (fixture_->is_2D_routing_enabled() &&
                            sync_traffic_parameters.chip_send_type == ChipSendType::CHIP_MULTICAST) {
                            mcast_start_node_id =
                                fixture_->get_mcast_start_node_id(sync_sender.device, single_direction_hops);
                        }

                        TestTrafficSenderConfig sync_config = {
                            .parameters = sync_traffic_parameters,
                            .src_node_id = sync_sender.device,
                            .dst_node_ids = dst_node_ids,   // Empty for multicast sync
                            .hops = single_direction_hops,  // Use already single-direction hops
                            .mcast_start_node_id = mcast_start_node_id,
                            .dst_logical_core = dummy_dst_core,
                            .target_address = sync_address,
                            .atomic_inc_address = sync_address,
                            .dst_noc_encoding = dst_noc_encoding};

                        // Add sync config to the master sender on this device
                        this->test_devices_.at(device_coord).add_sender_sync_config(sync_core, std::move(sync_config));
                    }
                }
            }

            // Validate that all sync cores have the same coordinate
            if (!device_global_sync_cores_.empty()) {
                CoreCoord reference_sync_core = device_global_sync_cores_.begin()->second;
                for (const auto& [device_id, sync_core] : device_global_sync_cores_) {
                    if (sync_core.x != reference_sync_core.x || sync_core.y != reference_sync_core.y) {
                        TT_THROW(
                            "Global sync requires all devices to use the same sync core coordinate. "
                            "Device {} uses sync core ({}, {}) but expected ({}, {}) based on first device.",
                            device_id.chip_id,
                            sync_core.x,
                            sync_core.y,
                            reference_sync_core.x,
                            reference_sync_core.y);
                    }
                }
                log_info(
                    tt::LogTest,
                    "Validated sync core consistency: all {} devices use sync core ({}, {})",
                    device_global_sync_cores_.size(),
                    reference_sync_core.x,
                    reference_sync_core.y);
            }
        }

        for (const auto& sender : config.senders) {
            for (const auto& pattern : sender.patterns) {
                // Track local sync core for this device
                device_local_sync_cores_[sender.device].push_back(sender.core.value());

                // The allocator has already filled in all the necessary details.
                // We just need to construct the TrafficConfig and pass it to add_traffic_config.
                const auto& dest = pattern.destination.value();

                TrafficParameters traffic_parameters = {
                    .chip_send_type = pattern.ftype.value(),
                    .noc_send_type = pattern.ntype.value(),
                    .payload_size_bytes = pattern.size.value(),
                    .num_packets = pattern.num_packets.value(),
                    .atomic_inc_val = pattern.atomic_inc_val,
                    .atomic_inc_wrap = pattern.atomic_inc_wrap,
                    .mcast_start_hops = pattern.mcast_start_hops,
                    .seed = config.seed,
                    .is_2D_routing_enabled = fixture_->is_2D_routing_enabled(),
                    .is_dynamic_routing_enabled = fixture_->is_dynamic_routing_enabled(),
                    .mesh_shape = this->fixture_->get_mesh_shape(),
                };

                TestTrafficConfig traffic_config = {
                    .parameters = traffic_parameters,
                    .src_node_id = sender.device,
                    .src_logical_core = sender.core,
                    .dst_logical_core = dest.core,
                    .target_address = dest.target_address,
                    .atomic_inc_address = dest.atomic_inc_address,
                    .link_id = sender.link_id,
                };

                if (dest.device.has_value()) {
                    traffic_config.dst_node_ids = {dest.device.value()};
                }
                if (dest.hops.has_value()) {
                    traffic_config.hops = dest.hops;
                }

                this->add_traffic_config(traffic_config);
            }
        }
    }

    void open_devices(const TestFabricSetup& fabric_setup) { fixture_->open_devices(fabric_setup); }

    void initialize_sync_memory() {
        if (!global_sync_) {
            return;  // Only initialize sync memory if line sync is enabled
        }

        log_info(tt::LogTest, "Initializing sync memory for line sync");

        // Initialize sync memory location with 16 bytes of zeros on all devices
        uint32_t global_sync_address = this->sender_memory_map_.get_global_sync_address();
        uint32_t global_sync_memory_size = this->sender_memory_map_.get_global_sync_region_size();
        uint32_t local_sync_address = this->sender_memory_map_.get_local_sync_address();
        uint32_t local_sync_memory_size = this->sender_memory_map_.get_local_sync_region_size();

        // clear the global sync cores in device_global_sync_cores_ using zero_out_buffer_on_cores
        for (const auto& [device_id, global_sync_core] : device_global_sync_cores_) {
            if (fixture_->is_local_fabric_node_id(device_id)) {
                const auto& device_coord = fixture_->get_device_coord(device_id);
                std::vector<CoreCoord> cores = {global_sync_core};
                // zero out the global sync address for global sync core
                fixture_->zero_out_buffer_on_cores(device_coord, cores, global_sync_address, global_sync_memory_size);
                // also need to zero out the local sync address for global sync core
                fixture_->zero_out_buffer_on_cores(device_coord, cores, local_sync_address, global_sync_memory_size);
            }
        }

        // clear the local sync cores in device_local_sync_cores_ using zero_out_buffer_on_cores
        for (const auto& [device_id, local_sync_cores] : device_local_sync_cores_) {
            if (fixture_->is_local_fabric_node_id(device_id)) {
                const auto& device_coord = fixture_->get_device_coord(device_id);
                fixture_->zero_out_buffer_on_cores(
                    device_coord, local_sync_cores, local_sync_address, local_sync_memory_size);
            }
        }

        log_info(
            tt::LogTest,
            "Sync memory initialization complete at address: {} and address: {}",
            global_sync_address,
            local_sync_address);
    }

    void compile_programs() {
        fixture_->setup_workload();
        // TODO: should we be taking const ref?
        for (auto& [coord, test_device] : test_devices_) {
            test_device.set_benchmark_mode(benchmark_mode_);
            test_device.set_global_sync(global_sync_);
            test_device.set_global_sync_val(global_sync_val_);

            auto device_id = test_device.get_node_id();
            test_device.set_sync_core(device_global_sync_cores_[device_id]);

            test_device.create_kernels();
            auto& program_handle = test_device.get_program_handle();
            if (program_handle.num_kernels()) {
                fixture_->enqueue_program(coord, std::move(program_handle));
            }
        }
    }

    void launch_programs() { fixture_->run_programs(); }

    void wait_for_programs() { fixture_->wait_for_programs(); }

    void validate_results() {
        for (const auto& [_, test_device] : test_devices_) {
            test_device.validate_results();
        }
    }

    void profile_results(const TestConfig& config) {
        // calculate per device boundary/direction traffic passing through
        calculate_outgoing_traffics_through_device_boundaries();

        // Read performance results from sender cores
        read_performance_results();

        // Convert core cycles to direction-based cycles
        convert_core_cycles_to_direction_cycles();

        // Calculate and print bandwidth (Bytes/cycle)
        calculate_bandwidth(config);

        // Generate CSV file with bandwidth results
        generate_bandwidth_csv(config);

        // validate perf with golden csv
        generate_comparison_csv(config);
        validate_against_golden();
    }

    void initialize_csv_file() {
        // Create output directory
        std::filesystem::path output_path =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / output_dir;

        if (!std::filesystem::exists(output_path)) {
            std::filesystem::create_directories(output_path);
        }

        auto arch_name = tt::tt_metal::hal::get_arch_name();

        // Generate detailed CSV filename
        std::ostringstream oss;
        oss << "bandwidth_results_" << arch_name << ".csv";
        csv_file_path_ = output_path / oss.str();

        // Create detailed CSV file with header
        std::ofstream csv_stream(csv_file_path_, std::ios::out | std::ios::trunc);  // Truncate file
        if (!csv_stream.is_open()) {
            log_error(tt::LogTest, "Failed to create CSV file: {}", csv_file_path_.string());
            return;
        }

        // Write detailed header
        csv_stream
            << "test_name,ftype,ntype,topology,num_devices,device,num_links,direction,total_traffic_count,num_packets,"
               "packet_size,cycles,"
               "bandwidth_gb_s,packets_per_second\n";
        csv_stream.close();

        log_info(tt::LogTest, "Initialized CSV file: {}", csv_file_path_.string());

        // Generate summary CSV filename
        std::ostringstream summary_oss;
        summary_oss << "bandwidth_summary_results_" << arch_name << ".csv";
        csv_summary_file_path_ = output_path / summary_oss.str();

        // Create summary CSV file with header
        std::ofstream summary_csv_stream(csv_summary_file_path_, std::ios::out | std::ios::trunc);  // Truncate file
        if (!summary_csv_stream.is_open()) {
            log_error(tt::LogTest, "Failed to create summary CSV file: {}", csv_summary_file_path_.string());
            return;
        }

        // Write summary header
        summary_csv_stream << "test_name,ftype,ntype,topology,num_devices,num_links,packet_size,cycles,bandwidth_gb_s,"
                              "packets_per_second,tolerance_percent\n";
        summary_csv_stream.close();

        log_info(tt::LogTest, "Initialized summary CSV file: {}", csv_summary_file_path_.string());

        // Initialize diff CSV file for golden comparison
        std::ostringstream diff_oss;
        diff_oss << "bandwidth_summary_results_" << arch_name << "_diff.csv";
        diff_csv_file_path_ = output_path / diff_oss.str();

        // Create diff CSV file with header
        std::ofstream diff_csv_stream(diff_csv_file_path_, std::ios::out | std::ios::trunc);  // Truncate file
        if (!diff_csv_stream.is_open()) {
            log_error(tt::LogTest, "Failed to create diff CSV file: {}", diff_csv_file_path_.string());
            return;
        }

        // Write diff header
        diff_csv_stream << "test_name,ftype,ntype,topology,num_devices,num_links,packet_size,"
                           "current_bandwidth_gb_s,golden_bandwidth_gb_s,difference_percent,status\n";
        diff_csv_stream.close();

        log_info(tt::LogTest, "Initialized diff CSV file: {}", diff_csv_file_path_.string());

        // load golden csv based on arch and cluster type
        load_golden_csv();
    }

    void close_devices() { fixture_->close_devices(); }

    void set_benchmark_mode(bool benchmark_mode) { benchmark_mode_ = benchmark_mode; }

    bool get_benchmark_mode() { return benchmark_mode_; }

    void set_global_sync(bool global_sync) { global_sync_ = global_sync; }

    void set_global_sync_val(uint32_t val) { global_sync_val_ = val; }

    bool has_test_failures() const { return has_test_failures_; }

    const std::vector<std::string>& get_all_failed_tests() const { return all_failed_tests_; }

    // Determine tolerance percentage by looking up from golden CSV entries
    double get_tolerance_percent(
        const std::string& test_name,
        const std::string& ftype,
        const std::string& ntype,
        const std::string& topology,
        const std::string& num_devices,
        uint32_t num_links,
        uint32_t packet_size) const {
        // Search for matching entry in golden CSV
        auto golden_it =
            std::find_if(golden_csv_entries_.begin(), golden_csv_entries_.end(), [&](const GoldenCsvEntry& golden) {
                return golden.test_name == test_name && golden.ftype == ftype && golden.ntype == ntype &&
                       golden.topology == topology && golden.num_devices == num_devices &&
                       golden.num_links == num_links && golden.packet_size == packet_size;
            });

        if (golden_it != golden_csv_entries_.end()) {
            return golden_it->tolerance_percent;
        }

        log_warning(
            tt::LogTest,
            "No golden entry found for tolerance lookup: {}, {}, {}, {}, {}, {}, {} - using default 5.0%",
            test_name,
            ftype,
            ntype,
            topology,
            num_devices,
            num_links,
            packet_size);
        return 0.0;
    }

private:
    void reset_local_variables() {
        benchmark_mode_ = false;
        global_sync_ = false;
        global_sync_val_ = 0;
        outgoing_traffic_.clear();
        device_direction_cycles_.clear();
        device_core_cycles_.clear();
        bandwidth_results_.clear();
        bandwidth_results_summary_.clear();
        comparison_results_.clear();
        failed_tests_.clear();
        // Note: has_test_failures_ is NOT reset here to preserve failures across tests
        // Note: golden_csv_entries_ is kept loaded for reuse across tests
    }

    void add_traffic_config(const TestTrafficConfig& traffic_config) {
        // This function now assumes all allocation has been done by the GlobalAllocator.
        // It is responsible for taking the planned config and setting up the TestDevice objects.
        const auto& src_node_id = traffic_config.src_node_id;

        CoreCoord src_logical_core = traffic_config.src_logical_core.value();
        CoreCoord dst_logical_core = traffic_config.dst_logical_core.value();
        uint32_t target_address = traffic_config.target_address.value_or(0);
        uint32_t atomic_inc_address = traffic_config.atomic_inc_address.value_or(0);

        std::vector<FabricNodeId> dst_node_ids;
        std::optional<std::unordered_map<RoutingDirection, uint32_t>> hops = std::nullopt;

        if (traffic_config.hops.has_value()) {
            hops = traffic_config.hops;
            dst_node_ids = this->fixture_->get_dst_node_ids_from_hops(
                traffic_config.src_node_id, hops.value(), traffic_config.parameters.chip_send_type);
        } else {
            dst_node_ids = traffic_config.dst_node_ids.value();

            // assign hops for 2d LL and 1D
            if (!(fixture_->is_dynamic_routing_enabled())) {
                hops = this->fixture_->get_hops_to_chip(src_node_id, dst_node_ids[0]);
            }
        }

        // for 2d, we need to spcify the mcast start node id
        // TODO: in future, we should be able to specify the mcast start node id in the traffic config
        std::optional<FabricNodeId> mcast_start_node_id = std::nullopt;
        if (fixture_->is_2D_routing_enabled() &&
            traffic_config.parameters.chip_send_type == ChipSendType::CHIP_MULTICAST) {
            mcast_start_node_id = fixture_->get_mcast_start_node_id(src_node_id, hops.value());
        }

        uint32_t dst_noc_encoding = this->fixture_->get_worker_noc_encoding(dst_logical_core);
        uint32_t sender_id = fixture_->get_worker_id(traffic_config.src_node_id, src_logical_core);

        // Get payload buffer size from receiver memory map (cached during initialization)
        uint32_t payload_buffer_size = receiver_memory_map_.get_payload_chunk_size();

        TestTrafficSenderConfig sender_config = {
            .parameters = traffic_config.parameters,
            .src_node_id = traffic_config.src_node_id,
            .dst_node_ids = dst_node_ids,
            .hops = hops,
            .mcast_start_node_id = mcast_start_node_id,
            .dst_logical_core = dst_logical_core,
            .target_address = target_address,
            .atomic_inc_address = atomic_inc_address,
            .dst_noc_encoding = dst_noc_encoding,
            .payload_buffer_size = payload_buffer_size,
            .link_id = traffic_config.link_id};

        TestTrafficReceiverConfig receiver_config = {
            .parameters = traffic_config.parameters,
            .sender_id = sender_id,
            .target_address = target_address,
            .atomic_inc_address = atomic_inc_address,
            .payload_buffer_size = payload_buffer_size};

        if (fixture_->is_local_fabric_node_id(src_node_id)) {
            const auto& src_coord = this->fixture_->get_device_coord(src_node_id);
            auto& src_test_device = this->test_devices_.at(src_coord);
            src_test_device.add_sender_traffic_config(src_logical_core, std::move(sender_config));
        }

        for (const auto& dst_node_id : dst_node_ids) {
            if (fixture_->is_local_fabric_node_id(dst_node_id)) {
                const auto& dst_coord = this->fixture_->get_device_coord(dst_node_id);
                this->test_devices_.at(dst_coord).add_receiver_traffic_config(dst_logical_core, receiver_config);
            }
        }
    }

    void initialize_memory_maps() {
        // Get uniform L1 memory layout (same across all devices)
        uint32_t l1_unreserved_base = this->fixture_->get_l1_unreserved_base();
        uint32_t l1_unreserved_size = this->fixture_->get_l1_unreserved_size();
        uint32_t l1_alignment = this->fixture_->get_l1_alignment();
        uint32_t default_payload_chunk_size = allocation_policies_.default_payload_chunk_size;
        uint32_t max_configs_per_core = std::max(
            allocation_policies_.sender_config.max_configs_per_core,
            allocation_policies_.receiver_config.max_configs_per_core);

        // Create memory maps directly using constructors
        sender_memory_map_ = tt::tt_fabric::fabric_tests::SenderMemoryMap(
            l1_unreserved_base, l1_unreserved_size, l1_alignment, max_configs_per_core);

        receiver_memory_map_ = tt::tt_fabric::fabric_tests::ReceiverMemoryMap(
            l1_unreserved_base, l1_unreserved_size, l1_alignment, default_payload_chunk_size, max_configs_per_core);

        // Validate memory maps
        if (!sender_memory_map_.is_valid() || !receiver_memory_map_.is_valid()) {
            TT_THROW("Invalid memory map configuration");
        }
    }

    std::map<FabricNodeId, std::map<RoutingDirection, uint32_t>>
    calculate_outgoing_traffics_through_device_boundaries() {
        outgoing_traffic_.clear();  // Clear previous data

        log_debug(tt::LogTest, "Calculating outgoing traffic through device boundaries");

        // Process each test device and its sender configurations
        for (const auto& [device_coord, test_device] : test_devices_) {
            const auto& src_node_id = test_device.get_node_id();

            // Process regular senders only (ignore sync senders)
            for (const auto& [core_coord, sender] : test_device.get_senders()) {
                for (const auto& [config, fabric_conn_idx] : sender.get_configs()) {
                    // trace only one of the links, use link 0 as default
                    uint32_t link_id = config.link_id.value_or(0);
                    if (link_id == 0) {
                        trace_traffic_path(src_node_id, config);
                    }
                }
            }
        }

        // Log the results for debugging (automatically sorted)
        for (const auto& [node_id, device_traffic] : outgoing_traffic_) {
            if (!device_traffic.empty()) {
                for (const auto& [direction, count] : device_traffic) {
                    if (count > 0) {
                        log_debug(
                            tt::LogTest, "Device {} Direction {} Traffic Count: {}", node_id.chip_id, direction, count);
                    }
                }
            }
        }

        return outgoing_traffic_;
    }

    void trace_traffic_path(const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config) {
        const auto& hops = config.hops;

        // Use proper topology detection from fixture
        if (fixture_->get_topology() == Topology::Ring) {
            // Ring topology - use ring traversal logic with boundary turning
            trace_ring_traffic_path(src_node_id, config);
        } else {
            // Regular hop-based tracing for linear/mesh topologies
            trace_line_or_mesh_traffic_path(src_node_id, config);
        }
    }

    void trace_ring_traffic_path(const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config) {
        const auto& hops = config.hops;

        // Find the initial direction and total hops for ring traversal
        for (const auto& [initial_direction, hop_count] : *hops) {
            if (hop_count == 0) {
                continue;
            }

            // Use the appropriate ring traversal helper based on mesh type
            std::vector<std::pair<FabricNodeId, RoutingDirection>> ring_path;

            // Check if this is a wrap-around mesh
            bool is_wrap_around = fixture_->wrap_around_mesh(src_node_id);

            if (is_wrap_around) {
                // Use the existing wrap-around mesh logic
                ring_path = fixture_->trace_wrap_around_mesh_ring_path(src_node_id, initial_direction, hop_count);
            } else {
                // Use the new non wrap-around mesh logic
                ring_path = fixture_->trace_ring_path(src_node_id, initial_direction, hop_count);
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

    void trace_line_or_mesh_traffic_path(const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config) {
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
            RoutingDirection next_direction = fixture_->get_forwarding_direction(remaining_hops);

            // Check if we have any remaining hops in this direction
            if (remaining_hops.count(next_direction) == 0 || remaining_hops[next_direction] == 0) {
                break;  // No more hops to process
            }

            uint32_t hops_in_direction = remaining_hops[next_direction];

            // Trace all hops in this direction sequentially
            for (uint32_t hop = 0; hop < hops_in_direction; hop++) {
                // Record traffic from current node in this direction
                outgoing_traffic_[current_node][next_direction]++;

                // Move to next node in this direction
                current_node = fixture_->get_neighbor_node_id(current_node, next_direction);
            }

            // Mark this direction as completed
            remaining_hops[next_direction] = 0;
        }
    }

    void read_performance_results() {
        // Clear previous data
        device_core_cycles_.clear();

        log_info(tt::LogTest, "Reading performance results from sender cores");

        // Process each test device
        for (const auto& [device_coord, test_device] : test_devices_) {
            const auto& device_node_id = test_device.get_node_id();

            // Get sender cores (excluding sync cores)
            std::vector<CoreCoord> sender_cores;
            sender_cores.reserve(test_device.get_senders().size());
            for (const auto& [core, _] : test_device.get_senders()) {
                sender_cores.push_back(core);
            }

            if (sender_cores.empty()) {
                continue;
            }

            // Read buffer data from sender cores
            auto data = fixture_->read_buffer_from_cores(
                device_coord,
                sender_cores,
                sender_memory_map_.get_result_buffer_address(),
                sender_memory_map_.get_result_buffer_size());

            // Extract cycles from each core and store in map
            for (const auto& [core, core_data] : data) {
                // Cycles are stored as 64-bit value split across two 32-bit words
                uint32_t cycles_low = core_data[TT_FABRIC_CYCLES_INDEX];
                uint32_t cycles_high = core_data[TT_FABRIC_CYCLES_INDEX + 1];
                uint64_t total_cycles = static_cast<uint64_t>(cycles_high) << 32 | cycles_low;

                device_core_cycles_[device_node_id][core] = total_cycles;
            }
        }

        // Print results for checking
        log_debug(tt::LogTest, "Performance profiling results:");
        // Results are automatically sorted by device ID and core coordinates
        for (const auto& [device_id, core_cycles] : device_core_cycles_) {
            for (const auto& [core, cycles] : core_cycles) {
                log_debug(tt::LogTest, "Device {} Core ({},{}) Cycles: {}", device_id.chip_id, core.x, core.y, cycles);
            }
        }
    }

    void convert_core_cycles_to_direction_cycles() {
        // Clear previous data
        device_direction_cycles_.clear();

        log_debug(tt::LogTest, "Converting core cycles to direction cycles");

        for (const auto& [device_coord, test_device] : test_devices_) {
            const auto& device_node_id = test_device.get_node_id();

            // Process each sender core
            for (const auto& [core, sender] : test_device.get_senders()) {
                // Get cycles for this core (if available)
                if (device_core_cycles_.count(device_node_id) == 0 ||
                    device_core_cycles_[device_node_id].count(core) == 0) {
                    continue;
                }

                uint64_t core_cycles = device_core_cycles_[device_node_id][core];

                // Get unique (direction, link_id) pairs this core sends traffic to
                std::set<std::pair<RoutingDirection, uint32_t>> core_direction_links;
                for (const auto& [config, fabric_conn_idx] : sender.get_configs()) {
                    RoutingDirection direction = fixture_->get_forwarding_direction(*config.hops);
                    uint32_t link_id = config.link_id.value_or(0);  // Default to link 0 if not specified
                    core_direction_links.insert({direction, link_id});
                }

                // Add cycles to each (direction, link_id) pair this core sends to
                // Only one core per device should send in each (direction, link) combination
                for (const auto& [direction, link_id] : core_direction_links) {
                    if (device_direction_cycles_[device_node_id][direction].count(link_id) > 0) {
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

    void calculate_bandwidth(const TestConfig& config) {
        log_info(tt::LogTest, "Calculating bandwidth (GB/s) by direction:");

        // Clear previous bandwidth results
        bandwidth_results_.clear();
        // Clear previous summary
        bandwidth_results_summary_.clear();

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

        for (const auto& [device_id, direction_map] : device_direction_cycles_) {
            for (const auto& [direction, link_map] : direction_map) {
                for (const auto& [link_id, cycles] : link_map) {
                    if (cycles == 0) {
                        continue;  // Skip to avoid division by zero
                    }

                    // Get traffic count for this device and direction
                    if (outgoing_traffic_.count(device_id) > 0 && outgoing_traffic_[device_id].count(direction) > 0) {
                        total_traffic_count = outgoing_traffic_[device_id][direction];
                    }

                    // calculate the max for summary info
                    max_cycles = std::max(max_cycles, cycles);
                    max_traffic_count = std::max(max_traffic_count, total_traffic_count);

                    // Find sender configs that send in this direction and link to get payload size and packet count
                    for (const auto& [device_coord, test_device] : test_devices_) {
                        if (test_device.get_node_id() != device_id) {
                            continue;
                        }

                        bool found_connected_core = false;
                        for (const auto& [core, sender] : test_device.get_senders()) {
                            for (const auto& [config, fabric_conn_idx] : sender.get_configs()) {
                                RoutingDirection config_direction = fixture_->get_forwarding_direction(config.hops.value());
                                uint32_t config_link_id = config.link_id.value_or(0);
                                if (config_direction == direction && config_link_id == link_id) {
                                    uint32_t payload_size_bytes = config.parameters.payload_size_bytes;
                                    num_packets = config.parameters.num_packets;
                                    total_bytes =
                                        static_cast<uint64_t>(payload_size_bytes) * num_packets * total_traffic_count;
                                    total_packets = static_cast<uint64_t>(num_packets) * total_traffic_count;
                                    packet_size = payload_size_bytes;
                                    found_connected_core = true;
                                    break;
                                }
                            }
                            if (found_connected_core) {
                                break;
                            }
                        }
                    }

                    // Calculate bandwidth in Bytes/cycle and convert to GB/s
                    const auto physical_chip_id = tt::tt_metal::MetalContext::instance()
                                                      .get_control_plane()
                                                      .get_physical_chip_id_from_fabric_node_id(device_id);
                    const auto device_frequency_mhz =
                        tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(physical_chip_id);
                    uint32_t device_frequency_hz = device_frequency_mhz * 1e6;
                    // use min frequency (in real senario we will have the same freq)
                    device_freq = std::min(device_freq, device_frequency_hz);
                    const auto duration_seconds =
                        static_cast<double>(cycles) / static_cast<double>(device_frequency_hz);

                    double bandwidth_bytes_per_cycle = static_cast<double>(total_bytes) / static_cast<double>(cycles);
                    double bandwidth_gb_s = (bandwidth_bytes_per_cycle * device_frequency_mhz) / 1e3;
                    double packets_per_second = static_cast<double>(total_packets) / duration_seconds;

                    // TODO: need to figure out a better way to show the number of devices in a test.
                    // Ex, we compute number of devices for linear topology test as NS and EW separated.
                    // But in a mesh topology setup, how do we run linear topology and still show separate
                    // number of devices? There will be even more choices for arbitrary unicast setups.
                    uint32_t num_devices = 0;
                    const auto mesh_shape = fixture_->get_mesh_shape();
                    const auto topology = fixture_->get_topology();
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
                    // save all possible num devices
                    num_devices_set.insert(num_devices);

                    log_info(
                        tt::LogTest,
                        "Device {} Direction {} Link {} Bandwidth: {:.6f} GB/s (Total Packets: {}, Packet Size: {}, "
                        "Total Bytes: "
                        "{}, "
                        "Cycles: {})",
                        device_id.chip_id,
                        direction,
                        link_id,
                        bandwidth_gb_s,
                        total_packets,
                        packet_size,
                        total_bytes,
                        cycles);

                    // Store result for CSV generation (using GB/s)
                    bandwidth_results_.emplace_back(BandwidthResult{
                        .num_devices = num_devices,
                        .device_id = device_id.chip_id,
                        .direction = direction,
                        .total_traffic_count = total_traffic_count,
                        .num_packets = num_packets,
                        .packet_size = packet_size,
                        .cycles = cycles,
                        .bandwidth_gb_s = bandwidth_gb_s,
                        .packets_per_second = packets_per_second});
                }
            }
        }

        total_bytes = static_cast<uint64_t>(packet_size) * num_packets * max_traffic_count;
        double bandwidth_bytes_per_cycle = static_cast<double>(total_bytes) / static_cast<double>(max_cycles);
        double bandwidth_gb_s = (bandwidth_bytes_per_cycle * device_freq) / 1e9;

        // Calculate packets per second
        double duration_seconds = static_cast<double>(max_cycles) / static_cast<double>(device_freq);
        double packets_per_second = static_cast<double>(max_traffic_count * num_packets) / duration_seconds;

        bandwidth_results_summary_.push_back(BandwidthResultSummary{
            .num_devices = std::vector<uint32_t>(num_devices_set.begin(), num_devices_set.end()),
            .packet_size = packet_size,
            .cycles = max_cycles,
            .bandwidth_gb_s = bandwidth_gb_s,
            .packets_per_second = packets_per_second});
    }

    void generate_bandwidth_csv(const TestConfig& config) {
        // Extract representative ftype and ntype from first sender's first pattern
        std::string ftype_str = "None";
        std::string ntype_str = "None";
        if (!config.senders.empty() && !config.senders[0].patterns.empty()) {
            const auto& first_pattern = config.senders[0].patterns[0];
            if (first_pattern.ftype.has_value()) {
                ftype_str = enchantum::to_string(first_pattern.ftype.value()).data();
            }
            if (first_pattern.ntype.has_value()) {
                ntype_str = enchantum::to_string(first_pattern.ntype.value()).data();
            }
        }

        // Open CSV file in append mode
        std::ofstream csv_stream(csv_file_path_, std::ios::out | std::ios::app);
        if (!csv_stream.is_open()) {
            log_error(tt::LogTest, "Failed to open CSV file for appending: {}", csv_file_path_.string());
            return;
        }

        // Write data rows (header already written in initialize_csv_file)
        for (const auto& result : bandwidth_results_) {
            csv_stream << config.name << "," << ftype_str << "," << ntype_str << ","
                       << enchantum::to_string(config.fabric_setup.topology) << "," << result.num_devices << ","
                       << result.device_id << "," << config.fabric_setup.num_links << ","
                       << enchantum::to_string(result.direction) << "," << result.total_traffic_count << ","
                       << result.num_packets << "," << result.packet_size << "," << result.cycles << "," << std::fixed
                       << std::setprecision(6) << result.bandwidth_gb_s << "," << std::fixed << std::setprecision(3)
                       << result.packets_per_second << "\n";
        }

        csv_stream.close();
        log_info(tt::LogTest, "Bandwidth results appended to CSV file: {}", csv_file_path_.string());

        // Open CSV file in append mode
        std::ofstream summary_csv_stream(csv_summary_file_path_, std::ios::out | std::ios::app);
        if (!summary_csv_stream.is_open()) {
            log_error(
                tt::LogTest, "Failed to open summary CSV file for appending: {}", csv_summary_file_path_.string());
            return;
        }

        // Write data rows (header already written in initialize_csv_file)
        for (const auto& result : bandwidth_results_summary_) {
            // Convert vector of num_devices to a string representation
            std::string num_devices_str = "[";
            for (size_t i = 0; i < result.num_devices.size(); ++i) {
                if (i > 0) {
                    num_devices_str += ",";
                }
                num_devices_str += std::to_string(result.num_devices[i]);
            }
            num_devices_str += "]";

            std::string topology_str = enchantum::to_string(config.fabric_setup.topology).data();
            double tolerance = get_tolerance_percent(
                config.name,
                ftype_str,
                ntype_str,
                topology_str,
                num_devices_str,
                config.fabric_setup.num_links,
                result.packet_size);

            summary_csv_stream << config.name << "," << ftype_str << "," << ntype_str << "," << topology_str << ",\""
                               << num_devices_str << "\"," << config.fabric_setup.num_links << "," << result.packet_size
                               << "," << result.cycles << "," << std::fixed << std::setprecision(6)
                               << result.bandwidth_gb_s << "," << std::fixed << std::setprecision(3)
                               << result.packets_per_second << "," << std::fixed << std::setprecision(1) << tolerance
                               << "\n";
        }

        summary_csv_stream.close();
        log_info(tt::LogTest, "Bandwidth summary results appended to CSV file: {}", csv_summary_file_path_.string());
    }

    std::string get_golden_csv_filename() {
        auto arch_name = tt::tt_metal::hal::get_arch_name();
        auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();

        // Convert cluster type enum to lowercase string
        std::string cluster_name = enchantum::to_string(cluster_type).data();
        std::transform(cluster_name.begin(), cluster_name.end(), cluster_name.begin(), ::tolower);

        std::string file_name = "golden_bandwidth_summary_" + arch_name + "_" + cluster_name + ".csv";
        return file_name;
    }

    bool load_golden_csv() {
        golden_csv_entries_.clear();

        std::string golden_filename = get_golden_csv_filename();
        std::filesystem::path golden_path =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
            "tests/tt_metal/tt_metal/perf_microbenchmark/routing/golden" / golden_filename;

        if (!std::filesystem::exists(golden_path)) {
            log_warning(tt::LogTest, "Golden CSV file not found: {}", golden_path.string());
            return false;
        }

        std::ifstream golden_file(golden_path);
        if (!golden_file.is_open()) {
            log_error(tt::LogTest, "Failed to open golden CSV file: {}", golden_path.string());
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
                // Handle quoted strings for num_devices
                if (token.front() == '"' && token.back() != '"') {
                    std::string quoted_token = token;
                    while (std::getline(ss, token, ',') && token.back() != '"') {
                        quoted_token += "," + token;
                    }
                    quoted_token += "," + token;
                    // Remove quotes
                    quoted_token = quoted_token.substr(1, quoted_token.length() - 2);
                    tokens.push_back(quoted_token);
                } else if (token.front() == '"' && token.back() == '"') {
                    // Remove quotes from single quoted token
                    tokens.push_back(token.substr(1, token.length() - 2));
                } else {
                    tokens.push_back(token);
                }
            }

            // Validate we have enough tokens for the new format with tolerance
            if (tokens.size() < 11) {
                log_error(tt::LogTest, "Invalid CSV format in golden file. Expected 11 fields, got {}", tokens.size());
                continue;
            }

            GoldenCsvEntry entry;
            entry.test_name = tokens[0];
            entry.ftype = tokens[1];
            entry.ntype = tokens[2];
            entry.topology = tokens[3];
            entry.num_devices = tokens[4];
            entry.num_links = std::stoul(tokens[5]);
            entry.packet_size = std::stoul(tokens[6]);
            entry.cycles = std::stoull(tokens[7]);
            entry.bandwidth_gb_s = std::stod(tokens[8]);
            entry.packets_per_second = std::stod(tokens[9]);
            entry.tolerance_percent = std::stod(tokens[10]);

            golden_csv_entries_.push_back(entry);
        }

        golden_file.close();
        log_info(tt::LogTest, "Loaded {} golden entries from: {}", golden_csv_entries_.size(), golden_path.string());
        return true;
    }

    void generate_comparison_csv(const TestConfig& config) {
        // Clear previous results
        comparison_results_.clear();
        failed_tests_.clear();

        // Load golden CSV (will warn if not found)
        if (golden_csv_entries_.empty()) {
            log_warning(tt::LogTest, "Skipping golden CSV comparison - no golden file found");
            return;
        }

        // Extract ftype and ntype from config
        std::string ftype_str = "None";
        std::string ntype_str = "None";
        if (!config.senders.empty() && !config.senders[0].patterns.empty()) {
            const auto& first_pattern = config.senders[0].patterns[0];
            if (first_pattern.ftype.has_value()) {
                ftype_str = enchantum::to_string(first_pattern.ftype.value()).data();
            }
            if (first_pattern.ntype.has_value()) {
                ntype_str = enchantum::to_string(first_pattern.ntype.value()).data();
            }
        }

        // Compare current results with golden
        for (const auto& summary_result : bandwidth_results_summary_) {
            // Convert vector of num_devices to string representation
            std::string num_devices_str = "[";
            for (size_t i = 0; i < summary_result.num_devices.size(); ++i) {
                if (i > 0) {
                    num_devices_str += ",";
                }
                num_devices_str += std::to_string(summary_result.num_devices[i]);
            }
            num_devices_str += "]";

            std::string topology_str = enchantum::to_string(config.fabric_setup.topology).data();

            // Find matching golden entry
            auto golden_it =
                std::find_if(golden_csv_entries_.begin(), golden_csv_entries_.end(), [&](const GoldenCsvEntry& golden) {
                    return golden.test_name == config.name && golden.ftype == ftype_str && golden.ntype == ntype_str &&
                           golden.topology == topology_str && golden.num_devices == num_devices_str &&
                           golden.num_links == config.fabric_setup.num_links &&
                           golden.packet_size == summary_result.packet_size;
                });

            ComparisonResult comp_result;
            comp_result.test_name = config.name;
            comp_result.ftype = ftype_str;
            comp_result.ntype = ntype_str;
            comp_result.topology = topology_str;
            comp_result.num_devices = num_devices_str;
            comp_result.num_links = config.fabric_setup.num_links;
            comp_result.packet_size = summary_result.packet_size;
            comp_result.current_bandwidth_gb_s = summary_result.bandwidth_gb_s;

            if (golden_it != golden_csv_entries_.end()) {
                comp_result.golden_bandwidth_gb_s = golden_it->bandwidth_gb_s;
                comp_result.difference_percent =
                    ((comp_result.current_bandwidth_gb_s - comp_result.golden_bandwidth_gb_s) /
                     comp_result.golden_bandwidth_gb_s) *
                    100.0;

                // Use per-test tolerance from golden CSV instead of global tolerance
                double test_tolerance = golden_it->tolerance_percent;
                comp_result.within_tolerance = std::abs(comp_result.difference_percent) <= test_tolerance;

                if (comp_result.within_tolerance) {
                    comp_result.status = "PASS";
                } else {
                    comp_result.status = "FAIL";
                    failed_tests_.push_back(
                        config.name + " (" + ftype_str + "," + ntype_str + "," + topology_str + "," + num_devices_str +
                        ") - diff: " + std::to_string(comp_result.difference_percent) +
                        "%, tolerance: " + std::to_string(test_tolerance) + "%");
                }
            } else {
                comp_result.golden_bandwidth_gb_s = 0.0;
                comp_result.difference_percent = 0.0;
                comp_result.within_tolerance = false;
                comp_result.status = "NO_GOLDEN";
                failed_tests_.push_back(config.name + " (NO GOLDEN ENTRY)");
            }

            comparison_results_.push_back(comp_result);
        }

        // Open diff CSV file in append mode (header already written in initialize_csv_file)
        std::ofstream diff_csv(diff_csv_file_path_, std::ios::out | std::ios::app);
        if (!diff_csv.is_open()) {
            log_error(tt::LogTest, "Failed to open diff CSV file for appending: {}", diff_csv_file_path_.string());
            return;
        }

        // Write comparison results (header already written in initialize_csv_file)
        for (const auto& result : comparison_results_) {
            diff_csv << result.test_name << "," << result.ftype << "," << result.ntype << "," << result.topology
                     << ",\"" << result.num_devices << "\"," << result.num_links << "," << result.packet_size << ","
                     << std::fixed << std::setprecision(6) << result.current_bandwidth_gb_s << ","
                     << result.golden_bandwidth_gb_s << "," << std::setprecision(2) << result.difference_percent << ","
                     << result.status << "\n";
        }

        diff_csv.close();
        log_info(tt::LogTest, "Comparison diff CSV results appended to: {}", diff_csv_file_path_.string());
    }

    void validate_against_golden() {
        if (comparison_results_.empty()) {
            log_info(tt::LogTest, "No golden comparison performed (no golden file found)");
            return;
        }

        if (!failed_tests_.empty()) {
            has_test_failures_ = true;
            log_error(tt::LogTest, "The following tests failed golden comparison (using per-test tolerance):");
            for (const auto& failed_test : failed_tests_) {
                log_error(tt::LogTest, "  - {}", failed_test);
                all_failed_tests_.push_back(failed_test);  // Accumulate for final summary
            }
        } else {
            log_info(tt::LogTest, "All tests passed golden comparison using per-test tolerance values");
        }
    }

    // Track sync cores for each device
    std::unordered_map<FabricNodeId, CoreCoord> device_global_sync_cores_;
    std::unordered_map<FabricNodeId, std::vector<CoreCoord>> device_local_sync_cores_;

    std::shared_ptr<TestFixture> fixture_;
    std::unordered_map<MeshCoordinate, TestDevice> test_devices_;
    std::unique_ptr<tt::tt_fabric::fabric_tests::GlobalAllocator> allocator_;

    // Uniform memory maps shared across all devices
    tt::tt_fabric::fabric_tests::SenderMemoryMap sender_memory_map_;
    tt::tt_fabric::fabric_tests::ReceiverMemoryMap receiver_memory_map_;
    tt::tt_fabric::fabric_tests::AllocatorPolicies allocation_policies_;
    bool benchmark_mode_ = false;  // Benchmark mode for current test
    bool global_sync_ = false;     // Line sync for current test
    uint32_t global_sync_val_ = 0;

    // Performance profiling data
    // TODO: add link index into the result
    std::map<FabricNodeId, std::map<RoutingDirection, uint32_t>> outgoing_traffic_;
    std::map<FabricNodeId, std::map<RoutingDirection, std::map<uint32_t, uint64_t>>> device_direction_cycles_;
    std::map<FabricNodeId, std::map<CoreCoord, uint64_t>> device_core_cycles_;
    std::vector<BandwidthResult> bandwidth_results_;
    std::vector<BandwidthResultSummary> bandwidth_results_summary_;
    std::filesystem::path csv_file_path_;
    std::filesystem::path csv_summary_file_path_;

    // Golden CSV comparison data
    std::vector<GoldenCsvEntry> golden_csv_entries_;
    std::vector<ComparisonResult> comparison_results_;
    std::vector<std::string> failed_tests_;      // Per-test failed tests (reset each test)
    std::vector<std::string> all_failed_tests_;  // Accumulates all failed tests across test run
    std::filesystem::path diff_csv_file_path_;
    bool has_test_failures_ = false;  // Track if any tests failed validation
};
