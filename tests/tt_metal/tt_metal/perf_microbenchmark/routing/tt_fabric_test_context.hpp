// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
#include <fstream>
#include <iomanip>
#include <optional>
#include <set>
#include <sstream>

#include "tt_fabric_test_config.hpp"
#include "tt_fabric_test_common.hpp"
#include "tt_fabric_test_device_setup.hpp"
#include "tt_fabric_test_traffic.hpp"
#include "tt_fabric_test_allocator.hpp"
#include "tt_fabric_test_memory_map.hpp"
#include "tt_fabric_telemetry.hpp"
#include "tt_fabric_test_progress_monitor.hpp"
#include "tt_fabric_test_results.hpp"
#include "tt_fabric_test_eth_readback.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_coord.hpp>

// Constants
const std::string output_dir = "generated/fabric";
const std::string default_built_tests_dump_file = "built_tests.yaml";
// CI will always check the following folder for artifacts to upload
const std::string ci_artifacts_dir = "generated/test_reports";

using TestFixture = tt::tt_fabric::fabric_tests::TestFixture;
using TestDevice = tt::tt_fabric::fabric_tests::TestDevice;
using TestConfig = tt::tt_fabric::fabric_tests::TestConfig;
using TestFabricSetup = tt::tt_fabric::fabric_tests::TestFabricSetup;
using TrafficParameters = tt::tt_fabric::fabric_tests::TrafficParameters;
using TestTrafficConfig = tt::tt_fabric::fabric_tests::TestTrafficConfig;
using TestTrafficSenderConfig = tt::tt_fabric::fabric_tests::TestTrafficSenderConfig;
using TestTrafficReceiverConfig = tt::tt_fabric::fabric_tests::TestTrafficReceiverConfig;
using SenderCreditInfo = tt::tt_fabric::fabric_tests::SenderCreditInfo;
using ReceiverCreditInfo = tt::tt_fabric::fabric_tests::ReceiverCreditInfo;
using TestWorkerType = tt::tt_fabric::fabric_tests::TestWorkerType;
using CommonMemoryMap = tt::tt_fabric::fabric_tests::CommonMemoryMap;
using ProgressMonitorConfig = tt::tt_fabric::fabric_tests::ProgressMonitorConfig;
using TestProgressMonitor = tt::tt_fabric::fabric_tests::TestProgressMonitor;
using SenderMemoryMap = tt::tt_fabric::fabric_tests::SenderMemoryMap;
using IDeviceInfoProvider = tt::tt_fabric::fabric_tests::IDeviceInfoProvider;
using TrafficPatternConfig = tt::tt_fabric::fabric_tests::TrafficPatternConfig;

using ChipSendType = tt::tt_fabric::ChipSendType;
using NocSendType = tt::tt_fabric::NocSendType;
using FabricNodeId = tt::tt_fabric::FabricNodeId;
using MeshId = tt::tt_fabric::MeshId;
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
using FabricTensixConfig = tt::tt_fabric::FabricTensixConfig;

using BandwidthResult = tt::tt_fabric::fabric_tests::BandwidthResult;
using BandwidthResultSummary = tt::tt_fabric::fabric_tests::BandwidthResultSummary;
using GoldenCsvEntry = tt::tt_fabric::fabric_tests::GoldenCsvEntry;
using ComparisonResult = tt::tt_fabric::fabric_tests::ComparisonResult;
using PostComparisonAnalyzer = tt::tt_fabric::fabric_tests::PostComparisonAnalyzer;

// Helper functions for parsing traffic pattern parameters
using tt::tt_fabric::fabric_tests::fetch_first_traffic_pattern;
using tt::tt_fabric::fabric_tests::fetch_pattern_ftype;
using tt::tt_fabric::fabric_tests::fetch_pattern_ntype;
using tt::tt_fabric::fabric_tests::fetch_pattern_num_packets;
using tt::tt_fabric::fabric_tests::fetch_pattern_packet_size;

// Bandwidth Summary Statistics
// If you want to add new statistics, populate this enum with their names
enum class BandwidthStatistics {
    BandwidthMean,
    BandwidthMin,
    BandwidthMax,
    BandwidthStdDev,
    PacketsPerSecondMean,
    CyclesMean
};
// The header of each statistic in the Bandwidth Summary CSV
const std::unordered_map<BandwidthStatistics, std::string> BandwidthStatisticsHeader = {
    {BandwidthStatistics::BandwidthMean, "Avg Bandwidth (GB/s)"},
    {BandwidthStatistics::BandwidthMin, "BW Min (GB/s)"},
    {BandwidthStatistics::BandwidthMax, "BW Max (GB/s)"},
    {BandwidthStatistics::BandwidthStdDev, "BW Std Dev (GB/s)"},
    {BandwidthStatistics::PacketsPerSecondMean, "Avg Packets/s"},
    {BandwidthStatistics::CyclesMean, "Avg Cycles"},
};

// Access to internal API: ProgramImpl::num_kernel
#include "impl/program/program_impl.hpp"

class TestContext {
public:
    void init(
        std::shared_ptr<TestFixture> fixture,
        const tt::tt_fabric::fabric_tests::AllocatorPolicies& policies,
        bool use_dynamic_policies = true) {
        fixture_ = std::move(fixture);
        allocation_policies_ = policies;
        use_dynamic_policies_ = use_dynamic_policies;  // Store for prepare_for_test()

        // Initialize memory maps for all available devices
        initialize_memory_maps();

        // Create dynamic policy manager if needed
        if (use_dynamic_policies_) {
            policy_manager_ =
                std::make_unique<tt::tt_fabric::fabric_tests::DynamicPolicyManager>(*this->fixture_, *this->fixture_);
        }

        // Create allocator with memory maps
        // Note: Memory maps will be updated in prepare_for_test() if using dynamic policies
        this->allocator_ = std::make_unique<tt::tt_fabric::fabric_tests::GlobalAllocator>(
            *this->fixture_, *this->fixture_, policies, sender_memory_map_, receiver_memory_map_);
    }

    void prepare_for_test(const TestConfig& config) {
        // Skip reconstruction entirely for explicit YAML policies
        if (!use_dynamic_policies_) {
            return;  // Early return - allocator and maps already correct, reset() will clean up state
        }

        // Ask policy manager if a new policy is needed
        // Returns nullopt if cached policy should be reused, otherwise returns new policy
        auto new_policy = policy_manager_->get_new_policy_for_test(config);

        if (new_policy.has_value()) {
            // New policy computed - need to reconstruct allocator and memory maps
            update_memory_maps(new_policy.value());

            allocator_.reset();
            allocator_ = std::make_unique<tt::tt_fabric::fabric_tests::GlobalAllocator>(
                *fixture_, *fixture_, new_policy.value(), sender_memory_map_, receiver_memory_map_);
        }

        // Validate packet size (uses either new policy or cached policy)
        const auto& policy_to_validate =
            new_policy.has_value() ? new_policy.value() : policy_manager_->get_cached_policy();
        validate_packet_sizes_for_policy(config, policy_to_validate.default_payload_chunk_size);
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
        // Allocate resources
        log_debug(tt::LogTest, "Allocating resources for test config");
        this->allocator_->allocate_resources(config);
        log_debug(tt::LogTest, "Resource allocation complete");

        // Use unified connection manager when BOTH sync AND flow control are enabled
        // - This ensures sync and credit returns use the same link tracking for correct mux detection
        // - When only sync is enabled (no flow control), separate managers avoid mux overhead
        if (config.enable_flow_control && config.global_sync) {
            for (auto& [_, device] : test_devices_) {
                device.set_use_unified_connection_manager(true);
            }
        }

        // Transfer pristine cores from allocator to each device
        for (auto& [coord, device] : test_devices_) {
            auto node_id = device.get_node_id();
            auto pristine_cores = allocator_->get_pristine_cores_for_device(node_id);
            device.set_pristine_cores(std::move(pristine_cores));
        }

        if (config.global_sync) {
            // set it only after the test_config is built since it needs set the sync value during expand the high-level
            // patterns.
            this->set_global_sync(config.global_sync);
            this->set_global_sync_val(config.global_sync_val);
            this->set_benchmark_mode(config.benchmark_mode);

            log_debug(tt::LogTest, "Enabled sync, global sync value: {}, ", global_sync_val_);
            log_debug(tt::LogTest, "Ubenchmark mode: {}, ", benchmark_mode_);

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
                            .mcast_start_hops = sync_pattern.mcast_start_hops,
                            .seed = config.seed,
                            .is_2D_routing_enabled = fixture_->is_2D_routing_enabled(),
                            .mesh_shape = this->fixture_->get_mesh_shape(),
                            .topology = this->fixture_->get_topology()};

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
                            .dst_noc_encoding = dst_noc_encoding,
                            .link_id = sync_sender.link_id};  // Derive from SenderConfig (always 0 for sync)

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
                log_debug(
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
                    .mcast_start_hops = pattern.mcast_start_hops,
                    .enable_flow_control = config.enable_flow_control,  // Propagate from test-level config
                    .seed = config.seed,
                    .is_2D_routing_enabled = fixture_->is_2D_routing_enabled(),
                    .mesh_shape = this->fixture_->get_mesh_shape(),
                    .topology = this->fixture_->get_topology()};

                TestTrafficConfig traffic_config = {
                    .parameters = traffic_parameters,
                    .src_node_id = sender.device,
                    .src_logical_core = sender.core,
                    .dst_logical_core = dest.core,
                    .target_address = dest.target_address,
                    .atomic_inc_address = dest.atomic_inc_address,
                    .link_id = sender.link_id,
                    .sender_credit_info = pattern.sender_credit_info,
                    .credit_return_batch_size = pattern.credit_return_batch_size,
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

    void compile_programs() {
        fixture_->setup_workload();
        // TODO: should we be taking const ref?
        for (auto& [coord, test_device] : test_devices_) {
            test_device.set_benchmark_mode(benchmark_mode_);
            test_device.set_global_sync(global_sync_);
            test_device.set_global_sync_val(global_sync_val_);
            test_device.set_progress_monitoring_enabled(progress_config_.enabled);

            auto device_id = test_device.get_node_id();
            test_device.set_sync_core(device_global_sync_cores_[device_id]);

            test_device.create_kernels();
            auto& program_handle = test_device.get_program_handle();
            if (program_handle.impl().num_kernels()) {
                fixture_->enqueue_program(coord, std::move(program_handle));
            }
        }
    }

    void launch_programs() { fixture_->run_programs(); }

    void wait_for_programs() { fixture_->wait_for_programs(); }

    void enable_progress_monitoring(const ProgressMonitorConfig& config) {
        progress_config_ = config;
        progress_config_.enabled = true;
    }

    void wait_for_programs_with_progress();

    // Accessors for progress monitor
    const std::unordered_map<MeshCoordinate, TestDevice>& get_test_devices() const { return test_devices_; }

    const SenderMemoryMap& get_sender_memory_map() const { return sender_memory_map_; }

    IDeviceInfoProvider* get_device_info_provider() const { return fixture_.get(); }

    void process_telemetry_data(TestConfig& built_test_config) {
        if (this->get_telemetry_enabled()) {
            this->read_telemetry();
            this->process_telemetry_for_golden();
            this->dump_raw_telemetry_csv(built_test_config);
        }
    }

    void validate_results() {
        constexpr uint32_t MAX_CONCURRENT_DEVICES = 16;

        // Convert map to vector for easier indexing
        std::vector<std::pair<MeshCoordinate, const TestDevice*>> devices;
        devices.reserve(test_devices_.size());
        for (const auto& [coord, device] : test_devices_) {
            devices.push_back({coord, &device});
        }

        // Process in groups
        for (size_t i = 0; i < devices.size(); i += MAX_CONCURRENT_DEVICES) {
            size_t group_end = std::min(i + MAX_CONCURRENT_DEVICES, devices.size());

            // Initiate reads for this group
            std::vector<TestDevice::ValidationReadOps> read_ops;
            read_ops.reserve(group_end - i);
            for (size_t j = i; j < group_end; ++j) {
                read_ops.push_back(devices[j].second->initiate_results_readback());
            }

            // Barrier
            fixture_->barrier_reads();

            // Validate results
            for (size_t j = i; j < group_end; ++j) {
                devices[j].second->validate_results_after_readback(read_ops[j - i]);
            }
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
    }

    void generate_bandwidth_summary() {
        // Load golden CSV file
        load_golden_csv();

        // Calculate bandwidth statistics for multi-iteration tests
        calculate_bandwidth_summary_statistics();

        // Generate bandwidth summary CSV file
        generate_bandwidth_summary_csv();

        // Compare summary results with golden CSV
        compare_summary_results_with_golden();

        // Generate statistics based on golden comparison
        PostComparisonAnalyzer post_comparison_analyzer(comparison_results_);
        post_comparison_analyzer.generate_comparison_statistics();

        // Generate comparison statistics CSV file
        set_comparison_statistics_csv_file_path();
        post_comparison_analyzer.generate_comparison_statistics_csv(comparison_statistics_csv_file_path_);

        validate_against_golden();
    }

    void initialize_bandwidth_results_csv_file() {
        // Create output directory
        std::filesystem::path tt_metal_home =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
        std::filesystem::path bandwidth_results_path = tt_metal_home / output_dir;

        if (!std::filesystem::exists(bandwidth_results_path)) {
            std::filesystem::create_directories(bandwidth_results_path);
        }

        // Generate detailed CSV filename
        auto arch_name = tt::tt_metal::hal::get_arch_name();
        std::ostringstream oss;
        oss << "bandwidth_results_" << arch_name << ".csv";
        csv_file_path_ = bandwidth_results_path / oss.str();

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
               "bandwidth_GB_s,packets_per_second";
        if (this->telemetry_enabled_) {
            csv_stream << ",telemetry_bw_GB_s_min,telemetry_bw_GB_s_avg,telemetry_bw_GB_s_max";
        }
        csv_stream << "\n";
        csv_stream.close();

        log_info(tt::LogTest, "Initialized CSV file: {}", csv_file_path_.string());
    }

    void close_devices() { fixture_->close_devices(); }

    void set_benchmark_mode(bool benchmark_mode) { benchmark_mode_ = benchmark_mode; }

    void set_telemetry_enabled(bool enabled) { telemetry_enabled_ = enabled; }

    bool get_benchmark_mode() { return benchmark_mode_; }

    bool get_telemetry_enabled() { return telemetry_enabled_; }

    // Code profiling getters/setters
    bool get_code_profiling_enabled() const { return code_profiling_enabled_; }
    void set_code_profiling_enabled(bool enabled) { code_profiling_enabled_ = enabled; }
    const std::vector<CodeProfilingEntry>& get_code_profiling_entries() const { return code_profiling_entries_; }

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

    void setup_ci_artifacts() {
        std::filesystem::path tt_metal_home =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
        std::filesystem::path bandwidth_results_path = tt_metal_home / output_dir;
        std::filesystem::path ci_artifacts_path = tt_metal_home / ci_artifacts_dir;
        // Create CI artifacts directory if it doesn't exist
        if (!std::filesystem::exists(ci_artifacts_path)) {
            try {
                std::filesystem::create_directories(ci_artifacts_path);
            } catch (const std::filesystem::filesystem_error& e) {
                log_error(
                    tt::LogTest,
                    "Failed to create CI artifacts directory, skipping CI artifacts creation: {}",
                    e.what());
                return;
            }
        }

        // Copy CSV files to CI artifacts directory
        for (const std::filesystem::path& csv_filepath :
             {csv_file_path_, csv_summary_file_path_, diff_csv_file_path_, comparison_statistics_csv_file_path_}) {
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
        log_trace(tt::LogTest, "Copied CSV files to CI artifacts directory: {}", ci_artifacts_path.string());
    }

    void read_telemetry();

    void clear_telemetry();

    void process_telemetry_for_golden();

    void dump_raw_telemetry_csv(const TestConfig& config);

    // Code profiling methods
    void read_code_profiling_results();

    void clear_code_profiling_buffers();

    void report_code_profiling_results();

private:
    void reset_local_variables() {
        benchmark_mode_ = false;
        global_sync_ = false;
        global_sync_val_ = 0;
        outgoing_traffic_.clear();
        device_direction_cycles_.clear();
        device_core_cycles_.clear();
        bandwidth_results_.clear();
        code_profiling_entries_.clear();
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
            if (src_node_id.mesh_id == dst_node_ids[0].mesh_id) {
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
            .payload_buffer_size = payload_buffer_size,
            .link_id = traffic_config.link_id};  // Derive from sender's link_id

        if (traffic_config.parameters.enable_flow_control) {
            TT_FATAL(
                traffic_config.sender_credit_info.has_value(),
                "Sender credit info not allocated for sender {} with flow control enabled",
                traffic_config.src_node_id);

            sender_config.sender_credit_info = traffic_config.sender_credit_info.value();

            TT_FATAL(
                traffic_config.credit_return_batch_size.has_value(),
                "Credit batch size not calculated for sender {} with flow control enabled",
                traffic_config.src_node_id);
            uint32_t credit_return_batch_size = traffic_config.credit_return_batch_size.value();

            receiver_config.receiver_credit_info = ReceiverCreditInfo{
                .receiver_node_id = FabricNodeId(MeshId{0}, 0),
                .sender_node_id = traffic_config.src_node_id,
                .sender_logical_core = src_logical_core,
                .sender_noc_encoding = fixture_->get_worker_noc_encoding(src_logical_core),
                .credit_return_address = 0,
                .credit_return_batch_size = credit_return_batch_size,
                .hops = std::nullopt};
        } else {
            // If flow control is disabled, ensure sender_credit_info is not set
            sender_config.sender_credit_info = std::nullopt;
            receiver_config.receiver_credit_info = std::nullopt;
        }

        // CRITICAL: receiver_idx must be global across ALL receivers (local + remote)
        uint32_t receiver_idx = 0;
        for (const auto& dst_node_id : dst_node_ids) {
            if (fixture_->is_local_fabric_node_id(dst_node_id)) {
                const auto& dst_coord = this->fixture_->get_device_coord(dst_node_id);
                TestTrafficReceiverConfig per_receiver_config = receiver_config;

                if (traffic_config.parameters.enable_flow_control) {
                    TT_FATAL(
                        per_receiver_config.receiver_credit_info.has_value(),
                        "Receiver credit info not allocated for receiver with flow control enabled");

                    uint32_t credit_chunk_base = sender_config.sender_credit_info->credit_reception_address_base;
                    uint32_t credit_return_address =
                        SenderMemoryMap::get_receiver_credit_address(credit_chunk_base, receiver_idx);

                    per_receiver_config.receiver_credit_info->receiver_node_id = dst_node_id;
                    per_receiver_config.receiver_credit_info->credit_return_address = credit_return_address;

                    std::optional<std::unordered_map<RoutingDirection, uint32_t>> reverse_hops = std::nullopt;
                    if (src_node_id.mesh_id == dst_node_id.mesh_id) {
                        reverse_hops = fixture_->get_hops_to_chip(dst_node_id, src_node_id);
                    }
                    per_receiver_config.receiver_credit_info->hops = reverse_hops;
                }

                this->test_devices_.at(dst_coord).add_receiver_traffic_config(dst_logical_core, per_receiver_config);
            }

            // CRITICAL: Increment for EVERY receiver (local + remote)
            receiver_idx++;
        }

        if (fixture_->is_local_fabric_node_id(src_node_id)) {
            const auto& src_coord = this->fixture_->get_device_coord(src_node_id);
            auto& src_test_device = this->test_devices_.at(src_coord);
            src_test_device.add_sender_traffic_config(src_logical_core, std::move(sender_config));
        }
    }

    void initialize_memory_maps() {
        // Use allocation_policies_ from init() call
        update_memory_maps(allocation_policies_);
    }

    void update_memory_maps(const tt::tt_fabric::fabric_tests::AllocatorPolicies& policies) {
        // Get uniform L1 memory layout (same across all devices)
        auto l1_unreserved_base = fixture_->get_l1_unreserved_base();
        auto l1_unreserved_size = fixture_->get_l1_unreserved_size();
        auto l1_alignment = fixture_->get_l1_alignment();

        sender_memory_map_ =
            tt::tt_fabric::fabric_tests::SenderMemoryMap(l1_unreserved_base, l1_unreserved_size, l1_alignment);

        receiver_memory_map_ = tt::tt_fabric::fabric_tests::ReceiverMemoryMap(
            l1_unreserved_base,
            l1_unreserved_size,
            l1_alignment,
            policies.default_payload_chunk_size,
            policies.receiver_config.max_configs_per_core);

        if (!sender_memory_map_.is_valid() || !receiver_memory_map_.is_valid()) {
            TT_THROW("Invalid memory map configuration");
        }
    }

    void validate_packet_sizes_for_policy(const TestConfig& config, uint32_t payload_chunk_size) {
        uint32_t max_packet_size = 0;
        for (const auto& sender : config.senders) {
            for (const auto& pattern : sender.patterns) {
                if (pattern.size.has_value()) {
                    max_packet_size = std::max(max_packet_size, pattern.size.value());
                }
            }
        }

        if (max_packet_size > payload_chunk_size) {
            TT_FATAL(
                false,
                "Test '{}' configuration is INVALID!\n"
                "  Max packet size: {} bytes\n"
                "  Computed buffer size: {} bytes\n"
                "  The packet size exceeds buffer capacity.\n"
                "  Fix: Reduce packet size to <= {} bytes or adjust parametrization.",
                config.parametrized_name,
                max_packet_size,
                payload_chunk_size,
                payload_chunk_size);
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

    void trace_traffic_path(const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config) {
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
        for (const auto& [device_coord, test_device] : test_devices_) {
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
                device.read_op = fixture_->initiate_read_buffer_from_cores(
                    device.device_coord,
                    device.sender_cores,
                    sender_memory_map_.get_result_buffer_address(),
                    sender_memory_map_.get_result_buffer_size());
            }

            // Barrier to wait for all reads in this group to complete
            fixture_->barrier_reads();

            // Second loop: Process completed results
            for (size_t i = group_start; i < group_end; ++i) {
                auto& device = all_devices[i];
                auto data = fixture_->complete_read_buffer_from_cores(device.read_op);

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
                for (const auto& [config, _] : sender.get_configs()) {
                    RoutingDirection direction = fixture_->get_forwarding_direction(*config.hops);
                    uint32_t link_id = config.link_id;
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

    unsigned int get_device_frequency_mhz(const FabricNodeId& device_id) {
        if (!device_freq_mhz_map_.contains(device_id)) {
            auto& metal_context = tt::tt_metal::MetalContext::instance();
            auto physical_chip_id =
                metal_context.get_control_plane().get_physical_chip_id_from_fabric_node_id(device_id);
            device_freq_mhz_map_[device_id] = metal_context.get_cluster().get_device_aiclk(physical_chip_id);
        }
        auto freq_mhz = device_freq_mhz_map_.at(device_id);
        TT_FATAL(freq_mhz != 0, "Device frequency reported as 0 MHz for device {}", device_id.chip_id);
        return freq_mhz;
    }

    void calculate_bandwidth(const TestConfig& config) {
        log_debug(tt::LogTest, "Calculating bandwidth (GB/s) by direction:");

        // Clear previous bandwidth results
        bandwidth_results_.clear();

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
        const auto mesh_shape = fixture_->get_mesh_shape();
        const auto topology = fixture_->get_topology();
        // Pre-compute sender config lookup cache to avoid O(n³) search in inner loop
        std::unordered_map<std::string, std::tuple<uint32_t, uint32_t, uint32_t>> config_cache;
        for (const auto& [device_coord, test_device] : test_devices_) {
            const auto& device_id = test_device.get_node_id();
            for (const auto& [core, sender] : test_device.get_senders()) {
                for (const auto& [config, _] : sender.get_configs()) {
                    RoutingDirection config_direction = fixture_->get_forwarding_direction(config.hops.value());
                    uint32_t config_link_id = config.link_id;

                    // Create cache key: device_id + direction + link_id
                    std::string cache_key = std::to_string(device_id.chip_id) + "_" +
                                            std::to_string(static_cast<int>(config_direction)) + "_" +
                                            std::to_string(config_link_id);

                    config_cache[cache_key] = std::make_tuple(
                        config.parameters.payload_size_bytes,
                        config.parameters.num_packets,
                        config.parameters.payload_size_bytes  // packet_size
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
                    if (outgoing_traffic_.count(device_id) > 0 && outgoing_traffic_[device_id].count(direction) > 0) {
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
                    const auto device_frequency_mhz = get_device_frequency_mhz(device_id);
                    uint32_t device_frequency_hz = device_frequency_mhz * 1e6;
                    // use min frequency (in real senario we will have the same freq)
                    device_freq = std::min(device_freq, device_frequency_hz);
                    const auto duration_seconds =
                        static_cast<double>(cycles) / static_cast<double>(device_frequency_hz);

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

                    if (telemetry_enabled_) {
                        bw_result.telemetry_bw_GB_s_min = measured_bw_min_;
                        bw_result.telemetry_bw_GB_s_avg = measured_bw_avg_;
                        bw_result.telemetry_bw_GB_s_max = measured_bw_max_;
                    }

                    // Store result for CSV generation (using GB/s)
                    bandwidth_results_.emplace_back(bw_result);
                }
            }
        }

        // Calculate and store a summary of this test
        total_bytes = static_cast<uint64_t>(packet_size) * num_packets * max_traffic_count;
        double bandwidth_bytes_per_cycle = static_cast<double>(total_bytes) / static_cast<double>(max_cycles);
        double bandwidth_GB_s = (bandwidth_bytes_per_cycle * device_freq) / 1e9;

        // Calculate packets per second
        double duration_seconds = static_cast<double>(max_cycles) / static_cast<double>(device_freq);
        double packets_per_second = static_cast<double>(max_traffic_count * num_packets) / duration_seconds;

        // Case 1: This test is the first iteration of a new test, or is a single iteration test
        // Generate a new entry for the test, grouping multi-iteration tests into the same entry
        if (config.iteration_number == 0) {
            // Use base name for test name, rather than name with _iter_0 suffix
            const std::string& test_name = config.name;
            // Find test parameters based on the test's first test pattern
            const TrafficPatternConfig& first_pattern = fetch_first_traffic_pattern(config);
            std::string ftype_str = fetch_pattern_ftype(first_pattern);
            std::string ntype_str = fetch_pattern_ntype(first_pattern);
            uint32_t num_packets_first_pattern = fetch_pattern_num_packets(first_pattern);
            uint32_t packet_size_first_pattern = fetch_pattern_packet_size(first_pattern);

            // Create a new entry that represents all iterations of the same test
            bandwidth_results_summary_.emplace_back(BandwidthResultSummary{
                .test_name = test_name,
                .num_iterations = 1,
                .ftype = ftype_str,
                .ntype = ntype_str,
                .topology = std::string(enchantum::to_string(config.fabric_setup.topology)),
                .num_links = config.fabric_setup.num_links,
                .num_packets = num_packets_first_pattern,
                .num_devices = std::vector<uint32_t>(num_devices_set.begin(), num_devices_set.end()),
                .packet_size = packet_size_first_pattern,
                // Push in results for the first iteration
                .cycles_vector = {static_cast<double>(max_cycles)},
                .bandwidth_vector_GB_s = {bandwidth_GB_s},
                .packets_per_second_vector = {packets_per_second},
            });
        }
        // Case 2: This is not the first iteration of a test.
        // Multi-iteration tests are executed sequentially, so we can just append to the last-created test entry
        else {
            BandwidthResultSummary& test_result = bandwidth_results_summary_.back();
            test_result.cycles_vector.push_back(static_cast<double>(max_cycles));
            test_result.bandwidth_vector_GB_s.push_back(bandwidth_GB_s);
            test_result.packets_per_second_vector.push_back(packets_per_second);
            test_result.num_iterations++;
        }
    }

    void calculate_mean(const BandwidthStatistics& stat, const auto& lambda_measurement_vector) {
        // Push statistics name into results summary csv header
        stat_order_.push_back(stat);
        for (auto& result : bandwidth_results_summary_) {
            const std::vector<double>& measurements_vector = lambda_measurement_vector(result);
            double sum = std::accumulate(measurements_vector.begin(), measurements_vector.end(), 0.0);
            double mean = sum / result.num_iterations;
            result.statistics_vector.push_back(mean);
        }
    }

    void calculate_cycles_mean() {
        calculate_mean(BandwidthStatistics::CyclesMean, [](const auto& result) { return result.cycles_vector; });
    }

    void calculate_packets_per_second_mean() {
        calculate_mean(BandwidthStatistics::PacketsPerSecondMean, [](const auto& result) {
            return result.packets_per_second_vector;
        });
    }

    void calculate_bandwidth_mean() {
        calculate_mean(
            BandwidthStatistics::BandwidthMean, [](const auto& result) { return result.bandwidth_vector_GB_s; });
    }

    void calculate_bandwidth_min() {
        // Push statistics name into results summary csv header
        stat_order_.push_back(BandwidthStatistics::BandwidthMin);
        for (auto& result : bandwidth_results_summary_) {
            result.statistics_vector.push_back(
                *std::min_element(result.bandwidth_vector_GB_s.begin(), result.bandwidth_vector_GB_s.end()));
        }
    }

    void calculate_bandwidth_max() {
        // Push statistics name into results summary csv header
        stat_order_.push_back(BandwidthStatistics::BandwidthMax);
        for (auto& result : bandwidth_results_summary_) {
            result.statistics_vector.push_back(
                *std::max_element(result.bandwidth_vector_GB_s.begin(), result.bandwidth_vector_GB_s.end()));
        }
    }

    void calculate_bandwidth_std_dev() {
        // Push statistics name into results summary csv header
        stat_order_.push_back(BandwidthStatistics::BandwidthStdDev);
        for (auto& result : bandwidth_results_summary_) {
            double sum = std::accumulate(result.bandwidth_vector_GB_s.begin(), result.bandwidth_vector_GB_s.end(), 0.0);
            double mean = sum / result.num_iterations;
            double variance = 0.0;
            for (auto& bandwidth_gb_s : result.bandwidth_vector_GB_s) {
                variance += std::pow(bandwidth_gb_s - mean, 2);
            }
            variance /= result.num_iterations;
            double std_dev = std::sqrt(variance);
            result.statistics_vector.push_back(std_dev);
        }
    }

    void calculate_bandwidth_summary_statistics() {
        // Add new statistics here
        // The statistics will be displayed in the bandwidth summary CSV file in this order
        // The name of each statistic collected is maintained in-order in the stat_order_ vector
        // The statistics are calculated for each test in the same order and are stored in each test's
        // BandwidthResultSummary.statistics_vector Each function here should calculate the statistics for every test
        // within a single invocation (see functions for details) NOTE: If you add new statistics, you must re-generate
        // the golden CSV file, otherwise benchmarking will fail.
        calculate_cycles_mean();
        calculate_packets_per_second_mean();
        calculate_bandwidth_mean();
        calculate_bandwidth_min();
        calculate_bandwidth_max();
        calculate_bandwidth_std_dev();
    }

    void generate_bandwidth_csv(const TestConfig& config) {
        // Extract representative ftype and ntype from first sender's first pattern
        const TrafficPatternConfig& first_pattern = fetch_first_traffic_pattern(config);
        std::string ftype_str = fetch_pattern_ftype(first_pattern);
        std::string ntype_str = fetch_pattern_ntype(first_pattern);

        // Open CSV file in append mode
        std::ofstream csv_stream(csv_file_path_, std::ios::out | std::ios::app);
        if (!csv_stream.is_open()) {
            log_error(tt::LogTest, "Failed to open CSV file for appending: {}", csv_file_path_.string());
            return;
        }

        // Write data rows (header already written in initialize_bandwidth_results_csv_file)
        for (const auto& result : bandwidth_results_) {
            csv_stream << config.name << "," << ftype_str << "," << ntype_str << ","
                       << enchantum::to_string(config.fabric_setup.topology) << "," << result.num_devices << ","
                       << result.device_id << "," << config.fabric_setup.num_links << ","
                       << enchantum::to_string(result.direction) << "," << result.total_traffic_count << ","
                       << result.num_packets << "," << result.packet_size << "," << result.cycles << "," << std::fixed
                       << std::setprecision(6) << result.bandwidth_GB_s << "," << std::fixed << std::setprecision(3)
                       << result.packets_per_second;

            if (this->telemetry_enabled_) {
                csv_stream << "," << std::fixed << std::setprecision(3) << result.telemetry_bw_GB_s_min.value() << ","
                           << std::fixed << std::setprecision(3) << result.telemetry_bw_GB_s_avg.value() << ","
                           << std::fixed << std::setprecision(3) << result.telemetry_bw_GB_s_max.value();
            }
            csv_stream << "\n";
        }

        csv_stream.close();
        log_info(tt::LogTest, "Bandwidth results appended to CSV file: {}", csv_file_path_.string());
    }

    std::vector<GoldenCsvEntry>::iterator fetch_corresponding_golden_entry(const BandwidthResultSummary& test_result);

    void generate_bandwidth_summary_csv() {
        // Bandwidth summary CSV file is generated separately from Bandwidth CSV because we need to wait for all
        // multirun tests to complete Generate detailed CSV filename
        std::ostringstream summary_oss;
        auto arch_name = tt::tt_metal::hal::get_arch_name();
        summary_oss << "bandwidth_summary_results_" << arch_name << ".csv";
        // Output directory already set in initialize_bandwidth_results_csv_file()
        std::filesystem::path output_path =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / output_dir;
        csv_summary_file_path_ = output_path / summary_oss.str();

        // Create detailed CSV file with header
        std::ofstream summary_csv_stream(csv_summary_file_path_, std::ios::out | std::ios::trunc);  // Truncate file
        if (!summary_csv_stream.is_open()) {
            log_error(tt::LogTest, "Failed to create summary CSV file: {}", csv_summary_file_path_.string());
            return;
        }

        // Write detailed header
        summary_csv_stream << "test_name,ftype,ntype,topology,num_devices,num_links,packet_size,iterations";
        for (BandwidthStatistics stat : stat_order_) {
            const std::string& stat_name = BandwidthStatisticsHeader.at(stat);
            summary_csv_stream << "," << stat_name;
        }
        summary_csv_stream << ",tolerance_percent";
        summary_csv_stream << "\n";
        log_info(tt::LogTest, "Initialized summary CSV file: {}", csv_summary_file_path_.string());

        // Write data rows
        for (const auto& result : bandwidth_results_summary_) {
            // Convert vector of num_devices to a string representation
            std::string num_devices_str = convert_num_devices_to_string(result.num_devices);
            summary_csv_stream << result.test_name << "," << result.ftype << "," << result.ntype << ","
                               << result.topology << ",\"" << num_devices_str << "\"," << result.num_links << ","
                               << result.packet_size << "," << result.num_iterations;
            for (double stat : result.statistics_vector) {
                summary_csv_stream << "," << std::fixed << std::setprecision(6) << stat;
            }
            // Find the corresponding golden entry for this test result
            auto golden_it = fetch_corresponding_golden_entry(result);
            if (golden_it == golden_csv_entries_.end()) {
                log_warning(
                    tt::LogTest,
                    "Golden CSV entry not found for test {}, putting tolerance of 1.0 in summary CSV",
                    result.test_name);
                summary_csv_stream << "," << 1.0;
            } else {
                summary_csv_stream << "," << golden_it->tolerance_percent;
            }
            summary_csv_stream << "\n";
        }
        summary_csv_stream.close();
        log_info(tt::LogTest, "Bandwidth summary results appended to CSV file: {}", csv_summary_file_path_.string());
    }

    std::string get_golden_csv_filename() {
        auto arch_name = tt::tt_metal::hal::get_arch_name();
        auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();

        // Convert cluster type enum to lowercase string
        std::string cluster_name = std::string(enchantum::to_string(cluster_type));
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
            entry.num_iterations = std::stoul(tokens[7]);
            entry.cycles = std::stod(tokens[8]);
            entry.packets_per_second = std::stod(tokens[9]);
            entry.bandwidth_GB_s = std::stod(tokens[10]);
            // Skip min, max, std dev
            entry.tolerance_percent = std::stod(tokens[14]);
            golden_csv_entries_.push_back(entry);
        }

        golden_file.close();
        log_info(tt::LogTest, "Loaded {} golden entries from: {}", golden_csv_entries_.size(), golden_path.string());
        return true;
    }

    void populate_comparison_result_bandwidth(
        double result_bandwidth_GB_s, ComparisonResult& comp_result, auto& golden_it) {
        comp_result.current_bandwidth_GB_s = result_bandwidth_GB_s;

        double test_tolerance = 1.0;  // Default tolerance for no golden case
        if (golden_it != golden_csv_entries_.end()) {
            comp_result.golden_bandwidth_GB_s = golden_it->bandwidth_GB_s;
            // Use per-test tolerance from golden CSV instead of global tolerance
            test_tolerance = golden_it->tolerance_percent;
            comp_result.within_tolerance = std::abs(comp_result.difference_percent()) <= test_tolerance;

            if (comp_result.within_tolerance) {
                comp_result.status = "PASS";
            } else {
                comp_result.status = "FAIL";
            }
        } else {
            log_warning(tt::LogTest, "Golden CSV entry not found for test {}", comp_result.test_name);
            comp_result.golden_bandwidth_GB_s = 0.0;
            // Set within_tolerance explicitly to prevent an edge case, where golden is not found and test result is
            // 0.0, which would cause subsequent within_tolerance checks to be true
            comp_result.within_tolerance = false;
            comp_result.status = "NO_GOLDEN";
        }
    }

    ComparisonResult create_comparison_result(const BandwidthResultSummary& test_result);

    std::string convert_num_devices_to_string(const std::vector<uint32_t>& num_devices);

    std::string generate_failed_test_format_string(
        const BandwidthResultSummary& test_result,
        double test_result_avg_bandwidth,
        double difference_percent,
        double acceptable_tolerance);

    void compare_summary_results_with_golden() {
        if (golden_csv_entries_.empty()) {
            log_warning(tt::LogTest, "Skipping golden CSV comparison - no golden file found");
            return;
        }
        if (bandwidth_results_summary_.size() != golden_csv_entries_.size()) {
            log_warning(
                tt::LogTest,
                "Number of test results ({}) does not match number of golden entries ({})",
                bandwidth_results_summary_.size(),
                golden_csv_entries_.size());
        }

        for (int i = 0; i < bandwidth_results_summary_.size(); i++) {
            BandwidthResultSummary& test_result = bandwidth_results_summary_[i];
            // Find Average bandwidth result for the test
            // Statistic name for average bandwidth is set in calculate_bandwidth_mean()
            auto bandwidth_stat_location =
                std::find(stat_order_.begin(), stat_order_.end(), BandwidthStatistics::BandwidthMean);
            if (bandwidth_stat_location == stat_order_.end()) {
                log_error(tt::LogTest, "Average bandwidth statistic not found, was it calculated?");
                return;
            }
            int bandwidth_stat_index = std::distance(stat_order_.begin(), bandwidth_stat_location);
            double test_result_avg_bandwidth = test_result.statistics_vector[bandwidth_stat_index];

            // Search for the corresponding golden entry for this test result
            auto golden_it = fetch_corresponding_golden_entry(test_result);

            // Compare the test result with the golden entry
            ComparisonResult comp_result = create_comparison_result(test_result);
            populate_comparison_result_bandwidth(test_result_avg_bandwidth, comp_result, golden_it);
            comparison_results_.push_back(comp_result);

            if (!comp_result.within_tolerance) {
                double acceptable_tolerance = 0.0;
                if (golden_it != golden_csv_entries_.end()) {
                    acceptable_tolerance = golden_it->tolerance_percent;
                }
                std::string csv_format_string = generate_failed_test_format_string(
                    test_result, test_result_avg_bandwidth, comp_result.difference_percent(), acceptable_tolerance);
                all_failed_tests_.push_back(csv_format_string);
            }
        }

        // Write comparison results to diff CSV file
        // Output directory already created in initialize_bandwidth_results_csv_file()
        std::filesystem::path output_path =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) / output_dir;
        std::ostringstream diff_oss;
        auto arch_name = tt::tt_metal::hal::get_arch_name();
        diff_oss << "bandwidth_summary_results_" << arch_name << "_diff.csv";
        diff_csv_file_path_ = output_path / diff_oss.str();
        // Create diff CSV file with header
        std::ofstream diff_csv_stream(diff_csv_file_path_, std::ios::out | std::ios::trunc);  // Truncate file
        if (!diff_csv_stream.is_open()) {
            log_error(tt::LogTest, "Failed to create diff CSV file: {}", diff_csv_file_path_.string());
            return;
        }
        // Write diff header
        diff_csv_stream << "test_name,ftype,ntype,topology,num_devices,num_links,packet_size,num_iterations,"
                           "current_avg_bandwidth_gb_s,golden_avg_bandwidth_gb_s,difference_percent,status\n";
        log_info(tt::LogTest, "Initialized diff CSV file: {}", diff_csv_file_path_.string());

        for (const auto& result : comparison_results_) {
            diff_csv_stream << result.test_name << "," << result.ftype << "," << result.ntype << "," << result.topology
                            << ",\"" << result.num_devices << "\"," << result.num_links << "," << result.packet_size
                            << "," << result.num_iterations << "," << std::fixed << std::setprecision(6)
                            << result.current_bandwidth_GB_s << "," << result.golden_bandwidth_GB_s << ","
                            << std::setprecision(2) << result.difference_percent() << "," << result.status << "\n";
        }
        diff_csv_stream.close();
        log_info(tt::LogTest, "Comparison diff CSV results appended to: {}", diff_csv_file_path_.string());
    }

    void validate_against_golden() {
        if (comparison_results_.empty()) {
            log_info(tt::LogTest, "No golden comparison performed (no golden file found)");
            return;
        }

        if (!all_failed_tests_.empty()) {
            has_test_failures_ = true;
            log_error(tt::LogTest, "The following tests failed golden comparison (using per-test tolerance):");
            for (const auto& failed_test : all_failed_tests_) {
                log_error(tt::LogTest, "  - {}", failed_test);
            }
        } else {
            log_info(tt::LogTest, "All tests passed golden comparison using per-test tolerance values");
        }
    }

    void set_comparison_statistics_csv_file_path();

    // Track sync cores for each device
    std::unordered_map<FabricNodeId, CoreCoord> device_global_sync_cores_;
    std::unordered_map<FabricNodeId, std::vector<CoreCoord>> device_local_sync_cores_;

    std::shared_ptr<TestFixture> fixture_;
    std::unordered_map<MeshCoordinate, TestDevice> test_devices_;
    std::unique_ptr<tt::tt_fabric::fabric_tests::GlobalAllocator> allocator_;
    std::unique_ptr<tt::tt_fabric::fabric_tests::DynamicPolicyManager>
        policy_manager_;  // Manages dynamic policy computation and caching

    // Uniform memory maps shared across all devices
    tt::tt_fabric::fabric_tests::SenderMemoryMap sender_memory_map_;
    tt::tt_fabric::fabric_tests::ReceiverMemoryMap receiver_memory_map_;
    tt::tt_fabric::fabric_tests::AllocatorPolicies allocation_policies_;

    // Dynamic allocation policy control
    bool use_dynamic_policies_ = true;  // Whether to compute dynamic policies per test

    bool benchmark_mode_ = false;     // Benchmark mode for current test
    bool telemetry_enabled_ = false;  // Telemetry enabled for current test
    bool global_sync_ = false;        // Line sync for current test
    uint32_t global_sync_val_ = 0;

    // Performance profiling data
    // TODO: add link index into the result
    std::map<FabricNodeId, std::map<RoutingDirection, uint32_t>> outgoing_traffic_;
    std::map<FabricNodeId, std::map<RoutingDirection, std::map<uint32_t, uint64_t>>> device_direction_cycles_;
    std::map<FabricNodeId, std::map<CoreCoord, uint64_t>> device_core_cycles_;
    std::vector<BandwidthResult> bandwidth_results_;
    std::vector<BandwidthResultSummary> bandwidth_results_summary_;
    std::vector<TelemetryEntry> telemetry_entries_;  // Per-test raw data
    std::vector<CodeProfilingEntry> code_profiling_entries_;  // Per-test code profiling data
    bool code_profiling_enabled_ = false;

    // Device frequency cache to avoid repeated calculations
    std::unordered_map<FabricNodeId, uint32_t> device_freq_mhz_map_;
    double measured_bw_min_ = 0.0;
    double measured_bw_avg_ = 0.0;
    double measured_bw_max_ = 0.0;

    // Progress monitoring
    ProgressMonitorConfig progress_config_;
    std::filesystem::path raw_telemetry_csv_path_;
    std::vector<BandwidthStatistics> stat_order_;
    std::filesystem::path csv_file_path_;
    std::filesystem::path csv_summary_file_path_;

    // Golden CSV comparison data
    std::vector<GoldenCsvEntry> golden_csv_entries_;
    std::vector<ComparisonResult> comparison_results_;
    std::vector<std::string> all_failed_tests_;  // Accumulates all failed tests across test run
    std::filesystem::path diff_csv_file_path_;
    bool has_test_failures_ = false;  // Track if any tests failed validation

    // Golden CSV comparison statistics
    std::filesystem::path comparison_statistics_csv_file_path_;

    // Ethernet core buffer readback helper
    std::unique_ptr<EthCoreBufferReadback> eth_readback_;

    // Getter for lazy initialization of eth_readback_
    EthCoreBufferReadback& get_eth_readback() {
        if (!eth_readback_) {
            eth_readback_ = std::make_unique<EthCoreBufferReadback>(test_devices_, *fixture_);
        }
        return *eth_readback_;
    }
};
