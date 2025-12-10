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
#include <chrono>
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
#include "tt_fabric_test_bandwidth_profiler.hpp"
#include "tt_fabric_test_eth_readback.hpp"
#include "tt_fabric_test_code_profiler.hpp"
#include "tt_fabric_telemetry_manager.hpp"
#include "tt_fabric_test_constants.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_coord.hpp>

using tt::tt_fabric::fabric_tests::CI_ARTIFACTS_DIR;
using tt::tt_fabric::fabric_tests::DEFAULT_BUILT_TESTS_DUMP_FILE;
using tt::tt_fabric::fabric_tests::OUTPUT_DIR;

using TestFixture = tt::tt_fabric::fabric_tests::TestFixture;
using TestDevice = tt::tt_fabric::fabric_tests::TestDevice;
using TestConfig = tt::tt_fabric::fabric_tests::TestConfig;
using TestFabricSetup = tt::tt_fabric::fabric_tests::TestFabricSetup;
using TrafficParameters = tt::tt_fabric::fabric_tests::TrafficParameters;
using PerformanceTestMode = tt::tt_fabric::fabric_tests::PerformanceTestMode;
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
using FabricTensixConfig = tt::tt_fabric::FabricTensixConfig;

using BandwidthResult = tt::tt_fabric::fabric_tests::BandwidthResult;
using BandwidthResultSummary = tt::tt_fabric::fabric_tests::BandwidthResultSummary;
using LatencyResult = tt::tt_fabric::fabric_tests::LatencyResult;
using GoldenCsvEntry = tt::tt_fabric::fabric_tests::GoldenCsvEntry;
using GoldenLatencyEntry = tt::tt_fabric::fabric_tests::GoldenLatencyEntry;
using ComparisonResult = tt::tt_fabric::fabric_tests::ComparisonResult;
using LatencyComparisonResult = tt::tt_fabric::fabric_tests::LatencyComparisonResult;
using PostComparisonAnalyzer = tt::tt_fabric::fabric_tests::PostComparisonAnalyzer;
using BandwidthStatistics = tt::tt_fabric::fabric_tests::BandwidthStatistics;
using BandwidthProfiler = ::BandwidthProfiler;
using BandwidthResultsManager = tt::tt_fabric::fabric_tests::BandwidthResultsManager;

// Helper functions for parsing traffic pattern parameters
using tt::tt_fabric::fabric_tests::fetch_first_traffic_pattern;
using tt::tt_fabric::fabric_tests::fetch_pattern_ftype;
using tt::tt_fabric::fabric_tests::fetch_pattern_ntype;
using tt::tt_fabric::fabric_tests::fetch_pattern_num_packets;
using tt::tt_fabric::fabric_tests::fetch_pattern_packet_size;

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

        // Initialize bandwidth managers (telemetry/code profiler are lazy)
        bandwidth_profiler_ =
            std::make_unique<BandwidthProfiler>(*fixture_, *fixture_, *fixture_);  // fixture implements interfaces
        bandwidth_results_manager_ = std::make_unique<BandwidthResultsManager>();
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

        // Destroy managers that hold references to eth_readback before clearing it
        code_profiler_.reset();
        telemetry_manager_.reset();
        eth_readback_.reset();

        reset_local_variables();
    }

    void process_traffic_config(TestConfig& config) {
        // Latency test mode: manually populate senders_ and receivers_ maps
        // with latency-specific kernels and configurations
        if (config.performance_test_mode == PerformanceTestMode::LATENCY) {
            setup_latency_test_workers(config);
            return;  // Skip normal resource allocation and traffic config setup
        }

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
            this->set_performance_test_mode(config.performance_test_mode);

            log_debug(tt::LogTest, "Enabled sync, global sync value: {}, ", global_sync_val_);
            log_debug(tt::LogTest, "Performance test mode: {}", enchantum::to_string(performance_test_mode_));

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

    bool open_devices(const TestFabricSetup& fabric_setup) { return fixture_->open_devices(fabric_setup); }

    void compile_programs() {
        fixture_->setup_workload();
        // TODO: should we be taking const ref?
        for (auto& [coord, test_device] : test_devices_) {
            test_device.set_benchmark_mode(performance_test_mode_ == PerformanceTestMode::BANDWIDTH);
            test_device.set_global_sync(global_sync_);
            test_device.set_global_sync_val(global_sync_val_);
            test_device.set_progress_monitoring_enabled(progress_config_.enabled);

            auto device_id = test_device.get_node_id();
            test_device.set_sync_core(device_global_sync_cores_[device_id]);

            // Create kernels (latency or normal)
            if (performance_test_mode_ == PerformanceTestMode::LATENCY) {
                create_latency_kernels_for_device(test_device);
            } else {
                // Normal mode: create standard kernels for all devices
                test_device.create_kernels();
            }
        }

        // Enqueue all programs
        for (auto& [coord, test_device] : test_devices_) {
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
        // Skip telemetry readback in latency test mode because we don't actually care about the values of the telemetry.
        // We only enable it so that latency tests take into account the overheads of having telemetry enabled
        if (this->get_telemetry_enabled() && performance_test_mode_ != PerformanceTestMode::LATENCY) {
            this->read_telemetry();
            this->process_telemetry_for_golden();
            this->dump_raw_telemetry_csv(built_test_config);
        }
    }

    void validate_results() {
        // Skip validation in benchmark or latency mode (neither validates packet contents)
        if (performance_test_mode_ != PerformanceTestMode::NONE) {
            log_info(
                tt::LogTest,
                "Skipping validation (performance_test_mode: {})",
                enchantum::to_string(performance_test_mode_));
            return;
        }

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
        TT_FATAL(bandwidth_profiler_ && bandwidth_results_manager_, "Bandwidth managers not initialized");

        bandwidth_profiler_->profile_results(config, test_devices_, sender_memory_map_);

        // Inject telemetry before retrieving result
        if (telemetry_enabled_) {
            auto& telemetry = get_telemetry_manager();
            bandwidth_profiler_->set_telemetry_bandwidth(
                telemetry.get_measured_bw_min(), telemetry.get_measured_bw_avg(), telemetry.get_measured_bw_max());
        }

        const auto& latest_results = bandwidth_profiler_->get_latest_results();
        for (const auto& result : latest_results) {
            bandwidth_results_manager_->add_result(config, result);
        }
        bandwidth_results_manager_->add_summary(config, bandwidth_profiler_->get_latest_summary());

        // Append last result (matches previous behavior)
        if (!latest_results.empty()) {
            bandwidth_results_manager_->append_to_csv(config, latest_results.back());
        }
    }

    void collect_latency_results();
    void report_latency_results(const TestConfig& config);

    void generate_latency_summary() {
        // Load golden latency CSV file
        load_golden_latency_csv();

        // Generate latency results CSV file with all results
        generate_latency_results_csv();

        // Compare latency results with golden CSV
        compare_latency_results_with_golden();

        // Validate latency results against golden (uses common validation method)
        validate_against_golden();
    }

    void generate_bandwidth_summary() {
        TT_FATAL(bandwidth_results_manager_, "Bandwidth results manager not initialized");
        bandwidth_results_manager_->load_golden_csv();
        bandwidth_results_manager_->generate_summary();
        bandwidth_results_manager_->validate_against_golden();
        // Track failures from bandwidth manager
        if (bandwidth_results_manager_->has_failures()) {
            const auto failed = bandwidth_results_manager_->get_failed_tests();
            all_failed_bandwidth_tests_.insert(all_failed_bandwidth_tests_.end(), failed.begin(), failed.end());
            has_test_failures_ = true;
        }
    }

    void initialize_bandwidth_results_csv_file() {
        TT_FATAL(bandwidth_results_manager_, "Bandwidth results manager not initialized");
        bandwidth_results_manager_->initialize_bandwidth_csv_file(this->telemetry_enabled_);
    }

    void initialize_latency_results_csv_file();

    void close_devices() { fixture_->close_devices(); }

    void set_performance_test_mode(PerformanceTestMode mode) { performance_test_mode_ = mode; }

    void set_telemetry_enabled(bool enabled) { telemetry_enabled_ = enabled; }

    PerformanceTestMode get_performance_test_mode() { return performance_test_mode_; }

    bool get_telemetry_enabled() { return telemetry_enabled_; }

    // Code profiling getters/setters
    bool get_code_profiling_enabled() const { return code_profiling_enabled_; }
    void set_code_profiling_enabled(bool enabled) {
        code_profiling_enabled_ = enabled;
        if (code_profiler_) {
            code_profiler_->set_enabled(enabled);
        }
    }
    const std::vector<CodeProfilingEntry>& get_code_profiling_entries() const {
        static const std::vector<CodeProfilingEntry> empty_entries{};
        if (code_profiler_) {
            return code_profiler_->get_entries();
        }
        return empty_entries;
    }

    void set_global_sync(bool global_sync) { global_sync_ = global_sync; }

    void set_global_sync_val(uint32_t val) { global_sync_val_ = val; }

    bool has_test_failures() const { return has_test_failures_; }

    std::vector<std::string> get_all_failed_tests() const;

    void setup_ci_artifacts() {
        std::filesystem::path tt_metal_home =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir());
        std::filesystem::path ci_artifacts_path = tt_metal_home / std::string(CI_ARTIFACTS_DIR);
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

        // Delegate bandwidth artifacts to results manager
        if (bandwidth_results_manager_) {
            bandwidth_results_manager_->setup_ci_artifacts();
        }

        // Latency artifacts (unchanged)
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

    // Configures latency test mode by extracting and storing parameters from the test config
    void setup_latency_test_mode(const TestConfig& config);

private:
    void reset_local_variables() {
        performance_test_mode_ = PerformanceTestMode::NONE;
        global_sync_ = false;
        global_sync_val_ = 0;
        // Note: latency_results_ is NOT cleared here to preserve for golden comparison at end
        // Note: has_test_failures_ is NOT reset here to preserve failures across tests
        // Note: golden_csv_entries_ is kept loaded for reuse across tests
        // Note: latency_results_ is kept for golden comparison after all tests complete
    }

    /**
     * Setup latency test workers and configurations.
     *
     * Latency tests differ from bandwidth tests in several key ways:
     * 1. Use specialized kernels (tt_fabric_latency_sender.cpp / tt_fabric_latency_responder.cpp)
     *    that measure round-trip latency using hardware timestamps
     * 2. Bypass the GlobalAllocator - latency tests use fixed memory layouts and don't need
     *    dynamic allocation of send/receive buffers
     * 3. Store latency samples in result buffers rather than throughput metrics
     * 4. Use a single sender-responder pair rather than arbitrary traffic patterns
     * 5. Constrained to 1 message per sample (no batching)
     *
     * This function manually populates the senders_ and receivers_ maps for latency test
     * workers and sets their latency-specific kernel sources.
     */
    void setup_latency_test_workers(TestConfig& config);

    // Helper struct for latency worker location information
    struct LatencyWorkerLocation {
        TestDevice* device = nullptr;
        MeshCoordinate mesh_coord{0, 0};
        CoreCoord core;
        FabricNodeId node_id{MeshId{0}, 0};
    };

    /**
     * Find a latency worker device by checking for non-empty workers map.
     * Used internally by get_latency_sender_location() and get_latency_receiver_location().
     */
    template <typename GetWorkersMapFunc>
    LatencyWorkerLocation find_latency_worker_device(
        GetWorkersMapFunc get_workers_map, const std::string& worker_type) {
        LatencyWorkerLocation info;
        for (auto& [coord, device] : test_devices_) {
            const auto& workers_map = get_workers_map(device);
            if (!workers_map.empty()) {
                info.device = &device;
                info.mesh_coord = coord;
                info.core = workers_map.begin()->first;
                info.node_id = device.get_node_id();
                break;
            }
        }
        TT_FATAL(info.device != nullptr, "Could not find latency {} device", worker_type);
        return info;
    }

    /**
     * Create latency kernels for a device based on its role (sender, responder, or neither).
     * For sender devices: creates latency sender kernel with responder coordinates.
     * For responder devices: creates latency responder kernel with sender buffer addresses.
     * For other devices: creates normal kernels.
     */
    void create_latency_kernels_for_device(TestDevice& test_device);

    /**
     * Get the location of the latency sender device/core.
     * Searches for the first device with a non-empty senders_ map.
     */
    LatencyWorkerLocation get_latency_sender_location() {
        return find_latency_worker_device([](TestDevice& d) -> const auto& { return d.get_senders(); }, "sender");
    }

    /**
     * Get the location of the latency receiver/responder device/core.
     * Searches for the first device with a non-empty receivers_ map.
     */
    LatencyWorkerLocation get_latency_receiver_location() {
        return find_latency_worker_device([](TestDevice& d) -> const auto& { return d.get_receivers(); }, "receiver");
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

    void generate_latency_results_csv();

    std::vector<GoldenLatencyEntry>::iterator fetch_corresponding_golden_latency_entry(const LatencyResult& test_result);

    std::string get_golden_latency_csv_filename();

    bool load_golden_latency_csv();

    void compare_latency_results_with_golden();

    // Common helper to populate tolerance and status fields
    template <typename CompResultType, typename GoldenIterType>
    void populate_comparison_tolerance_and_status(
        CompResultType& comp_result,
        GoldenIterType golden_it,
        GoldenIterType golden_end,
        double golden_tolerance_default = 1.0) {
        double test_tolerance = golden_tolerance_default;
        if (golden_it != golden_end) {
            test_tolerance = golden_it->tolerance_percent;
            comp_result.within_tolerance = std::abs(comp_result.difference_percent()) <= test_tolerance;
            comp_result.status = comp_result.within_tolerance ? "PASS" : "FAIL";
        } else {
            log_warning(tt::LogTest, "Golden entry not found for test {}", comp_result.test_name);
            comp_result.within_tolerance = false;
            comp_result.status = "NO_GOLDEN";
        }
    }

    // Common CSV diff file initialization
    std::ofstream init_diff_csv_file(std::filesystem::path& diff_csv_path, const std::string& csv_header, const std::string& test_type) {
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

    void validate_against_golden();

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

    PerformanceTestMode performance_test_mode_ = PerformanceTestMode::NONE;  // Performance test mode for current test
    bool telemetry_enabled_ = false;                                         // Telemetry enabled for current test
    bool global_sync_ = false;        // Line sync for current test
    uint32_t global_sync_val_ = 0;

    // Managers (bandwidth)
    std::unique_ptr<BandwidthProfiler> bandwidth_profiler_;
    std::unique_ptr<BandwidthResultsManager> bandwidth_results_manager_;
    std::vector<LatencyResult> latency_results_;
    std::vector<TelemetryEntry> telemetry_entries_;  // Per-test raw data
    bool code_profiling_enabled_ = false;

    // Progress monitoring
    ProgressMonitorConfig progress_config_;
    std::filesystem::path raw_telemetry_csv_path_;
    std::filesystem::path latency_csv_file_path_;

    // Golden CSV comparison data (latency only; bandwidth handled in manager)
    std::vector<GoldenLatencyEntry> golden_latency_entries_;
    std::vector<LatencyComparisonResult> latency_comparison_results_;
    std::vector<std::string> all_failed_bandwidth_tests_;  // Accumulates failed bandwidth tests
    std::vector<std::string> all_failed_latency_tests_;    // Accumulates failed latency tests
    std::filesystem::path latency_diff_csv_file_path_;
    bool has_test_failures_ = false;  // Track if any tests failed validation

    // Ethernet core buffer readback helper
    std::unique_ptr<EthCoreBufferReadback> eth_readback_;
    std::unique_ptr<CodeProfiler> code_profiler_;
    std::unique_ptr<TelemetryManager> telemetry_manager_;

    // Getter for lazy initialization of eth_readback_
    EthCoreBufferReadback& get_eth_readback() {
        if (!eth_readback_) {
            eth_readback_ = std::make_unique<EthCoreBufferReadback>(test_devices_, *fixture_);
        }
        return *eth_readback_;
    }

    CodeProfiler& get_code_profiler() {
        if (!code_profiler_) {
            code_profiler_ = std::make_unique<CodeProfiler>(get_eth_readback());
        }
        return *code_profiler_;
    }

    TelemetryManager& get_telemetry_manager() {
        if (!telemetry_manager_) {
            telemetry_manager_ = std::make_unique<TelemetryManager>(*fixture_, get_eth_readback());
        }
        return *telemetry_manager_;
    }
};
