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
#include "tt_fabric_test_latency_manager.hpp"
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
using TestTrafficSyncConfig = tt::tt_fabric::fabric_tests::TestTrafficSyncConfig;
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
using LatencyTestManager = ::LatencyTestManager;

// Helper functions for parsing traffic pattern parameters
using tt::tt_fabric::fabric_tests::fetch_first_traffic_pattern;
using tt::tt_fabric::fabric_tests::fetch_pattern_ftype;
using tt::tt_fabric::fabric_tests::fetch_pattern_ntype;
using tt::tt_fabric::fabric_tests::fetch_pattern_num_packets;
using tt::tt_fabric::fabric_tests::fetch_pattern_packet_size;

// Helper functions for code profiling
using tt::tt_fabric::convert_code_profiling_timer_type_to_str;
using tt::tt_fabric::convert_to_code_profiling_timer_type;

// Access to internal API: ProgramImpl::num_kernel
#include "impl/program/program_impl.hpp"

class TestContext {
public:
    void init(
        std::shared_ptr<TestFixture> fixture,
        const tt::tt_fabric::fabric_tests::AllocatorPolicies& policies,
        bool use_dynamic_policies = true);

    void prepare_for_test(const TestConfig& config);

    uint32_t get_randomized_master_seed() const { return fixture_->get_randomized_master_seed(); }

    void setup_devices();

    void reset_devices();

    void add_sync_traffic_to_devices(const TestConfig& config);

    void process_traffic_config(TestConfig& config);

    bool open_devices(const TestFabricSetup& fabric_setup) { return fixture_->open_devices(fabric_setup); }

    void compile_programs();

    void launch_programs() { fixture_->run_programs(); }

    void wait_for_programs() { fixture_->wait_for_programs(); }

    void enable_progress_monitoring(const ProgressMonitorConfig& config);

    void wait_for_programs_with_progress();

    // Accessors for progress monitor
    const std::unordered_map<MeshCoordinate, TestDevice>& get_test_devices() const { return test_devices_; }

    const SenderMemoryMap& get_sender_memory_map() const { return sender_memory_map_; }

    IDeviceInfoProvider* get_device_info_provider() const { return fixture_.get(); }

    void process_telemetry_data(TestConfig& built_test_config);

    void validate_results();

    void profile_results(const TestConfig& config);

    void collect_latency_results();

    void report_latency_results(const TestConfig& config);

    void generate_latency_summary();

    void generate_bandwidth_summary();

    void initialize_bandwidth_results_csv_file();

    void initialize_latency_results_csv_file();

    void close_devices() { fixture_->close_devices(); }

    void set_performance_test_mode(PerformanceTestMode mode) { performance_test_mode_ = mode; }

    void set_telemetry_enabled(bool enabled) { telemetry_enabled_ = enabled; }

    PerformanceTestMode get_performance_test_mode() { return performance_test_mode_; }

    bool get_telemetry_enabled() { return telemetry_enabled_; }

    void set_skip_packet_validation(bool skip_packet_validation) { skip_packet_validation_ = skip_packet_validation; }

    bool get_skip_packet_validation() { return skip_packet_validation_; }

    // Code profiling getters/setters
    bool get_code_profiling_enabled() const { return code_profiling_enabled_; }

    void set_code_profiling_enabled(bool enabled);

    const std::vector<CodeProfilingEntry>& get_code_profiling_entries() const {
        static const std::vector<CodeProfilingEntry> empty_entries{};
        if (code_profiler_) {
            return code_profiler_->get_entries();
        }
        return empty_entries;
    }

    void set_global_sync(bool global_sync) { global_sync_ = global_sync; }

    bool has_test_failures() const { return has_test_failures_; }

    std::vector<std::string> get_all_failed_tests() const;

    void setup_ci_artifacts();

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
    
    void initialize_code_profiling_results_csv_file();

    std::string convert_coord_to_string(const ::tt::tt_metal::distributed::MeshCoordinate& coord);

    void dump_code_profiling_results_to_csv(const TestConfig& config);

private:
    void reset_local_variables();

    void setup_latency_test_workers(TestConfig& config);

    void create_latency_kernels_for_device(TestDevice& test_device);

    LatencyTestManager::LatencyWorkerLocation get_latency_sender_location();

    LatencyTestManager::LatencyWorkerLocation get_latency_receiver_location();

    void add_traffic_config(const TestTrafficConfig& traffic_config);

    void initialize_memory_maps();

    void update_memory_maps(const tt::tt_fabric::fabric_tests::AllocatorPolicies& policies);

    void validate_packet_sizes_for_policy(const TestConfig& config, uint32_t payload_chunk_size);

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
    bool skip_packet_validation_ = false;  // Enable benchmark mode in kernels only (skips validation)
    bool global_sync_ = false;             // Line sync for current test

    // Managers (bandwidth)
    std::unique_ptr<BandwidthProfiler> bandwidth_profiler_;
    std::unique_ptr<BandwidthResultsManager> bandwidth_results_manager_;
    std::unique_ptr<LatencyTestManager> latency_test_manager_;
    std::vector<TelemetryEntry> telemetry_entries_;  // Per-test raw data
    bool code_profiling_enabled_ = false;

    // Progress monitoring
    ProgressMonitorConfig progress_config_;
    std::filesystem::path raw_telemetry_csv_path_;

    std::vector<std::string> all_failed_bandwidth_tests_;  // Accumulates failed bandwidth tests
    bool has_test_failures_ = false;  // Track if any tests failed validation

    // Code profiling results CSV file path (TODO: Move to tt_fabric_test_Results)
    std::filesystem::path code_profiling_csv_file_path_;

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
