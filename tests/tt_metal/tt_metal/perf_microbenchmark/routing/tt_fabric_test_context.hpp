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

class TestContext {
public:
    void init(std::shared_ptr<TestFixture> fixture, const tt::tt_fabric::fabric_tests::AllocatorPolicies& policies);
    void setup_devices();
    void reset_devices();
    void process_traffic_config(TestConfig& config);
    void open_devices(Topology topology, RoutingType routing_type);
    void initialize_sync_memory();
    void compile_programs();
    void launch_programs();
    void wait_for_prorgams();
    void validate_results();
    void profile_results(const TestConfig& config);
    void initialize_csv_file();
    void close_devices();
    void set_benchmark_mode(bool benchmark_mode) { benchmark_mode_ = benchmark_mode; }
    void set_global_sync(bool global_sync) { global_sync_ = global_sync; }
    void set_global_sync_val(uint32_t val) { global_sync_val_ = val; }

private:
    void add_traffic_config(const TestTrafficConfig& traffic_config);
    void initialize_memory_maps();
    std::map<FabricNodeId, std::map<RoutingDirection, uint32_t>>
    calculate_outgoing_traffics_through_device_boundaries();
    void trace_traffic_path(const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config);
    void trace_ring_traffic_path(const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config);
    void trace_line_or_mesh_traffic_path(const FabricNodeId& src_node_id, const TestTrafficSenderConfig& config);
    void calculate_bandwidth(const TestConfig& config);
    void generate_bandwidth_csv(const TestConfig& config);

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
    std::map<FabricNodeId, std::map<RoutingDirection, uint64_t>> device_direction_cycles_;
    std::map<FabricNodeId, std::map<CoreCoord, uint64_t>> device_core_cycles_;
    std::vector<BandwidthResult> bandwidth_results_;
    std::vector<BandwidthResultSummary> bandwidth_results_summary_;
    std::filesystem::path csv_file_path_;
    std::filesystem::path csv_summary_file_path_;

    void reset_local_variables();
    void read_performance_results();
    void convert_core_cycles_to_direction_cycles();
};
