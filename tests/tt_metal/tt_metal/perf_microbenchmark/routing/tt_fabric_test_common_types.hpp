// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <variant>
#include <optional>
#include <cstdint>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include <tt-metalium/tt_align.hpp>

namespace tt::tt_fabric::fabric_tests {

// Performance test mode - replaces separate latency_test_mode and benchmark_mode booleans
enum class PerformanceTestMode {
    NONE,       // No performance testing (functional test only)
    BANDWIDTH,  // Bandwidth/throughput test mode (formerly benchmark_mode)
    LATENCY     // Latency measurement test mode (formerly latency_test_mode)
};

// Device identifier that can be resolved later (used during parsing)
using DeviceIdentifier = std::variant<
    FabricNodeId,                      // Already resolved
    ChipId,                            // Physical chip ID
    std::pair<MeshId, ChipId>,         // [mesh_id, chip_id]
    std::pair<MeshId, MeshCoordinate>  // [mesh_id, [row, col]]
    >;

// A map to hold various parametrization options parsed from the YAML.
using ParametrizationValues = std::variant<std::vector<std::string>, std::vector<uint32_t>>;
using ParametrizationOptionsMap = std::unordered_map<std::string, ParametrizationValues>;

// Parsed structures (before resolution) - use DeviceIdentifier
struct ParsedDestinationConfig {
    std::optional<DeviceIdentifier> device;
    std::optional<CoreCoord> core;
    std::optional<std::unordered_map<RoutingDirection, uint32_t>> hops;
    std::optional<uint32_t> target_address;
    std::optional<uint32_t> atomic_inc_address;
};

struct ParsedTrafficPatternConfig {
    std::optional<ChipSendType> ftype;
    std::optional<NocSendType> ntype;
    std::optional<uint32_t> size;
    std::optional<uint32_t> num_packets;
    std::optional<ParsedDestinationConfig> destination;
    std::optional<uint32_t> atomic_inc_val;
    std::optional<uint32_t> mcast_start_hops;
};

struct ParsedSenderConfig {
    DeviceIdentifier device = FabricNodeId(MeshId{0}, 0);
    std::optional<CoreCoord> core;
    std::vector<ParsedTrafficPatternConfig> patterns;
    std::optional<uint32_t> link_id;  // Link ID for multi-link tests
};

// Resolved structures (after resolution) - use FabricNodeId
struct DestinationConfig {
    std::optional<FabricNodeId> device;
    std::optional<CoreCoord> core;
    std::optional<std::unordered_map<RoutingDirection, uint32_t>> hops;
    std::optional<uint32_t> target_address;
    std::optional<uint32_t> atomic_inc_address;
};

// Credit flow structures for bidirectional sender-receiver communication
struct SenderCreditInfo {
    uint32_t expected_receiver_count{};        // How many receivers to wait for
    uint32_t credit_reception_address_base{};  // Base L1 address for credit chunk (mcast support)
    uint32_t initial_credits{};                // Initial credit capacity (based on receiver buffer size)
};

struct TrafficPatternConfig {
    std::optional<ChipSendType> ftype;
    std::optional<NocSendType> ntype;
    std::optional<uint32_t> size;
    std::optional<uint32_t> num_packets;
    std::optional<DestinationConfig> destination;
    std::optional<uint32_t> atomic_inc_val;
    std::optional<uint32_t> mcast_start_hops;

    // Credit info
    std::optional<SenderCreditInfo> sender_credit_info;  // For sender
    std::optional<uint32_t> credit_return_batch_size;    // For receivers
};

struct SenderConfig {
    FabricNodeId device = FabricNodeId(MeshId{0}, 0);
    std::optional<CoreCoord> core;
    std::vector<TrafficPatternConfig> patterns;
    uint32_t link_id = 0;  // Link ID for multi-link tests
};

// Sync configuration for a single device
struct SyncConfig {
    uint32_t sync_val = 0;       // Sync value for this device
    SenderConfig sender_config;  // Sync messages sent by this device
};

enum class HighLevelTrafficPattern {
    AllToAll,
    OneToAll,
    AllToOne,
    AllToOneRandom,
    FullDeviceRandomPairing,
    UnidirectionalLinear,
    FullRing,
    HalfRing,
    AllDevicesUniformPattern,
    NeighborExchange,
    SequentialAllToAll,
};

struct TestFabricSetup {
    tt::tt_fabric::Topology topology{0};
    std::optional<tt_fabric::FabricTensixConfig> fabric_tensix_config;
    std::optional<tt_fabric::FabricReliabilityMode> fabric_reliability_mode;
    uint32_t num_links{};
    std::optional<std::string> torus_config;  // For Torus topology: "X", "Y", or "XY"
    std::optional<uint32_t> max_packet_size;  // Custom max packet size for router
};

struct HighLevelPatternConfig {
    std::string type;
    std::optional<uint32_t> iterations;
};

struct ParsedTestConfig {
    std::string name;               // Original base name for golden lookup
    std::string parametrized_name;  // Enhanced name for debugging and logging
    TestFabricSetup fabric_setup;
    std::optional<std::vector<std::string>> skip;  // Platforms on which this test should be skipped
    std::optional<std::string> on_missing_param_policy;
    std::optional<ParsedTrafficPatternConfig> defaults;
    std::optional<ParametrizationOptionsMap> parametrization_params;
    // A test can be defined by either a concrete list of senders or a high-level pattern.
    std::optional<std::vector<HighLevelPatternConfig>> patterns;
    // add sync sender configs here, each config contains current device and the patterns
    std::vector<SyncConfig> sync_configs;
    std::vector<ParsedSenderConfig> senders;
    std::optional<std::string> bw_calc_func;
    PerformanceTestMode performance_test_mode =
        PerformanceTestMode::NONE;   // Performance testing mode (NONE, BANDWIDTH, or LATENCY)
    bool telemetry_enabled = false;  // Enable telemetry for performance testing
    bool global_sync = false;  // Enable sync for device synchronization. Typically used for benchmarking to minimize
                               // cross-chip start-skew effects
    bool enable_flow_control = false;  // Enable flow control for all patterns in this test
    bool skip_packet_validation = false;  // Enable benchmark mode in sender and receiver kernels (skips validation)
    uint32_t seed{};
    uint32_t num_top_level_iterations = 1;  // Number of times to repeat a built test
};

struct TestConfig {
    std::string name;               // Original base name for golden lookup
    std::string parametrized_name;  // Enhanced name for debugging and logging
    uint32_t iteration_number = 0;  // For multi-iteration tests, notes the specific iteration of this test
    TestFabricSetup fabric_setup;
    std::optional<std::string> on_missing_param_policy;
    std::optional<TrafficPatternConfig> defaults;
    std::optional<ParametrizationOptionsMap> parametrization_params;
    // A test can be defined by either a concrete list of senders or a high-level pattern.
    std::optional<std::vector<HighLevelPatternConfig>> patterns;
    // add sync sender configs here, each config contains current device and the patterns
    std::vector<SyncConfig> sync_configs;
    std::vector<SenderConfig> senders;
    std::optional<std::string> bw_calc_func;
    PerformanceTestMode performance_test_mode =
        PerformanceTestMode::NONE;  // Performance testing mode (NONE, BANDWIDTH, or LATENCY)
    bool telemetry_enabled = false;
    bool global_sync = false;  // Enable sync for device synchronization. Typically used for benchmarking to minimize
                               // cross-chip start-skew effects
    bool enable_flow_control = false;  // Enable flow control for all patterns in this test
    bool skip_packet_validation = false;  // Enable benchmark mode in sender and receiver kernels (skips validation)
    uint32_t seed{};
};

// Latency test results structure (parallel to bandwidth results)
struct LatencyResults {
    std::string test_name;
    uint32_t num_samples;
    uint32_t message_size_bytes;
    std::vector<uint64_t> latencies_cycles;  // raw cycle counts
    std::vector<double> latencies_ns;        // converted to ns
    uint64_t min_latency_cycles;
    uint64_t max_latency_cycles;
    double avg_latency_ns;
    double p50_latency_ns;
    double p99_latency_ns;
};

// ======================================================================================
// Allocation Policies
// ======================================================================================
namespace detail {
constexpr uint32_t DEFAULT_MAX_SENDER_CONFIGS_PER_CORE = 1;
constexpr uint32_t DEFAULT_MAX_RECEIVER_CONFIGS_PER_CORE = 2;
constexpr uint32_t DEFAULT_SENDER_INITIAL_POOL_SIZE = 1;
constexpr uint32_t DEFAULT_SENDER_POOL_REFILL_SIZE = 1;
constexpr uint32_t DEFAULT_PAYLOAD_CHUNK_SIZE_BYTES = 0x80000;  // 512KB
constexpr uint32_t DEFAULT_RECEIVER_L1_SIZE = 0x100000;
}  // namespace detail

enum class CoreType {
    SENDER,
    RECEIVER,
};

constexpr size_t SENDER_TYPE_IDX = static_cast<size_t>(CoreType::SENDER);
constexpr size_t RECEIVER_TYPE_IDX = static_cast<size_t>(CoreType::RECEIVER);

enum class CoreAllocationPolicy {
    RoundRobin,    // Cycle through available cores to distribute load.
    ExhaustFirst,  // Fill one core with workers before moving to the next.
};

struct CoreAllocationConfig {
    CoreAllocationPolicy policy = CoreAllocationPolicy::RoundRobin;
    uint32_t max_configs_per_core = 1;

    // Size of the initial pool of active cores to cycle through.
    uint32_t initial_pool_size = 0;
    // When the pool is exhausted, how many new cores to add to the active set.
    uint32_t pool_refill_size = 1;

    static CoreAllocationConfig get_default_sender_allocation_config() {
        // Default sender policy: one sender per core.
        return CoreAllocationConfig{
            .policy = CoreAllocationPolicy::ExhaustFirst,
            .max_configs_per_core = detail::DEFAULT_MAX_SENDER_CONFIGS_PER_CORE,
            .initial_pool_size = 1,  // ExhaustFirst is equivalent to a pool size of 1.
            .pool_refill_size = detail::DEFAULT_SENDER_POOL_REFILL_SIZE,
        };
    }

    static CoreAllocationConfig get_default_receiver_allocation_config() {
        // Default receiver policy: reuse a core until it's full.
        return CoreAllocationConfig{
            .policy = CoreAllocationPolicy::ExhaustFirst,
            .max_configs_per_core = detail::DEFAULT_MAX_RECEIVER_CONFIGS_PER_CORE,
            // No default pool sizes for receivers. The pool will be populated with all remaining cores
            // after senders have been allocated.
        };
    }
};

struct AllocatorPolicies {
    CoreAllocationConfig sender_config;
    CoreAllocationConfig receiver_config;
    uint32_t default_payload_chunk_size;

    AllocatorPolicies(
        std::optional<CoreAllocationConfig> sender_config = std::nullopt,
        std::optional<CoreAllocationConfig> receiver_config = std::nullopt,
        std::optional<uint32_t> default_payload_chunk_size = std::nullopt) {
        if (sender_config.has_value()) {
            this->sender_config = sender_config.value();
        } else {
            this->sender_config = CoreAllocationConfig::get_default_sender_allocation_config();
        }

        if (receiver_config.has_value()) {
            this->receiver_config = receiver_config.value();
        } else {
            this->receiver_config = CoreAllocationConfig::get_default_receiver_allocation_config();
        }

        if (default_payload_chunk_size.has_value()) {
            this->default_payload_chunk_size = default_payload_chunk_size.value();
        } else {
            // derive a reasonable default based on the number of configs served per receiver core
            auto payload_chunk_size = detail::DEFAULT_RECEIVER_L1_SIZE / this->receiver_config.max_configs_per_core;
            // since L1 alignment is not available here, align to 64 bytes as a safe minimum
            this->default_payload_chunk_size = tt::align(payload_chunk_size, 64);
        }
    }
};

struct PhysicalMeshConfig {
    std::string mesh_descriptor_path;
    std::vector<std::vector<EthCoord>> eth_coord_mapping;

    PhysicalMeshConfig() : eth_coord_mapping({}) {
        // Default path to the mesh descriptor.
    }
};

// Helper functions for fetching pattern parameters
TrafficPatternConfig fetch_first_traffic_pattern(const TestConfig& config);

std::string fetch_pattern_test_type(const TrafficPatternConfig& pattern, auto lambda_test_type);

std::string fetch_pattern_ftype(const TrafficPatternConfig& pattern);

std::string fetch_pattern_ntype(const TrafficPatternConfig& pattern);

uint32_t fetch_pattern_int(const TrafficPatternConfig& pattern, auto lambda_parameter);

uint32_t fetch_pattern_num_packets(const TrafficPatternConfig& pattern);

uint32_t fetch_pattern_packet_size(const TrafficPatternConfig& pattern);

}  // namespace tt::tt_fabric::fabric_tests
