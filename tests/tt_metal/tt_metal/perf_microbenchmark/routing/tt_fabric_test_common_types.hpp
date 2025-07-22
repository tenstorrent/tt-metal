// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <variant>
#include <optional>
#include <cstdint>

#include <tt-metalium/fabric_edm_types.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/device.hpp>

namespace tt::tt_fabric::fabric_tests {

// Device identifier that can be resolved later (used during parsing)
using DeviceIdentifier = std::variant<
    FabricNodeId,                      // Already resolved
    chip_id_t,                         // Physical chip ID
    std::pair<MeshId, chip_id_t>,      // [mesh_id, chip_id]
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
    std::optional<uint16_t> atomic_inc_val;
    std::optional<uint16_t> atomic_inc_wrap;
    std::optional<uint32_t> mcast_start_hops;
};

struct ParsedSenderConfig {
    DeviceIdentifier device = FabricNodeId(MeshId{0}, 0);
    std::optional<CoreCoord> core;
    std::vector<ParsedTrafficPatternConfig> patterns;
};

// Resolved structures (after resolution) - use FabricNodeId
struct DestinationConfig {
    std::optional<FabricNodeId> device;
    std::optional<CoreCoord> core;
    std::optional<std::unordered_map<RoutingDirection, uint32_t>> hops;
    std::optional<uint32_t> target_address;
    std::optional<uint32_t> atomic_inc_address;
};

struct TrafficPatternConfig {
    std::optional<ChipSendType> ftype;
    std::optional<NocSendType> ntype;
    std::optional<uint32_t> size;
    std::optional<uint32_t> num_packets;
    std::optional<DestinationConfig> destination;
    std::optional<uint16_t> atomic_inc_val;
    std::optional<uint16_t> atomic_inc_wrap;
    std::optional<uint32_t> mcast_start_hops;
};

struct SenderConfig {
    FabricNodeId device = FabricNodeId(MeshId{0}, 0);
    std::optional<CoreCoord> core;
    std::vector<TrafficPatternConfig> patterns;
};

enum class RoutingType {
    LowLatency,
    Dynamic,
};

enum class HighLevelTrafficPattern {
    AllToAllUnicast,
    FullDeviceRandomPairing,
    AllToAllMulticast,
    UnidirectionalLinearMulticast,
    FullRingMulticast,
    HalfRingMulticast,
};

struct TestFabricSetup {
    tt::tt_fabric::Topology topology;
    std::optional<RoutingType> routing_type;
};

struct HighLevelPatternConfig {
    std::string type;
    std::optional<uint32_t> iterations;
};

struct ParsedTestConfig {
    std::string name;
    TestFabricSetup fabric_setup;
    std::optional<std::string> on_missing_param_policy;
    std::optional<ParsedTrafficPatternConfig> defaults;
    std::optional<ParametrizationOptionsMap> parametrization_params;
    // A test can be defined by either a concrete list of senders or a high-level pattern.
    std::optional<std::vector<HighLevelPatternConfig>> patterns;
    // add sync sender configs here, each config contains current device and the patterns
    std::vector<SenderConfig> global_sync_configs;
    std::vector<ParsedSenderConfig> senders;
    std::optional<std::string> bw_calc_func;
    bool benchmark_mode = false;  // Enable benchmark mode for performance testing
    bool global_sync = false;     // Enable sync for device synchronization. Typically used for benchmarking to minimize
                                  // cross-chip start-skew effects
    uint32_t global_sync_val = 0;
    uint32_t seed;
};

struct TestConfig {
    std::string name;
    TestFabricSetup fabric_setup;
    std::optional<std::string> on_missing_param_policy;
    std::optional<TrafficPatternConfig> defaults;
    std::optional<ParametrizationOptionsMap> parametrization_params;
    // A test can be defined by either a concrete list of senders or a high-level pattern.
    std::optional<std::vector<HighLevelPatternConfig>> patterns;
    // add sync sender configs here, each config contains current device and the patterns
    std::vector<SenderConfig> global_sync_configs;
    std::vector<SenderConfig> senders;
    std::optional<std::string> bw_calc_func;
    bool benchmark_mode = false;  // Enable benchmark mode for performance testing
    bool global_sync = false;     // Enable sync for device synchronization. Typically used for benchmarking to minimize
                                  // cross-chip start-skew effects
    uint32_t global_sync_val = 0;
    uint32_t seed;
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
            this->default_payload_chunk_size =
                detail::DEFAULT_RECEIVER_L1_SIZE / this->receiver_config.max_configs_per_core;
        }
    }
};

}  // namespace tt::tt_fabric::fabric_tests
