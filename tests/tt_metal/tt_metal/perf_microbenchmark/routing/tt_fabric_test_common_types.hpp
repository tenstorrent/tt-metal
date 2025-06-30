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

// A map to hold various parametrization options parsed from the YAML.
using ParametrizationValues = std::variant<std::vector<std::string>, std::vector<uint32_t>>;
using ParametrizationOptionsMap = std::unordered_map<std::string, ParametrizationValues>;

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

struct TestFabricSetup {
    tt::tt_fabric::Topology topology;
    std::optional<RoutingType> routing_type;
};

struct HighLevelPatternConfig {
    std::string type;
    std::optional<uint32_t> iterations;
};

struct TestConfig {
    std::string name;
    TestFabricSetup fabric_setup;
    std::optional<std::string> on_missing_param_policy;
    std::optional<TrafficPatternConfig> defaults;
    std::optional<ParametrizationOptionsMap> parametrization_params;
    // A test can be defined by either a concrete list of senders or a high-level pattern.
    std::optional<std::vector<HighLevelPatternConfig>> patterns;
    std::vector<SenderConfig> senders;
    std::optional<std::string> bw_calc_func;
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
};

struct AllocatorPolicies {
    CoreAllocationConfig sender_config;
    CoreAllocationConfig receiver_config;
    std::optional<uint32_t> default_payload_chunk_size;

    AllocatorPolicies() {
        // Default sender policy: one sender per core to isolate performance.
        sender_config.policy = CoreAllocationPolicy::ExhaustFirst;
        sender_config.max_configs_per_core = detail::DEFAULT_MAX_SENDER_CONFIGS_PER_CORE;
        sender_config.pool_refill_size = detail::DEFAULT_SENDER_POOL_REFILL_SIZE;
        sender_config.initial_pool_size = 1;  // ExhaustFirst is equivalent to a pool size of 1.

        // Default receiver policy: reuse a core until it's full (shared receiver model).
        receiver_config.policy = CoreAllocationPolicy::ExhaustFirst;
        receiver_config.max_configs_per_core = detail::DEFAULT_MAX_RECEIVER_CONFIGS_PER_CORE;
        // No default pool sizes for receivers. The pool will be populated with all remaining cores
        // after senders have been allocated.
        default_payload_chunk_size = detail::DEFAULT_PAYLOAD_CHUNK_SIZE_BYTES;
    }
};

}  // namespace tt::tt_fabric::fabric_tests
