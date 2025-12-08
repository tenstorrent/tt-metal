// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <cabling_generator/cabling_generator.hpp>
#include "protobuf/cluster_config.pb.h"

namespace tt::scaleout_tools {

namespace proto = cabling_generator::proto;

// =============================================================================
// Test Profile Configuration
// =============================================================================

/// Test profile determines the complexity and coverage of generated tests
/// Tests are ordered from easiest (top) to hardest (bottom) in the output YAML
enum class TestProfile {
    SANITY,     // Quick functional validation - simple patterns, low packet counts
    STRESS,     // High-volume stress testing - flow control, high packet counts
    BENCHMARK,  // Performance measurement - bandwidth/latency focused
    COVERAGE    // Full coverage - all patterns, all mesh pairs, all noc types
};

/// NoC send types to include in parametrization
enum class NocSendTypeSet {
    BASIC,     // unicast_write only
    STANDARD,  // unicast_write, atomic_inc, fused_atomic_inc
    FULL       // All types including unicast_scatter_write
};

/// Fabric type patterns to include
enum class FabricTypeSet {
    UNICAST_ONLY,  // Only unicast patterns
    ALL            // Unicast and multicast
};

// =============================================================================
// Traffic Test Configuration
// =============================================================================

struct TrafficTestConfig {
    TestProfile profile = TestProfile::SANITY;

    // MGD configuration
    bool generate_mgd = true;  // Auto-generate MGD file alongside tests
    std::filesystem::path mgd_output_path;
    std::optional<std::filesystem::path> existing_mgd_path;  // Use existing MGD instead of generating

    // Test content configuration
    NocSendTypeSet noc_types = NocSendTypeSet::STANDARD;
    FabricTypeSet fabric_types = FabricTypeSet::UNICAST_ONLY;
    bool include_flow_control = false;
    bool include_sync = true;  // Enable sync for timing consistency

    // Packet configuration (defaults vary by profile)
    std::optional<std::vector<uint32_t>> packet_sizes;
    std::optional<std::vector<uint32_t>> packet_counts;
    std::optional<uint32_t> top_level_iterations;

    // Platform skip configuration
    std::vector<std::string> skip_platforms;  // e.g., ["GALAXY", "BLACKHOLE_GALAXY"]

    // Test naming
    std::string test_name_prefix;  // Optional prefix for generated test names
};

// =============================================================================
// Topology Information (extracted from cabling descriptor)
// =============================================================================

struct MeshTopologyInfo {
    size_t num_meshes{};
    std::vector<int> device_dims;  // [rows, cols] per mesh (from node type)
    std::string node_type;
    std::string architecture;  // "WORMHOLE_B0" or "BLACKHOLE"

    // Mesh connectivity graph: mesh_id -> { connected_mesh_id -> channel_count }
    std::map<uint32_t, std::map<uint32_t, uint32_t>> mesh_connections;

    // Ordered list of unique mesh pairs (for deterministic test generation)
    std::vector<std::pair<uint32_t, uint32_t>> connected_pairs;
};

// =============================================================================
// Public API
// =============================================================================

/// Extract topology information from a cabling descriptor
[[nodiscard]] MeshTopologyInfo extract_topology_info(const proto::ClusterDescriptor& cluster_desc, bool verbose = false);

/// Extract topology information from a cabling descriptor file
[[nodiscard]] MeshTopologyInfo extract_topology_info(
    const std::filesystem::path& cabling_descriptor_path, bool verbose = false);

/// Generate traffic test YAML content as a string
[[nodiscard]] std::string generate_traffic_tests_yaml(
    const proto::ClusterDescriptor& cluster_desc,
    const TrafficTestConfig& config = {},
    bool verbose = false);

/// Generate traffic test YAML content from topology info
[[nodiscard]] std::string generate_traffic_tests_yaml(
    const MeshTopologyInfo& topology, const std::filesystem::path& mgd_path, const TrafficTestConfig& config = {});

/// Write traffic test YAML to file
void write_traffic_tests_to_file(const std::string& yaml_content, const std::filesystem::path& output_path);

/// Convenience function: generate and write traffic tests in one call
/// Also generates MGD if config.generate_mgd is true
void generate_traffic_tests(
    const std::filesystem::path& cabling_descriptor_path,
    const std::filesystem::path& output_path,
    const TrafficTestConfig& config = {},
    bool verbose = false);

// =============================================================================
// Profile Presets
// =============================================================================

/// Get default configuration for sanity testing (quick validation)
[[nodiscard]] TrafficTestConfig get_sanity_config();

/// Get default configuration for stress testing
[[nodiscard]] TrafficTestConfig get_stress_config();

/// Get default configuration for benchmark testing
[[nodiscard]] TrafficTestConfig get_benchmark_config();

/// Get default configuration for full coverage testing
[[nodiscard]] TrafficTestConfig get_coverage_config();

}  // namespace tt::scaleout_tools
