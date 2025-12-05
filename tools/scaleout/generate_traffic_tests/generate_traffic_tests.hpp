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
// Test Categories - users can enable/disable specific test types
// =============================================================================

/// Individual test categories that can be enabled/disabled
struct TestCategories {
    bool simple_unicast = true;   // Basic intra-mesh unicast (always recommended)
    bool inter_mesh = true;       // Mesh-to-mesh connectivity tests
    bool all_to_all = true;       // All devices send to all devices
    bool random_pairing = false;  // Random sender/receiver pairing
    bool all_to_one = false;      // Convergence test (many->one)
    bool flow_control = false;    // High-volume flow control stress
    bool sequential = false;      // Sequential all-to-all (slowest)
};

// =============================================================================
// Test Profile - preset configurations
// =============================================================================

/// Test profile determines default settings
/// - SANITY: Quick validation (<1 min), basic patterns only
/// - STRESS: Thorough testing (5-15 min), flow control, high packet counts
/// - BENCHMARK: Performance focused, varied packet sizes, consistent counts
enum class TestProfile { SANITY, STRESS, BENCHMARK };

// =============================================================================
// Traffic Test Configuration
// =============================================================================

struct TrafficTestConfig {
    // --- Profile (sets sensible defaults) ---
    TestProfile profile = TestProfile::SANITY;

    // --- Test selection ---
    TestCategories categories;  // Which test types to generate

    // --- MGD configuration ---
    bool generate_mgd = true;
    std::filesystem::path mgd_output_path;
    std::optional<std::filesystem::path> existing_mgd_path;

    // --- Packet configuration ---
    // If not set, uses profile defaults
    std::optional<std::vector<uint32_t>> packet_sizes;  // e.g., {1024, 2048, 4096}
    std::optional<uint32_t> num_packets;                // Packets per sender (single value for simplicity)

    // --- NoC types to test ---
    // If empty, uses profile defaults. Common types:
    // - "unicast_write" (basic, always works)
    // - "atomic_inc", "fused_atomic_inc" (stress testing)
    // - "unicast_scatter_write" (advanced)
    std::vector<std::string> noc_types;

    // --- Test behavior ---
    bool include_sync = true;  // Synchronize timing across devices

    // --- Platform skip ---
    std::vector<std::string> skip_platforms;  // e.g., {"GALAXY", "BLACKHOLE"}

    // --- Naming ---
    std::string test_name_prefix;
};

// =============================================================================
// Topology Information (extracted from cabling descriptor)
// =============================================================================

struct MeshTopologyInfo {
    size_t num_meshes{};
    std::vector<int> device_dims;  // [rows, cols] per mesh
    std::string node_type;
    std::string architecture;  // "WORMHOLE_B0" or "BLACKHOLE"

    // Mesh connectivity: mesh_id -> { connected_mesh_id -> channel_count }
    std::map<uint32_t, std::map<uint32_t, uint32_t>> mesh_connections;

    // Ordered unique mesh pairs for deterministic output
    std::vector<std::pair<uint32_t, uint32_t>> connected_pairs;

    // Computed stats
    size_t total_devices() const { return num_meshes * device_dims[0] * device_dims[1]; }
};

// =============================================================================
// Public API
// =============================================================================

/// Extract topology from cabling descriptor protobuf
[[nodiscard]] MeshTopologyInfo extract_topology_info(
    const proto::ClusterDescriptor& cluster_desc, bool verbose = false);

/// Extract topology from cabling descriptor file
[[nodiscard]] MeshTopologyInfo extract_topology_info(
    const std::filesystem::path& cabling_descriptor_path, bool verbose = false);

/// Generate traffic test YAML as string
[[nodiscard]] std::string generate_traffic_tests_yaml(
    const MeshTopologyInfo& topology, const std::filesystem::path& mgd_path, const TrafficTestConfig& config = {});

/// Generate traffic test YAML from cabling descriptor
[[nodiscard]] std::string generate_traffic_tests_yaml(
    const proto::ClusterDescriptor& cluster_desc, const TrafficTestConfig& config = {}, bool verbose = false);

/// Write YAML content to file (adds license header)
void write_traffic_tests_to_file(const std::string& yaml_content, const std::filesystem::path& output_path);

/// Convenience: generate and write in one call
void generate_traffic_tests(
    const std::filesystem::path& cabling_descriptor_path,
    const std::filesystem::path& output_path,
    const TrafficTestConfig& config = {},
    bool verbose = false);

// =============================================================================
// Profile Presets - get config with sensible defaults for each profile
// =============================================================================

/// Sanity: Quick validation, ~30s-1min
/// Tests: simple_unicast, inter_mesh, all_to_all
/// Packets: 100, Sizes: 1024, 2048
[[nodiscard]] TrafficTestConfig get_sanity_config();

/// Stress: Thorough testing, ~5-15min depending on cluster size
/// Tests: all categories enabled
/// Packets: 1000-10000, Sizes: 1024, 2048, 4096
[[nodiscard]] TrafficTestConfig get_stress_config();

/// Benchmark: Performance measurement, ~2-5min
/// Tests: simple_unicast, inter_mesh, all_to_all
/// Packets: 1000, Sizes: 512, 1024, 2048, 4096, 8192
[[nodiscard]] TrafficTestConfig get_benchmark_config();

// =============================================================================
// Utility
// =============================================================================

/// Apply profile defaults to config (fills in unset optional fields)
void apply_profile_defaults(TrafficTestConfig& config);

/// Estimate test duration based on config and topology
/// Returns rough estimate in seconds
[[nodiscard]] uint32_t estimate_test_duration_seconds(
    const MeshTopologyInfo& topology, const TrafficTestConfig& config);

}  // namespace tt::scaleout_tools
