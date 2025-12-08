// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <set>
#include <optional>
#include <cabling_generator/cabling_generator.hpp>
#include <vector>
#include <tuple>

namespace YAML {
    class Node;
}

namespace tt::scaleout_tools {

// An AsicId uniquely identifies an ASIC by (hostname, tray_id, asic_location)
using AsicId = std::tuple<std::string, uint32_t, uint32_t>;

// Check if all expected ASIC pairs have at least min_connections discovered connections
// Returns true if all pairs satisfy the minimum, false otherwise
// insufficient_connections is populated with pairs that don't meet the minimum
bool check_min_connection_count_satisfied(
    const std::set<PhysicalChannelConnection>& discovered_connections,
    const std::set<std::pair<PhysicalChannelEndpoint, PhysicalChannelEndpoint>>& generated_connections,
    uint32_t min_connections,
    std::vector<std::pair<std::pair<AsicId, AsicId>, uint32_t>>& insufficient_connections);

// Common utility function for validating FSD against discovered GSD
// Validates that the Factory System Descriptor (FSD) matches the Global System Descriptor (GSD)
// Parameters:
//   - fsd_filename: Path to the FSD protobuf text format file
//   - gsd_filename: Path to the GSD YAML file
//   - strict_validation: If true, checks that all connections match bidirectionally
//                        If false, only checks that GSD connections exist in FSD
//   - assert_on_connection_mismatch: If true, throws an error if there are connection mismatches
//                                    If false, missing connections are logged without an error being thrown
//   - min_connections: If specified, enables relaxed validation mode where missing connections
//                      are not reported as errors if each expected ASIC pair has at least this many
//                      discovered connections (checked at per-ASIC-pair granularity)
std::set<PhysicalChannelConnection> validate_fsd_against_gsd(
    const std::string& fsd_filename,
    const std::string& gsd_filename,
    bool strict_validation = true,
    bool assert_on_connection_mismatch = true,
    bool log_output = true,
    std::optional<uint32_t> min_connections = std::nullopt);

// In-memory overload for validating FSD against GSD without disk I/O
// Validates that the Factory System Descriptor (FSD) protobuf matches the Global System Descriptor (GSD) YAML node
// This overload operates on in-memory data structures, avoiding file I/O operations
// Parameters:
//   - fsd_proto: The FSD protobuf object (in-memory)
//   - gsd_yaml_node: The GSD YAML node (in-memory)
//   - strict_validation: If true, checks that all connections match bidirectionally
//                        If false, only checks that GSD connections exist in FSD
//   - assert_on_connection_mismatch: If true, throws an error if there are connection mismatches
//                                    If false, missing connections are logged without an error being thrown
//   - log_output: If true, logs validation results
//   - min_connections: If specified, enables relaxed validation mode where missing connections
//                      are not reported as errors if each expected ASIC pair has at least this many
//                      discovered connections (checked at per-ASIC-pair granularity)
// Returns: Set of physical channel connections that are missing or mismatched
std::set<PhysicalChannelConnection> validate_fsd_against_gsd(
    const fsd::proto::FactorySystemDescriptor& fsd_proto,
    const YAML::Node& gsd_yaml_node,
    bool strict_validation = true,
    bool assert_on_connection_mismatch = true,
    bool log_output = true,
    std::optional<uint32_t> min_connections = std::nullopt);

// Validate cabling descriptor against discovered system topology
std::set<PhysicalChannelConnection> validate_cabling_descriptor_against_gsd(
    const std::string& cabling_descriptor_path,
    const std::vector<std::string>& hostnames,
    const YAML::Node& gsd_yaml_node,
    bool strict_validation = true,
    bool assert_on_connection_mismatch = true,
    bool log_output = true);

// Generate cluster descriptor(s) from a factory system descriptor
// For single-host systems, a single YAML file is generated with the cluster configuration.
// For multi-host systems, one YAML file is generated per host
// along with a mapping file that maps each rank to its corresponding cluster descriptor file.
std::string generate_cluster_descriptor_from_fsd(
    const std::string& fsd_filename, const std::string& output_dir, const std::string& base_filename);

}  // namespace tt::scaleout_tools
