// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>
#include <optional>

namespace tt::scaleout_tools::cabling_generator::proto {
class ClusterDescriptor;
class GraphTemplate;
class PortConnections;
class Connection;
class Port;
}  // namespace tt::scaleout_tools::cabling_generator::proto

namespace tt::scaleout_tools {

struct MergeValidationResult {
    bool success = true;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;

    void add_warning(const std::string& msg) { warnings.push_back(msg); }

    void add_error(const std::string& msg) {
        errors.push_back(msg);
        success = false;
    }

    void merge(const MergeValidationResult& other) {
        if (!other.success) {
            success = false;
        }
        warnings.insert(warnings.end(), other.warnings.begin(), other.warnings.end());
        errors.insert(errors.end(), other.errors.begin(), other.errors.end());
    }

    std::string format_messages() const;
};

struct ConnectionEndpoint {
    std::string graph_template_name;
    std::vector<std::string> path;
    uint32_t tray_id;
    uint32_t port_id;

    bool operator<(const ConnectionEndpoint& other) const;
    bool operator==(const ConnectionEndpoint& other) const;
    std::string to_string() const;
};

struct ConnectionPair {
    ConnectionEndpoint endpoint_a;
    ConnectionEndpoint endpoint_b;
    std::string port_type;

    bool operator<(const ConnectionPair& other) const;
    bool operator==(const ConnectionPair& other) const;
    std::string to_string() const;
};

struct MergeStatistics {
    size_t total_descriptors = 0;
    size_t total_graph_templates = 0;
    size_t total_node_descriptors = 0;
    size_t total_connections = 0;
    size_t duplicate_connections_removed = 0;
    std::set<uint32_t> host_ids_found;

    size_t expected_host_count() const { return host_ids_found.empty() ? 0 : (*host_ids_found.rbegin() + 1); }
};

// Utility class for merging multiple cabling descriptors into a single unified descriptor.
class DescriptorMerger {
public:
    // Merge descriptors from file paths. Throws on conflicts or parse errors.
    static cabling_generator::proto::ClusterDescriptor merge_descriptors(
        const std::vector<std::string>& descriptor_paths);

    // Merge already-loaded descriptors with validation output.
    static cabling_generator::proto::ClusterDescriptor merge_descriptors(
        const std::vector<cabling_generator::proto::ClusterDescriptor>& descriptors,
        MergeValidationResult& validation_result);

    // Find all .textproto files in a directory recursively.
    static std::vector<std::string> find_descriptor_files(const std::string& directory_path);

    // Validate structural consistency of a merged descriptor.
    static MergeValidationResult validate_merged_descriptor(
        const cabling_generator::proto::ClusterDescriptor& descriptor);

    // Check host configuration consistency across descriptors.
    static MergeValidationResult validate_host_consistency(const std::vector<std::string>& descriptor_paths);

    static bool is_directory(const std::string& path);

    static MergeStatistics get_merge_statistics(const cabling_generator::proto::ClusterDescriptor& descriptor);

private:
    static cabling_generator::proto::ClusterDescriptor load_descriptor(const std::string& file_path);

    static bool merge_graph_templates(
        cabling_generator::proto::ClusterDescriptor& target,
        const cabling_generator::proto::ClusterDescriptor& source,
        const std::string& source_file,
        MergeValidationResult& result);

    static bool merge_node_descriptors(
        cabling_generator::proto::ClusterDescriptor& target,
        const cabling_generator::proto::ClusterDescriptor& source,
        const std::string& source_file,
        MergeValidationResult& result);

    static void merge_internal_connections(
        cabling_generator::proto::GraphTemplate& target_template,
        const cabling_generator::proto::GraphTemplate& source_template,
        const std::string& template_name,
        const std::string& source_file,
        MergeValidationResult& result);

    using ConnectionMap = std::map<ConnectionEndpoint, ConnectionEndpoint>;
    static ConnectionMap build_connection_map(const cabling_generator::proto::ClusterDescriptor& descriptor);

    static void detect_connection_conflicts(
        const cabling_generator::proto::ClusterDescriptor& desc1,
        const std::string& file1,
        const cabling_generator::proto::ClusterDescriptor& desc2,
        const std::string& file2,
        MergeValidationResult& result);

    static std::set<uint32_t> extract_host_ids(const cabling_generator::proto::ClusterDescriptor& descriptor);

    static bool graph_templates_equal(
        const cabling_generator::proto::GraphTemplate& a, const cabling_generator::proto::GraphTemplate& b);

    static bool connections_equal(
        const cabling_generator::proto::Connection& a, const cabling_generator::proto::Connection& b);

    static ConnectionEndpoint port_to_endpoint(
        const cabling_generator::proto::Port& port, const std::string& graph_template_name);
};

}  // namespace tt::scaleout_tools
