// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace tt::scaleout_tools::cabling_generator::proto {
class ClusterDescriptor;
class GraphTemplate;
class NodeDescriptor;
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
        success = success && other.success;
        warnings.insert(warnings.end(), other.warnings.begin(), other.warnings.end());
        errors.insert(errors.end(), other.errors.begin(), other.errors.end());
    }

    std::string format_messages() const;
};

struct ConnectionEndpoint {
    std::string graph_template_name;
    std::vector<std::string> path;
    uint32_t tray_id = 0;
    uint32_t port_id = 0;

    bool operator<(const ConnectionEndpoint& other) const;
    bool operator==(const ConnectionEndpoint& other) const;
};

std::ostream& operator<<(std::ostream& os, const ConnectionEndpoint& endpoint);

struct ConnectionPair {
    ConnectionEndpoint endpoint_a;
    ConnectionEndpoint endpoint_b;
    std::string port_type;

    bool operator<(const ConnectionPair& other) const;
    bool operator==(const ConnectionPair& other) const;
};

std::ostream& operator<<(std::ostream& os, const ConnectionPair& pair);

class DescriptorMerger {
public:
    static cabling_generator::proto::ClusterDescriptor merge_descriptors(
        const std::vector<std::string>& descriptor_paths);

    static cabling_generator::proto::ClusterDescriptor merge_descriptors(
        const std::vector<cabling_generator::proto::ClusterDescriptor>& descriptors,
        MergeValidationResult& validation_result);

    static std::vector<std::string> find_descriptor_files(const std::string& directory_path);

    static MergeValidationResult validate_host_consistency(const std::vector<std::string>& descriptor_paths);

    static bool is_directory(const std::string& path);

private:
    using ConnectionMap = std::map<ConnectionEndpoint, ConnectionEndpoint>;

    static void merge_internal_connections(
        cabling_generator::proto::GraphTemplate& target_template,
        const cabling_generator::proto::GraphTemplate& source_template,
        const std::string& template_name,
        const std::string& source_file,
        MergeValidationResult& result);

    static ConnectionMap build_connection_map(const cabling_generator::proto::ClusterDescriptor& descriptor);

    static void detect_connection_conflicts(
        const cabling_generator::proto::ClusterDescriptor& desc1,
        const std::string& file1,
        const cabling_generator::proto::ClusterDescriptor& desc2,
        const std::string& file2,
        MergeValidationResult& result);

    static std::set<uint32_t> extract_host_ids(const cabling_generator::proto::ClusterDescriptor& descriptor);

    static ConnectionEndpoint port_to_endpoint(
        const cabling_generator::proto::Port& port, const std::string& graph_template_name);
};

}  // namespace tt::scaleout_tools
