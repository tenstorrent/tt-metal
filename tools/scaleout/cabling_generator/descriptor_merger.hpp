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
class ChildInstance;
class PortConnections;
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

    static std::vector<std::string> find_descriptor_files(const std::string& directory_path);

    static bool is_directory(const std::string& path);

private:
    static cabling_generator::proto::ClusterDescriptor merge_descriptors_impl(
        const std::vector<cabling_generator::proto::ClusterDescriptor>& descriptors,
        MergeValidationResult& validation_result);

    static MergeValidationResult validate_host_consistency(const std::vector<std::string>& descriptor_paths);

private:
    using ConnectionMap = std::map<ConnectionEndpoint, ConnectionEndpoint>;

    static void merge_internal_connections(
        cabling_generator::proto::GraphTemplate& target_template,
        const cabling_generator::proto::GraphTemplate& source_template,
        const std::string& template_name,
        const std::string& source_file,
        MergeValidationResult& result);

    static void normalize_torus_descriptors(
        cabling_generator::proto::ClusterDescriptor& target,
        const cabling_generator::proto::ClusterDescriptor& source,
        const std::string& template_name);

    static void add_torus_internal_connections(
        cabling_generator::proto::GraphTemplate& target_template,
        const std::string& target_desc,
        const std::string& source_desc,
        const std::string& child_name);

    static std::set<uint32_t> extract_host_ids(const cabling_generator::proto::ClusterDescriptor& descriptor);

    static ConnectionEndpoint port_to_endpoint(
        const cabling_generator::proto::Port& port, const std::string& graph_template_name);

    static void validate_structure_identity(
        const cabling_generator::proto::ClusterDescriptor& desc1,
        const std::string& file1,
        const cabling_generator::proto::ClusterDescriptor& desc2,
        const std::string& file2,
        MergeValidationResult& result);

    static void validate_node_descriptors_identity(
        const cabling_generator::proto::ClusterDescriptor& desc1,
        const std::string& file1,
        const cabling_generator::proto::ClusterDescriptor& desc2,
        const std::string& file2,
        MergeValidationResult& result);

    static void validate_graph_template_children_identity(
        const cabling_generator::proto::GraphTemplate& tmpl1,
        const cabling_generator::proto::GraphTemplate& tmpl2,
        const std::string& template_name,
        const std::string& file1,
        const std::string& file2,
        MergeValidationResult& result);

    static void validate_child_identity(
        const cabling_generator::proto::ChildInstance& child1,
        const cabling_generator::proto::ChildInstance& child2,
        const std::string& template_name,
        const std::string& file1,
        const std::string& file2,
        MergeValidationResult& result);

    static bool is_torus_compatible(const std::string& desc1, const std::string& desc2);

    static bool is_a_torus(const std::string& desc);
    static bool has_same_torus_architecture(const std::string& desc1, const std::string& desc2);
    static std::string extract_torus_architecture(const std::string& desc);

    static bool has_torus_variant(const std::string& desc, const std::string& variant);
};

}  // namespace tt::scaleout_tools
