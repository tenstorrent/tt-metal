// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "descriptor_merger.hpp"
#include "protobuf_utils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <google/protobuf/text_format.h>

#include "protobuf/cluster_config.pb.h"
#include "node/node.hpp"
#include "node/node_types.hpp"

namespace tt::scaleout_tools {

std::string MergeValidationResult::format_messages() const {
    std::ostringstream oss;
    if (!warnings.empty()) {
        oss << "Warnings (" << warnings.size() << "):\n";
        for (const auto& warning : warnings) {
            oss << "  - " << warning << "\n";
        }
    }
    if (!errors.empty()) {
        oss << "Errors (" << errors.size() << "):\n";
        for (const auto& error : errors) {
            oss << "  - " << error << "\n";
        }
    }
    return oss.str();
}

bool ConnectionEndpoint::operator<(const ConnectionEndpoint& other) const {
    if (graph_template_name != other.graph_template_name) {
        return graph_template_name < other.graph_template_name;
    }
    if (path != other.path) {
        return path < other.path;
    }
    if (tray_id != other.tray_id) {
        return tray_id < other.tray_id;
    }
    return port_id < other.port_id;
}

bool ConnectionEndpoint::operator==(const ConnectionEndpoint& other) const {
    return graph_template_name == other.graph_template_name && path == other.path && tray_id == other.tray_id &&
           port_id == other.port_id;
}

std::ostream& operator<<(std::ostream& os, const ConnectionEndpoint& endpoint) {
    os << endpoint.graph_template_name << ":[";
    bool first = true;
    for (const auto& path_element : endpoint.path) {
        if (!first) {
            os << "/";
        }
        os << path_element;
        first = false;
    }
    os << "]:tray" << endpoint.tray_id << ":port" << endpoint.port_id;
    return os;
}

bool ConnectionPair::operator<(const ConnectionPair& other) const {
    auto normalize = [](const ConnectionPair& p) -> std::pair<ConnectionEndpoint, ConnectionEndpoint> {
        return (p.endpoint_b < p.endpoint_a) ? std::make_pair(p.endpoint_b, p.endpoint_a)
                                             : std::make_pair(p.endpoint_a, p.endpoint_b);
    };
    auto [a1, a2] = normalize(*this);
    auto [b1, b2] = normalize(other);

    if (port_type != other.port_type) {
        return port_type < other.port_type;
    }
    if (!(a1 == b1)) {
        return a1 < b1;
    }
    return a2 < b2;
}

bool ConnectionPair::operator==(const ConnectionPair& other) const {
    bool same_order = (endpoint_a == other.endpoint_a && endpoint_b == other.endpoint_b);
    bool reverse_order = (endpoint_a == other.endpoint_b && endpoint_b == other.endpoint_a);
    return port_type == other.port_type && (same_order || reverse_order);
}

std::ostream& operator<<(std::ostream& os, const ConnectionPair& pair) {
    os << pair.port_type << ": " << pair.endpoint_a << " <-> " << pair.endpoint_b;
    return os;
}

bool DescriptorMerger::is_directory(const std::string& path) { return std::filesystem::is_directory(path); }

std::vector<std::string> DescriptorMerger::find_descriptor_files(const std::string& directory_path) {
    if (!std::filesystem::exists(directory_path)) {
        throw std::runtime_error("Directory does not exist: " + directory_path);
    }
    if (!std::filesystem::is_directory(directory_path)) {
        throw std::runtime_error("Path is not a directory: " + directory_path);
    }

    std::vector<std::string> files;
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".textproto") {
                files.push_back(entry.path().string());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        throw std::runtime_error("Error reading directory " + directory_path + ": " + e.what());
    }

    std::sort(files.begin(), files.end());

    if (files.empty()) {
        throw std::runtime_error("No .textproto files found in directory: " + directory_path);
    }
    return files;
}

cabling_generator::proto::ClusterDescriptor DescriptorMerger::merge_descriptors(
    const std::vector<std::string>& descriptor_paths) {
    if (descriptor_paths.empty()) {
        throw std::runtime_error("No descriptor paths provided for merging");
    }
    if (descriptor_paths.size() == 1) {
        return load_cluster_descriptor(descriptor_paths[0]);
    }

    std::vector<cabling_generator::proto::ClusterDescriptor> descriptors;
    descriptors.reserve(descriptor_paths.size());
    for (const auto& path : descriptor_paths) {
        descriptors.push_back(load_cluster_descriptor(path));
    }

    MergeValidationResult validation_result;
    validation_result.merge(validate_host_consistency(descriptor_paths));

    for (size_t i = 0; i < descriptor_paths.size(); ++i) {
        for (size_t j = i + 1; j < descriptor_paths.size(); ++j) {
            detect_connection_conflicts(
                descriptors[i], descriptor_paths[i], descriptors[j], descriptor_paths[j], validation_result);
        }
    }

    if (!validation_result.success) {
        throw std::runtime_error("Merge validation failed:\n" + validation_result.format_messages());
    }

    auto merged = merge_descriptors(descriptors, validation_result);

    for (const auto& warning : validation_result.warnings) {
        std::cerr << "WARNING: Descriptor merge: " << warning << std::endl;
    }

    // Check if merge_descriptors added any new errors
    if (!validation_result.success) {
        throw std::runtime_error("Merge failed:\n" + validation_result.format_messages());
    }
    return merged;
}

cabling_generator::proto::ClusterDescriptor DescriptorMerger::merge_descriptors(
    const std::vector<cabling_generator::proto::ClusterDescriptor>& descriptors,
    MergeValidationResult& validation_result) {
    if (descriptors.empty()) {
        validation_result.add_error("No descriptors provided for merging");
        return {};
    }
    if (descriptors.size() == 1) {
        return descriptors[0];
    }

    cabling_generator::proto::ClusterDescriptor merged = descriptors[0];

    for (size_t i = 1; i < descriptors.size(); ++i) {
        const auto& source = descriptors[i];
        const std::string source_identifier = "descriptor[" + std::to_string(i) + "]";

        // Handle graph_templates: merge internal_connections for existing templates
        for (const auto& [name, tmpl] : source.graph_templates()) {
            if (!merged.graph_templates().contains(name)) {
                // New template: copy entire template
                (*merged.mutable_graph_templates())[name] = tmpl;
            } else {
                // Existing template: only merge internal_connections
                // Assumes children and other attributes are identical
                merge_internal_connections(
                    (*merged.mutable_graph_templates())[name], tmpl, name, source_identifier, validation_result);
            }
        }

        // Copy node_descriptors if not already present (first occurrence wins)
        for (const auto& [name, desc] : source.node_descriptors()) {
            if (!merged.node_descriptors().contains(name)) {
                (*merged.mutable_node_descriptors())[name] = desc;
            }
        }
    }

    // Use root_instance from first descriptor that has one
    // Assumes root_instance structure is identical across files
    if (!merged.has_root_instance()) {
        for (const auto& desc : descriptors) {
            if (desc.has_root_instance()) {
                *merged.mutable_root_instance() = desc.root_instance();
                break;
            }
        }
    }

    return merged;
}

void DescriptorMerger::merge_internal_connections(
    cabling_generator::proto::GraphTemplate& target_template,
    const cabling_generator::proto::GraphTemplate& source_template,
    const std::string& template_name,
    const std::string& source_file,
    MergeValidationResult& result) {
    std::set<ConnectionPair> existing_connections;

    for (const auto& [port_type, connections] : target_template.internal_connections()) {
        for (const auto& conn : connections.connections()) {
            existing_connections.insert(
                {port_to_endpoint(conn.port_a(), template_name),
                 port_to_endpoint(conn.port_b(), template_name),
                 port_type});
        }
    }

    for (const auto& [port_type, source_connections] : source_template.internal_connections()) {
        auto* target_connections = &(*target_template.mutable_internal_connections())[port_type];

        for (const auto& conn : source_connections.connections()) {
            const ConnectionPair pair{
                port_to_endpoint(conn.port_a(), template_name),
                port_to_endpoint(conn.port_b(), template_name),
                port_type};

            if (existing_connections.contains(pair)) {
                std::ostringstream oss;
                oss << "Duplicate connection in template '" << template_name << "' from " << source_file << ": " << pair
                    << ". Using first occurrence.";
                result.add_warning(oss.str());
            } else {
                *target_connections->add_connections() = conn;
                existing_connections.insert(pair);
            }
        }
    }
}

DescriptorMerger::ConnectionMap DescriptorMerger::build_connection_map(
    const cabling_generator::proto::ClusterDescriptor& descriptor) {
    ConnectionMap map;
    for (const auto& [template_name, graph_template] : descriptor.graph_templates()) {
        for (const auto& [port_type, connections] : graph_template.internal_connections()) {
            for (const auto& conn : connections.connections()) {
                auto endpoint_a = port_to_endpoint(conn.port_a(), template_name);
                auto endpoint_b = port_to_endpoint(conn.port_b(), template_name);
                map[endpoint_a] = endpoint_b;
                map[endpoint_b] = endpoint_a;
            }
        }
    }
    return map;
}

void DescriptorMerger::detect_connection_conflicts(
    const cabling_generator::proto::ClusterDescriptor& desc1,
    const std::string& file1,
    const cabling_generator::proto::ClusterDescriptor& desc2,
    const std::string& file2,
    MergeValidationResult& result) {
    const auto map1 = build_connection_map(desc1);
    const auto map2 = build_connection_map(desc2);

    for (const auto& [endpoint, connected_to_1] : map1) {
        if (const auto it = map2.find(endpoint); it != map2.end() && !(connected_to_1 == it->second)) {
            std::ostringstream oss;
            oss << "Connection conflict: " << endpoint << " connects to " << connected_to_1 << " in " << file1
                << " but connects to " << it->second << " in " << file2;
            result.add_error(oss.str());
        }
    }
}

std::set<uint32_t> DescriptorMerger::extract_host_ids(const cabling_generator::proto::ClusterDescriptor& descriptor) {
    std::set<uint32_t> host_ids;

    const auto extract_from_instance = [&](const auto& instance, auto& self) -> void {
        for (const auto& [child_name, mapping] : instance.child_mappings()) {
            if (mapping.has_host_id()) {
                host_ids.insert(mapping.host_id());
            } else if (mapping.has_sub_instance()) {
                self(mapping.sub_instance(), self);
            }
        }
    };

    if (descriptor.has_root_instance()) {
        extract_from_instance(descriptor.root_instance(), extract_from_instance);
    }
    return host_ids;
}

MergeValidationResult DescriptorMerger::validate_host_consistency(const std::vector<std::string>& descriptor_paths) {
    MergeValidationResult result;
    if (descriptor_paths.size() < 2) {
        return result;
    }

    std::optional<size_t> first_host_count;

    for (const auto& path : descriptor_paths) {
        const auto host_ids = extract_host_ids(load_cluster_descriptor(path));
        if (host_ids.empty()) {
            continue;
        }

        // Calculate host count from highest host_id (assuming 0-based indexing)
        const size_t count = *host_ids.rbegin() + 1;
        if (!first_host_count) {
            first_host_count = count;
        } else if (count != *first_host_count) {
            result.add_warning(
                "Host count mismatch between descriptors: " + std::to_string(*first_host_count) + " vs " +
                std::to_string(count) + " hosts (file: " + path + ")");
            break;
        }
    }
    return result;
}

ConnectionEndpoint DescriptorMerger::port_to_endpoint(
    const cabling_generator::proto::Port& port, const std::string& graph_template_name) {
    ConnectionEndpoint endpoint{graph_template_name, {}, port.tray_id(), port.port_id()};
    endpoint.path.reserve(port.path_size());
    for (const auto& path_element : port.path()) {
        endpoint.path.push_back(path_element);
    }
    return endpoint;
}

}  // namespace tt::scaleout_tools
