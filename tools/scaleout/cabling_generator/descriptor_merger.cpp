// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "descriptor_merger.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <google/protobuf/text_format.h>
#include <google/protobuf/util/message_differencer.h>

#include "protobuf/cluster_config.pb.h"
#include "node/node.hpp"
#include "node/node_types.hpp"

namespace tt::scaleout_tools {

namespace {

template <typename Descriptor>
Descriptor load_descriptor_from_textproto(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    Descriptor descriptor;
    if (!google::protobuf::TextFormat::ParseFromString(file_content, &descriptor)) {
        throw std::runtime_error("Failed to parse textproto file: " + file_path);
    }
    return descriptor;
}

}  // namespace

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

std::string ConnectionEndpoint::to_string() const {
    std::ostringstream oss;
    oss << graph_template_name << ":[";
    for (size_t i = 0; i < path.size(); ++i) {
        if (i > 0) {
            oss << "/";
        }
        oss << path[i];
    }
    oss << "]:tray" << tray_id << ":port" << port_id;
    return oss.str();
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

std::string ConnectionPair::to_string() const {
    return port_type + ": " + endpoint_a.to_string() + " <-> " + endpoint_b.to_string();
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
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".textproto") {
            files.push_back(entry.path().string());
        }
    }

    std::sort(files.begin(), files.end());

    if (files.empty()) {
        throw std::runtime_error("No .textproto files found in directory: " + directory_path);
    }
    return files;
}

cabling_generator::proto::ClusterDescriptor DescriptorMerger::load_descriptor(const std::string& file_path) {
    return load_descriptor_from_textproto<cabling_generator::proto::ClusterDescriptor>(file_path);
}

cabling_generator::proto::ClusterDescriptor DescriptorMerger::merge_descriptors(
    const std::vector<std::string>& descriptor_paths) {
    if (descriptor_paths.empty()) {
        throw std::runtime_error("No descriptor paths provided for merging");
    }
    if (descriptor_paths.size() == 1) {
        return load_descriptor(descriptor_paths[0]);
    }

    std::vector<cabling_generator::proto::ClusterDescriptor> descriptors;
    descriptors.reserve(descriptor_paths.size());
    for (const auto& path : descriptor_paths) {
        descriptors.push_back(load_descriptor(path));
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

    // 1. Union all graph_templates and inline node_descriptors
    for (size_t i = 1; i < descriptors.size(); ++i) {
        for (const auto& [name, tmpl] : descriptors[i].graph_templates()) {
            if (merged.graph_templates().find(name) == merged.graph_templates().end()) {
                (*merged.mutable_graph_templates())[name] = tmpl;
            } else {
                // Same name - merge internal_connections
                merge_internal_connections(
                    (*merged.mutable_graph_templates())[name],
                    tmpl,
                    name,
                    "descriptor[" + std::to_string(i) + "]",
                    validation_result);
            }
        }
        for (const auto& [name, desc] : descriptors[i].node_descriptors()) {
            if (merged.node_descriptors().find(name) == merged.node_descriptors().end()) {
                (*merged.mutable_node_descriptors())[name] = desc;
            }
        }
    }

    // 2. Collect all node_descriptors used across all root templates
    std::map<std::string, std::set<std::string>> child_to_descriptors;  // child_name -> set of node_descriptors
    for (const auto& desc : descriptors) {
        const auto& root_template_name = desc.root_instance().template_name();
        auto it = desc.graph_templates().find(root_template_name);
        if (it == desc.graph_templates().end()) {
            continue;
        }

        for (const auto& child : it->second.children()) {
            if (child.has_node_ref()) {
                child_to_descriptors[child.name()].insert(child.node_ref().node_descriptor());
            }
        }
    }

    // 3. Merge node_descriptor connections where multiple types exist for same child
    for (const auto& [child_name, desc_names] : child_to_descriptors) {
        if (desc_names.size() <= 1) {
            continue;
        }

        // Merge all into the first one
        auto it = desc_names.begin();
        std::string base_name = *it++;
        auto base_desc = get_node_descriptor(base_name, merged);

        while (it != desc_names.end()) {
            auto other_desc = get_node_descriptor(*it, merged);
            base_desc = merge_node_descriptor_connections(base_desc, other_desc);
            validation_result.add_warning("Merged connections from '" + base_name + "' and '" + *it + "'");
            ++it;
        }
        (*merged.mutable_node_descriptors())[base_name] = base_desc;
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
            ConnectionPair pair;
            pair.port_type = port_type;
            pair.endpoint_a = port_to_endpoint(conn.port_a(), template_name);
            pair.endpoint_b = port_to_endpoint(conn.port_b(), template_name);
            existing_connections.insert(pair);
        }
    }

    for (const auto& [port_type, source_connections] : source_template.internal_connections()) {
        auto* target_connections = &(*target_template.mutable_internal_connections())[port_type];

        for (const auto& conn : source_connections.connections()) {
            ConnectionPair pair;
            pair.port_type = port_type;
            pair.endpoint_a = port_to_endpoint(conn.port_a(), template_name);
            pair.endpoint_b = port_to_endpoint(conn.port_b(), template_name);

            if (existing_connections.count(pair) > 0) {
                result.add_warning(
                    "Duplicate connection in template '" + template_name + "' from " + source_file + ": " +
                    pair.to_string() + ". Using first occurrence.");
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
    auto map1 = build_connection_map(desc1);
    auto map2 = build_connection_map(desc2);

    for (const auto& [endpoint, connected_to_1] : map1) {
        auto it = map2.find(endpoint);
        if (it != map2.end() && !(connected_to_1 == it->second)) {
            result.add_error(
                "Connection conflict: " + endpoint.to_string() + " connects to " + connected_to_1.to_string() + " in " +
                file1 + " but connects to " + it->second.to_string() + " in " + file2);
        }
    }
}

std::set<uint32_t> DescriptorMerger::extract_host_ids(const cabling_generator::proto::ClusterDescriptor& descriptor) {
    std::set<uint32_t> host_ids;

    std::function<void(const cabling_generator::proto::GraphInstance&)> extract_from_instance;
    extract_from_instance = [&](const cabling_generator::proto::GraphInstance& instance) {
        for (const auto& [child_name, mapping] : instance.child_mappings()) {
            if (mapping.has_host_id()) {
                host_ids.insert(mapping.host_id());
            } else if (mapping.has_sub_instance()) {
                extract_from_instance(mapping.sub_instance());
            }
        }
    };

    if (descriptor.has_root_instance()) {
        extract_from_instance(descriptor.root_instance());
    }
    return host_ids;
}

MergeValidationResult DescriptorMerger::validate_merged_descriptor(
    const cabling_generator::proto::ClusterDescriptor& descriptor) {
    MergeValidationResult result;

    std::function<void(const cabling_generator::proto::GraphInstance&, const std::string&)> validate_instance;
    validate_instance = [&](const cabling_generator::proto::GraphInstance& instance, const std::string& path) {
        const auto& template_name = instance.template_name();

        if (descriptor.graph_templates().find(template_name) == descriptor.graph_templates().end()) {
            result.add_error("Graph template '" + template_name + "' referenced at " + path + " not found");
            return;
        }

        const auto& graph_template = descriptor.graph_templates().at(template_name);

        for (const auto& child : graph_template.children()) {
            const auto& child_name = child.name();
            std::string child_path = path.empty() ? child_name : path + "/" + child_name;

            if (instance.child_mappings().find(child_name) == instance.child_mappings().end()) {
                result.add_error("Missing child mapping for '" + child_name + "' in template '" + template_name + "'");
                continue;
            }

            const auto& mapping = instance.child_mappings().at(child_name);

            if (child.has_graph_ref()) {
                if (!mapping.has_sub_instance()) {
                    result.add_error("Graph child '" + child_name + "' requires sub_instance mapping, not host_id");
                } else {
                    validate_instance(mapping.sub_instance(), child_path);
                }
            } else if (child.has_node_ref() && !mapping.has_host_id()) {
                result.add_error("Node child '" + child_name + "' requires host_id mapping, not sub_instance");
            }
        }
    };

    if (descriptor.has_root_instance()) {
        validate_instance(descriptor.root_instance(), "");
    }

    auto host_ids = extract_host_ids(descriptor);
    if (!host_ids.empty()) {
        uint32_t max_host_id = *host_ids.rbegin();
        for (uint32_t i = 0; i <= max_host_id; ++i) {
            if (host_ids.find(i) == host_ids.end()) {
                result.add_warning(
                    "Host ID " + std::to_string(i) + " is missing from sequence 0.." + std::to_string(max_host_id));
            }
        }
    }
    return result;
}

MergeValidationResult DescriptorMerger::validate_host_consistency(const std::vector<std::string>& descriptor_paths) {
    MergeValidationResult result;
    if (descriptor_paths.size() < 2) {
        return result;
    }

    // Get host counts from descriptors that have root_instances
    std::optional<size_t> first_host_count;
    std::string first_path;

    for (const auto& path : descriptor_paths) {
        auto host_ids = extract_host_ids(load_descriptor(path));
        if (host_ids.empty()) {
            continue;
        }

        size_t count = *host_ids.rbegin() + 1;
        if (!first_host_count) {
            first_host_count = count;
            first_path = path;
        } else if (count != *first_host_count) {
            result.add_warning(
                "Host count mismatch: " + std::to_string(*first_host_count) + " vs " + std::to_string(count) +
                " hosts");
            break;
        }
    }
    return result;
}

MergeStatistics DescriptorMerger::get_merge_statistics(const cabling_generator::proto::ClusterDescriptor& descriptor) {
    MergeStatistics stats;
    stats.total_graph_templates = descriptor.graph_templates().size();
    stats.total_node_descriptors = descriptor.node_descriptors().size();

    for (const auto& [name, graph_template] : descriptor.graph_templates()) {
        for (const auto& [port_type, connections] : graph_template.internal_connections()) {
            stats.total_connections += connections.connections().size();
        }
    }

    stats.host_ids_found = extract_host_ids(descriptor);
    return stats;
}

cabling_generator::proto::NodeDescriptor DescriptorMerger::get_node_descriptor(
    const std::string& node_descriptor_name, const cabling_generator::proto::ClusterDescriptor& descriptor) {
    // First check inline descriptors
    auto it = descriptor.node_descriptors().find(node_descriptor_name);
    if (it != descriptor.node_descriptors().end()) {
        return it->second;
    }
    // Fall back to creating from node type
    auto node_type = get_node_type_from_string(node_descriptor_name);
    return create_node_descriptor(node_type);
}

cabling_generator::proto::NodeDescriptor DescriptorMerger::merge_node_descriptor_connections(
    const cabling_generator::proto::NodeDescriptor& a, const cabling_generator::proto::NodeDescriptor& b) {
    cabling_generator::proto::NodeDescriptor merged = a;

    // Merge boards - use boards from 'a' as base, add any missing from 'b'
    std::set<int32_t> existing_tray_ids;
    for (const auto& board : merged.boards().board()) {
        existing_tray_ids.insert(board.tray_id());
    }
    for (const auto& board : b.boards().board()) {
        if (existing_tray_ids.find(board.tray_id()) == existing_tray_ids.end()) {
            *merged.mutable_boards()->add_board() = board;
        }
    }

    // Merge port_type_connections - combine connections from both
    for (const auto& [port_type, b_connections] : b.port_type_connections()) {
        auto& merged_connections = (*merged.mutable_port_type_connections())[port_type];

        // Build set of existing connections for deduplication
        std::set<std::tuple<int32_t, int32_t, int32_t, int32_t>> existing;
        for (const auto& conn : merged_connections.connections()) {
            auto key = std::make_tuple(
                conn.port_a().tray_id(), conn.port_a().port_id(), conn.port_b().tray_id(), conn.port_b().port_id());
            existing.insert(key);
            // Also insert reverse
            auto rev_key = std::make_tuple(
                conn.port_b().tray_id(), conn.port_b().port_id(), conn.port_a().tray_id(), conn.port_a().port_id());
            existing.insert(rev_key);
        }

        // Add connections from 'b' that don't already exist
        for (const auto& conn : b_connections.connections()) {
            auto key = std::make_tuple(
                conn.port_a().tray_id(), conn.port_a().port_id(), conn.port_b().tray_id(), conn.port_b().port_id());
            if (existing.find(key) == existing.end()) {
                *merged_connections.add_connections() = conn;
                existing.insert(key);
                auto rev_key = std::make_tuple(
                    conn.port_b().tray_id(), conn.port_b().port_id(), conn.port_a().tray_id(), conn.port_a().port_id());
                existing.insert(rev_key);
            }
        }
    }

    return merged;
}

ConnectionEndpoint DescriptorMerger::port_to_endpoint(
    const cabling_generator::proto::Port& port, const std::string& graph_template_name) {
    ConnectionEndpoint endpoint;
    endpoint.graph_template_name = graph_template_name;
    endpoint.tray_id = port.tray_id();
    endpoint.port_id = port.port_id();
    for (const auto& path_element : port.path()) {
        endpoint.path.push_back(path_element);
    }
    return endpoint;
}

}  // namespace tt::scaleout_tools
