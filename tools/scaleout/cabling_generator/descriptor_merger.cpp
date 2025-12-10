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

    const std::string file_content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

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

    for (size_t i = 1; i < descriptors.size(); ++i) {
        const auto& source = descriptors[i];

        for (const auto& [name, tmpl] : source.graph_templates()) {
            if (!merged.graph_templates().contains(name)) {
                (*merged.mutable_graph_templates())[name] = tmpl;
            } else {
                merge_graph_template(
                    (*merged.mutable_graph_templates())[name],
                    tmpl,
                    name,
                    "descriptor[" + std::to_string(i) + "]",
                    validation_result);
            }
        }

        for (const auto& [name, desc] : source.node_descriptors()) {
            if (!merged.node_descriptors().contains(name)) {
                (*merged.mutable_node_descriptors())[name] = desc;
            }
        }
    }

    std::map<std::string, std::set<std::string>> child_to_descriptors;
    for (const auto& desc : descriptors) {
        const auto& root_template_name = desc.root_instance().template_name();
        const auto it = desc.graph_templates().find(root_template_name);
        if (it == desc.graph_templates().end()) {
            continue;
        }

        for (const auto& child : it->second.children()) {
            if (child.has_node_ref()) {
                child_to_descriptors[child.name()].insert(child.node_ref().node_descriptor());
            }
        }
    }

    for (const auto& [child_name, desc_names] : child_to_descriptors) {
        if (desc_names.size() <= 1) {
            continue;
        }

        auto it = desc_names.begin();
        const std::string base_name = *it++;
        auto base_desc = get_node_descriptor(base_name, merged);

        while (it != desc_names.end()) {
            const auto other_desc = get_node_descriptor(*it, merged);
            base_desc = merge_node_descriptor_connections(base_desc, other_desc);
            validation_result.add_warning("Merged connections from '" + base_name + "' and '" + *it + "'");
            ++it;
        }
        (*merged.mutable_node_descriptors())[base_name] = base_desc;
    }

    merge_root_instances(merged, descriptors, validation_result);

    return merged;
}

void DescriptorMerger::merge_root_instances(
    cabling_generator::proto::ClusterDescriptor& merged,
    const std::vector<cabling_generator::proto::ClusterDescriptor>& descriptors,
    MergeValidationResult& validation_result) {
    // Find the first descriptor with a root_instance
    const cabling_generator::proto::GraphInstance* base_root_instance = nullptr;
    size_t base_idx = 0;

    if (merged.has_root_instance()) {
        base_root_instance = &merged.root_instance();
    } else {
        // Find first descriptor with root_instance
        for (size_t i = 0; i < descriptors.size(); ++i) {
            if (descriptors[i].has_root_instance()) {
                *merged.mutable_root_instance() = descriptors[i].root_instance();
                base_root_instance = &merged.root_instance();
                base_idx = i + 1;  // Start merging from next descriptor
                break;
            }
        }

        if (!base_root_instance) {
            return;  // No descriptor has root_instance
        }
    }

    const std::string base_template_name = base_root_instance->template_name();

    for (size_t i = base_idx; i < descriptors.size(); ++i) {
        const auto& source = descriptors[i];

        if (!source.has_root_instance()) {
            continue;
        }

        if (source.root_instance().template_name() != base_template_name) {
            validation_result.add_warning(
                "root_instance template_name mismatch in descriptor[" + std::to_string(i) + "]: '" +
                source.root_instance().template_name() + "' vs '" + base_template_name + "'. Using first occurrence.");
            continue;
        }

        for (const auto& [child_name, child_mapping] : source.root_instance().child_mappings()) {
            if (merged.root_instance().child_mappings().contains(child_name)) {
                const auto& existing_mapping = merged.root_instance().child_mappings().at(child_name);

                if (existing_mapping.mapping_case() != child_mapping.mapping_case()) {
                    validation_result.add_error(
                        "Child '" + child_name + "' has conflicting mapping types in root_instance");
                    continue;
                }

                if (child_mapping.has_host_id()) {
                    if (existing_mapping.host_id() != child_mapping.host_id()) {
                        validation_result.add_error(
                            "Child '" + child_name +
                            "' mapped to different host_ids: " + std::to_string(existing_mapping.host_id()) + " vs " +
                            std::to_string(child_mapping.host_id()));
                    }
                }
            } else {
                (*merged.mutable_root_instance()->mutable_child_mappings())[child_name] = child_mapping;
            }
        }
    }
}

void DescriptorMerger::merge_graph_template(
    cabling_generator::proto::GraphTemplate& target_template,
    const cabling_generator::proto::GraphTemplate& source_template,
    const std::string& template_name,
    const std::string& source_file,
    MergeValidationResult& result) {
    std::set<std::string> existing_children;
    for (const auto& child : target_template.children()) {
        existing_children.insert(child.name());
    }

    for (const auto& child : source_template.children()) {
        if (!existing_children.contains(child.name())) {
            *target_template.add_children() = child;
            existing_children.insert(child.name());
        } else {
            const auto& existing_child = std::find_if(
                target_template.children().begin(), target_template.children().end(), [&child](const auto& c) {
                    return c.name() == child.name();
                });

            if (existing_child != target_template.children().end()) {
                if (existing_child->has_node_ref() && child.has_node_ref()) {
                    if (existing_child->node_ref().node_descriptor() != child.node_ref().node_descriptor()) {
                        result.add_warning(
                            "Child '" + child.name() + "' in template '" + template_name +
                            "' has different node_descriptor in " + source_file + " ('" +
                            child.node_ref().node_descriptor() + "' vs '" +
                            existing_child->node_ref().node_descriptor() + "'). Using first occurrence.");
                    }
                } else if (existing_child->has_graph_ref() && child.has_graph_ref()) {
                    if (existing_child->graph_ref().graph_template() != child.graph_ref().graph_template()) {
                        result.add_warning(
                            "Child '" + child.name() + "' in template '" + template_name +
                            "' has different graph_template in " + source_file + " ('" +
                            child.graph_ref().graph_template() + "' vs '" +
                            existing_child->graph_ref().graph_template() + "'). Using first occurrence.");
                    }
                }
            }
        }
    }

    merge_internal_connections(target_template, source_template, template_name, source_file, result);
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
    const auto map1 = build_connection_map(desc1);
    const auto map2 = build_connection_map(desc2);

    for (const auto& [endpoint, connected_to_1] : map1) {
        if (const auto it = map2.find(endpoint); it != map2.end() && !(connected_to_1 == it->second)) {
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

MergeValidationResult DescriptorMerger::validate_host_consistency(const std::vector<std::string>& descriptor_paths) {
    MergeValidationResult result;
    if (descriptor_paths.size() < 2) {
        return result;
    }

    std::optional<size_t> first_host_count;

    for (const auto& path : descriptor_paths) {
        const auto host_ids = extract_host_ids(load_descriptor(path));
        if (host_ids.empty()) {
            continue;
        }

        const size_t count = *host_ids.rbegin() + 1;
        if (!first_host_count) {
            first_host_count = count;
        } else if (count != *first_host_count) {
            result.add_warning(
                "Host count mismatch: " + std::to_string(*first_host_count) + " vs " + std::to_string(count) +
                " hosts");
            break;
        }
    }
    return result;
}

cabling_generator::proto::NodeDescriptor DescriptorMerger::get_node_descriptor(
    const std::string& node_descriptor_name, const cabling_generator::proto::ClusterDescriptor& descriptor) {
    if (const auto it = descriptor.node_descriptors().find(node_descriptor_name);
        it != descriptor.node_descriptors().end()) {
        return it->second;
    }
    const auto node_type = get_node_type_from_string(node_descriptor_name);
    return create_node_descriptor(node_type);
}

cabling_generator::proto::NodeDescriptor DescriptorMerger::merge_node_descriptor_connections(
    const cabling_generator::proto::NodeDescriptor& a, const cabling_generator::proto::NodeDescriptor& b) {
    cabling_generator::proto::NodeDescriptor merged = a;

    std::set<int32_t> existing_tray_ids;
    for (const auto& board : merged.boards().board()) {
        existing_tray_ids.insert(board.tray_id());
    }
    for (const auto& board : b.boards().board()) {
        if (!existing_tray_ids.contains(board.tray_id())) {
            *merged.mutable_boards()->add_board() = board;
        }
    }

    for (const auto& [port_type, b_connections] : b.port_type_connections()) {
        auto& merged_connections = (*merged.mutable_port_type_connections())[port_type];

        std::set<std::tuple<int32_t, int32_t, int32_t, int32_t>> existing;
        for (const auto& conn : merged_connections.connections()) {
            const auto key = std::make_tuple(
                conn.port_a().tray_id(), conn.port_a().port_id(), conn.port_b().tray_id(), conn.port_b().port_id());
            existing.insert(key);
            const auto rev_key = std::make_tuple(
                conn.port_b().tray_id(), conn.port_b().port_id(), conn.port_a().tray_id(), conn.port_a().port_id());
            existing.insert(rev_key);
        }

        for (const auto& conn : b_connections.connections()) {
            const auto key = std::make_tuple(
                conn.port_a().tray_id(), conn.port_a().port_id(), conn.port_b().tray_id(), conn.port_b().port_id());
            if (!existing.contains(key)) {
                *merged_connections.add_connections() = conn;
                existing.insert(key);
                const auto rev_key = std::make_tuple(
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
