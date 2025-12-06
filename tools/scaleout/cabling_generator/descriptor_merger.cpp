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

    for (size_t i = 1; i < descriptors.size(); ++i) {
        std::string source_file = "descriptor[" + std::to_string(i) + "]";
        merge_graph_templates(merged, descriptors[i], source_file, validation_result);
        merge_node_descriptors(merged, descriptors[i], source_file, validation_result);
    }

    const auto& base_root = descriptors[0].root_instance();
    for (size_t i = 1; i < descriptors.size(); ++i) {
        const auto& other_root = descriptors[i].root_instance();
        if (!base_root.template_name().empty() && !other_root.template_name().empty() &&
            base_root.template_name() != other_root.template_name()) {
            validation_result.add_warning(
                "Multiple root_instance templates found. Using '" + base_root.template_name() +
                "' from first descriptor. Found '" + other_root.template_name() + "' in descriptor[" +
                std::to_string(i) + "]");
        }
    }
    return merged;
}

bool DescriptorMerger::merge_graph_templates(
    cabling_generator::proto::ClusterDescriptor& target,
    const cabling_generator::proto::ClusterDescriptor& source,
    const std::string& source_file,
    MergeValidationResult& result) {
    bool success = true;

    for (const auto& [name, source_template] : source.graph_templates()) {
        auto it = target.graph_templates().find(name);
        if (it == target.graph_templates().end()) {
            (*target.mutable_graph_templates())[name] = source_template;
        } else if (graph_templates_equal(it->second, source_template)) {
            merge_internal_connections(
                (*target.mutable_graph_templates())[name], source_template, name, source_file, result);
        } else {
            result.add_error(
                "Conflicting graph template '" + name + "' in " + source_file +
                ". Template already exists with different structure.");
            success = false;
        }
    }
    return success;
}

bool DescriptorMerger::merge_node_descriptors(
    cabling_generator::proto::ClusterDescriptor& target,
    const cabling_generator::proto::ClusterDescriptor& source,
    const std::string& source_file,
    MergeValidationResult& result) {
    bool success = true;

    for (const auto& [name, source_descriptor] : source.node_descriptors()) {
        auto it = target.node_descriptors().find(name);
        if (it == target.node_descriptors().end()) {
            (*target.mutable_node_descriptors())[name] = source_descriptor;
        } else {
            google::protobuf::util::MessageDifferencer differencer;
            if (!differencer.Compare(it->second, source_descriptor)) {
                result.add_error(
                    "Conflicting node descriptor '" + name + "' in " + source_file +
                    ". Node descriptor already exists with different content.");
                success = false;
            }
        }
    }
    return success;
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
    if (descriptor_paths.empty()) {
        return result;
    }

    std::vector<std::pair<std::string, std::set<uint32_t>>> descriptor_host_ids;
    descriptor_host_ids.reserve(descriptor_paths.size());

    for (const auto& path : descriptor_paths) {
        try {
            auto descriptor = load_descriptor(path);
            descriptor_host_ids.emplace_back(path, extract_host_ids(descriptor));
        } catch (const std::exception& e) {
            result.add_error("Failed to load descriptor '" + path + "': " + e.what());
            return result;
        }
    }

    std::vector<std::pair<std::string, std::set<uint32_t>>> descriptors_with_hosts;
    for (const auto& [path, host_ids] : descriptor_host_ids) {
        if (!host_ids.empty()) {
            descriptors_with_hosts.emplace_back(path, host_ids);
        }
    }

    if (descriptors_with_hosts.empty()) {
        return result;
    }

    const auto& [first_path, first_host_ids] = descriptors_with_hosts[0];
    size_t first_host_count = first_host_ids.empty() ? 0 : (*first_host_ids.rbegin() + 1);

    for (size_t i = 1; i < descriptors_with_hosts.size(); ++i) {
        const auto& [path, host_ids] = descriptors_with_hosts[i];
        size_t host_count = host_ids.empty() ? 0 : (*host_ids.rbegin() + 1);

        if (host_count != first_host_count) {
            result.add_warning(
                "Host count mismatch: '" + first_path + "' has " + std::to_string(first_host_count) + " hosts, but '" +
                path + "' has " + std::to_string(host_count) + " hosts");
        }

        if (first_host_ids != host_ids) {
            std::set<uint32_t> only_in_first, only_in_second;
            std::set_difference(
                first_host_ids.begin(),
                first_host_ids.end(),
                host_ids.begin(),
                host_ids.end(),
                std::inserter(only_in_first, only_in_first.begin()));
            std::set_difference(
                host_ids.begin(),
                host_ids.end(),
                first_host_ids.begin(),
                first_host_ids.end(),
                std::inserter(only_in_second, only_in_second.begin()));

            if (!only_in_first.empty() || !only_in_second.empty()) {
                std::ostringstream msg;
                msg << "Host ID mismatch between '" << first_path << "' and '" << path << "': ";
                if (!only_in_first.empty()) {
                    msg << "IDs only in first: {";
                    bool first = true;
                    for (auto id : only_in_first) {
                        if (!first) {
                            msg << ", ";
                        }
                        msg << id;
                        first = false;
                    }
                    msg << "} ";
                }
                if (!only_in_second.empty()) {
                    msg << "IDs only in second: {";
                    bool first = true;
                    for (auto id : only_in_second) {
                        if (!first) {
                            msg << ", ";
                        }
                        msg << id;
                        first = false;
                    }
                    msg << "}";
                }
                result.add_warning(msg.str());
            }
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

bool DescriptorMerger::graph_templates_equal(
    const cabling_generator::proto::GraphTemplate& a, const cabling_generator::proto::GraphTemplate& b) {
    if (a.children().size() != b.children().size()) {
        return false;
    }

    std::map<std::string, const cabling_generator::proto::ChildInstance*> a_children;
    for (const auto& child : a.children()) {
        a_children[child.name()] = &child;
    }

    for (const auto& b_child : b.children()) {
        auto it = a_children.find(b_child.name());
        if (it == a_children.end()) {
            return false;
        }

        const auto& a_child = *it->second;
        if (a_child.has_node_ref() != b_child.has_node_ref() || a_child.has_graph_ref() != b_child.has_graph_ref()) {
            return false;
        }
        if (a_child.has_node_ref() && a_child.node_ref().node_descriptor() != b_child.node_ref().node_descriptor()) {
            return false;
        }
        if (a_child.has_graph_ref() && a_child.graph_ref().graph_template() != b_child.graph_ref().graph_template()) {
            return false;
        }
    }
    return true;
}

bool DescriptorMerger::connections_equal(
    const cabling_generator::proto::Connection& a, const cabling_generator::proto::Connection& b) {
    auto ports_equal = [](const cabling_generator::proto::Port& p1, const cabling_generator::proto::Port& p2) {
        if (p1.path().size() != p2.path().size()) {
            return false;
        }
        for (int i = 0; i < p1.path().size(); ++i) {
            if (p1.path(i) != p2.path(i)) {
                return false;
            }
        }
        return p1.tray_id() == p2.tray_id() && p1.port_id() == p2.port_id();
    };

    return (ports_equal(a.port_a(), b.port_a()) && ports_equal(a.port_b(), b.port_b())) ||
           (ports_equal(a.port_a(), b.port_b()) && ports_equal(a.port_b(), b.port_a()));
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
