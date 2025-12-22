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

#include <enchantum/enchantum.hpp>
#include <tt_stl/caseless_comparison.hpp>

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

    // Validate structure identity: compare all pairs of descriptors (O(n^2))
    // Ensures all files are consistent with each other
    for (size_t i = 0; i < descriptors.size(); ++i) {
        for (size_t j = i + 1; j < descriptors.size(); ++j) {
            validate_structure_identity(
                descriptors[i], descriptor_paths[i], descriptors[j], descriptor_paths[j], validation_result);
        }
    }

    if (!validation_result.success) {
        throw std::runtime_error("Merge validation failed:\n" + validation_result.format_messages());
    }

    auto merged = merge_descriptors_impl(descriptors, validation_result);

    for (const auto& warning : validation_result.warnings) {
        std::cerr << "WARNING: Descriptor merge: " << warning << std::endl;
    }

    // Check if merge_descriptors_impl added any new errors
    if (!validation_result.success) {
        throw std::runtime_error("Merge failed:\n" + validation_result.format_messages());
    }
    return merged;
}

cabling_generator::proto::ClusterDescriptor DescriptorMerger::merge_descriptors_impl(
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
                // Existing template: merge internal_connections
                auto* merged_template = &(*merged.mutable_graph_templates())[name];
                merge_internal_connections(*merged_template, tmpl, name, source_identifier, validation_result);
                // Check for errors immediately after merge to fail fast
                if (!validation_result.success) {
                    return merged;
                }
                // Normalize torus descriptors to base descriptor (topology defined by connections)
                normalize_torus_descriptors(merged, source, name);
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
    ConnectionMap existing_map;

    for (const auto& [port_type, connections] : target_template.internal_connections()) {
        for (const auto& conn : connections.connections()) {
            const auto endpoint_a = port_to_endpoint(conn.port_a(), template_name);
            const auto endpoint_b = port_to_endpoint(conn.port_b(), template_name);
            existing_connections.insert({endpoint_a, endpoint_b, port_type});
            existing_map[endpoint_a] = endpoint_b;
            existing_map[endpoint_b] = endpoint_a;
        }
    }

    for (const auto& [port_type, source_connections] : source_template.internal_connections()) {
        auto* target_connections = &(*target_template.mutable_internal_connections())[port_type];

        for (const auto& conn : source_connections.connections()) {
            const auto endpoint_a = port_to_endpoint(conn.port_a(), template_name);
            const auto endpoint_b = port_to_endpoint(conn.port_b(), template_name);
            const ConnectionPair pair{endpoint_a, endpoint_b, port_type};

            if (existing_connections.contains(pair)) {
                std::ostringstream oss;
                oss << "Duplicate connection in template '" << template_name << "' from " << source_file << ": " << pair
                    << ". Using first occurrence.";
                result.add_warning(oss.str());
            } else {
                if (const auto it = existing_map.find(endpoint_a);
                    it != existing_map.end() && !(it->second == endpoint_b)) {
                    std::ostringstream oss;
                    oss << "Connection conflict in template '" << template_name << "' from " << source_file << ": "
                        << endpoint_a << " connects to " << it->second << " in merged descriptor but connects to "
                        << endpoint_b << " in " << source_file;
                    result.add_error(oss.str());
                } else if (const auto it = existing_map.find(endpoint_b);
                           it != existing_map.end() && !(it->second == endpoint_a)) {
                    std::ostringstream oss;
                    oss << "Connection conflict in template '" << template_name << "' from " << source_file << ": "
                        << endpoint_b << " connects to " << it->second << " in merged descriptor but connects to "
                        << endpoint_a << " in " << source_file;
                    result.add_error(oss.str());
                } else {
                    *target_connections->add_connections() = conn;
                    existing_connections.insert(pair);
                    existing_map[endpoint_a] = endpoint_b;
                    existing_map[endpoint_b] = endpoint_a;
                }
            }
        }
    }
}

void DescriptorMerger::normalize_torus_descriptors(
    cabling_generator::proto::ClusterDescriptor& target,
    const cabling_generator::proto::ClusterDescriptor& source,
    const std::string& template_name) {
    auto& target_template = (*target.mutable_graph_templates())[template_name];
    const auto& source_template = source.graph_templates().at(template_name);

    std::map<std::string, int> target_children_indices;
    for (int i = 0; i < target_template.children_size(); ++i) {
        target_children_indices[target_template.children(i).name()] = i;
    }

    for (const auto& source_child : source_template.children()) {
        const auto it = target_children_indices.find(source_child.name());
        if (it == target_children_indices.end()) {
            continue;
        }

        auto* target_child = target_template.mutable_children(it->second);
        if (!target_child->has_node_ref() || !source_child.has_node_ref()) {
            continue;
        }

        const std::string target_desc = target_child->node_ref().node_descriptor();
        const std::string source_desc = source_child.node_ref().node_descriptor();

        if (target_desc == source_desc) {
            continue;
        }

        // Normalize to base architecture and convert torus topology to internal_connections
        const std::string base_arch = extract_torus_architecture(target_desc);
        if (!base_arch.empty()) {
            target_child->mutable_node_ref()->set_node_descriptor(base_arch);

            // Add torus connections as internal_connections (self-connections from node to node)
            add_torus_internal_connections(target_template, target_desc, source_desc, source_child.name());
        }
    }
}

bool DescriptorMerger::is_torus_compatible(const std::string& desc1, const std::string& desc2) {
    if (!is_a_torus(desc1) || !is_a_torus(desc2)) {
        return false;
    }

    if (!has_same_torus_architecture(desc1, desc2)) {
        return false;
    }

    const bool desc1_x = has_torus_variant(desc1, "_X_TORUS");
    const bool desc1_y = has_torus_variant(desc1, "_Y_TORUS");
    const bool desc1_xy = has_torus_variant(desc1, "_XY_TORUS");
    const bool desc2_x = has_torus_variant(desc2, "_X_TORUS");
    const bool desc2_y = has_torus_variant(desc2, "_Y_TORUS");
    const bool desc2_xy = has_torus_variant(desc2, "_XY_TORUS");

    return (desc1_x && desc2_y) || (desc1_y && desc2_x) ||    // X + Y
           (desc1_xy && desc2_x) || (desc1_x && desc2_xy) ||  // XY + X
           (desc1_xy && desc2_y) || (desc1_y && desc2_xy) ||  // XY + Y
           (desc1_xy && desc2_xy);                            // XY + XY
}

bool DescriptorMerger::is_a_torus(const std::string& desc) {
    return has_torus_variant(desc, "_X_TORUS") || has_torus_variant(desc, "_Y_TORUS") ||
           has_torus_variant(desc, "_XY_TORUS");
}

bool DescriptorMerger::has_same_torus_architecture(const std::string& desc1, const std::string& desc2) {
    const bool both_wh = desc1.find("WH_GALAXY") == 0 && desc2.find("WH_GALAXY") == 0;
    const bool both_bh = desc1.find("BH_GALAXY") == 0 && desc2.find("BH_GALAXY") == 0;
    return both_wh || both_bh;
}

std::string DescriptorMerger::extract_torus_architecture(const std::string& desc) {
    if (desc.find("WH_GALAXY") == 0) {
        return "WH_GALAXY";
    }
    if (desc.find("BH_GALAXY") == 0) {
        return "BH_GALAXY";
    }
    return "";
}

bool DescriptorMerger::has_torus_variant(const std::string& desc, const std::string& variant) {
    return desc.find(variant) != std::string::npos;
}

// Helper to get the combined torus type for merging
NodeType get_combined_torus_type(const std::string& desc1, const std::string& desc2) {
    const bool desc1_x = desc1.find("_X_TORUS") != std::string::npos;
    const bool desc1_y = desc1.find("_Y_TORUS") != std::string::npos;
    const bool desc1_xy = desc1.find("_XY_TORUS") != std::string::npos;
    const bool desc2_x = desc2.find("_X_TORUS") != std::string::npos;
    const bool desc2_y = desc2.find("_Y_TORUS") != std::string::npos;
    const bool desc2_xy = desc2.find("_XY_TORUS") != std::string::npos;

    // Determine if we need XY torus
    const bool needs_x = desc1_x || desc1_xy || desc2_x || desc2_xy;
    const bool needs_y = desc1_y || desc1_xy || desc2_y || desc2_xy;

    if (needs_x && needs_y) {
        if (desc1.find("WH_GALAXY") == 0) {
            return NodeType::WH_GALAXY_XY_TORUS;
        } else if (desc1.find("BH_GALAXY") == 0) {
            return NodeType::BH_GALAXY_XY_TORUS;
        }
    } else if (needs_x) {
        if (desc1.find("WH_GALAXY") == 0) {
            return NodeType::WH_GALAXY_X_TORUS;
        } else if (desc1.find("BH_GALAXY") == 0) {
            return NodeType::BH_GALAXY_X_TORUS;
        }
    } else if (needs_y) {
        if (desc1.find("WH_GALAXY") == 0) {
            return NodeType::WH_GALAXY_Y_TORUS;
        } else if (desc1.find("BH_GALAXY") == 0) {
            return NodeType::BH_GALAXY_Y_TORUS;
        }
    }
    
    throw std::runtime_error("Unable to determine combined torus type from: " + desc1 + " and " + desc2);
}

void DescriptorMerger::add_torus_internal_connections(
    cabling_generator::proto::GraphTemplate& target_template,
    const std::string& target_desc,
    const std::string& source_desc,
    const std::string& child_name) {
    
    // Get the combined torus type
    NodeType combined_type = get_combined_torus_type(target_desc, source_desc);
    
    // Create a NodeDescriptor using node.cpp (which has all the torus connection logic!)
    auto combined_node_descriptor = create_node_descriptor(combined_type);
    
    // Extract base architecture name (e.g., "WH_GALAXY" from "WH_GALAXY_XY_TORUS")
    std::string base_arch = extract_torus_architecture(target_desc);
    
    // Only copy QSFP_DD connections (the torus-specific ones)
    // Don't copy LINKING_BOARD connections as those are already in the merged descriptor
    if (combined_node_descriptor.port_type_connections().contains("QSFP_DD")) {
        const auto& qsfp_connections = combined_node_descriptor.port_type_connections().at("QSFP_DD");
        auto& target_qsfp_connections = (*target_template.mutable_internal_connections())["QSFP_DD"];
        
        for (const auto& conn : qsfp_connections.connections()) {
            auto* new_conn = target_qsfp_connections.add_connections();
            
            // Copy port_a with child_name as path
            auto* port_a = new_conn->mutable_port_a();
            port_a->add_path(child_name);
            port_a->set_tray_id(conn.port_a().tray_id());
            port_a->set_port_id(conn.port_a().port_id());
            
            // Copy port_b with child_name as path
            auto* port_b = new_conn->mutable_port_b();
            port_b->add_path(child_name);
            port_b->set_tray_id(conn.port_b().tray_id());
            port_b->set_port_id(conn.port_b().port_id());
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

void DescriptorMerger::validate_structure_identity(
    const cabling_generator::proto::ClusterDescriptor& desc1,
    const std::string& file1,
    const cabling_generator::proto::ClusterDescriptor& desc2,
    const std::string& file2,
    MergeValidationResult& result) {
    validate_node_descriptors_identity(desc1, file1, desc2, file2, result);

    for (const auto& [template_name, tmpl1] : desc1.graph_templates()) {
        if (!desc2.graph_templates().contains(template_name)) {
            continue;
        }

        const auto& tmpl2 = desc2.graph_templates().at(template_name);
        const bool has_children1 = tmpl1.children_size() > 0;
        const bool has_children2 = tmpl2.children_size() > 0;

        if (has_children1 && has_children2) {
            validate_graph_template_children_identity(tmpl1, tmpl2, template_name, file1, file2, result);
        }
    }
}

void DescriptorMerger::validate_node_descriptors_identity(
    const cabling_generator::proto::ClusterDescriptor& desc1,
    const std::string& file1,
    const cabling_generator::proto::ClusterDescriptor& desc2,
    const std::string& file2,
    MergeValidationResult& result) {
    for (const auto& [name, node_desc1] : desc1.node_descriptors()) {
        if (desc2.node_descriptors().contains(name)) {
            const auto& node_desc2 = desc2.node_descriptors().at(name);
            std::string serialized1, serialized2;
            node_desc1.SerializeToString(&serialized1);
            node_desc2.SerializeToString(&serialized2);
            if (serialized1 != serialized2) {
                std::ostringstream oss;
                oss << "Node descriptor '" << name << "' differs between " << file1 << " and " << file2
                    << ". Node descriptors must be identical across files.";
                result.add_error(oss.str());
            }
        }
    }
}

void DescriptorMerger::validate_graph_template_children_identity(
    const cabling_generator::proto::GraphTemplate& tmpl1,
    const cabling_generator::proto::GraphTemplate& tmpl2,
    const std::string& template_name,
    const std::string& file1,
    const std::string& file2,
    MergeValidationResult& result) {
    if (tmpl1.children_size() != tmpl2.children_size()) {
        std::ostringstream oss;
        oss << "Graph template '" << template_name << "' has different number of children between " << file1 << " ("
            << tmpl1.children_size() << ") and " << file2 << " (" << tmpl2.children_size()
            << "). Children must be identical across files.";
        result.add_error(oss.str());
        return;
    }

    std::map<std::string, int> children1_indices;
    for (int i = 0; i < tmpl1.children_size(); ++i) {
        const auto& child = tmpl1.children(i);
        if (children1_indices.contains(child.name())) {
            std::ostringstream oss;
            oss << "Graph template '" << template_name << "' has duplicate child name '" << child.name() << "' in "
                << file1 << ". Child names must be unique.";
            result.add_error(oss.str());
            continue;
        }
        children1_indices[child.name()] = i;
    }

    for (int i = 0; i < tmpl2.children_size(); ++i) {
        const auto& child2 = tmpl2.children(i);
        const auto it = children1_indices.find(child2.name());
        if (it == children1_indices.end()) {
            std::ostringstream oss;
            oss << "Graph template '" << template_name << "' has child '" << child2.name() << "' in " << file2
                << " but not in " << file1 << ". Children must be identical across files.";
            result.add_error(oss.str());
            continue;
        }

        const auto& child1 = tmpl1.children(it->second);
        validate_child_identity(child1, child2, template_name, file1, file2, result);
    }
}

void DescriptorMerger::validate_child_identity(
    const cabling_generator::proto::ChildInstance& child1,
    const cabling_generator::proto::ChildInstance& child2,
    const std::string& template_name,
    const std::string& file1,
    const std::string& file2,
    MergeValidationResult& result) {
    if (child1.has_node_ref() != child2.has_node_ref()) {
        std::ostringstream oss;
        oss << "Graph template '" << template_name << "' child '" << child1.name()
            << "' has different reference types between " << file1 << " and " << file2
            << ". Children must be identical across files.";
        result.add_error(oss.str());
        return;
    }

    if (child1.has_node_ref()) {
        const std::string desc1 = child1.node_ref().node_descriptor();
        const std::string desc2 = child2.node_ref().node_descriptor();
        if (desc1 != desc2) {
            // Special case: allow torus variants to differ
            // XY_TORUS is a superset, so it can be combined with X_TORUS or Y_TORUS
            const bool is_torus_variant = is_torus_compatible(desc1, desc2);
            if (!is_torus_variant) {
                std::ostringstream oss;
                oss << "Graph template '" << template_name << "' child '" << child1.name()
                    << "' has different node_descriptor between " << file1 << " ('" << desc1 << "') and " << file2
                    << " ('" << desc2
                    << "'). Only torus variants (X_TORUS, Y_TORUS, XY_TORUS) differences are allowed.";
                result.add_error(oss.str());
            }
        }
    } else if (child1.has_graph_ref()) {
        if (child1.graph_ref().graph_template() != child2.graph_ref().graph_template()) {
            std::ostringstream oss;
            oss << "Graph template '" << template_name << "' child '" << child1.name()
                << "' has different graph_template reference between " << file1 << " ('"
                << child1.graph_ref().graph_template() << "') and " << file2 << " ('"
                << child2.graph_ref().graph_template() << "'). Children must be identical across files.";
            result.add_error(oss.str());
        }
    }
}

}  // namespace tt::scaleout_tools
