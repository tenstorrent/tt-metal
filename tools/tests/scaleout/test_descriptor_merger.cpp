// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <google/protobuf/text_format.h>

#include <cabling_generator/descriptor_merger.hpp>
#include <cabling_generator/cabling_generator.hpp>
#include "protobuf/cluster_config.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools {

class DescriptorMergerTest : public ::testing::Test {
protected:

    std::vector<std::string> split_descriptor(
        const std::string& source_path,
        const std::string& output_dir,
        const std::string& template_name = "",
        int num_splits = 2) {
        EXPECT_GT(num_splits, 0) << "num_splits must be positive";

        std::ifstream file(source_path);
        EXPECT_TRUE(file.is_open()) << "Failed to open " << source_path;
        const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        cabling_generator::proto::ClusterDescriptor original_desc;
        EXPECT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &original_desc))
            << "Failed to parse " << source_path;

        const std::string target_template =
            template_name.empty() ? original_desc.graph_templates().begin()->first : template_name;

        EXPECT_TRUE(original_desc.graph_templates().contains(target_template))
            << "Template '" << target_template << "' not found";

        // Create empty parts (don't copy everything)
        std::vector<cabling_generator::proto::ClusterDescriptor> parts(num_splits);

        // Split graph_templates - distribute each template's children and connections
        for (const auto& [tmpl_name, tmpl] : original_desc.graph_templates()) {
            // Collect children to distribute
            std::vector<cabling_generator::proto::ChildInstance> children_vec;
            for (const auto& child : tmpl.children()) {
                children_vec.push_back(child);
            }

            // Distribute children round-robin
            for (size_t child_idx = 0; child_idx < children_vec.size(); child_idx++) {
                const int part_idx = child_idx % num_splits;
                *(*parts[part_idx].mutable_graph_templates())[tmpl_name].add_children() = children_vec[child_idx];
            }

            // Split internal_connections for this template
            for (const auto& [port_type, port_conns] : tmpl.internal_connections()) {
                for (int conn_idx = 0; conn_idx < port_conns.connections_size(); conn_idx++) {
                    const int part_idx = conn_idx % num_splits;
                    auto* template_conns =
                        (*parts[part_idx].mutable_graph_templates())[tmpl_name].mutable_internal_connections();
                    *(*template_conns)[port_type].add_connections() = port_conns.connections(conn_idx);
                }
            }
        }

        // Split inline node_descriptors if any
        for (const auto& [node_name, node_desc] : original_desc.node_descriptors()) {
            // Since we don't have a natural way to split node_descriptors, just duplicate them in all parts
            // (They're typically small and not the main source of duplication)
            for (int i = 0; i < num_splits; i++) {
                (*parts[i].mutable_node_descriptors())[node_name] = node_desc;
            }
        }

        // Split root_instance child_mappings
        if (original_desc.has_root_instance()) {
            std::vector<std::pair<std::string, cabling_generator::proto::ChildMapping>> mappings_vec;
            for (const auto& [key, mapping] : original_desc.root_instance().child_mappings()) {
                mappings_vec.push_back({key, mapping});
            }

            // Distribute mappings round-robin
            for (size_t mapping_idx = 0; mapping_idx < mappings_vec.size(); mapping_idx++) {
                const int part_idx = mapping_idx % num_splits;
                const auto& [key, mapping] = mappings_vec[mapping_idx];
                auto* root = parts[part_idx].mutable_root_instance();
                root->set_template_name(original_desc.root_instance().template_name());
                (*root->mutable_child_mappings())[key] = mapping;
            }
        }

        std::filesystem::create_directories(output_dir);

        std::vector<std::string> paths;
        for (int i = 0; i < num_splits; i++) {
            const std::string path = output_dir + "/part" + std::to_string(i + 1) + ".textproto";
            std::string part_str;
            google::protobuf::TextFormat::PrintToString(parts[i], &part_str);
            std::ofstream(path) << part_str;
            paths.push_back(path);
        }

        return paths;
    }

    // Split into nodes-only and connections-only files
    std::pair<std::string, std::string> split_nodes_vs_connections(
        const std::string& source_path, const std::string& output_dir) {
        std::ifstream file(source_path);
        EXPECT_TRUE(file.is_open());
        const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        cabling_generator::proto::ClusterDescriptor original;
        EXPECT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &original));

        // Create nodes-only descriptor
        cabling_generator::proto::ClusterDescriptor nodes_only;
        for (const auto& [key, tmpl] : original.graph_templates()) {
            for (const auto& child : tmpl.children()) {
                *(*nodes_only.mutable_graph_templates())[key].add_children() = child;
            }
        }
        if (original.has_root_instance()) {
            *nodes_only.mutable_root_instance() = original.root_instance();
        }

        // Create connections-only descriptor
        cabling_generator::proto::ClusterDescriptor conns_only;
        for (const auto& [key, tmpl] : original.graph_templates()) {
            *(*conns_only.mutable_graph_templates())[key].mutable_internal_connections() = tmpl.internal_connections();
        }

        std::filesystem::create_directories(output_dir);
        std::string nodes_path = output_dir + "nodes.textproto";
        std::string conns_path = output_dir + "connections.textproto";

        std::string nodes_str, conns_str;
        google::protobuf::TextFormat::PrintToString(nodes_only, &nodes_str);
        google::protobuf::TextFormat::PrintToString(conns_only, &conns_str);
        std::ofstream(nodes_path) << nodes_str;
        std::ofstream(conns_path) << conns_str;

        return {nodes_path, conns_path};
    }
};

TEST_F(DescriptorMergerTest, FindDescriptorFilesInDirectory) {
    // Use an existing directory with descriptors
    const auto files = DescriptorMerger::find_descriptor_files("tools/tests/scaleout/cabling_descriptors");

    EXPECT_GT(files.size(), 0);
    for (const auto& file : files) {
        EXPECT_TRUE(file.ends_with(".textproto"));
    }

    EXPECT_TRUE(std::is_sorted(files.begin(), files.end()));
}

TEST_F(DescriptorMergerTest, FindDescriptorFilesEmptyDirectory) {
    const std::string empty_dir = "generated/tests/empty_merge_test_dir/";
    std::filesystem::create_directories(empty_dir);

    EXPECT_THROW(DescriptorMerger::find_descriptor_files(empty_dir), std::runtime_error);

    std::filesystem::remove_all(empty_dir);
}

TEST_F(DescriptorMergerTest, FindDescriptorFilesNonexistentDirectory) {
    EXPECT_THROW(DescriptorMerger::find_descriptor_files("nonexistent_directory_12345"), std::runtime_error);
}

TEST_F(DescriptorMergerTest, ConnectionEndpointComparison) {
    const ConnectionEndpoint a{"template1", {"node1"}, 0, 0};
    const ConnectionEndpoint b{"template1", {"node1"}, 0, 0};
    const ConnectionEndpoint c{"template1", {"node1"}, 0, 1};
    const ConnectionEndpoint d{"template1", {"node2"}, 0, 0};

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, d);
    EXPECT_FALSE(a < b);
    EXPECT_TRUE(a < c);
}

TEST_F(DescriptorMergerTest, ConnectionPairEquality) {
    const ConnectionEndpoint a{"t", {"n1"}, 0, 0};
    const ConnectionEndpoint b{"t", {"n2"}, 0, 0};

    const ConnectionPair pair1{a, b, "QSFP_DD"};
    const ConnectionPair pair2{b, a, "QSFP_DD"};
    const ConnectionPair pair3{a, b, "TRACE"};

    EXPECT_EQ(pair1, pair2);
    EXPECT_NE(pair1, pair3);
}

TEST_F(DescriptorMergerTest, EmptyPathsThrows) {
    const std::vector<std::string> empty_paths;
    EXPECT_THROW(DescriptorMerger::merge_descriptors(empty_paths), std::runtime_error);
}

TEST_F(DescriptorMergerTest, NonexistentFileThrows) {
    const std::vector<std::string> paths = {"nonexistent_file.textproto"};
    EXPECT_THROW(DescriptorMerger::merge_descriptors(paths), std::runtime_error);
}

TEST_F(DescriptorMergerTest, MergeConnectionsFromSplitDescriptor) {
    const std::string source = "tools/tests/scaleout/cabling_descriptors/8x16_wh_galaxy_xy_torus_superpod.textproto";
    const std::string test_dir = "generated/tests/connection_merge_test/";

    // Use split_nodes_vs_connections to split only connections
    auto [nodes_path, conns_path] = split_nodes_vs_connections(source, test_dir);

    const auto merged = DescriptorMerger::merge_descriptors({nodes_path, conns_path});

    // Load original for comparison
    std::ifstream file(source);
    ASSERT_TRUE(file.is_open());
    const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    cabling_generator::proto::ClusterDescriptor original;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &original));

    // Verify structure matches
    EXPECT_EQ(merged.graph_templates_size(), original.graph_templates_size());
    for (const auto& [key, orig_tmpl] : original.graph_templates()) {
        ASSERT_TRUE(merged.graph_templates().contains(key));
        const auto& merged_tmpl = merged.graph_templates().at(key);
        EXPECT_EQ(merged_tmpl.children_size(), orig_tmpl.children_size());

        // Verify connections merged correctly
        for (const auto& [port_type, orig_conns] : orig_tmpl.internal_connections()) {
            EXPECT_TRUE(merged_tmpl.internal_connections().contains(port_type));
            EXPECT_EQ(
                merged_tmpl.internal_connections().at(port_type).connections_size(), orig_conns.connections_size());
        }
    }
}

TEST_F(DescriptorMergerTest, MergeTorusDescriptorsKeepsSeparateTemplates) {
    // Test merging X and Y torus descriptors with different template names
    const auto merged = DescriptorMerger::merge_descriptors(
        {"tt_metal/fabric/cabling_descriptors/wh_galaxy_x_torus.textproto",
         "tt_metal/fabric/cabling_descriptors/wh_galaxy_y_torus.textproto"});

    // Both templates should be present since they have different names
    EXPECT_EQ(merged.graph_templates_size(), 2);
    EXPECT_TRUE(merged.graph_templates().contains("wh_galaxy_x_torus"));
    EXPECT_TRUE(merged.graph_templates().contains("wh_galaxy_y_torus"));

    // root_instance should use first file's template
    ASSERT_TRUE(merged.has_root_instance());
    EXPECT_EQ(merged.root_instance().template_name(), "wh_galaxy_x_torus");
}

}  // namespace tt::scaleout_tools
