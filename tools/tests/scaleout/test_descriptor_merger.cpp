// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <google/protobuf/text_format.h>

#include <cabling_generator/descriptor_merger.hpp>
#include <cabling_generator/cabling_generator.hpp>
#include <factory_system_descriptor/utils.hpp>
#include "protobuf/cluster_config.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools {

class DescriptorMergerTest : public ::testing::Test {
protected:
    // Helper to create a test directory
    static std::string create_test_dir(const std::string& test_name) {
        const std::string dir = "generated/tests/" + test_name + "/";
        std::filesystem::create_directories(dir);
        return dir;
    }

    // Helper to write a string to a textproto file
    static void write_textproto(const std::string& path, const std::string& content) { std::ofstream(path) << content; }

    // Helper to write a protobuf message to a textproto file
    template <typename T>
    static void write_proto_to_textproto(const std::string& path, const T& proto) {
        std::string proto_str;
        google::protobuf::TextFormat::PrintToString(proto, &proto_str);
        write_textproto(path, proto_str);
    }

    // Helper to create a simple single-node descriptor
    static void create_simple_descriptor(
        const std::string& path,
        const std::string& template_name,
        const std::string& node_name,
        const std::string& node_descriptor,
        uint32_t host_id = 0) {
        write_textproto(
            path,
            "graph_templates {\n"
            "  key: \"" +
                template_name +
                "\"\n"
                "  value {\n"
                "    children {\n"
                "      name: \"" +
                node_name +
                "\"\n"
                "      node_ref { node_descriptor: \"" +
                node_descriptor +
                "\" }\n"
                "    }\n"
                "  }\n"
                "}\n"
                "root_instance {\n"
                "  template_name: \"" +
                template_name +
                "\"\n"
                "  child_mappings { key: \"" +
                node_name + "\" value { host_id: " + std::to_string(host_id) +
                " } }\n"
                "}\n");
    }

    // Helper to generate FSD and YAML for a descriptor
    static std::string generate_yaml_from_descriptor(
        const std::string& descriptor_path,
        const std::string& output_dir,
        const std::string& yaml_prefix,
        const std::vector<std::string>& hostnames = {"host0"}) {
        CablingGenerator gen(descriptor_path, hostnames);
        const std::string fsd_path = output_dir + yaml_prefix + "_fsd.textproto";
        gen.emit_factory_system_descriptor(fsd_path);
        return generate_cluster_descriptor_from_fsd(fsd_path, output_dir, yaml_prefix + "_cluster");
    }

    // Compare YAML ClusterDescriptors - the universal format where all connections are flattened
    static void assert_yaml_cluster_descriptors_equal(
        const std::string& yaml_path1, const std::string& yaml_path2, const std::string& context = "") {
        std::ifstream file1(yaml_path1);
        ASSERT_TRUE(file1.is_open()) << context << ": Failed to open " << yaml_path1;
        const std::string yaml1((std::istreambuf_iterator<char>(file1)), std::istreambuf_iterator<char>());

        std::ifstream file2(yaml_path2);
        ASSERT_TRUE(file2.is_open()) << context << ": Failed to open " << yaml_path2;
        const std::string yaml2((std::istreambuf_iterator<char>(file2)), std::istreambuf_iterator<char>());

        EXPECT_EQ(yaml1, yaml2) << context << ": YAML ClusterDescriptor mismatch\n"
                                << "  File 1: " << yaml_path1 << "\n"
                                << "  File 2: " << yaml_path2;
    }

    // Helper to create a two-node descriptor with a connection
    static void create_two_node_descriptor_with_connection(
        const std::string& path,
        const std::string& template_name,
        const std::string& node1_descriptor,
        const std::string& node2_descriptor,
        const std::string& conn_path_b,  // "node2", "node3", etc.
        uint32_t tray_a = 1,
        uint32_t port_a = 1,
        uint32_t tray_b = 1,
        uint32_t port_b = 1) {
        write_textproto(
            path,
            "graph_templates {\n"
            "  key: \"" +
                template_name +
                "\"\n"
                "  value {\n"
                "    children { name: \"node1\" node_ref { node_descriptor: \"" +
                node1_descriptor +
                "\" } }\n"
                "    children { name: \"node2\" node_ref { node_descriptor: \"" +
                node2_descriptor +
                "\" } }\n"
                "    internal_connections {\n"
                "      key: \"QSFP_DD\"\n"
                "      value {\n"
                "        connections {\n"
                "          port_a { path: [\"node1\"] tray_id: " +
                std::to_string(tray_a) + " port_id: " + std::to_string(port_a) +
                " }\n"
                "          port_b { path: [\"" +
                conn_path_b + "\"] tray_id: " + std::to_string(tray_b) + " port_id: " + std::to_string(port_b) +
                " }\n"
                "        }\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "}\n"
                "root_instance {\n"
                "  template_name: \"" +
                template_name +
                "\"\n"
                "  child_mappings { key: \"node1\" value { host_id: 0 } }\n"
                "  child_mappings { key: \"node2\" value { host_id: 1 } }\n"
                "}\n");
    }

    // Helper to create a multi-node descriptor with a connection
    static void create_multi_node_descriptor_with_connection(
        const std::string& path,
        const std::string& template_name,
        const std::vector<std::string>& node_names,
        const std::string& node_descriptor,
        const std::string& conn_from,
        const std::string& conn_to,
        uint32_t tray_a = 1,
        uint32_t port_a = 1,
        uint32_t tray_b = 1,
        uint32_t port_b = 1) {
        std::string content = "graph_templates {\n  key: \"" + template_name + "\"\n  value {\n";

        // Add children
        for (const auto& node_name : node_names) {
            content += "    children { name: \"" + node_name + "\" node_ref { node_descriptor: \"" + node_descriptor +
                       "\" } }\n";
        }

        // Add connection
        content +=
            "    internal_connections {\n"
            "      key: \"QSFP_DD\"\n"
            "      value {\n"
            "        connections {\n"
            "          port_a { path: [\"" +
            conn_from + "\"] tray_id: " + std::to_string(tray_a) + " port_id: " + std::to_string(port_a) +
            " }\n"
            "          port_b { path: [\"" +
            conn_to + "\"] tray_id: " + std::to_string(tray_b) + " port_id: " + std::to_string(port_b) +
            " }\n"
            "        }\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n"
            "root_instance {\n"
            "  template_name: \"" +
            template_name + "\"\n";

        // Add child mappings
        for (size_t i = 0; i < node_names.size(); ++i) {
            content +=
                "  child_mappings { key: \"" + node_names[i] + "\" value { host_id: " + std::to_string(i) + " } }\n";
        }

        content += "}\n";
        write_textproto(path, content);
    }

    // Helper to load a ClusterDescriptor from file
    static cabling_generator::proto::ClusterDescriptor load_descriptor(const std::string& path) {
        std::ifstream file(path);
        EXPECT_TRUE(file.is_open()) << "Failed to open " << path;
        const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        cabling_generator::proto::ClusterDescriptor desc;
        EXPECT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &desc)) << "Failed to parse " << path;
        return desc;
    }

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
            write_proto_to_textproto(path, parts[i]);
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

        write_proto_to_textproto(nodes_path, nodes_only);
        write_proto_to_textproto(conns_path, conns_only);

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
    const std::string empty_dir = create_test_dir("empty_merge_test_dir");
    EXPECT_THROW(DescriptorMerger::find_descriptor_files(empty_dir), std::runtime_error);
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
    const std::string test_dir = create_test_dir("connection_merge_test");

    auto [nodes_path, conns_path] = split_nodes_vs_connections(source, test_dir);
    const auto merged = DescriptorMerger::merge_descriptors({nodes_path, conns_path});
    const auto original = load_descriptor(source);

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

TEST_F(DescriptorMergerTest, ValidateStructureIdentityAllowsXAndYTorusMerge) {
    const std::string test_dir = create_test_dir("xy_torus_internal_connections");

    // Create test descriptors
    create_simple_descriptor(test_dir + "x_torus.textproto", "test_cluster", "node1", "WH_GALAXY_X_TORUS");
    create_simple_descriptor(test_dir + "y_torus.textproto", "test_cluster", "node1", "WH_GALAXY_Y_TORUS");
    create_simple_descriptor(test_dir + "xy_torus.textproto", "test_cluster", "node1", "WH_GALAXY_XY_TORUS");

    // Merge X_TORUS + Y_TORUS → should produce WH_GALAXY + internal_connections
    const auto merged =
        DescriptorMerger::merge_descriptors({test_dir + "x_torus.textproto", test_dir + "y_torus.textproto"});

    // Verify merged descriptor has WH_GALAXY base and internal_connections
    ASSERT_EQ(merged.graph_templates_size(), 1);
    const auto& merged_template = merged.graph_templates().at("test_cluster");
    ASSERT_EQ(merged_template.children_size(), 1);
    EXPECT_EQ(merged_template.children(0).node_ref().node_descriptor(), "WH_GALAXY");
    EXPECT_GT(merged_template.internal_connections_size(), 0) << "Should have torus as internal_connections";

    // Save merged descriptor and generate YAMLs for comparison
    write_proto_to_textproto(test_dir + "merged.textproto", merged);

    const std::string merged_yaml = generate_yaml_from_descriptor(test_dir + "merged.textproto", test_dir, "merged");
    const std::string xy_yaml = generate_yaml_from_descriptor(test_dir + "xy_torus.textproto", test_dir, "xy");

    assert_yaml_cluster_descriptors_equal(
        merged_yaml, xy_yaml, "Merged X_TORUS + Y_TORUS (as internal_connections) vs direct XY_TORUS");
}

TEST_F(DescriptorMergerTest, ValidateStructureIdentityRejectsDifferentChildren) {
    const std::string test_dir = create_test_dir("structure_validation_test2");

    // Minimal test: just one node with different descriptors between files
    create_simple_descriptor(test_dir + "file1.textproto", "test_cluster", "node1", "WH_GALAXY_Y_TORUS");
    create_simple_descriptor(test_dir + "file2.textproto", "test_cluster", "node1", "N300_T3K_NODE");

    // Should throw because node1 has different node_descriptor (not X/Y torus pair)
    EXPECT_THROW(
        {
            try {
                DescriptorMerger::merge_descriptors({test_dir + "file1.textproto", test_dir + "file2.textproto"});
                FAIL() << "Expected std::runtime_error for different children";
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("node_descriptor"), std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(DescriptorMergerTest, DetectConnectionConflictBetweenDescriptor1And2) {
    const std::string test_dir = create_test_dir("conflict_test_1_2");

    // Minimal test: 3 files, conflict happens between file1 and file2 (not 0 and 1)
    const std::vector<std::string> nodes = {"node1", "node2", "node3", "node4"};

    // File 0: node1 port 1 -> node2
    create_multi_node_descriptor_with_connection(
        test_dir + "file0.textproto", "test_cluster", nodes, "WH_GALAXY", "node1", "node2", 1, 1);

    // File 1: node1 port 2 -> node3 (no conflict with file 0)
    create_multi_node_descriptor_with_connection(
        test_dir + "file1.textproto", "test_cluster", nodes, "WH_GALAXY", "node1", "node3", 1, 2);

    // File 2: node1 port 2 -> node4 (CONFLICTS with file 1 on node1 port 2)
    create_multi_node_descriptor_with_connection(
        test_dir + "file2.textproto", "test_cluster", nodes, "WH_GALAXY", "node1", "node4", 1, 2);

    EXPECT_THROW(
        {
            try {
                DescriptorMerger::merge_descriptors(
                    {test_dir + "file0.textproto", test_dir + "file1.textproto", test_dir + "file2.textproto"});
                FAIL() << "Expected std::runtime_error for connection conflict";
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("Connection conflict"), std::string::npos);
                EXPECT_NE(std::string(e.what()).find("node1"), std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(DescriptorMergerTest, ErrorPropagatesFromConstructor) {
    const std::string test_dir = create_test_dir("error_propagation_test");

    // Minimal test: same endpoint connecting to different destinations = conflict
    create_two_node_descriptor_with_connection(
        test_dir + "file1.textproto", "test_cluster", "WH_GALAXY", "WH_GALAXY", "node2");
    create_two_node_descriptor_with_connection(
        test_dir + "file2.textproto", "test_cluster", "WH_GALAXY", "WH_GALAXY", "node3");

    EXPECT_THROW(
        {
            try {
                CablingGenerator gen(test_dir, std::vector<std::string>{"host0", "host1"});
                FAIL() << "Expected std::runtime_error from CablingGenerator constructor";
            } catch (const std::runtime_error& e) {
                EXPECT_NE(std::string(e.what()).find("Connection conflict"), std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(DescriptorMergerTest, MultipleConflictsDetectedDuringMerge) {
    const std::string test_dir = create_test_dir("multiple_conflicts_test");

    // Minimal test: file1 has node1->node2, file2 has node1->node3 AND node2->node4 (both conflict)
    create_two_node_descriptor_with_connection(
        test_dir + "file1.textproto", "test_cluster", "WH_GALAXY", "WH_GALAXY", "node2");

    write_textproto(test_dir + "file2.textproto", R"(
graph_templates {
  key: "test_cluster"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node2" node_ref { node_descriptor: "WH_GALAXY" } }
    internal_connections {
      key: "QSFP_DD"
      value {
        connections {
          port_a { path: ["node1"] tray_id: 1 port_id: 1 }
          port_b { path: ["node3"] tray_id: 1 port_id: 1 }
        }
        connections {
          port_a { path: ["node2"] tray_id: 1 port_id: 1 }
          port_b { path: ["node4"] tray_id: 1 port_id: 1 }
        }
      }
    }
  }
}
root_instance {
  template_name: "test_cluster"
  child_mappings { key: "node1" value { host_id: 0 } }
  child_mappings { key: "node2" value { host_id: 1 } }
}
)");

    EXPECT_THROW(
        {
            try {
                DescriptorMerger::merge_descriptors({test_dir + "file1.textproto", test_dir + "file2.textproto"});
                FAIL() << "Expected std::runtime_error for multiple conflicts";
            } catch (const std::runtime_error& e) {
                const std::string error_msg = e.what();
                EXPECT_NE(error_msg.find("node1"), std::string::npos);
                EXPECT_NE(error_msg.find("node2"), std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(DescriptorMergerTest, MergeXTorusAndYTorusProducesSameFSDAsXYTorus) {
    const std::string test_dir = create_test_dir("xy_torus_merge_test");

    // Create test descriptors
    create_simple_descriptor(test_dir + "x_torus.textproto", "test_cluster", "node1", "WH_GALAXY_X_TORUS");
    create_simple_descriptor(test_dir + "y_torus.textproto", "test_cluster", "node1", "WH_GALAXY_Y_TORUS");
    create_simple_descriptor(test_dir + "xy_torus.textproto", "test_cluster", "node1", "WH_GALAXY_XY_TORUS");

    // Merge and save
    const auto merged =
        DescriptorMerger::merge_descriptors({test_dir + "x_torus.textproto", test_dir + "y_torus.textproto"});
    write_proto_to_textproto(test_dir + "merged.textproto", merged);

    // Generate and compare YAMLs
    const std::string merged_yaml = generate_yaml_from_descriptor(test_dir + "merged.textproto", test_dir, "merged");
    const std::string xy_yaml = generate_yaml_from_descriptor(test_dir + "xy_torus.textproto", test_dir, "xy");

    assert_yaml_cluster_descriptors_equal(merged_yaml, xy_yaml, "X_TORUS + Y_TORUS merge");
}

}  // namespace tt::scaleout_tools
