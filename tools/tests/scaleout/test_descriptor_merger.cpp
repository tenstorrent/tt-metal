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

static const std::string test_fixtures_dir = "tools/tests/scaleout/cabling_descriptors/merge_tests/";

class DescriptorMergerTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(std::filesystem::exists(test_fixtures_dir))
            << "Test fixtures directory not found: " << test_fixtures_dir;
    }

    std::string fixture_path(const std::string& filename) const { return test_fixtures_dir + filename; }

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

    int count_hosts_recursive(
        const cabling_generator::proto::GraphTemplate& graph_template,
        const std::map<std::string, cabling_generator::proto::GraphTemplate>& all_templates) {
        int count = 0;
        for (const auto& child : graph_template.children()) {
            if (child.has_node_ref()) {
                count++;
            } else if (child.has_graph_ref()) {
                const auto& ref_name = child.graph_ref().graph_template();
                if (all_templates.contains(ref_name)) {
                    count += count_hosts_recursive(all_templates.at(ref_name), all_templates);
                }
            }
        }
        return count;
    }

    void test_split_merge_produces_same_fsd(
        const std::string& descriptor_path,
        const std::string& template_name,
        const std::string& temp_dir_base,
        int num_splits = 2) {
        std::ifstream file(descriptor_path);
        ASSERT_TRUE(file.is_open()) << "Failed to open " << descriptor_path;
        const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

        cabling_generator::proto::ClusterDescriptor original_desc;
        ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &original_desc));

        const auto& graph_template = original_desc.graph_templates().at(template_name);

        const int num_hosts = count_hosts_recursive(
            graph_template,
            std::map<std::string, cabling_generator::proto::GraphTemplate>(
                original_desc.graph_templates().begin(), original_desc.graph_templates().end()));
        ASSERT_GT(num_hosts, 0) << "No hosts found in template";

        std::string node_type;
        if (graph_template.children_size() > 0) {
            const auto& first_child = graph_template.children(0);
            if (first_child.has_node_ref() && !first_child.node_ref().node_descriptor().empty()) {
                node_type = first_child.node_ref().node_descriptor();
            } else if (first_child.has_graph_ref()) {
                const auto& ref_name = first_child.graph_ref().graph_template();
                const auto& ref_template = original_desc.graph_templates().at(ref_name);
                if (ref_template.children_size() > 0 && ref_template.children(0).has_node_ref()) {
                    node_type = ref_template.children(0).node_ref().node_descriptor();
                }
            }
        }
        ASSERT_FALSE(node_type.empty()) << "Could not determine node_type from descriptor";

        const int total_conns = graph_template.internal_connections().at("QSFP_DD").connections_size();

        const std::string cabling_dir = temp_dir_base + "cabling/";
        const auto part_paths = split_descriptor(descriptor_path, cabling_dir, template_name, num_splits);

        const auto merged = DescriptorMerger::merge_descriptors(part_paths);

        ASSERT_TRUE(merged.graph_templates().contains(template_name));
        const auto& merged_template = merged.graph_templates().at(template_name);
        ASSERT_TRUE(merged_template.internal_connections().contains("QSFP_DD"));

        const int merged_conn_count = merged_template.internal_connections().at("QSFP_DD").connections_size();
        EXPECT_EQ(merged_conn_count, total_conns);

        // Save merged descriptor to file
        std::string merged_desc_str;
        google::protobuf::TextFormat::PrintToString(merged, &merged_desc_str);
        std::ofstream(temp_dir_base + "merged_descriptor.textproto") << merged_desc_str;

        const std::string deployment_path = temp_dir_base + "deployment.textproto";
        std::ostringstream deployment_oss;
        deployment_oss << "rack_capacity: 1\n";
        for (int i = 0; i < num_hosts; i++) {
            deployment_oss << "hosts { hall: \"0\" aisle: \"0\" rack: 0 shelf_u: " << i << " node_type: \"" << node_type
                           << "\" host: \"h" << i << "\" }\n";
        }
        std::ofstream(deployment_path) << deployment_oss.str();

        const std::string original_temp_path = temp_dir_base + "original.textproto";
        std::ofstream(original_temp_path) << content;

        const CablingGenerator gen_original(original_temp_path, deployment_path);
        const auto fsd_original = gen_original.generate_factory_system_descriptor();

        const CablingGenerator gen_merged(cabling_dir, deployment_path);
        const auto fsd_merged = gen_merged.generate_factory_system_descriptor();

        // Save FSDs to files for inspection
        std::string fsd_original_str, fsd_merged_str;
        google::protobuf::TextFormat::PrintToString(fsd_original, &fsd_original_str);
        google::protobuf::TextFormat::PrintToString(fsd_merged, &fsd_merged_str);
        std::ofstream(temp_dir_base + "fsd_original.textproto") << fsd_original_str;
        std::ofstream(temp_dir_base + "fsd_merged.textproto") << fsd_merged_str;

        EXPECT_EQ(fsd_original.hosts_size(), fsd_merged.hosts_size());

        const int original_eth_conns = fsd_original.eth_connections().connection_size();
        const int merged_eth_conns = fsd_merged.eth_connections().connection_size();

        EXPECT_EQ(original_eth_conns, merged_eth_conns);
        EXPECT_GT(original_eth_conns, 0);

        // std::filesystem::remove_all(temp_dir_base);
    }
};

TEST_F(DescriptorMergerTest, FindDescriptorFilesInDirectory) {
    const auto files = DescriptorMerger::find_descriptor_files(test_fixtures_dir);

    EXPECT_GE(files.size(), 4);
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

TEST_F(DescriptorMergerTest, IsDirectoryCheck) {
    EXPECT_TRUE(DescriptorMerger::is_directory(test_fixtures_dir));
    EXPECT_FALSE(DescriptorMerger::is_directory(fixture_path("base_intrapod.textproto")));
    EXPECT_FALSE(DescriptorMerger::is_directory("nonexistent_path"));
}

TEST_F(DescriptorMergerTest, LoadSingleDescriptor) {
    const std::vector<std::string> paths = {fixture_path("base_intrapod.textproto")};
    const auto merged = DescriptorMerger::merge_descriptors(paths);

    EXPECT_TRUE(merged.graph_templates().contains("test_pod"));
    EXPECT_EQ(merged.graph_templates().at("test_pod").children().size(), 4);
    EXPECT_TRUE(merged.has_root_instance());
    EXPECT_EQ(merged.root_instance().template_name(), "test_pod");
}

TEST_F(DescriptorMergerTest, MergeComplementaryDescriptors) {
    const std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("additional_interpod.textproto")};
    const auto merged = DescriptorMerger::merge_descriptors(paths);

    ASSERT_TRUE(merged.graph_templates().contains("test_pod"));
    const auto& template_def = merged.graph_templates().at("test_pod");
    ASSERT_TRUE(template_def.internal_connections().contains("QSFP_DD"));
    EXPECT_EQ(template_def.internal_connections().at("QSFP_DD").connections().size(), 4);
}

TEST_F(DescriptorMergerTest, MergeDifferentTemplates) {
    const std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("different_template.textproto")};
    const auto merged = DescriptorMerger::merge_descriptors(paths);

    EXPECT_TRUE(merged.graph_templates().contains("test_pod"));
    EXPECT_TRUE(merged.graph_templates().contains("test_superpod"));
}

TEST_F(DescriptorMergerTest, DetectConflictingConnections) {
    const std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("conflicting_connection.textproto")};

    EXPECT_THROW(
        {
            try {
                DescriptorMerger::merge_descriptors(paths);
                FAIL() << "Expected std::runtime_error";
            } catch (const std::runtime_error& e) {
                const std::string error_msg = e.what();
                EXPECT_NE(error_msg.find("conflict"), std::string::npos) << "Error message should contain 'conflict'";
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(DescriptorMergerTest, HandleDuplicateConnections) {
    const std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("duplicate_connection.textproto")};
    const auto merged = DescriptorMerger::merge_descriptors(paths);

    ASSERT_TRUE(merged.graph_templates().contains("test_pod"));
    ASSERT_TRUE(merged.graph_templates().at("test_pod").internal_connections().contains("QSFP_DD"));
    const auto& connections = merged.graph_templates().at("test_pod").internal_connections().at("QSFP_DD");
    EXPECT_EQ(connections.connections().size(), 3);
}

TEST_F(DescriptorMergerTest, ValidateHostConsistencySameHosts) {
    const std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("additional_interpod.textproto")};
    const auto validation = DescriptorMerger::validate_host_consistency(paths);

    EXPECT_TRUE(validation.success);
    const bool has_mismatch =
        std::any_of(validation.warnings.begin(), validation.warnings.end(), [](const std::string& w) {
            return w.find("Host count mismatch") != std::string::npos;
        });
    EXPECT_FALSE(has_mismatch);
}

TEST_F(DescriptorMergerTest, ValidateHostConsistencyWithTemplateOnlyDescriptor) {
    const std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("different_template.textproto")};
    const auto validation = DescriptorMerger::validate_host_consistency(paths);

    EXPECT_TRUE(validation.success);
}

TEST_F(DescriptorMergerTest, ValidateHostConsistencyDifferentHostCounts) {
    const std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("different_host_count.textproto")};
    const auto validation = DescriptorMerger::validate_host_consistency(paths);

    EXPECT_TRUE(validation.success);
    EXPECT_FALSE(validation.warnings.empty());
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

TEST_F(DescriptorMergerTest, MergeFromDirectoryExcludingConflicts) {
    const std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"),
        fixture_path("additional_interpod.textproto"),
        fixture_path("different_template.textproto")};
    const auto merged = DescriptorMerger::merge_descriptors(paths);

    ASSERT_TRUE(merged.graph_templates().contains("test_pod"));
    ASSERT_TRUE(merged.graph_templates().contains("test_superpod"));
    EXPECT_EQ(merged.graph_templates().at("test_pod").internal_connections().at("QSFP_DD").connections().size(), 4);
    EXPECT_EQ(
        merged.graph_templates().at("test_superpod").internal_connections().at("QSFP_DD").connections().size(), 1);
}

TEST_F(DescriptorMergerTest, MergeSplitDescriptorProducesSameFSD) {
    for (int i = 1; i <= 16; i++) {
        test_split_merge_produces_same_fsd(
            "tools/tests/scaleout/cabling_descriptors/8x16_wh_galaxy_xy_torus_superpod.textproto",
            "8x16_wh_galaxy_xy_torus_superpod",
            "generated/tests/split_merge_test_" + std::to_string(i) + "parts/",
            i);
    }
}

TEST_F(DescriptorMergerTest, MergeSplit5NodeSuperpodProducesSameFSD) {
    for (int i = 1; i <= 5; i++) {
        test_split_merge_produces_same_fsd(
            "tools/tests/scaleout/cabling_descriptors/5_wh_galaxy_y_torus_superpod.textproto",
            "5_wh_galaxy_y_torus_superpod",
            "generated/tests/split_5node_test_" + std::to_string(i) + "parts/",
            i);
    }
}

TEST_F(DescriptorMergerTest, MergeSplit5N300LBSuperpodProducesSameFSD) {
    for (int i = 1; i <= 5; i++) {
        test_split_merge_produces_same_fsd(
            "tools/tests/scaleout/cabling_descriptors/5_n300_lb_superpod.textproto",
            "5lb",
            "generated/tests/split_5n300lb_test_" + std::to_string(i) + "parts/",
            i);
    }
}

TEST_F(DescriptorMergerTest, MergeSplit16N300LBClusterProducesSameFSD) {
    for (int i = 1; i <= 16; i++) {
        test_split_merge_produces_same_fsd(
            "tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto",
            "n300_lb_cluster",
            "generated/tests/split_16n300lb_test_" + std::to_string(i) + "parts/",
            i);
    }
}

TEST_F(DescriptorMergerTest, MergeSplit4xBHQuietboxProducesSameFSD) {
    for (int i = 1; i <= 4; i++) {
        test_split_merge_produces_same_fsd(
            "tests/scale_out/4x_bh_quietbox/cabling_descriptors/4x_bh_quietbox.textproto",
            "4x_bh_quietbox",
            "generated/tests/split_4xbh_test_" + std::to_string(i) + "parts/",
            i);
    }
}

TEST_F(DescriptorMergerTest, MergeChildrenFromMultipleFiles) {
    const std::string test_dir = "generated/tests/children_union_test/";
    std::filesystem::create_directories(test_dir);

    std::ofstream(test_dir + "file1.textproto") << R"(
graph_templates {
  key: "test_cluster"
  value {
    children {
      name: "node1"
      node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" }
    }
    children {
      name: "node2"
      node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" }
    }
    internal_connections {
      key: "QSFP_DD"
      value {
        connections {
          port_a { path: ["node1"] tray_id: 1 port_id: 1 }
          port_b { path: ["node2"] tray_id: 1 port_id: 1 }
        }
      }
    }
  }
}
root_instance {
  template_name: "test_cluster"
  child_mappings { key: "node1" value { host_id: 0 } }
  child_mappings { key: "node2" value { host_id: 1 } }
  child_mappings { key: "node3" value { host_id: 2 } }
  child_mappings { key: "node4" value { host_id: 3 } }
}
)";

    std::ofstream(test_dir + "file2.textproto") << R"(
graph_templates {
  key: "test_cluster"
  value {
    children {
      name: "node3"
      node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" }
    }
    children {
      name: "node4"
      node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" }
    }
    internal_connections {
      key: "QSFP_DD"
      value {
        connections {
          port_a { path: ["node3"] tray_id: 2 port_id: 2 }
          port_b { path: ["node4"] tray_id: 2 port_id: 2 }
        }
      }
    }
  }
}
)";

    const auto merged =
        DescriptorMerger::merge_descriptors({test_dir + "file1.textproto", test_dir + "file2.textproto"});

    ASSERT_TRUE(merged.graph_templates().contains("test_cluster"));
    const auto& tmpl = merged.graph_templates().at("test_cluster");

    EXPECT_EQ(tmpl.children_size(), 4) << "Should have all 4 children from both files";

    std::set<std::string> child_names;
    for (const auto& child : tmpl.children()) {
        child_names.insert(child.name());
    }

    EXPECT_TRUE(child_names.contains("node1"));
    EXPECT_TRUE(child_names.contains("node2"));
    EXPECT_TRUE(child_names.contains("node3"));
    EXPECT_TRUE(child_names.contains("node4"));

    EXPECT_EQ(tmpl.internal_connections().at("QSFP_DD").connections_size(), 2) << "Should have both connections merged";

    std::filesystem::remove_all(test_dir);
}

TEST_F(DescriptorMergerTest, MergeRootInstanceChildMappings) {
    const std::string test_dir = "generated/tests/root_instance_merge_test/";
    std::filesystem::create_directories(test_dir);

    std::ofstream(test_dir + "subset1.textproto") << R"(
graph_templates {
  key: "full_cluster"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
    children { name: "node2" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
    children { name: "node3" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
    internal_connections { key: "QSFP_DD" value {} }
  }
}
root_instance {
  template_name: "full_cluster"
  child_mappings { key: "node1" value { host_id: 0 } }
}
)";

    std::ofstream(test_dir + "subset2.textproto") << R"(
graph_templates {
  key: "full_cluster"
  value {
    children { name: "node4" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
    children { name: "node5" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
    internal_connections { key: "QSFP_DD" value {} }
  }
}
root_instance {
  template_name: "full_cluster"
  child_mappings { key: "node2" value { host_id: 1 } }
  child_mappings { key: "node3" value { host_id: 2 } }
}
)";

    std::ofstream(test_dir + "subset3.textproto") << R"(
graph_templates {
  key: "full_cluster"
  value {
    children { name: "node6" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
    internal_connections { key: "QSFP_DD" value {} }
  }
}
root_instance {
  template_name: "full_cluster"
  child_mappings { key: "node4" value { host_id: 3 } }
  child_mappings { key: "node5" value { host_id: 4 } }
  child_mappings { key: "node6" value { host_id: 5 } }
}
)";

    const auto merged = DescriptorMerger::merge_descriptors(
        {test_dir + "subset1.textproto", test_dir + "subset2.textproto", test_dir + "subset3.textproto"});

    ASSERT_TRUE(merged.has_root_instance());
    EXPECT_EQ(merged.root_instance().template_name(), "full_cluster");

    const auto& mappings = merged.root_instance().child_mappings();
    EXPECT_EQ(mappings.size(), 6) << "Should have all 6 child_mappings merged";

    ASSERT_TRUE(mappings.contains("node1"));
    ASSERT_TRUE(mappings.contains("node2"));
    ASSERT_TRUE(mappings.contains("node3"));
    ASSERT_TRUE(mappings.contains("node4"));
    ASSERT_TRUE(mappings.contains("node5"));
    ASSERT_TRUE(mappings.contains("node6"));

    EXPECT_EQ(mappings.at("node1").host_id(), 0);
    EXPECT_EQ(mappings.at("node2").host_id(), 1);
    EXPECT_EQ(mappings.at("node3").host_id(), 2);
    EXPECT_EQ(mappings.at("node4").host_id(), 3);
    EXPECT_EQ(mappings.at("node5").host_id(), 4);
    EXPECT_EQ(mappings.at("node6").host_id(), 5);

    ASSERT_TRUE(merged.graph_templates().contains("full_cluster"));
    EXPECT_EQ(merged.graph_templates().at("full_cluster").children_size(), 6)
        << "Should have all 6 children from all files";

    std::filesystem::remove_all(test_dir);
}

TEST_F(DescriptorMergerTest, ConflictingNodeDescriptorsWarnsAndUsesFirst) {
    const std::string test_dir = "generated/tests/conflicting_descriptors_test/";
    std::filesystem::create_directories(test_dir);

    std::ofstream(test_dir + "file1.textproto") << R"(
graph_templates {
  key: "cluster"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
    internal_connections { key: "QSFP_DD" value {} }
  }
}
root_instance {
  template_name: "cluster"
  child_mappings { key: "node1" value { host_id: 0 } }
}
)";

    std::ofstream(test_dir + "file2.textproto") << R"(
graph_templates {
  key: "cluster"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY_X_TORUS" } }
    internal_connections { key: "QSFP_DD" value {} }
  }
}
)";

    const auto merged =
        DescriptorMerger::merge_descriptors({test_dir + "file1.textproto", test_dir + "file2.textproto"});

    ASSERT_TRUE(merged.graph_templates().contains("cluster"));
    const auto& tmpl = merged.graph_templates().at("cluster");

    EXPECT_EQ(tmpl.children_size(), 1) << "Should have only one 'node1' child";

    const auto& child = tmpl.children(0);
    EXPECT_EQ(child.name(), "node1");
    EXPECT_TRUE(child.has_node_ref());
    EXPECT_EQ(child.node_ref().node_descriptor(), "WH_GALAXY_Y_TORUS") << "Should use node_descriptor from first file";

    std::filesystem::remove_all(test_dir);
}

TEST_F(DescriptorMergerTest, MergingDifferentTemplateNamesKeepsBoth) {
    const std::string test_dir = "generated/tests/different_template_names_test/";
    std::filesystem::create_directories(test_dir);

    std::ofstream(test_dir + "x_torus.textproto") << R"(
graph_templates {
  key: "wh_galaxy_x_torus"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY_X_TORUS" } }
  }
}
root_instance {
  template_name: "wh_galaxy_x_torus"
  child_mappings { key: "node1" value { host_id: 0 } }
}
)";

    std::ofstream(test_dir + "y_torus.textproto") << R"(
graph_templates {
  key: "wh_galaxy_y_torus"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY_Y_TORUS" } }
  }
}
root_instance {
  template_name: "wh_galaxy_y_torus"
  child_mappings { key: "node1" value { host_id: 0 } }
}
)";

    const auto merged =
        DescriptorMerger::merge_descriptors({test_dir + "x_torus.textproto", test_dir + "y_torus.textproto"});

    EXPECT_EQ(merged.graph_templates().size(), 2) << "Should have both templates since they have different names";

    EXPECT_TRUE(merged.graph_templates().contains("wh_galaxy_x_torus"));
    EXPECT_TRUE(merged.graph_templates().contains("wh_galaxy_y_torus"));

    ASSERT_TRUE(merged.has_root_instance());
    EXPECT_EQ(merged.root_instance().template_name(), "wh_galaxy_x_torus")
        << "root_instance should use first file's template_name";

    EXPECT_EQ(merged.root_instance().child_mappings().size(), 1)
        << "Only first file's child_mappings used (second file's root_instance ignored)";

    std::filesystem::remove_all(test_dir);
}

}  // namespace tt::scaleout_tools
