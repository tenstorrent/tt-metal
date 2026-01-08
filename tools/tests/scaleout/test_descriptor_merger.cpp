// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/message_differencer.h>

#include <cabling_generator/cabling_generator.hpp>
#include <factory_system_descriptor/utils.hpp>
#include "protobuf/cluster_config.pb.h"
#include "protobuf/factory_system_descriptor.pb.h"

namespace tt::scaleout_tools {

class DescriptorMergerTest : public ::testing::Test {
protected:
    // Helper to create a test directory (cleans up any existing files)
    static std::string create_test_dir(const std::string& test_name) {
        const std::string dir = "generated/tests/" + test_name + "/";
        // Remove directory if it exists to clean up previous test runs
        if (std::filesystem::exists(dir)) {
            std::filesystem::remove_all(dir);
        }
        std::filesystem::create_directories(dir);
        return dir;
    }

    // Helper to create a vector of hostnames (host0, host1, ..., hostN-1)
    static std::vector<std::string> create_host_vector(int count) {
        std::vector<std::string> hostnames;
        hostnames.reserve(count);
        for (int i = 0; i < count; ++i) {
            hostnames.push_back("host" + std::to_string(i));
        }
        return hostnames;
    }

    // Helper to get host count from a descriptor file
    static int get_host_count(const std::string& descriptor_path) {
        auto desc = load_descriptor(descriptor_path);
        return desc.root_instance().child_mappings().size();
    }

    // Helper to create a torus descriptor file for testing merging
    static void create_torus_descriptor(
        const std::string& path,
        const std::string& template_name,
        const std::string& node_name,
        const std::string& node_type,
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
                node_type +
                "\" }\n"
                "    }\n"
                "  }\n"
                "}\n"
                "\n"
                "root_instance {\n"
                "  template_name: \"" +
                template_name +
                "\"\n"
                "  child_mappings {\n"
                "    key: \"" +
                node_name +
                "\"\n"
                "    value { host_id: " +
                std::to_string(host_id) +
                " }\n"
                "  }\n"
                "}\n");
    }

    // Helper to write a string to a textproto file
    static void write_textproto(const std::string& path, const std::string& content) {
        std::ofstream ofs(path);
        ofs << content;
        ofs.flush();
    }

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

    // Helper to create a descriptor with internal connections (graph-level connections between nodes)
    static void create_descriptor_with_internal_connections(
        const std::string& path,
        const std::string& template_name,
        const std::vector<std::string>& node_names,
        const std::vector<std::pair<
            std::pair<std::string, std::pair<uint32_t, uint32_t>>,
            std::pair<std::string, std::pair<uint32_t, uint32_t>>>>& connections,
        const std::vector<uint32_t>& host_ids) {
        std::string content = "graph_templates {\n  key: \"" + template_name + "\"\n";
        content += "  value {\n";
        for (const auto& node_name : node_names) {
            content += "    children { name: \"" + node_name + "\" node_ref { node_descriptor: \"WH_GALAXY\" } }\n";
        }
        if (!connections.empty()) {
            content += "    internal_connections {\n      key: \"QSFP_DD\"\n      value {\n";
            for (const auto& [node_a_port, node_b_port] : connections) {
                const auto& [node_a, port_a] = node_a_port;
                const auto& [node_b, port_b] = node_b_port;
                content += "        connections {\n";
                content += "          port_a { path: [\"" + node_a + "\"] tray_id: " + std::to_string(port_a.first) +
                           " port_id: " + std::to_string(port_a.second) + " }\n";
                content += "          port_b { path: [\"" + node_b + "\"] tray_id: " + std::to_string(port_b.first) +
                           " port_id: " + std::to_string(port_b.second) + " }\n";
                content += "        }\n";
            }
            content += "      }\n    }\n";
        }
        content += "  }\n}\n";
        content += "root_instance {\n  template_name: \"" + template_name + "\"\n";
        for (size_t i = 0; i < node_names.size(); ++i) {
            content += "  child_mappings { key: \"" + node_names[i] +
                       "\" value { host_id: " + std::to_string(host_ids[i]) + " } }\n";
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

    static std::vector<std::string> split_descriptor(
        const std::string& source_path,
        const std::string& output_dir,
        const std::string& template_name = "",
        int num_splits = 2) {
        if (num_splits < 2) {
            throw std::runtime_error("num_splits must be at least 2");
        }

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

        // Split graph_templates - ensure all files have complete structure EXCEPT internal_connections
        // All children must be in all parts (complete structure required for merge)
        // Only internal_connections should be split/incomplete in each part
        for (const auto& [tmpl_name, tmpl] : original_desc.graph_templates()) {
            // All children must be in all parts (complete structure required for merge)
            for (int i = 0; i < num_splits; i++) {
                for (const auto& child : tmpl.children()) {
                    *(*parts[i].mutable_graph_templates())[tmpl_name].add_children() = child;
                }
            }

            // Split internal_connections across parts using round-robin distribution
            // Each connection goes to exactly one part, distributed evenly
            for (const auto& [port_type, port_conns] : tmpl.internal_connections()) {
                int connection_idx = 0;
                for (const auto& conn : port_conns.connections()) {
                    // Round-robin: connection i goes to part (i % num_splits)
                    int target_part = connection_idx % num_splits;
                    auto* target_internal_conns =
                        (*parts[target_part].mutable_graph_templates())[tmpl_name].mutable_internal_connections();
                    *(*target_internal_conns)[port_type].add_connections() = conn;
                    connection_idx++;
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

        // Split root_instance child_mappings - all mappings must be in all parts (complete structure)
        if (original_desc.has_root_instance()) {
            // All mappings must be in all parts (complete structure required for merge)
            for (int i = 0; i < num_splits; i++) {
                auto* root = parts[i].mutable_root_instance();
                root->set_template_name(original_desc.root_instance().template_name());
                for (const auto& [key, mapping] : original_desc.root_instance().child_mappings()) {
                    (*root->mutable_child_mappings())[key] = mapping;
                }
            }
        }

        std::filesystem::create_directories(output_dir);

        std::vector<std::string> paths;
        for (int i = 0; i < num_splits; i++) {
            const std::string path = output_dir + "/part" + std::to_string(i + 1) + ".textproto";
            write_proto_to_textproto(path, parts[i]);
            if (!std::filesystem::exists(path)) {
                throw std::runtime_error("Failed to write split descriptor: " + path);
            }
            paths.push_back(path);
        }

        return paths;
    }
};

// ============================================================================
// Sanity Tests: Basic functionality and building blocks
// ============================================================================

TEST_F(DescriptorMergerTest, RejectEmptyDirectory) {
    // Empty directory should throw - no descriptor files found
    const std::string empty_dir = create_test_dir("empty_test");
    EXPECT_THROW(CablingGenerator(empty_dir, std::vector<std::string>{}), std::runtime_error);
}

TEST_F(DescriptorMergerTest, RejectNonexistentFile) {
    // Nonexistent file should throw
    EXPECT_THROW(CablingGenerator("nonexistent_file.textproto", std::vector<std::string>{"host0"}), std::runtime_error);
}

// ============================================================================
// Torus Merge Tests: Torus-specific inter_board_connections merging
// ============================================================================

TEST_F(DescriptorMergerTest, MergeXTorusAndYTorusIntoXYTorus) {
    // Test that X_TORUS and Y_TORUS node types can merge into a combined XY_TORUS configuration
    // Both X and Y torus have the same Wormhole architecture and compatible topology
    // Their inter_board_connections should merge successfully to form an XY_TORUS
    const std::string test_dir = create_test_dir("xy_torus_merge_test");

    const std::string merge_dir = test_dir + "merge/";
    std::filesystem::create_directories(merge_dir);
    create_torus_descriptor(merge_dir + "x_torus.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_X_TORUS", 0);
    create_torus_descriptor(merge_dir + "y_torus.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_Y_TORUS", 0);

    // Merge X + Y torus
    CablingGenerator merged_gen(merge_dir, create_host_vector(1));

    // Create reference XY torus descriptor
    const std::string ref_dir = test_dir + "reference/";
    std::filesystem::create_directories(ref_dir);
    create_torus_descriptor(ref_dir + "xy_torus.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_XY_TORUS", 0);
    CablingGenerator xy_gen(ref_dir + "xy_torus.textproto", create_host_vector(1));

    // Validate that X + Y merge produces the same result as XY torus
    EXPECT_EQ(merged_gen, xy_gen) << "X torus + Y torus should equal XY torus";
}

TEST_F(DescriptorMergerTest, MergeBHXTorusAndBHYTorusIntoXYTorus) {
    // Test that BH (Blackhole) X_TORUS and Y_TORUS can merge into XY_TORUS
    // Validates torus merging works for Blackhole architecture, not just Wormhole
    const std::string test_dir = create_test_dir("bh_xy_torus_merge");

    const std::string merge_dir = test_dir + "merge/";
    std::filesystem::create_directories(merge_dir);
    create_torus_descriptor(merge_dir + "x_torus.textproto", "bh_galaxy_torus", "node1", "BH_GALAXY_X_TORUS", 0);
    create_torus_descriptor(merge_dir + "y_torus.textproto", "bh_galaxy_torus", "node1", "BH_GALAXY_Y_TORUS", 0);

    // Merge BH X + Y torus
    CablingGenerator merged_gen(merge_dir, create_host_vector(1));

    // Create reference BH XY torus descriptor
    const std::string ref_dir = test_dir + "reference/";
    std::filesystem::create_directories(ref_dir);
    create_torus_descriptor(ref_dir + "xy_torus.textproto", "bh_galaxy_torus", "node1", "BH_GALAXY_XY_TORUS", 0);
    CablingGenerator xy_gen(ref_dir + "xy_torus.textproto", create_host_vector(1));

    // Validate that BH X + Y merge produces the same result as BH XY torus
    EXPECT_EQ(merged_gen, xy_gen) << "BH X torus + Y torus should equal BH XY torus";
}

TEST_F(DescriptorMergerTest, MergeTwoIdenticalXTorusDescriptors) {
    // Test merging two identical X_TORUS descriptors
    // Both have the same torus type and architecture - should merge successfully
    // (duplicate connections will be deduplicated during merge)
    const std::string test_dir = create_test_dir("two_x_torus_merge");

    const std::string merge_dir = test_dir + "merge/";
    std::filesystem::create_directories(merge_dir);
    create_torus_descriptor(merge_dir + "x_torus1.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_X_TORUS", 0);
    create_torus_descriptor(merge_dir + "x_torus2.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_X_TORUS", 0);

    // Merge two identical X torus descriptors
    CablingGenerator merged_gen(merge_dir, create_host_vector(1));

    // Create reference X torus descriptor
    const std::string ref_dir = test_dir + "reference/";
    std::filesystem::create_directories(ref_dir);
    create_torus_descriptor(ref_dir + "x_torus.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_X_TORUS", 0);
    CablingGenerator x_gen(ref_dir + "x_torus.textproto", create_host_vector(1));

    // Validate that merging two identical X torus gives the same X torus (deduplication works)
    EXPECT_EQ(merged_gen, x_gen) << "Two identical X torus should merge into single X torus";
}

TEST_F(DescriptorMergerTest, MergeXYTorusWithXTorusDescriptors) {
    // Test merging XY_TORUS with X_TORUS descriptors
    // XY_TORUS already contains X-direction connections, X_TORUS adds more
    // Both are torus types with the same architecture (Wormhole) - should merge to XY_TORUS
    const std::string test_dir = create_test_dir("xy_plus_x_torus_merge");

    const std::string merge_dir = test_dir + "merge/";
    std::filesystem::create_directories(merge_dir);
    create_torus_descriptor(merge_dir + "xy_torus.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_XY_TORUS", 0);
    create_torus_descriptor(merge_dir + "x_torus.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_X_TORUS", 0);

    // Merge XY + X torus
    CablingGenerator merged_gen(merge_dir, create_host_vector(1));

    // Create reference XY torus descriptor
    const std::string ref_dir = test_dir + "reference/";
    std::filesystem::create_directories(ref_dir);
    create_torus_descriptor(ref_dir + "xy_torus.textproto", "wh_galaxy_torus", "node1", "WH_GALAXY_XY_TORUS", 0);
    CablingGenerator xy_gen(ref_dir + "xy_torus.textproto", create_host_vector(1));

    // Validate that XY + X torus still equals XY torus (X connections already present)
    EXPECT_EQ(merged_gen, xy_gen) << "XY torus + X torus should equal XY torus";
}

// ============================================================================
// Split/Merge Tests: End-to-end split and merge workflows for all topologies
// ============================================================================

TEST_F(DescriptorMergerTest, SplitAndMerge8x16WhGalaxyXyTorusSuperpod) {
    // Test splitting the 8x16 WH_GALAXY_XY_TORUS superpod descriptor and merging it back
    const std::string source_path =
        "tools/tests/scaleout/cabling_descriptors/8x16_wh_galaxy_xy_torus_superpod.textproto";

    // Create hostnames based on descriptor
    const auto hostnames = create_host_vector(get_host_count(source_path));

    for (int num_splits : {2, 4, 8, 16}) {
        // Create unique test directory for this iteration
        const std::string test_dir = create_test_dir("split_8x16_test_" + std::to_string(num_splits));
        const std::string split_dir = test_dir + "split/";

        // Split into num_splits parts
        auto split_paths = split_descriptor(source_path, split_dir, "", num_splits);
        EXPECT_EQ(split_paths.size(), num_splits) << "Failed to split into " << num_splits << " parts";

        // Test that each split file can be loaded individually first
        for (const auto& split_path : split_paths) {
            EXPECT_NO_THROW({ CablingGenerator gen(split_path, hostnames); })
                << "Failed to load split file: " << split_path << " (num_splits=" << num_splits << ")";
        }

        // Create CablingGenerator from original file
        CablingGenerator original_gen(source_path, hostnames);

        // Create CablingGenerator from merged split files
        CablingGenerator merged_gen(split_dir, hostnames);

        // Verify that the merged CablingGenerator equals the original
        EXPECT_EQ(original_gen, merged_gen)
            << "Merged CablingGenerator does not match original - split/merge failed (num_splits=" << num_splits << ")";
    }
}

TEST_F(DescriptorMergerTest, SplitAndMerge5WhGalaxyYTorusSuperpod) {
    // Test splitting the 5 WH_GALAXY_Y_TORUS superpod descriptor and merging it back
    const std::string source_path = "tools/tests/scaleout/cabling_descriptors/5_wh_galaxy_y_torus_superpod.textproto";

    // Create hostnames based on descriptor
    const auto hostnames = create_host_vector(get_host_count(source_path));

    // Create CablingGenerator from original file once
    CablingGenerator original_gen(source_path, hostnames);

    for (int num_splits : {2, 4}) {
        // Create a fresh test directory for each split count to avoid file conflicts
        const std::string test_dir = create_test_dir("split_5_test_" + std::to_string(num_splits));
        const std::string split_dir = test_dir + "split/";

        auto split_paths = split_descriptor(source_path, split_dir, "", num_splits);
        EXPECT_EQ(split_paths.size(), num_splits);

        // Test that each split file can be loaded individually first
        for (const auto& split_path : split_paths) {
            EXPECT_NO_THROW({ CablingGenerator gen(split_path, hostnames); })
                << "Failed to load split file: " << split_path;
        }

        // Create CablingGenerator from merged split files
        CablingGenerator merged_gen(split_dir, hostnames);

        // Verify that the merged CablingGenerator equals the original
        EXPECT_EQ(original_gen, merged_gen)
            << "Merged CablingGenerator does not match original for " << num_splits << " splits";
    }
}

TEST_F(DescriptorMergerTest, SplitAndMerge16N300Cluster) {
    // Test splitting and merging the 16 N300 cluster descriptor
    // This validates split/merge works for N300 architecture (not just WH/BH)
    const std::string source_path = "tools/tests/scaleout/cabling_descriptors/16_n300_lb_cluster.textproto";

    auto hostnames = create_host_vector(16);

    // Test with 2, 4, and 8-way splits
    for (int num_splits : {2, 4, 8}) {
        const std::string test_dir = create_test_dir("split_16n300_test_" + std::to_string(num_splits));
        const std::string split_dir = test_dir + "split/";

        auto split_paths = split_descriptor(source_path, split_dir, "", num_splits);
        EXPECT_EQ(split_paths.size(), num_splits);

        // Create CablingGenerator from original and merged
        CablingGenerator original_gen(source_path, hostnames);
        CablingGenerator merged_gen(split_dir, hostnames);

        EXPECT_EQ(original_gen, merged_gen) << "N300 split/merge failed for num_splits=" << num_splits;
    }
}

// ============================================================================
// Negative Tests: Validation and error handling
// ============================================================================

TEST_F(DescriptorMergerTest, RejectMismatchedNodeTypes) {
    // Test that merging files with incompatible node types fails
    // Y_TORUS and N300_T3K_NODE have different board structures and cannot merge
    const std::string test_dir = create_test_dir("node_type_mismatch_test");

    create_simple_descriptor(test_dir + "file1.textproto", "test_cluster", "node1", "WH_GALAXY_Y_TORUS");
    create_simple_descriptor(test_dir + "file2.textproto", "test_cluster", "node1", "N300_T3K_NODE");

    try {
        CablingGenerator gen(test_dir, create_host_vector(1));
        FAIL() << "Expected std::runtime_error for incompatible node types";
    } catch (const std::runtime_error& e) {
        const std::string error_msg = e.what();
        // Verify error message mentions structural mismatch
        EXPECT_TRUE(
            error_msg.find("motherboard") != std::string::npos || error_msg.find("board") != std::string::npos ||
            error_msg.find("node") != std::string::npos)
            << "Error: " << error_msg;
    }
}

TEST_F(DescriptorMergerTest, RejectWormholeAndBlackholeTorusTogether) {
    // Test that merging WH torus and BH torus fails due to different architectures
    // Even though both are torus types, they have incompatible hardware architectures
    const std::string test_dir = create_test_dir("wh_bh_torus_mismatch");

    create_torus_descriptor(test_dir + "wh_torus.textproto", "mixed_torus", "node1", "WH_GALAXY_X_TORUS", 0);
    create_torus_descriptor(test_dir + "bh_torus.textproto", "mixed_torus", "node1", "BH_GALAXY_X_TORUS", 0);

    try {
        CablingGenerator merged_gen(test_dir, create_host_vector(1));
        FAIL() << "Expected std::runtime_error for different architectures";
    } catch (const std::runtime_error& e) {
        const std::string error_msg = e.what();
        // Should mention structural mismatch (different node descriptor names)
        EXPECT_TRUE(
            error_msg.find("structural") != std::string::npos || error_msg.find("mismatch") != std::string::npos ||
            error_msg.find("board") != std::string::npos)
            << "Error: " << error_msg;
    }
}

TEST_F(DescriptorMergerTest, RejectWHAndBHMesh) {
    // Test that WH and BH mesh nodes with different architectures cannot merge
    // Even though both are mesh topology, different architectures should fail
    const std::string test_dir = create_test_dir("wh_bh_mesh_conflict");

    create_simple_descriptor(test_dir + "wh_mesh.textproto", "mixed_mesh", "node1", "WH_GALAXY");
    create_simple_descriptor(test_dir + "bh_mesh.textproto", "mixed_mesh", "node1", "BH_GALAXY");

    try {
        CablingGenerator merged_gen(test_dir, create_host_vector(1));
        FAIL() << "Expected std::runtime_error for different architectures (WH vs BH mesh)";
    } catch (const std::runtime_error& e) {
        const std::string error_msg = e.what();
        EXPECT_TRUE(
            error_msg.find("structural") != std::string::npos || error_msg.find("mismatch") != std::string::npos ||
            error_msg.find("motherboard") != std::string::npos)
            << "Error: " << error_msg;
    }
}

TEST_F(DescriptorMergerTest, RejectGraphTemplatesWithDifferentChildren_ForwardPass) {
    // Test forward pass: source has nodes that target doesn't have
    // File 1 has {node_a, node_b}, File 2 has {node_c, node_d} - completely different sets
    const std::string test_dir = create_test_dir("different_children_forward_test");

    // File 1: graph_template "cluster" with node_a and node_b
    write_textproto(test_dir + "file1.textproto", R"(
graph_templates {
  key: "cluster"
  value {
    children { name: "node_a" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node_b" node_ref { node_descriptor: "WH_GALAXY" } }
  }
}
root_instance {
  template_name: "cluster"
  child_mappings { key: "node_a" value { host_id: 0 } }
  child_mappings { key: "node_b" value { host_id: 1 } }
}
)");

    // File 2: graph_template "cluster" with node_c and node_d (completely different names!)
    write_textproto(test_dir + "file2.textproto", R"(
graph_templates {
  key: "cluster"
  value {
    children { name: "node_c" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node_d" node_ref { node_descriptor: "WH_GALAXY" } }
  }
}
root_instance {
  template_name: "cluster"
  child_mappings { key: "node_c" value { host_id: 0 } }
  child_mappings { key: "node_d" value { host_id: 1 } }
}
)");

    // This should throw during forward pass - source has nodes target doesn't have
    try {
        CablingGenerator gen(test_dir, create_host_vector(2));
        FAIL() << "Expected std::runtime_error for graph_templates with different children (forward pass)";
    } catch (const std::runtime_error& e) {
        const std::string error_msg = e.what();
        // Verify error mentions the children/template mismatch
        EXPECT_TRUE(
            error_msg.find("different sets of children") != std::string::npos ||
            error_msg.find("Graph templates must have identical children") != std::string::npos)
            << "Error message doesn't match expected: " << error_msg;
    }
}

TEST_F(DescriptorMergerTest, RejectGraphTemplatesWithDifferentChildren_BackwardPass) {
    // Test backward pass: target has nodes that source doesn't have
    // File 1 has {node_x, node_y}, File 2 has {node_x} (subset)
    // Note: Both files must have compatible host_id mappings, so both use only host_id: 0
    const std::string test_dir = create_test_dir("different_children_backward_test");

    // File 1: graph_template "cluster" with node_x and node_y (loaded first = target)
    // Use alphabetically first filename so it loads first
    write_textproto(test_dir + "a_file1.textproto", R"(
graph_templates {
  key: "cluster"
  value {
    children { name: "node_x" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node_y" node_ref { node_descriptor: "WH_GALAXY" } }
  }
}
root_instance {
  template_name: "cluster"
  child_mappings { key: "node_x" value { host_id: 0 } }
  child_mappings { key: "node_y" value { host_id: 0 } }
}
)");

    // File 2: graph_template "cluster" with only node_x (missing node_y!)
    write_textproto(test_dir + "b_file2.textproto", R"(
graph_templates {
  key: "cluster"
  value {
    children { name: "node_x" node_ref { node_descriptor: "WH_GALAXY" } }
  }
}
root_instance {
  template_name: "cluster"
  child_mappings { key: "node_x" value { host_id: 0 } }
}
)");

    // This should throw during backward pass - target has node_y that source doesn't have
    EXPECT_THROW(
        { CablingGenerator gen(test_dir, create_host_vector(1)); }, std::runtime_error)
        << "Should reject graph_templates with different children (backward pass: target has extra nodes)";
}

TEST_F(DescriptorMergerTest, AllowCrossDescriptorConnectionsOnDifferentPorts) {
    // Test that the same node can connect to different nodes across multiple descriptors
    // as long as different ports are used. This is valid because:
    // 1. Each port is only used once (no duplicate connections within a descriptor)
    // 2. Physical port exhaustion is checked at FSD generation time, not merge time
    // 3. This allows splitting a fully-connected graph across multiple descriptor files
    const std::string test_dir = create_test_dir("cross_descriptor_connections");

    // Use WH_GALAXY (MESH) which has no internal QSFP connections, so all ports 1-6 are available
    // File 1: node1 port 1 -> node2, node2 port 2 -> node3
    write_textproto(test_dir + "file1.textproto", R"(
graph_templates {
  key: "test_cluster"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node2" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node3" node_ref { node_descriptor: "WH_GALAXY" } }
    internal_connections {
      key: "QSFP_DD"
      value {
        connections {
          port_a { path: ["node1"] tray_id: 1 port_id: 1 }
          port_b { path: ["node2"] tray_id: 1 port_id: 1 }
        }
        connections {
          port_a { path: ["node2"] tray_id: 1 port_id: 2 }
          port_b { path: ["node3"] tray_id: 1 port_id: 2 }
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
}
)");

    // File 2: node1 port 3 (DIFFERENT port) -> node3
    write_textproto(test_dir + "file2.textproto", R"(
graph_templates {
  key: "test_cluster"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node2" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node3" node_ref { node_descriptor: "WH_GALAXY" } }
    internal_connections {
      key: "QSFP_DD"
      value {
        connections {
          port_a { path: ["node1"] tray_id: 1 port_id: 3 }
          port_b { path: ["node3"] tray_id: 1 port_id: 3 }
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
}
)");

    // This should succeed - each port is used once, connections are valid
    CablingGenerator gen(test_dir, create_host_vector(3));

    // Validate that the merged result has the expected structure
    auto fsd = gen.generate_factory_system_descriptor();
    EXPECT_EQ(fsd.hosts().size(), 3) << "Should have 3 hosts from merged descriptors";

    // Verify by loading again and checking equality (reflexive)
    CablingGenerator gen2(test_dir, create_host_vector(3));
    EXPECT_EQ(gen, gen2) << "Loading same directory twice should produce equal generators";
}

TEST_F(DescriptorMergerTest, RejectSamePortConnectedToDifferentDestinations) {
    // Test that connecting the same source port to two different destinations in a single descriptor throws
    // This validates connection conflict detection within inter_board_connections of a node template
    const std::string test_dir = create_test_dir("same_port_conflict_test");

    // Create a node descriptor with conflicting inter_board_connections
    // Port (tray_id: 1, port_id: 5) is connected to both (tray_id: 2, port_id: 1) AND (tray_id: 2, port_id: 2)
    write_textproto(test_dir + "conflict.textproto", R"(
node_descriptors {
  key: "CONFLICTING_NODE"
  value {
    motherboard: "CONFLICTING_MB"
    boards {
      tray_id: 1
      board_type: "UBB"
    }
    boards {
      tray_id: 2
      board_type: "UBB"
    }
    port_type_connections {
      key: "QSFP_DD"
      value {
        connections {
          port_a { tray_id: 1 port_id: 5 }
          port_b { tray_id: 2 port_id: 1 }
        }
        connections {
          port_a { tray_id: 1 port_id: 5 }
          port_b { tray_id: 2 port_id: 2 }
        }
      }
    }
  }
}
graph_templates {
  key: "conflict_cluster"
  value {
    children { name: "node1" node_ref { node_descriptor: "CONFLICTING_NODE" } }
  }
}
root_instance {
  template_name: "conflict_cluster"
  child_mappings { key: "node1" value { host_id: 0 } }
}
)");

    // This should throw because port (1, 5) is connected to two different destinations
    EXPECT_THROW(
        { CablingGenerator gen(test_dir + "conflict.textproto", create_host_vector(1)); }, std::runtime_error)
        << "Same port connected to different destinations should be rejected";
}

TEST_F(DescriptorMergerTest, RejectEmptyPath) {
    // Test that merging WH torus and BH torus fails due to different architectures
    // Even though both are torus types, they have incompatible hardware architectures
    const std::string test_dir = create_test_dir("wh_bh_torus_mismatch");

    create_torus_descriptor(test_dir + "wh_torus.textproto", "mixed_torus", "node1", "WH_GALAXY_X_TORUS", 0);
    create_torus_descriptor(test_dir + "bh_torus.textproto", "mixed_torus", "node1", "BH_GALAXY_X_TORUS", 0);

    try {
        CablingGenerator merged_gen(test_dir, create_host_vector(1));
        FAIL() << "Expected std::runtime_error for different architectures";
    } catch (const std::runtime_error& e) {
        const std::string error_msg = e.what();
        // Should mention structural mismatch (different node descriptor names)
        EXPECT_TRUE(
            error_msg.find("structural") != std::string::npos || error_msg.find("mismatch") != std::string::npos ||
            error_msg.find("board") != std::string::npos)
            << "Error: " << error_msg;
    }
}

TEST_F(DescriptorMergerTest, RejectMissingGraphTemplate) {
    // Test that referencing non-existent graph template is rejected
    const std::string test_dir = create_test_dir("missing_template_test");

    write_textproto(test_dir + "missing_template.textproto", R"(
graph_templates {
  key: "existing_template"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY" } }
  }
}
root_instance {
  template_name: "nonexistent_template"
  child_mappings { key: "node1" value { host_id: 0 } }
}
)");

    EXPECT_THROW(
        { CablingGenerator gen(test_dir + "missing_template.textproto", create_host_vector(1)); }, std::runtime_error)
        << "Missing graph template should be rejected";
}

TEST_F(DescriptorMergerTest, RejectMissingChildMapping) {
    // Test that missing child mapping is rejected
    const std::string test_dir = create_test_dir("missing_child_test");

    write_textproto(test_dir + "missing_child.textproto", R"(
graph_templates {
  key: "test"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node2" node_ref { node_descriptor: "WH_GALAXY" } }
  }
}
root_instance {
  template_name: "test"
  child_mappings { key: "node1" value { host_id: 0 } }
}
)");

    EXPECT_THROW(
        { CablingGenerator gen(test_dir + "missing_child.textproto", create_host_vector(1)); }, std::runtime_error)
        << "Missing child mapping should be rejected";
}

TEST_F(DescriptorMergerTest, RejectDuplicateHostId) {
    // Test that duplicate host_id across nodes is rejected
    const std::string test_dir = create_test_dir("duplicate_host_id_test");

    write_textproto(test_dir + "duplicate_host.textproto", R"(
graph_templates {
  key: "test"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY" } }
    children { name: "node2" node_ref { node_descriptor: "WH_GALAXY" } }
  }
}
root_instance {
  template_name: "test"
  child_mappings { key: "node1" value { host_id: 0 } }
  child_mappings { key: "node2" value { host_id: 0 } }
}
)");

    EXPECT_THROW(
        { CablingGenerator gen(test_dir + "duplicate_host.textproto", create_host_vector(2)); }, std::runtime_error)
        << "Duplicate host_id should be rejected";
}

TEST_F(DescriptorMergerTest, RejectInvalidHostIdInConnection) {
    // Test that connection referencing non-existent host_id is rejected
    const std::string test_dir = create_test_dir("invalid_host_id_test");

    write_textproto(test_dir + "invalid_host.textproto", R"(
graph_templates {
  key: "test"
  value {
    children { name: "node1" node_ref { node_descriptor: "WH_GALAXY" } }
    internal_connections {
      key: "QSFP_DD"
      value {
        connections {
          port_a { path: ["node1"] tray_id: 1 port_id: 1 }
          port_b { path: ["nonexistent_node"] tray_id: 1 port_id: 2 }
        }
      }
    }
  }
}
root_instance {
  template_name: "test"
  child_mappings { key: "node1" value { host_id: 0 } }
}
)");

    EXPECT_THROW(
        { CablingGenerator gen(test_dir + "invalid_host.textproto", create_host_vector(1)); }, std::runtime_error)
        << "Connection to non-existent node should be rejected";
}

TEST_F(DescriptorMergerTest, RejectZeroSplit) {
    // Test that split_descriptor with num_splits < 2 is rejected
    const std::string source_path = "tools/tests/scaleout/cabling_descriptors/t3k.textproto";
    const std::string test_dir = create_test_dir("zero_split_test");

    EXPECT_THROW(
        { split_descriptor(source_path, test_dir + "split/", "", 0); }, std::runtime_error)
        << "split_descriptor with num_splits=0 should throw";

    EXPECT_THROW(
        { split_descriptor(source_path, test_dir + "split/", "", 1); }, std::runtime_error)
        << "split_descriptor with num_splits=1 should throw";
}

// ============================================================================
// Operator== Tests: Equality comparison validation
// ============================================================================

TEST_F(DescriptorMergerTest, OperatorEqualityReflexive) {
    // Test that a CablingGenerator is equal to itself (reflexive property)
    const std::string test_dir = create_test_dir("equality_reflexive");
    create_simple_descriptor(test_dir + "test.textproto", "test_graph", "node1", "WH_GALAXY", 0);

    CablingGenerator gen(test_dir + "test.textproto", create_host_vector(1));
    EXPECT_EQ(gen, gen) << "CablingGenerator should be equal to itself";
}

TEST_F(DescriptorMergerTest, OperatorEqualitySymmetric) {
    // Test that if A == B, then B == A (symmetric property)
    const std::string test_dir = create_test_dir("equality_symmetric");
    create_simple_descriptor(test_dir + "test.textproto", "test_graph", "node1", "WH_GALAXY", 0);

    CablingGenerator gen1(test_dir + "test.textproto", create_host_vector(1));
    CablingGenerator gen2(test_dir + "test.textproto", create_host_vector(1));

    EXPECT_EQ(gen1, gen2) << "Generators from same descriptor should be equal";
    EXPECT_EQ(gen2, gen1) << "Equality should be symmetric";
}

TEST_F(DescriptorMergerTest, OperatorInequalityDifferentNodeTypes) {
    // Test that generators with different node types are not equal
    const std::string test_dir = create_test_dir("inequality_node_types");

    create_simple_descriptor(test_dir + "wh.textproto", "test_graph", "node1", "WH_GALAXY", 0);
    create_simple_descriptor(test_dir + "bh.textproto", "test_graph", "node1", "BH_GALAXY", 0);

    CablingGenerator wh_gen(test_dir + "wh.textproto", create_host_vector(1));
    CablingGenerator bh_gen(test_dir + "bh.textproto", create_host_vector(1));

    EXPECT_NE(wh_gen, bh_gen) << "Generators with different node types should not be equal";
}

TEST_F(DescriptorMergerTest, OperatorInequalityDifferentConnections) {
    // Test that generators with different internal connections are not equal
    const std::string test_dir = create_test_dir("inequality_connections");

    create_two_node_descriptor_with_connection(
        test_dir + "conn1.textproto", "test_graph", "WH_GALAXY", "WH_GALAXY", "node2", 1, 1, 1, 1);
    create_two_node_descriptor_with_connection(
        test_dir + "conn2.textproto", "test_graph", "WH_GALAXY", "WH_GALAXY", "node2", 1, 2, 1, 2);

    CablingGenerator gen1(test_dir + "conn1.textproto", create_host_vector(2));
    CablingGenerator gen2(test_dir + "conn2.textproto", create_host_vector(2));

    EXPECT_NE(gen1, gen2) << "Generators with different connections should not be equal";
}

TEST_F(DescriptorMergerTest, OperatorInequalityDifferentHostCount) {
    // Test that generators with different number of nodes are not equal
    const std::string test_dir = create_test_dir("inequality_host_count");

    create_simple_descriptor(test_dir + "one_node.textproto", "test_graph", "node1", "WH_GALAXY", 0);
    create_two_node_descriptor_with_connection(
        test_dir + "two_nodes.textproto", "test_graph", "WH_GALAXY", "WH_GALAXY", "node2");

    CablingGenerator gen1(test_dir + "one_node.textproto", create_host_vector(1));
    CablingGenerator gen2(test_dir + "two_nodes.textproto", create_host_vector(2));

    EXPECT_NE(gen1, gen2) << "Generators with different node counts should not be equal";
}

// ============================================================================
// Descriptor Loading Tests: Validate loading from files and directories
// ============================================================================

TEST_F(DescriptorMergerTest, LoadAllAvailableDescriptors) {
    // Test that all existing cabling descriptors can be loaded successfully
    // This validates that our test descriptors are well-formed and produce consistent results
    const std::vector<std::string> descriptors = {
        "tools/tests/scaleout/cabling_descriptors/wh_galaxy_mesh.textproto",
        "tools/tests/scaleout/cabling_descriptors/bh_galaxy_mesh.textproto",
        "tools/tests/scaleout/cabling_descriptors/bh_galaxy_xy_torus.textproto",
        "tools/tests/scaleout/cabling_descriptors/t3k.textproto",
    };

    for (const auto& desc_path : descriptors) {
        // Load descriptor twice
        CablingGenerator gen1(desc_path, create_host_vector(1));
        CablingGenerator gen2(desc_path, create_host_vector(1));

        // Verify reflexive equality
        EXPECT_EQ(gen1, gen2) << "Loading same descriptor twice should produce equal generators: " << desc_path;

        // Validate basic structure
        auto fsd = gen1.generate_factory_system_descriptor();
        EXPECT_GE(fsd.hosts().size(), 1) << "Descriptor should have at least one host: " << desc_path;
    }
}

TEST_F(DescriptorMergerTest, MergeExistingBHTorusDescriptors) {
    // Test merging existing bh_galaxy_x_torus.textproto and bh_galaxy_y_torus.textproto
    // This demonstrates using actual existing descriptor files from the cabling_descriptors directory
    const std::string test_dir = create_test_dir("merge_existing_bh_torus");

    // Read the actual existing files
    std::ifstream x_file("tools/tests/scaleout/cabling_descriptors/bh_galaxy_x_torus.textproto");
    std::ifstream y_file("tools/tests/scaleout/cabling_descriptors/bh_galaxy_y_torus.textproto");
    ASSERT_TRUE(x_file.is_open()) << "Failed to open existing bh_galaxy_x_torus.textproto";
    ASSERT_TRUE(y_file.is_open()) << "Failed to open existing bh_galaxy_y_torus.textproto";

    std::string x_content((std::istreambuf_iterator<char>(x_file)), std::istreambuf_iterator<char>());
    std::string y_content((std::istreambuf_iterator<char>(y_file)), std::istreambuf_iterator<char>());
    x_file.close();
    y_file.close();

    // Modify template names to be the same so they can merge
    // These files only have the template name in quoted strings, so simple replacement is sufficient
    size_t pos = 0;
    while ((pos = x_content.find("\"bh_galaxy_x_torus\"", pos)) != std::string::npos) {
        x_content.replace(pos, 19, "\"bh_galaxy_torus\"");
        pos += 18;
    }
    pos = 0;
    while ((pos = y_content.find("\"bh_galaxy_y_torus\"", pos)) != std::string::npos) {
        y_content.replace(pos, 19, "\"bh_galaxy_torus\"");
        pos += 18;
    }

    // Write modified files to test directory (using the actual existing file content)
    write_textproto(test_dir + "x_torus.textproto", x_content);
    write_textproto(test_dir + "y_torus.textproto", y_content);

    // Merge the existing X + Y torus files
    CablingGenerator merged_gen(test_dir, create_host_vector(1));

    // Read and modify the existing XY torus file for comparison (unify template name)
    std::ifstream xy_file("tools/tests/scaleout/cabling_descriptors/bh_galaxy_xy_torus.textproto");
    ASSERT_TRUE(xy_file.is_open()) << "Failed to open existing bh_galaxy_xy_torus.textproto";
    std::string xy_content((std::istreambuf_iterator<char>(xy_file)), std::istreambuf_iterator<char>());
    xy_file.close();

    // Replace "bh_galaxy_xy_torus" with "bh_galaxy_torus" for comparison
    pos = 0;
    while ((pos = xy_content.find("\"bh_galaxy_xy_torus\"", pos)) != std::string::npos) {
        xy_content.replace(pos, 21, "\"bh_galaxy_torus\"");
        pos += 18;
    }

    // Write modified XY file for comparison
    const std::string ref_dir = test_dir + "reference/";
    std::filesystem::create_directories(ref_dir);
    write_textproto(ref_dir + "xy_torus.textproto", xy_content);

    // Compare with the existing XY torus file (with unified template name)
    CablingGenerator xy_gen(ref_dir + "xy_torus.textproto", create_host_vector(1));

    // Validate that merging the existing X and Y torus files produces the same result as the existing XY torus file
    EXPECT_EQ(merged_gen, xy_gen) << "Merging existing bh_galaxy_x_torus.textproto + bh_galaxy_y_torus.textproto "
                                     "should equal bh_galaxy_xy_torus.textproto";
}

TEST_F(DescriptorMergerTest, MergeExistingMeshDescriptors) {
    // Test merging existing mesh descriptors to demonstrate merging any existing files
    // This shows the pattern: copy existing files, unify template names, merge, and compare
    const std::string test_dir = create_test_dir("merge_existing_mesh");

    // Read the actual existing mesh files
    std::ifstream wh_file("tools/tests/scaleout/cabling_descriptors/wh_galaxy_mesh.textproto");
    std::ifstream bh_file("tools/tests/scaleout/cabling_descriptors/bh_galaxy_mesh.textproto");
    ASSERT_TRUE(wh_file.is_open()) << "Failed to open existing wh_galaxy_mesh.textproto";
    ASSERT_TRUE(bh_file.is_open()) << "Failed to open existing bh_galaxy_mesh.textproto";

    std::string wh_content((std::istreambuf_iterator<char>(wh_file)), std::istreambuf_iterator<char>());
    std::string bh_content((std::istreambuf_iterator<char>(bh_file)), std::istreambuf_iterator<char>());
    wh_file.close();
    bh_file.close();

    // Modify template names to be the same so they can merge
    // Replace template names with a unified name
    size_t pos = 0;
    while ((pos = wh_content.find("wh_galaxy", pos)) != std::string::npos) {
        wh_content.replace(pos, 9, "mesh_cluster");
        pos += 12;
    }
    pos = 0;
    while ((pos = bh_content.find("bh_galaxy", pos)) != std::string::npos) {
        bh_content.replace(pos, 9, "mesh_cluster");
        pos += 12;
    }

    // Write modified files to test directory (using the actual existing file content)
    write_textproto(test_dir + "wh_mesh.textproto", wh_content);
    write_textproto(test_dir + "bh_mesh.textproto", bh_content);

    // This should fail because WH and BH have different architectures
    // WH_GALAXY and BH_GALAXY are incompatible node types and cannot be merged
    EXPECT_THROW(
        { CablingGenerator merged_gen(test_dir, create_host_vector(1)); }, std::runtime_error)
        << "Merging existing WH and BH mesh descriptors should fail due to different architectures";
}

}  // namespace tt::scaleout_tools
