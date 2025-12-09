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

    // Helper function to split a descriptor's internal connections into two parts
    // Returns pair of (part1_path, part2_path)
    std::pair<std::string, std::string> split_descriptor(
        const std::string& source_path, const std::string& output_dir, const std::string& template_name = "") {
        // Load original descriptor
        std::ifstream file(source_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open " + source_path);
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        cabling_generator::proto::ClusterDescriptor original_desc;
        if (!google::protobuf::TextFormat::ParseFromString(content, &original_desc)) {
            throw std::runtime_error("Failed to parse " + source_path);
        }

        // Determine which template to split (use first one if not specified)
        std::string target_template = template_name;
        if (target_template.empty()) {
            if (original_desc.graph_templates().empty()) {
                throw std::runtime_error("No graph templates in descriptor");
            }
            target_template = original_desc.graph_templates().begin()->first;
        }

        if (!original_desc.graph_templates().contains(target_template)) {
            throw std::runtime_error("Template '" + target_template + "' not found");
        }

        // Create two copies
        cabling_generator::proto::ClusterDescriptor part1 = original_desc;
        cabling_generator::proto::ClusterDescriptor part2 = original_desc;

        // Split internal connections
        auto* template1 = part1.mutable_graph_templates()->at(target_template).mutable_internal_connections();
        auto* template2 = part2.mutable_graph_templates()->at(target_template).mutable_internal_connections();

        // Find first port type with connections
        std::string port_type_key;
        int total_conns = 0;
        for (const auto& [key, conns] : *template1) {
            if (conns.connections_size() > 0) {
                port_type_key = key;
                total_conns = conns.connections_size();
                break;
            }
        }

        if (total_conns == 0) {
            throw std::runtime_error("No connections found to split");
        }

        auto& qsfp_conns1 = (*template1)[port_type_key];
        auto& qsfp_conns2 = (*template2)[port_type_key];

        int half = total_conns / 2;

        // Copy connections and split
        auto conns_copy = qsfp_conns1.connections();
        qsfp_conns1.clear_connections();
        qsfp_conns2.clear_connections();

        for (int i = 0; i < half; i++) {
            *qsfp_conns1.add_connections() = conns_copy[i];
        }
        for (int i = half; i < total_conns; i++) {
            *qsfp_conns2.add_connections() = conns_copy[i];
        }

        // Write to files
        std::filesystem::create_directories(output_dir);

        std::string part1_path = output_dir + "/part1.textproto";
        std::string part2_path = output_dir + "/part2.textproto";

        std::string part1_str, part2_str;
        google::protobuf::TextFormat::PrintToString(part1, &part1_str);
        google::protobuf::TextFormat::PrintToString(part2, &part2_str);

        std::ofstream(part1_path) << part1_str;
        std::ofstream(part2_path) << part2_str;

        return {part1_path, part2_path};
    }
};

TEST_F(DescriptorMergerTest, FindDescriptorFilesInDirectory) {
    auto files = DescriptorMerger::find_descriptor_files(test_fixtures_dir);

    EXPECT_GE(files.size(), 4);
    for (const auto& file : files) {
        EXPECT_TRUE(file.ends_with(".textproto"));
    }

    auto sorted_files = files;
    std::sort(sorted_files.begin(), sorted_files.end());
    EXPECT_EQ(files, sorted_files);
}

TEST_F(DescriptorMergerTest, FindDescriptorFilesEmptyDirectory) {
    std::string empty_dir = "generated/tests/empty_merge_test_dir/";
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
    std::vector<std::string> paths = {fixture_path("base_intrapod.textproto")};
    auto merged = DescriptorMerger::merge_descriptors(paths);

    EXPECT_TRUE(merged.graph_templates().contains("test_pod"));
    EXPECT_EQ(merged.graph_templates().at("test_pod").children().size(), 4);
    EXPECT_TRUE(merged.has_root_instance());
    EXPECT_EQ(merged.root_instance().template_name(), "test_pod");
}

TEST_F(DescriptorMergerTest, MergeComplementaryDescriptors) {
    std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("additional_interpod.textproto")};
    auto merged = DescriptorMerger::merge_descriptors(paths);

    EXPECT_TRUE(merged.graph_templates().contains("test_pod"));
    const auto& template_def = merged.graph_templates().at("test_pod");
    EXPECT_TRUE(template_def.internal_connections().contains("QSFP_DD"));
    EXPECT_EQ(template_def.internal_connections().at("QSFP_DD").connections().size(), 4);
}

TEST_F(DescriptorMergerTest, MergeDifferentTemplates) {
    std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("different_template.textproto")};
    auto merged = DescriptorMerger::merge_descriptors(paths);

    EXPECT_TRUE(merged.graph_templates().contains("test_pod"));
    EXPECT_TRUE(merged.graph_templates().contains("test_superpod"));
}

TEST_F(DescriptorMergerTest, DetectConflictingConnections) {
    std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("conflicting_connection.textproto")};

    EXPECT_THROW(
        {
            try {
                DescriptorMerger::merge_descriptors(paths);
            } catch (const std::runtime_error& e) {
                std::string error_msg = e.what();
                EXPECT_TRUE(
                    error_msg.find("conflict") != std::string::npos || error_msg.find("Conflict") != std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(DescriptorMergerTest, HandleDuplicateConnections) {
    std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("duplicate_connection.textproto")};
    auto merged = DescriptorMerger::merge_descriptors(paths);

    const auto& connections = merged.graph_templates().at("test_pod").internal_connections().at("QSFP_DD");
    EXPECT_EQ(connections.connections().size(), 3);
}

TEST_F(DescriptorMergerTest, ValidateHostConsistencySameHosts) {
    std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("additional_interpod.textproto")};
    auto validation = DescriptorMerger::validate_host_consistency(paths);

    EXPECT_TRUE(validation.success);
    bool has_mismatch = std::any_of(validation.warnings.begin(), validation.warnings.end(), [](const std::string& w) {
        return w.find("Host count mismatch") != std::string::npos;
    });
    EXPECT_FALSE(has_mismatch);
}

TEST_F(DescriptorMergerTest, ValidateHostConsistencyWithTemplateOnlyDescriptor) {
    std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("different_template.textproto")};
    auto validation = DescriptorMerger::validate_host_consistency(paths);

    EXPECT_TRUE(validation.success);
}

TEST_F(DescriptorMergerTest, ValidateHostConsistencyDifferentHostCounts) {
    std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"), fixture_path("different_host_count.textproto")};
    auto validation = DescriptorMerger::validate_host_consistency(paths);

    EXPECT_TRUE(validation.success);
    EXPECT_FALSE(validation.warnings.empty());
}

TEST_F(DescriptorMergerTest, ConnectionEndpointComparison) {
    ConnectionEndpoint a{"template1", {"node1"}, 0, 0};
    ConnectionEndpoint b{"template1", {"node1"}, 0, 0};
    ConnectionEndpoint c{"template1", {"node1"}, 0, 1};
    ConnectionEndpoint d{"template1", {"node2"}, 0, 0};

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_FALSE(a == d);
    EXPECT_FALSE(a < b);
    EXPECT_TRUE(a < c);
}

TEST_F(DescriptorMergerTest, ConnectionPairEquality) {
    ConnectionEndpoint a{"t", {"n1"}, 0, 0};
    ConnectionEndpoint b{"t", {"n2"}, 0, 0};

    ConnectionPair pair1{a, b, "QSFP_DD"};
    ConnectionPair pair2{b, a, "QSFP_DD"};
    ConnectionPair pair3{a, b, "TRACE"};

    EXPECT_TRUE(pair1 == pair2);
    EXPECT_FALSE(pair1 == pair3);
}

TEST_F(DescriptorMergerTest, EmptyPathsThrows) {
    std::vector<std::string> empty_paths;
    EXPECT_THROW(DescriptorMerger::merge_descriptors(empty_paths), std::runtime_error);
}

TEST_F(DescriptorMergerTest, NonexistentFileThrows) {
    std::vector<std::string> paths = {"nonexistent_file.textproto"};
    EXPECT_THROW(DescriptorMerger::merge_descriptors(paths), std::runtime_error);
}

TEST_F(DescriptorMergerTest, MergeFromDirectoryExcludingConflicts) {
    std::vector<std::string> paths = {
        fixture_path("base_intrapod.textproto"),
        fixture_path("additional_interpod.textproto"),
        fixture_path("different_template.textproto")};
    auto merged = DescriptorMerger::merge_descriptors(paths);

    EXPECT_TRUE(merged.graph_templates().contains("test_pod"));
    EXPECT_TRUE(merged.graph_templates().contains("test_superpod"));
    EXPECT_EQ(merged.graph_templates().at("test_pod").internal_connections().at("QSFP_DD").connections().size(), 4);
    EXPECT_EQ(
        merged.graph_templates().at("test_superpod").internal_connections().at("QSFP_DD").connections().size(), 1);
}

TEST_F(DescriptorMergerTest, MergeSplitDescriptorProducesSameFSD) {
    // Load the original 8x16 superpod descriptor
    std::string original_path = "tools/tests/scaleout/cabling_descriptors/8x16_wh_galaxy_xy_torus_superpod.textproto";
    std::ifstream file(original_path);
    ASSERT_TRUE(file.is_open()) << "Failed to open " << original_path;
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    cabling_generator::proto::ClusterDescriptor original_desc;
    ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(content, &original_desc));

    int total_conns = original_desc.graph_templates()
                          .at("8x16_wh_galaxy_xy_torus_superpod")
                          .internal_connections()
                          .at("QSFP_DD")
                          .connections_size();

    // Split descriptor using helper function
    std::string temp_dir = "generated/tests/split_merge_test/";
    std::string cabling_dir = temp_dir + "cabling/";

    auto [part1_path, part2_path] = split_descriptor(original_path, cabling_dir, "8x16_wh_galaxy_xy_torus_superpod");

    // Merge the split descriptors
    auto merged = DescriptorMerger::merge_descriptors({part1_path, part2_path});

    // Verify merged has same number of connections as original
    ASSERT_TRUE(merged.graph_templates().contains("8x16_wh_galaxy_xy_torus_superpod"));
    const auto& merged_template = merged.graph_templates().at("8x16_wh_galaxy_xy_torus_superpod");
    ASSERT_TRUE(merged_template.internal_connections().contains("QSFP_DD"));

    int merged_conn_count = merged_template.internal_connections().at("QSFP_DD").connections_size();
    EXPECT_EQ(merged_conn_count, total_conns) << "Merged descriptor should have same number of connections as original";

    // Generate FSDs for both original and merged
    std::string deployment_path = temp_dir + "deployment.textproto";
    std::ofstream deployment_file(deployment_path);
    deployment_file << R"(
rack_capacity: 1
hosts { hall: "0" aisle: "0" rack: 0 shelf_u: 0 node_type: "WH_GALAXY_Y_TORUS" host: "h0" }
hosts { hall: "0" aisle: "0" rack: 0 shelf_u: 1 node_type: "WH_GALAXY_Y_TORUS" host: "h1" }
hosts { hall: "0" aisle: "0" rack: 0 shelf_u: 2 node_type: "WH_GALAXY_Y_TORUS" host: "h2" }
hosts { hall: "0" aisle: "0" rack: 0 shelf_u: 3 node_type: "WH_GALAXY_Y_TORUS" host: "h3" }
)";
    deployment_file.close();

    // Write original descriptor to file
    std::string original_temp_path = temp_dir + "original.textproto";
    std::ofstream(original_temp_path) << content;

    // Generate FSD from original
    CablingGenerator gen_original(original_temp_path, deployment_path);
    auto fsd_original = gen_original.generate_factory_system_descriptor();

    // Generate FSD from merged (use cabling directory not temp_dir)
    CablingGenerator gen_merged(cabling_dir, deployment_path);
    auto fsd_merged = gen_merged.generate_factory_system_descriptor();

    // Compare FSDs - they should be identical
    EXPECT_EQ(fsd_original.hosts_size(), fsd_merged.hosts_size());

    // Count ethernet connections in both
    int original_eth_conns = fsd_original.eth_connections().connection_size();
    int merged_eth_conns = fsd_merged.eth_connections().connection_size();

    EXPECT_EQ(original_eth_conns, merged_eth_conns)
        << "FSD from merged should have same ethernet connections as original";
    EXPECT_GT(original_eth_conns, 0) << "Should have at least some ethernet connections";

    // Cleanup
    std::filesystem::remove_all(temp_dir);
}

}  // namespace tt::scaleout_tools
