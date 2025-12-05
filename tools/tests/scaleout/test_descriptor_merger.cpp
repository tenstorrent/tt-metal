// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

#include <cabling_generator/descriptor_merger.hpp>
#include "protobuf/cluster_config.pb.h"

namespace tt::scaleout_tools {

static const std::string test_fixtures_dir = "tools/tests/scaleout/cabling_descriptors/merge_tests/";

class DescriptorMergerTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(std::filesystem::exists(test_fixtures_dir))
            << "Test fixtures directory not found: " << test_fixtures_dir;
    }

    std::string fixture_path(const std::string& filename) const { return test_fixtures_dir + filename; }
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

TEST_F(DescriptorMergerTest, ValidateMergedDescriptor) {
    std::vector<std::string> paths = {fixture_path("base_intrapod.textproto")};
    auto merged = DescriptorMerger::merge_descriptors(paths);
    auto validation = DescriptorMerger::validate_merged_descriptor(merged);

    EXPECT_TRUE(validation.success);
    EXPECT_TRUE(validation.errors.empty());
}

TEST_F(DescriptorMergerTest, GetMergeStatistics) {
    std::vector<std::string> paths = {fixture_path("base_intrapod.textproto")};
    auto merged = DescriptorMerger::merge_descriptors(paths);
    auto stats = DescriptorMerger::get_merge_statistics(merged);

    EXPECT_EQ(stats.total_graph_templates, 1);
    EXPECT_EQ(stats.total_connections, 2);
    EXPECT_EQ(stats.host_ids_found.size(), 4);
    EXPECT_EQ(stats.expected_host_count(), 4);
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

}  // namespace tt::scaleout_tools
