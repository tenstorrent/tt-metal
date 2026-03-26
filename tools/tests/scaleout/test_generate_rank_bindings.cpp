// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <yaml-cpp/yaml.h>

#include "generate_rank_bindings_helpers.hpp"

namespace {

std::filesystem::path make_temp_dir(const std::string& test_name) {
    auto base = std::filesystem::temp_directory_path() / ("grb_test_" + test_name);
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(base);
    return base;
}

RankBindingConfig make_binding(
    int rank, int mesh_id, int mesh_host_rank, std::string hostname, int slot, int psd_mpi_rank = 0) {
    RankBindingConfig b;
    b.rank = rank;
    b.mesh_id = mesh_id;
    b.mesh_host_rank = mesh_host_rank;
    b.hostname = std::move(hostname);
    b.slot = slot;
    b.psd_mpi_rank = psd_mpi_rank;
    return b;
}

}  // namespace

TEST(GenerateRankBindingsHelpersTest, WriteRankBindingsYaml_ContainsMeshGraphPathAndRanks) {
    const auto dir = make_temp_dir("yaml");
    const auto path = dir / "rank_bindings.yaml";
    std::vector<RankBindingConfig> bindings = {
        make_binding(0, 0, 0, "h0", 0),
        make_binding(1, 1, 0, "h1", 0),
    };
    bindings[0].env_overrides["TT_VISIBLE_DEVICES"] = "0,1";
    bindings[1].env_overrides["TT_VISIBLE_DEVICES"] = "2";

    const std::string mgd = "/path/to/mesh.textproto";
    write_rank_bindings_yaml(bindings, mgd, path.string());

    ASSERT_TRUE(std::filesystem::exists(path));
    const YAML::Node root = YAML::LoadFile(path.string());
    EXPECT_EQ(root["mesh_graph_desc_path"].as<std::string>(), mgd);
    ASSERT_TRUE(root["rank_bindings"]);
    EXPECT_EQ(root["rank_bindings"].size(), 2u);
    EXPECT_EQ(root["rank_bindings"][0]["rank"].as<int>(), 0);
    EXPECT_EQ(root["rank_bindings"][1]["rank"].as<int>(), 1);
    EXPECT_EQ(root["rank_bindings"][0]["env_overrides"]["TT_VISIBLE_DEVICES"].as<std::string>(), "0,1");
}

TEST(GenerateRankBindingsHelpersTest, WriteRankfile_SortsByRankAscending) {
    const auto dir = make_temp_dir("rankfile");
    const auto path = dir / "rankfile";
    std::vector<RankBindingConfig> bindings = {
        make_binding(2, 0, 0, "host-b", 1),
        make_binding(0, 0, 0, "host-a", 0),
        make_binding(1, 0, 0, "host-c", 0),
    };

    write_rankfile(bindings, path.string(), false);

    std::ifstream in(path.string());
    std::string line;
    std::vector<int> ranks;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        ASSERT_EQ(line.rfind("rank ", 0), 0u);
        const auto eq = line.find('=');
        ASSERT_NE(eq, std::string::npos);
        ranks.push_back(std::stoi(line.substr(5, eq - 5)));
    }
    ASSERT_EQ(ranks.size(), 3u);
    EXPECT_EQ(ranks[0], 0);
    EXPECT_EQ(ranks[1], 1);
    EXPECT_EQ(ranks[2], 2);
}

TEST(GenerateRankBindingsHelpersTest, WritePhase2MockMapping_WritesRankToClusterDesc) {
    const auto dir = make_temp_dir("phase2");
    const auto path = dir / "phase2_mock_mapping.yaml";
    std::vector<RankBindingConfig> bindings = {
        make_binding(0, 0, 0, "h", 0, 0),
        make_binding(1, 0, 0, "h", 1, 1),
    };
    std::map<int, std::string> mpi_rank_to_path = {
        {0, "/abs/first.yaml"},
        {1, "/abs/second.yaml"},
    };

    write_phase2_mock_mapping_yaml(bindings, mpi_rank_to_path, path.string());

    ASSERT_TRUE(std::filesystem::exists(path));
    const YAML::Node root = YAML::LoadFile(path.string());
    ASSERT_TRUE(root["rank_to_cluster_mock_cluster_desc"]);
    EXPECT_EQ(root["rank_to_cluster_mock_cluster_desc"]["0"].as<std::string>(), "/abs/first.yaml");
    EXPECT_EQ(root["rank_to_cluster_mock_cluster_desc"]["1"].as<std::string>(), "/abs/second.yaml");
}

TEST(GenerateRankBindingsHelpersTest, WritePhase2MockMapping_EmptyPathMapProducesNoFile) {
    const auto dir = make_temp_dir("phase2_empty");
    const auto path = dir / "phase2_mock_mapping.yaml";
    std::vector<RankBindingConfig> bindings = {make_binding(0, 0, 0, "h", 0, 0)};
    std::map<int, std::string> empty;

    write_phase2_mock_mapping_yaml(bindings, empty, path.string());

    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(GenerateRankBindingsHelpersTest, GetActualHostname_PassesThroughNonLocalhost) {
    EXPECT_EQ(get_actual_hostname("my.cluster.host"), "my.cluster.host");
}

TEST(GenerateRankBindingsHelpersTest, AllWritersProduceThreeFilesInTempDir) {
    const auto dir = make_temp_dir("three_files");
    std::vector<RankBindingConfig> bindings = {
        make_binding(0, 0, 0, "worker0", 0, 0),
        make_binding(1, 1, 0, "worker1", 0, 1),
    };
    bindings[0].env_overrides["TT_VISIBLE_DEVICES"] = "0";
    bindings[1].env_overrides["TT_VISIBLE_DEVICES"] = "1";

    const std::string mgd = "/mesh.textproto";
    write_rank_bindings_yaml(bindings, mgd, (dir / "rank_bindings.yaml").string());
    write_rankfile(bindings, (dir / "rankfile").string(), false);
    std::map<int, std::string> mock_paths = {{0, "/mock0.yaml"}, {1, "/mock1.yaml"}};
    write_phase2_mock_mapping_yaml(bindings, mock_paths, (dir / "phase2_mock_mapping.yaml").string());

    int n = 0;
    for ([[maybe_unused]] const auto& entry : std::filesystem::directory_iterator(dir)) {
        ++n;
    }
    EXPECT_EQ(n, 3);
}
