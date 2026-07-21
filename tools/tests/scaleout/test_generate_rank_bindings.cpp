// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

#include <yaml-cpp/yaml.h>

#include "generate_rank_bindings_helpers.hpp"

namespace {

std::filesystem::path make_temp_dir(const std::string& test_name) {
    using clock = std::chrono::high_resolution_clock;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now().time_since_epoch()).count();
    const std::filesystem::path base =
        std::filesystem::temp_directory_path() /
        (std::string{"grb_test_"} + test_name + "_" + std::to_string(getpid()) + "_" + std::to_string(ns));
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

    {
        std::ofstream stale(path.string());
        stale << "rank_to_cluster_mock_cluster_desc:\n  \"0\": /stale.yaml\n";
    }
    ASSERT_TRUE(std::filesystem::exists(path));

    write_phase2_mock_mapping_yaml(bindings, empty, path.string());

    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST(GenerateRankBindingsHelpersTest, WritePhase2MockMapping_NoRankEntriesRemovesStaleFile) {
    const auto dir = make_temp_dir("phase2_skip_all");
    const auto path = dir / "phase2_mock_mapping.yaml";
    std::vector<RankBindingConfig> bindings = {make_binding(0, 0, 0, "h", 0, -1)};
    std::map<int, std::string> mpi_rank_to_path = {{0, "/mock.yaml"}};

    {
        std::ofstream stale(path.string());
        stale << "rank_to_cluster_mock_cluster_desc:\n  \"0\": /stale.yaml\n";
    }
    ASSERT_TRUE(std::filesystem::exists(path));

    write_phase2_mock_mapping_yaml(bindings, mpi_rank_to_path, path.string());

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

// -----------------------------------------------------------------------------
// Multi-solution helpers (see README_generate_rank_bindings.md)
// -----------------------------------------------------------------------------

TEST(GenerateRankBindingsHelpersTest, SolutionHash_IsStableAndOrderIndependent) {
    std::vector<RankBindingConfig> a = {
        make_binding(0, 0, 0, "host-a", 0),
        make_binding(1, 1, 0, "host-b", 0),
    };
    a[0].env_overrides["TT_VISIBLE_DEVICES"] = "0,1";
    a[1].env_overrides["TT_VISIBLE_DEVICES"] = "2,3";

    // Same solution, bindings supplied in a different order -> identical hash.
    std::vector<RankBindingConfig> a_reordered = {a[1], a[0]};

    const std::string h1 = compute_solution_signature_hash(a);
    const std::string h2 = compute_solution_signature_hash(a_reordered);
    EXPECT_EQ(h1, h2) << "hash must be order-independent";
    EXPECT_EQ(h1.size(), 16u) << "hash is 16 hex chars";
    EXPECT_EQ(h1.find_first_not_of("0123456789abcdef"), std::string::npos) << "hash is lowercase hex";
}

TEST(GenerateRankBindingsHelpersTest, SolutionHash_DiffersForDifferentMappingOnSameHosts) {
    // Same host set {host-a, host-b}, but the (mesh,host-rank) -> host assignment is swapped.
    std::vector<RankBindingConfig> s1 = {
        make_binding(0, 0, 0, "host-a", 0),
        make_binding(1, 1, 0, "host-b", 0),
    };
    std::vector<RankBindingConfig> s2 = {
        make_binding(0, 0, 0, "host-b", 0),
        make_binding(1, 1, 0, "host-a", 0),
    };
    EXPECT_NE(compute_solution_signature_hash(s1), compute_solution_signature_hash(s2))
        << "same hosts but different mapping must produce different directories";
}

TEST(GenerateRankBindingsHelpersTest, WriteSolutionsIndex_HasEnumerationAndSolutions) {
    const auto dir = make_temp_dir("index");
    const auto path = dir / "solutions_index.yaml";

    std::vector<SolutionIndexEntry> entries;
    SolutionIndexEntry e0;
    e0.id = "3f9c1a20";
    e0.num_ranks = 2;
    e0.num_hosts = 2;
    e0.host_set = {"host-a", "host-b"};
    entries.push_back(e0);

    write_solutions_index_yaml(
        "/mesh.textproto", "distinct-host-sets", /*max_solutions=*/4, /*truncated=*/false, entries, path.string());

    ASSERT_TRUE(std::filesystem::exists(path));
    const YAML::Node root = YAML::LoadFile(path.string());
    EXPECT_EQ(root["mesh_graph_desc_path"].as<std::string>(), "/mesh.textproto");
    EXPECT_EQ(root["enumeration"]["mode"].as<std::string>(), "distinct-host-sets");
    EXPECT_EQ(root["enumeration"]["max_solutions"].as<int>(), 4);
    EXPECT_EQ(root["enumeration"]["found"].as<int>(), 1);
    EXPECT_FALSE(root["enumeration"]["truncated"].as<bool>());
    ASSERT_TRUE(root["solutions"]);
    ASSERT_EQ(root["solutions"].size(), 1u);
    EXPECT_EQ(root["solutions"][0]["id"].as<std::string>(), "3f9c1a20");
    EXPECT_EQ(root["solutions"][0]["dir"].as<std::string>(), "3f9c1a20");
    EXPECT_EQ(root["solutions"][0]["num_hosts"].as<int>(), 2);
    EXPECT_EQ(root["solutions"][0]["rank_bindings"].as<std::string>(), "3f9c1a20/rank_bindings.yaml");
    EXPECT_EQ(root["solutions"][0]["host_set"].size(), 2u);
}

TEST(GenerateRankBindingsHelpersTest, WriteSolutionMeta_ListsHostsAndCounts) {
    const auto dir = make_temp_dir("meta");
    const auto path = dir / "solution_meta.yaml";
    std::vector<RankBindingConfig> bindings = {
        make_binding(0, 0, 0, "host-a", 0),
        make_binding(1, 0, 1, "host-a", 1),
        make_binding(2, 1, 0, "host-b", 0),
    };

    write_solution_meta_yaml(bindings, "deadbeef00000000", "/mesh.textproto", path.string());

    ASSERT_TRUE(std::filesystem::exists(path));
    const YAML::Node root = YAML::LoadFile(path.string());
    EXPECT_EQ(root["solution_id"].as<std::string>(), "deadbeef00000000");
    EXPECT_EQ(root["num_ranks"].as<int>(), 3);
    EXPECT_EQ(root["num_hosts"].as<int>(), 2);  // host-a, host-b
    EXPECT_EQ(root["host_set"].size(), 2u);
}
