// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <tt-logger/tt-logger.hpp>
#include <yaml-cpp/yaml.h>

struct RankBindingConfig {
    int rank;               // Sequential MPI rank (0 to N-1, unique and contiguous) for rankfile and rank_bindings.yaml
    int psd_mpi_rank = -1;  // PSD MPI rank from discovery (used for phase2_mock_mapping.yaml lookup)
    int mesh_id;            // Mesh ID this rank belongs to
    int mesh_host_rank = 0;  // Host rank within the mesh (from MeshGraph), defaults to 0
    std::string hostname;    // Physical host for rankfile
    int slot;                // Slot number on host for rankfile (OpenMPI format)
    std::map<std::string, std::string> env_overrides;
};

inline std::string get_actual_hostname(const std::string& hostname) {
    if (hostname == "localhost") {
        char buf[256];
        if (gethostname(buf, sizeof(buf)) == 0) {
            return std::string(buf);
        }
        return "localhost";
    }
    return hostname;
}

inline void write_rank_bindings_yaml(
    const std::vector<RankBindingConfig>& rank_bindings,
    const std::string& mesh_graph_desc_path,
    const std::string& output_file) {
    YAML::Node root;

    YAML::Node rank_bindings_node;
    for (const auto& binding : rank_bindings) {
        YAML::Node binding_node;
        binding_node["rank"] = binding.rank;
        binding_node["mesh_id"] = binding.mesh_id;
        binding_node["mesh_host_rank"] = binding.mesh_host_rank;

        if (!binding.env_overrides.empty()) {
            YAML::Node env_overrides_node;
            for (const auto& [key, value] : binding.env_overrides) {
                env_overrides_node[key] = value;
            }
            binding_node["env_overrides"] = env_overrides_node;
        }

        rank_bindings_node.push_back(binding_node);
    }

    root["rank_bindings"] = rank_bindings_node;
    root["mesh_graph_desc_path"] = mesh_graph_desc_path;

    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }

    out_file << root << std::endl;
    out_file.close();
}

inline void write_phase2_mock_mapping_yaml(
    const std::vector<RankBindingConfig>& rank_bindings,
    const std::map<int, std::string>& mpi_rank_to_path,
    const std::string& output_file) {
    if (mpi_rank_to_path.empty()) {
        return;
    }

    YAML::Node root;
    YAML::Node rank_to_desc_node;

    for (const auto& binding : rank_bindings) {
        if (binding.psd_mpi_rank < 0) {
            log_warning(
                tt::LogFabric,
                "Rank {} has invalid psd_mpi_rank ({}), skipping phase2 mock mapping",
                binding.rank,
                binding.psd_mpi_rank);
            continue;
        }
        if (!mpi_rank_to_path.contains(binding.psd_mpi_rank)) {
            log_warning(
                tt::LogFabric,
                "PSD MPI rank {} (for sequential rank {}) has no cluster descriptor path, skipping",
                binding.psd_mpi_rank,
                binding.rank);
            continue;
        }
        rank_to_desc_node[std::to_string(binding.rank)] = mpi_rank_to_path.at(binding.psd_mpi_rank);
    }

    if (rank_to_desc_node.size() == 0) {
        return;
    }

    root["rank_to_cluster_mock_cluster_desc"] = rank_to_desc_node;

    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open output file for phase2 mock mapping: " + output_file);
    }
    out_file << root << std::endl;
    out_file.close();
}

inline void write_rankfile(
    const std::vector<RankBindingConfig>& rank_bindings, const std::string& output_file, bool mock_cluster_rankfile) {
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open rankfile for writing: " + output_file);
    }

    std::vector<RankBindingConfig> sorted_bindings = rank_bindings;
    std::sort(
        sorted_bindings.begin(), sorted_bindings.end(), [](const RankBindingConfig& a, const RankBindingConfig& b) {
            return a.rank < b.rank;
        });

    const std::string local_placement_host = mock_cluster_rankfile ? get_actual_hostname("localhost") : std::string{};

    for (const auto& binding : sorted_bindings) {
        std::string hostname = mock_cluster_rankfile ? local_placement_host : get_actual_hostname(binding.hostname);
        out_file << "rank " << binding.rank << "=" << hostname << " slot=" << binding.slot << "\n";
    }

    out_file.close();
}
