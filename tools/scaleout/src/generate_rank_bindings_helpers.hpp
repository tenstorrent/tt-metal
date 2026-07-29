// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <filesystem>
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
        if (gethostname(buf, sizeof(buf) - 1) != 0) {
            return "localhost";
        }
        buf[sizeof(buf) - 1] = '\0';
        return std::string(buf);
    }
    return hostname;
}

inline void write_rank_bindings_yaml(
    const std::vector<RankBindingConfig>& rank_bindings,
    const std::string& mesh_graph_desc_path,
    const std::string& output_file) {
    // Emit via streaming YAML::Emitter so env_override values are force-quoted.
    // The YAML::Node API emits a string like "5" as a bare scalar (5), which
    // yaml.safe_load on the tt-run side parses back as an int and fails the
    // TTRunConfig env_overrides: Dict[str, str] schema -- this happens when a
    // rank owns a single device (e.g. host_topology with 1 device/rank). Forcing
    // DoubleQuoted keeps every env_override value a string regardless of device count.
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "rank_bindings" << YAML::Value << YAML::BeginSeq;
    for (const auto& binding : rank_bindings) {
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "rank" << YAML::Value << binding.rank;
        emitter << YAML::Key << "mesh_id" << YAML::Value << binding.mesh_id;
        emitter << YAML::Key << "mesh_host_rank" << YAML::Value << binding.mesh_host_rank;
        if (!binding.env_overrides.empty()) {
            emitter << YAML::Key << "env_overrides" << YAML::Value << YAML::BeginMap;
            for (const auto& [key, value] : binding.env_overrides) {
                emitter << YAML::Key << key << YAML::Value << YAML::DoubleQuoted << value;
            }
            emitter << YAML::EndMap;
        }
        emitter << YAML::EndMap;
    }
    emitter << YAML::EndSeq;
    emitter << YAML::Key << "mesh_graph_desc_path" << YAML::Value << mesh_graph_desc_path;
    emitter << YAML::EndMap;

    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }

    out_file << emitter.c_str() << std::endl;
    out_file.close();
}

inline void write_phase2_mock_mapping_yaml(
    const std::vector<RankBindingConfig>& rank_bindings,
    const std::map<int, std::string>& mpi_rank_to_path,
    const std::string& output_file) {
    auto remove_output_if_present = [&output_file]() {
        std::error_code ec;
        std::filesystem::remove(output_file, ec);
    };

    if (mpi_rank_to_path.empty()) {
        remove_output_if_present();
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
        remove_output_if_present();
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
