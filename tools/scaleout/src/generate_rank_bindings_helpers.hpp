// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
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

// -----------------------------------------------------------------------------
// Multi-solution support (see README_generate_rank_bindings.md)
// -----------------------------------------------------------------------------

// Sorted set of distinct hostnames a solution occupies. Hostnames are used verbatim (no cleaning); the set
// deduplicates, so multiple ranks that share a host collapse to a single entry.
inline std::set<std::string> solution_host_set(const std::vector<RankBindingConfig>& rank_bindings) {
    std::set<std::string> hosts;
    for (const auto& b : rank_bindings) {
        hosts.insert(b.hostname);
    }
    return hosts;
}

// Join hosts as a comma-separated list with NO spaces (e.g. "hostA,hostB,hostC").
inline std::string host_set_csv(const std::vector<std::string>& hosts) {
    std::string csv;
    for (const auto& h : hosts) {
        if (!csv.empty()) {
            csv += ',';
        }
        csv += h;
    }
    return csv;
}

// Canonical, order-independent signature of a solution. Captures both the set of hosts used AND the
// assignment (per (mesh_id, mesh_host_rank): which host + which visible devices), so two solutions on the
// same hosts but different connectivity/mapping produce different signatures. Written verbatim to
// `.solution_key` so short-hash collisions can be disambiguated by comparing the full string.
inline std::string compute_solution_signature_string(const std::vector<RankBindingConfig>& rank_bindings) {
    std::set<std::string> assignment;  // sorted -> canonical regardless of input order
    for (const auto& b : rank_bindings) {
        std::string visible_devices;
        auto it = b.env_overrides.find("TT_VISIBLE_DEVICES");
        if (it != b.env_overrides.end()) {
            visible_devices = it->second;
        }
        assignment.insert(
            "m" + std::to_string(b.mesh_id) + "r" + std::to_string(b.mesh_host_rank) + "@" + b.hostname + "[" +
            visible_devices + "]");
    }

    std::string signature = "hosts:";
    for (const auto& host : solution_host_set(rank_bindings)) {
        signature += host;
        signature += ',';
    }
    signature += "|map:";
    for (const auto& entry : assignment) {
        signature += entry;
        signature += ';';
    }
    return signature;
}

// Stable 64-bit FNV-1a content hash of the canonical signature, rendered as 16 lowercase hex chars.
// Deterministic across runs/platforms (unlike std::hash), so the same solution always names the same
// directory -- enabling caching and dedupe. Not cryptographic; `.solution_key` holds the full signature
// for exact collision disambiguation.
inline std::string compute_solution_signature_hash(const std::vector<RankBindingConfig>& rank_bindings) {
    const std::string signature = compute_solution_signature_string(rank_bindings);
    std::uint64_t hash = 1469598103934665603ULL;  // FNV-1a 64-bit offset basis
    for (unsigned char c : signature) {
        hash ^= static_cast<std::uint64_t>(c);
        hash *= 1099511628211ULL;  // FNV-1a 64-bit prime
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
    return std::string(buf);
}

// Per-solution metadata file written inside each solution subdirectory.
inline void write_solution_meta_yaml(
    const std::vector<RankBindingConfig>& rank_bindings,
    const std::string& solution_id,
    const std::string& mesh_graph_desc_path,
    const std::string& output_file) {
    const auto hosts = solution_host_set(rank_bindings);

    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "solution_id" << YAML::Value << solution_id;
    emitter << YAML::Key << "mesh_graph_desc_path" << YAML::Value << mesh_graph_desc_path;
    emitter << YAML::Key << "num_ranks" << YAML::Value << static_cast<int>(rank_bindings.size());
    emitter << YAML::Key << "num_hosts" << YAML::Value << static_cast<int>(hosts.size());
    // Comma-separated host list, no spaces (e.g. "hostA,hostB").
    emitter << YAML::Key << "host_set" << YAML::Value
            << host_set_csv(std::vector<std::string>(hosts.begin(), hosts.end()));
    emitter << YAML::EndMap;

    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open solution meta file: " + output_file);
    }
    out_file << emitter.c_str() << std::endl;
    out_file.close();
}

// One entry in solutions_index.yaml.
struct SolutionIndexEntry {
    std::string id;  // == solution subdirectory name (the content hash)
    int num_ranks = 0;
    int num_hosts = 0;
    std::vector<std::string> host_set;
};

// Top-level index summarizing every enumerated solution, written at the base output directory.
inline void write_solutions_index_yaml(
    const std::string& mesh_graph_desc_path,
    const std::string& enumeration_mode,  // "all" or "distinct-host-sets"
    std::size_t max_solutions,
    bool truncated,
    const std::vector<SolutionIndexEntry>& entries,
    const std::string& output_file) {
    YAML::Emitter emitter;
    emitter << YAML::BeginMap;
    emitter << YAML::Key << "mesh_graph_desc_path" << YAML::Value << mesh_graph_desc_path;
    emitter << YAML::Key << "enumeration" << YAML::Value << YAML::BeginMap;
    emitter << YAML::Key << "mode" << YAML::Value << enumeration_mode;
    emitter << YAML::Key << "max_solutions" << YAML::Value << static_cast<int>(max_solutions);
    emitter << YAML::Key << "found" << YAML::Value << static_cast<int>(entries.size());
    emitter << YAML::Key << "truncated" << YAML::Value << truncated;
    emitter << YAML::EndMap;
    emitter << YAML::Key << "solutions" << YAML::Value << YAML::BeginSeq;
    for (const auto& entry : entries) {
        emitter << YAML::BeginMap;
        emitter << YAML::Key << "id" << YAML::Value << entry.id;
        emitter << YAML::Key << "dir" << YAML::Value << entry.id;
        emitter << YAML::Key << "num_hosts" << YAML::Value << entry.num_hosts;
        emitter << YAML::Key << "num_ranks" << YAML::Value << entry.num_ranks;
        // Comma-separated host list, no spaces (e.g. "hostA,hostB").
        emitter << YAML::Key << "host_set" << YAML::Value << host_set_csv(entry.host_set);
        emitter << YAML::Key << "rank_bindings" << YAML::Value << (entry.id + "/rank_bindings.yaml");
        emitter << YAML::Key << "rankfile" << YAML::Value << (entry.id + "/rankfile");
        emitter << YAML::EndMap;
    }
    emitter << YAML::EndSeq;
    emitter << YAML::EndMap;

    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open solutions index file: " + output_file);
    }
    out_file << emitter.c_str() << std::endl;
    out_file.close();
}
