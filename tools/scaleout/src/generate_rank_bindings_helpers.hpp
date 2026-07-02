// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <tt-logger/tt-logger.hpp>
#include <yaml-cpp/yaml.h>

struct RankBindingConfig {
    // Sequential MPI rank for rankfile / rank_bindings.yaml / phase2 mock: normally 0..N-1 in that file;
    // multi-MGD merged outputs use globally offset contiguous ranks in one combined rankfile and mock mapping.
    int rank;
    int psd_mpi_rank = -1;  // PSD MPI rank from discovery (used for phase2_mock_mapping.yaml lookup)
    int mesh_id;            // Mesh ID this rank belongs to
    int mesh_host_rank = 0;  // Host rank within the mesh (from MeshGraph), defaults to 0
    std::string hostname;    // Physical host for rankfile
    int slot;                // Slot on host for OpenMPI rankfile; multi-MGD merge renumbers globally by MPI rank order
    std::map<std::string, std::string> env_overrides;
};

// YAML key for multi-MGD: sub-context id (int) -> path to MGD .textproto (string).
inline constexpr const char* kSubcontextMgdMappingYamlKey = "subcontext_id_to_mesh_graph_descriptor";

// Same key as :func:`parse_rank_bindings_mapping` in :file:`ttrun.py` (--rank-bindings-mapping).
inline constexpr const char* kSubcontextRankBindingsMappingYamlKey = "subcontext_id_to_rank_bindings";

/**
 * Load and validate `subcontext_id_to_mesh_graph_descriptor` from a mapping YAML file.
 * Keys must be a dense set 0..N-1. For each value path: if relative, it is first resolved against the
 * mapping file's directory; if that file does not exist, then when \c TT_METAL_HOME is set, against
 * \c TT_METAL_HOME (paths relative to the repo root, like \c tests/tt_metal/tt_fabric/...).
 * Each value must be an existing regular file.
 */
inline std::map<int, std::filesystem::path> load_subcontext_id_to_mesh_graph_descriptor_mapping(
    const std::filesystem::path& mapping_yaml_path) {
    if (!std::filesystem::exists(mapping_yaml_path) || !std::filesystem::is_regular_file(mapping_yaml_path)) {
        throw std::invalid_argument("MGD mapping file does not exist: " + mapping_yaml_path.string());
    }

    YAML::Node root = YAML::LoadFile(mapping_yaml_path.string());
    if (!root[kSubcontextMgdMappingYamlKey]) {
        throw std::invalid_argument(
            std::string("MGD mapping YAML must contain top-level key '") + kSubcontextMgdMappingYamlKey + "'");
    }
    const YAML::Node& map_node = root[kSubcontextMgdMappingYamlKey];
    if (!map_node.IsMap()) {
        throw std::invalid_argument(std::string("'") + kSubcontextMgdMappingYamlKey + "' must be a mapping");
    }

    std::map<int, std::filesystem::path> out;
    const std::filesystem::path base_dir =
        mapping_yaml_path.parent_path().empty() ? std::filesystem::path(".") : mapping_yaml_path.parent_path();

    for (const auto& it : map_node) {
        int subctx_id;
        try {
            subctx_id = it.first.as<int>();
        } catch (const YAML::Exception&) {
            throw std::invalid_argument("MGD mapping: subcontext id key must be an integer");
        }
        if (subctx_id < 0) {
            throw std::invalid_argument(
                "MGD mapping: subcontext id must be non-negative, got " + std::to_string(subctx_id));
        }
        if (out.contains(subctx_id)) {
            throw std::invalid_argument("MGD mapping: duplicate subcontext id: " + std::to_string(subctx_id));
        }
        if (!it.second.IsScalar()) {
            throw std::invalid_argument(
                "MGD mapping: path for subcontext " + std::to_string(subctx_id) + " must be a string");
        }
        const std::string path_str = it.second.as<std::string>();
        std::filesystem::path mgd_path(path_str);
        if (mgd_path.is_relative()) {
            const std::filesystem::path from_mapping = (base_dir / mgd_path).lexically_normal();
            if (std::filesystem::exists(from_mapping) && std::filesystem::is_regular_file(from_mapping)) {
                mgd_path = from_mapping;
            } else {
                if (const char* tt_metal_home = std::getenv("TT_METAL_HOME")) {
                    const std::filesystem::path from_home =
                        (std::filesystem::path(tt_metal_home) / path_str).lexically_normal();
                    if (std::filesystem::exists(from_home) && std::filesystem::is_regular_file(from_home)) {
                        mgd_path = from_home;
                    } else {
                        throw std::invalid_argument(
                            "MGD file does not exist: " + from_mapping.string() + " (also tried " + from_home.string() +
                            " via TT_METAL_HOME)");
                    }
                } else {
                    throw std::invalid_argument("MGD file does not exist: " + from_mapping.string());
                }
            }
        } else {
            if (!std::filesystem::exists(mgd_path) || !std::filesystem::is_regular_file(mgd_path)) {
                throw std::invalid_argument("MGD file does not exist: " + mgd_path.string());
            }
        }
        out[subctx_id] = std::move(mgd_path);
    }

    if (out.empty()) {
        throw std::invalid_argument("MGD mapping: must list at least one subcontext_id");
    }

    const int max_id = out.rbegin()->first;
    for (int i = 0; i <= max_id; ++i) {
        if (!out.contains(i)) {
            throw std::invalid_argument(
                "MGD mapping: subcontext ids must be dense starting at 0; missing subcontext_id " + std::to_string(i));
        }
    }
    return out;
}

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

/** Writes sibling paths (relative filenames) resolved by tt-run alongside this file. \p pairs must follow sub-context
 * 0..N-1. */
inline void write_subcontext_rank_bindings_mapping_yaml(
    const std::vector<std::pair<int, std::string>>& subcontext_id_to_relative_rank_bindings_yaml,
    const std::string& output_file) {
    YAML::Node root;
    YAML::Node map_node;
    for (const auto& [sid, relpath] : subcontext_id_to_relative_rank_bindings_yaml) {
        map_node[std::to_string(sid)] = relpath;
    }
    root[kSubcontextRankBindingsMappingYamlKey] = map_node;

    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Failed to open rank bindings mapping file: " + output_file);
    }
    out_file << root << std::endl;
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

    // For a fully-mocked cluster every rank runs on the launching node. Emit the literal "localhost" rather than
    // resolving gethostname(): OpenMPI/PRRTE treats "localhost" as the local node and never launches via ssh, whereas
    // a real hostname is treated as a (possibly remote) host and triggers an ssh launch when it is not resolvable on
    // the launching machine (e.g. not in /etc/hosts or DNS), which breaks otherwise hardware-free mock runs.
    for (const auto& binding : sorted_bindings) {
        std::string hostname = mock_cluster_rankfile ? std::string{"localhost"} : get_actual_hostname(binding.hostname);
        out_file << "rank " << binding.rank << "=" << hostname << " slot=" << binding.slot << "\n";
    }

    out_file.close();
}
