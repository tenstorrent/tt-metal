// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include <yaml-cpp/yaml.h>

#ifdef OPEN_MPI
#include <mpi.h>
#endif

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::tt_fabric;

struct RankBindingConfig {
    int rank;
    int mesh_id;
    std::map<std::string, std::string> env_overrides;
};

/**
 * @brief Run Physical System Descriptor discovery via MPI
 *
 * This function should:
 * 1. Initialize MPI context
 * 2. Run physical discovery on all ranks
 * 3. Return the PhysicalSystemDescriptor
 *
 * TODO: Implement MPI-based PSD discovery
 */
PhysicalSystemDescriptor run_psd_discovery() {
    // TODO: Initialize MPI and run discovery
    // This should:
    // - Initialize MetalContext with distributed context
    // - Run PhysicalSystemDescriptor discovery
    // - Return the discovered PSD

    throw std::runtime_error("PSD discovery via MPI not yet implemented");
}

/**
 * @brief Run topology mapper to map logical meshes to physical ASICs
 *
 * This function should:
 * 1. Load mesh graph descriptor
 * 2. Load physical grouping descriptor
 * 3. Build physical multi-mesh graph from PSD
 * 4. Build logical multi-mesh graph from MGD
 * 5. Run map_multi_mesh_to_physical
 * 6. Return the mapping result
 *
 * TODO: Implement topology mapping
 */
TopologyMappingResult run_topology_mapping(const PhysicalSystemDescriptor& psd) {
    // TODO: Load mesh graph descriptor
    // MeshGraphDescriptor mgd = load_mesh_graph_descriptor(mesh_graph_desc_path);

    // TODO: Load physical grouping descriptor
    // PhysicalGroupingDescriptor pgd = load_physical_grouping_descriptor(physical_grouping_desc_path);

    // TODO: Build physical multi-mesh graph
    // PhysicalMultiMeshGraph physical_graph = build_physical_multi_mesh_adjacency_graph(
    //     psd, pgd, mgd);

    // TODO: Build logical multi-mesh graph from MGD
    // LogicalMultiMeshGraph logical_graph = build_logical_multi_mesh_adjacency_graph(mgd);

    // TODO: Build rank mappings from PSD
    // std::map<MeshId, std::map<AsicID, MeshHostRankId>> asic_id_to_mesh_rank;
    // std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> fabric_node_id_to_mesh_rank;

    // TODO: Configure topology mapping
    // TopologyMappingConfig config;
    // config.strict_mode = true;
    // Populate config.hostname_to_asics from PSD

    // TODO: Run mapping
    // TopologyMappingResult result = map_multi_mesh_to_physical(
    //     logical_graph, physical_graph, config, asic_id_to_mesh_rank, fabric_node_id_to_mesh_rank);

    throw std::runtime_error("Topology mapping not yet implemented");
}

/**
 * @brief Extract rank bindings from topology mapping result
 *
 * This function should:
 * 1. Group ASICs by mesh_id and rank
 * 2. Convert ASIC IDs to PCIe device IDs using PSD
 * 3. Generate RankBindingConfig for each rank
 *
 * TODO: Implement rank binding extraction
 */
std::vector<RankBindingConfig> extract_rank_bindings(
    const PhysicalSystemDescriptor& psd, const TopologyMappingResult& mapping_result) {
    std::vector<RankBindingConfig> rank_bindings;

    // TODO: Group fabric nodes by mesh_id and rank
    // For each mesh_id:
    //   - Find all fabric nodes mapped to that mesh
    //   - Group by the rank that owns those ASICs
    //   - Collect ASIC IDs for each rank

    // TODO: Convert ASIC IDs to PCIe device IDs
    // For each ASIC ID:
    //   - Get hostname from PSD
    //   - Get tray_id and asic_location from PSD
    //   - Use PSD.get_pcie_id_to_asic_location() to find PCIe device ID
    //   - Collect all PCIe device IDs for the rank

    // TODO: Build RankBindingConfig entries
    // For each (rank, mesh_id) pair:
    //   RankBindingConfig binding;
    //   binding.rank = rank;
    //   binding.mesh_id = mesh_id;
    //   binding.env_overrides["TT_VISIBLE_DEVICES"] = comma_separated_pcie_device_ids;
    //   rank_bindings.push_back(binding);

    throw std::runtime_error("Rank binding extraction not yet implemented");
}

/**
 * @brief Write rank bindings to YAML file
 */
void write_rank_bindings_yaml(
    const std::vector<RankBindingConfig>& rank_bindings,
    const std::string& mesh_graph_desc_path,
    const std::string& output_file) {
    YAML::Node root;

    YAML::Node rank_bindings_node;
    for (const auto& binding : rank_bindings) {
        YAML::Node binding_node;
        binding_node["rank"] = binding.rank;
        binding_node["mesh_id"] = binding.mesh_id;

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

int main(int argc, char** argv) {
    // Warn if any arguments are passed
    if (argc > 1) {
        log_warning(tt::LogFabric, "This tool does not accept any arguments. Ignoring {} argument(s).", argc - 1);
    }

    // Check if MPI is initialized (i.e., running under mpirun/srun/etc.)
    // When mpirun launches the program, MPI is already initialized
    // Initialize distributed context - this will detect MPI if available
    tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);

    // Verify that we have a valid MPI context
    // Check if context supports fault tolerance (MPI contexts do, SingleHost doesn't)
    // OR check if we have multiple processes (size > 1)
    auto context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    if (*context->size() == 1) {
        // Single host context with size 1 - likely not running under MPI
        log_error(
            tt::LogFabric,
            "This tool must be run with an MPI launcher (e.g., mpirun, srun). "
            "Example: mpirun -np 4 --tag-output ./build/tools/scaleout/generate_rank_bindings");
        return 1;
    }

    try {
        log_info(tt::LogFabric, "Generating rank bindings...");

        // Step 1: Run PSD discovery
        log_info(tt::LogFabric, "Step 1: Running Physical System Descriptor discovery...");
        PhysicalSystemDescriptor psd = run_psd_discovery();
        log_info(tt::LogFabric, "PSD discovery complete");

        // Step 2: Run topology mapping
        log_info(tt::LogFabric, "Step 2: Running topology mapping...");
        TopologyMappingResult mapping_result = run_topology_mapping(psd);

        if (!mapping_result.success) {
            log_error(tt::LogFabric, "Topology mapping failed: {}", mapping_result.error_message);
            return 1;
        }
        log_info(tt::LogFabric, "Topology mapping complete");

        // Step 3: Extract rank bindings
        log_info(tt::LogFabric, "Step 3: Extracting rank bindings...");
        std::vector<RankBindingConfig> rank_bindings = extract_rank_bindings(psd, mapping_result);
        log_info(tt::LogFabric, "Extracted {} rank binding(s)", rank_bindings.size());

        // Step 4: Write YAML file
        log_info(tt::LogFabric, "Step 4: Writing rank bindings to YAML...");
        // TODO: Determine mesh_graph_desc_path and output_file from configuration or discovery
        std::string mesh_graph_desc_path = "";           // TODO: Set from discovery/config
        std::string output_file = "rank_bindings.yaml";  // TODO: Set from discovery/config
        write_rank_bindings_yaml(rank_bindings, mesh_graph_desc_path, output_file);
        log_info(tt::LogFabric, "Successfully wrote: {}", output_file);

        log_info(tt::LogFabric, "Rank bindings generation complete!");

    } catch (const std::exception& e) {
        log_error(tt::LogFabric, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
