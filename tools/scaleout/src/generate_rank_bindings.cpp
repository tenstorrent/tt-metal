// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <cxxopts.hpp>
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>
#include <tt-metalium/experimental/fabric/topology_mapper_utils.hpp>
#include "tt_metal/fabric/physical_system_discovery.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph_descriptor.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/physical_grouping_descriptor.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>

#include "generate_rank_bindings_helpers.hpp"

#ifdef OPEN_MPI
#include <mpi.h>
#endif

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::tt_fabric;
using namespace tt::tt_fabric;

/**
 * @brief Run Physical System Descriptor discovery via MPI
 *
 * This function:
 * 1. Gets the MetalContext instance (which should already be initialized)
 * 2. Runs physical discovery on all ranks using the distributed context
 * 3. Returns the PhysicalSystemDescriptor
 */
PhysicalSystemDescriptor run_psd_discovery() {
    auto& context = tt::tt_metal::MetalContext::instance();
    auto distributed_context = context.get_distributed_context_ptr();
    const auto& cluster = context.get_cluster();
    const auto& rtoptions = context.rtoptions();
    auto& driver_ref = const_cast<tt::umd::Cluster&>(*cluster.get_driver());

    return tt::tt_metal::run_physical_system_discovery(*driver_ref.get_cluster_description(), distributed_context, rtoptions.get_target_device());
}

/**
 * @brief Run topology mapper to map logical meshes to physical ASICs
 *
 * This function:
 * 1. Builds physical multi-mesh graph from PSD, PGD, and MGD
 * 2. Builds logical multi-mesh graph from MGD (via MeshGraph)
 * 3. Configures topology mapping with strict mode and disabled rank bindings
 * 4. Runs map_multi_mesh_to_physical
 * 5. Returns the mapping result
 */
TopologyMappingResult run_topology_mapping(
    const PhysicalSystemDescriptor& psd,
    const PhysicalGroupingDescriptor& pgd,
    const MeshGraphDescriptor& mgd,
    const std::filesystem::path& mgd_path) {
    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = context.get_cluster();
    MeshGraph mesh_graph(cluster, mgd_path.string());

    // Configure topology mapping
    TopologyMappingConfig config;
    config.strict_mode = true;
    config.disable_rank_bindings = false;  // Pass the rank bindings to make sure there isn't host rank boundary issues

    // Apply the same galaxy corner pinnings as the control plane (Phase 2) so Phase 1 and Phase 2 place
    // the galaxy pins identically. Full galaxies (per-host slice >= 32) pin all four corners; sub-galaxy
    // slices pin only the NW corner to any tray-corner ASIC (asic_location==1 on trays 1..4).
    if (cluster.is_ubb_galaxy()) {
        const int world_size =
            static_cast<int>(*tt::tt_metal::distributed::multihost::DistributedContext::get_current_world()->size());
        for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
            const auto& mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
            const bool is_1d = mesh_shape[0] == 1 || mesh_shape[1] == 1;
            if (!is_1d && mesh_shape.mesh_size() % 32 == 0) {
                auto mesh_pinnings = get_galaxy_fixed_asic_position_pinnings_for_mesh(
                    mesh_id, mesh_shape, /*hard_pin_node_0=*/world_size == 1, /*nw_corner_only=*/false);
                for (const auto& [fabric_node, positions] : mesh_pinnings) {
                    for (const auto& position : positions) {
                        config.pinnings.emplace_back(position, fabric_node);
                    }
                }
            }
        }
    }

    // PSD hostname grouping and tray/ASIC-location map (logical mesh 0 anchor + pinnings support).
    for (const auto& [asic_id, desc] : psd.get_asic_descriptors()) {
        config.hostname_to_asics[desc.host_name].insert(asic_id);
        config.asic_positions[asic_id] = std::make_pair(desc.tray_id, desc.asic_location);
    }

    // Extract pinnings from MGD and add to config (same as control plane)
    const auto& pinnings = mgd.get_pinnings();
    for (const auto& [pos, fabric_node] : pinnings) {
        config.pinnings.emplace_back(pos, fabric_node);
    }

    // Set per-mesh validation modes based on mesh graph policy
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        config.mesh_validation_modes[mesh_id] = mesh_graph.is_intra_mesh_policy_relaxed(mesh_id)
                                                    ? ::tt::tt_fabric::ConnectionValidationMode::RELAXED
                                                    : ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    }

    // Set inter-mesh validation mode based on mesh graph policy
    // TODO: Enable per-connection inter-mesh validation mode. Currently, all inter-mesh connections
    // use the same validation mode based on the mesh graph's global inter-mesh policy. In the future,
    // we should support mixed STRICT and RELAXED policies where some inter-mesh connections are
    // device-level (strict) and others are mesh-level (relaxed).
    config.inter_mesh_validation_mode = mesh_graph.is_inter_mesh_policy_relaxed()
                                            ? ::tt::tt_fabric::ConnectionValidationMode::RELAXED
                                            : ::tt::tt_fabric::ConnectionValidationMode::STRICT;
    if (config.inter_mesh_validation_mode.value() == ::tt::tt_fabric::ConnectionValidationMode::RELAXED) {
        log_info(tt::LogFabric, "Inter-mesh validation mode: RELAXED");
    } else {
        log_info(tt::LogFabric, "Inter-mesh validation mode: STRICT");
    }

    // Build physical multi-mesh graph from PSD, PGD, and MGD
    log_info(tt::LogFabric, "Building physical multi-mesh adjacency graph...");
    PhysicalMultiMeshGraph physical_graph =
        build_physical_multi_mesh_adjacency_graph(psd, pgd, mgd, std::optional{config.pinnings});

    // Build logical multi-mesh graph from MGD
    log_info(tt::LogFabric, "Building logical multi-mesh adjacency graph...");
    LogicalMultiMeshGraph logical_graph = build_logical_multi_mesh_adjacency_graph(mesh_graph);

    // Print adjacency maps
    log_logical_multi_mesh_adjacency_histograms(logical_graph);
    log_physical_multi_mesh_adjacency_histograms(physical_graph);

    // Build logical rank bindings from mesh graph: each fabric node gets its mesh_host_rank from the mesh graph
    std::map<MeshId, std::map<FabricNodeId, MeshHostRankId>> fabric_node_id_to_mesh_rank;
    for (const auto& mesh_id : mesh_graph.get_all_mesh_ids()) {
        const auto& chip_ids = mesh_graph.get_chip_ids(mesh_id);
        for (const auto& [coord, chip_id] : chip_ids) {
            FabricNodeId fabric_node_id(mesh_id, chip_id);
            auto mesh_host_rank = mesh_graph.get_host_rank_for_chip(mesh_id, chip_id);
            if (mesh_host_rank.has_value()) {
                fabric_node_id_to_mesh_rank[mesh_id][fabric_node_id] = mesh_host_rank.value();
            }
        }
    }

    // Physical rank bindings: all ASICs set to UNSET - let topology mapper assign physical ASICs to hosts
    // Build asic_id_to_mesh_rank from physical graph mesh IDs to match the physical graph structure
    std::map<MeshId, std::map<tt::tt_metal::AsicID, MeshHostRankId>> asic_id_to_mesh_rank = {};

    // Run mapping with logical rank bindings from mesh graph
    log_info(tt::LogFabric, "Running topology mapping with mesh graph rank bindings...");
    TopologyMappingResult result = map_multi_mesh_to_physical(
        logical_graph, physical_graph, config, asic_id_to_mesh_rank, fabric_node_id_to_mesh_rank);

    return result;
}

/**
 * @brief Extract rank bindings from topology mapping result with topology-aware splitting.
 *
 * Bindings are one row per (mesh_id, PSD hostname, mesh_host_rank), sorted by mesh_id ascending,
 * then mesh_host_rank, hostname, PSD rank tiebreaker — so sequential MPI rank tracks mesh topology; inter-mesh
 * mapping in topology_mapper_utils biases logical mesh 0 toward this process's hostname and toward the physical
 * partition that contains tray id 1 / ASIC location 1 when uniquely resolvable from discovery.
 *
 * - **Multi-process Phase 1** (several MPI ranks / PSD hostnames): each distinct hostname usually owns
 *   ASICs for a single mesh_host_rank, so the per-hostname map has one entry — behavior matches the
 *   historical `(mesh_id, hostname)` grouping.
 * - **Single-process Phase 1** (`mpirun -np 1`, e.g. bare mock cluster YAML): every ASIC shares one
 *   PSD hostname; mesh_host_rank in the key splits logical mesh hosts so MGD `host_topology` still
 *   yields the correct number of Phase 2 MPI ranks.
 *
 * Assigns contiguous ranks 0..N-1 and per-hostname slot indices for the rankfile.
 */
std::vector<RankBindingConfig> extract_rank_bindings(
    const PhysicalSystemDescriptor& psd, const TopologyMappingResult& mapping_result, const MeshGraph& mesh_graph) {
    struct AsicGrouping {
        std::vector<AsicID> asic_ids;
        std::vector<tt::ChipId> chip_ids;
        std::optional<MeshHostRankId> mesh_host_rank;
    };

    // mesh_id -> hostname -> mesh_host_rank -> AsicGrouping
    std::map<int, std::map<std::string, std::map<int, AsicGrouping>>> mesh_host_asics;

    // Iterate through fabric_node_to_asic mapping
    for (const auto& [fabric_node_id, asic_id] : mapping_result.fabric_node_to_asic) {
        MeshId mesh_id = fabric_node_id.mesh_id;
        tt::ChipId chip_id_from_fabric_node = static_cast<tt::ChipId>(fabric_node_id.chip_id);

        std::optional<MeshHostRankId> mesh_host_rank =
            mesh_graph.get_host_rank_for_chip(mesh_id, chip_id_from_fabric_node);

        if (!mesh_host_rank.has_value()) {
            log_error(
                tt::LogFabric,
                "No mesh host rank found for mesh_id {} and chip_id {}",
                *mesh_id,
                chip_id_from_fabric_node);
            continue;
        }

        std::string hostname = psd.get_host_name_for_asic(asic_id);
        tt::ChipId chip_id = psd.get_umd_unique_id(asic_id);

        int mesh_id_int = static_cast<int>(*mesh_id);
        const int mesh_host_rank_int = static_cast<int>(*mesh_host_rank.value());
        auto& bucket = mesh_host_asics[mesh_id_int][hostname][mesh_host_rank_int];
        bucket.asic_ids.push_back(asic_id);
        bucket.chip_ids.push_back(chip_id);
        bucket.mesh_host_rank = mesh_host_rank;
    }

    // Build flat list of (mesh_id, hostname, chip_ids, mesh_host_rank, psd_rank) for canonical ordering
    // Order after sort: mesh_id ascending (rank 0 is first/lowest mesh), then mesh_host_rank, hostname,
    // PSD rank tiebreaker — so sequential MPI rank tracks mesh topology; psd_mpi_rank field keeps PSD identity
    using Entry = std::tuple<int, std::string, std::vector<tt::ChipId>, int, int>;
    std::vector<Entry> entries;
    for (const auto& [mesh_id, hostname_map] : mesh_host_asics) {
        for (const auto& [hostname, rank_map] : hostname_map) {
            for (const auto& [_, asic_data] : rank_map) {
                const auto& chip_ids = asic_data.chip_ids;
                const auto& mesh_host_rank = asic_data.mesh_host_rank;
                if (!mesh_host_rank.has_value()) {
                    continue;
                }
                uint32_t psd_rank = 0;
                if (psd.get_host_to_rank_map().contains(hostname)) {
                    psd_rank = psd.get_rank_for_hostname(hostname);
                } else {
                    log_warning(
                        tt::LogFabric, "Hostname {} not in PSD host_to_rank map, using 0 for rank ordering", hostname);
                }
                entries.emplace_back(
                    mesh_id, hostname, chip_ids, static_cast<int>(*mesh_host_rank.value()), static_cast<int>(psd_rank));
            }
        }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        // Primary: mesh_id ascending so binding.rank aligns with mesh order (rank 0 from mesh id 0 first)
        if (std::get<0>(a) != std::get<0>(b)) {
            return std::get<0>(a) < std::get<0>(b);
        }
        if (std::get<3>(a) != std::get<3>(b)) {
            return std::get<3>(a) < std::get<3>(b);
        }
        if (std::get<1>(a) != std::get<1>(b)) {
            return std::get<1>(a) < std::get<1>(b);
        }
        // PSD rank last for deterministic ties only
        return std::get<4>(a) < std::get<4>(b);
    });

    // Assign contiguous ranks 0..N-1 and track slot per host for rankfile
    std::vector<RankBindingConfig> rank_bindings;
    std::map<std::string, int> host_slot_counters;

    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& [mesh_id, hostname, chip_ids, mesh_host_rank, psd_rank] = entries[i];

        RankBindingConfig binding;
        // binding.rank is a sequential MPI rank (0, 1, 2, ...) that must be unique and contiguous,
        // assigned in mesh_id-major sorted order so low ranks correspond to low mesh ids.
        // It must match: (1) the rank in rank_bindings.yaml, and (2) the rank in rankfile.
        // binding.psd_mpi_rank stores the PSD MPI rank from discovery, used for phase2_mock_mapping.yaml lookup.
        binding.rank = static_cast<int>(i);  // Sequential rank for rankfile and rank_bindings.yaml

        // Store PSD MPI rank separately for phase2_mock_mapping.yaml lookup
        if (psd.get_host_to_rank_map().contains(hostname)) {
            binding.psd_mpi_rank = static_cast<int>(psd.get_rank_for_hostname(hostname));
        } else {
            log_warning(
                tt::LogFabric, "Hostname {} not found in PSD host_to_rank_map, psd_mpi_rank will be -1", hostname);
        }
        binding.mesh_id = mesh_id;
        binding.mesh_host_rank = mesh_host_rank;
        binding.hostname = hostname;
        binding.slot = host_slot_counters[hostname]++;
        binding.env_overrides = {};

        // Build TT_VISIBLE_DEVICES from ChipIds - only include MMIO devices, not remote devices.
        // For each chip, get its associated MMIO device; remote chips share an MMIO device.
        const auto& cluster = MetalContext::instance().get_cluster();
        std::set<tt::ChipId> mmio_device_ids;
        for (tt::ChipId chip_id : chip_ids) {
            tt::ChipId mmio_id = cluster.get_associated_mmio_device(chip_id);
            mmio_device_ids.insert(mmio_id);
        }
        std::string visible_devices;
        for (auto it = mmio_device_ids.begin(); it != mmio_device_ids.end(); ++it) {
            if (it != mmio_device_ids.begin()) {
                visible_devices += ",";
            }
            visible_devices += std::to_string(*it);
        }
        if (!visible_devices.empty()) {
            binding.env_overrides["TT_VISIBLE_DEVICES"] = visible_devices;
        }

        rank_bindings.push_back(binding);
    }

    return rank_bindings;
}

/**
 * @brief Gather mock cluster descriptor paths from all MPI ranks to rank 0.
 *
 * When TT_METAL_MOCK_CLUSTER_DESC_PATH is set (mock mode), each rank may have its own descriptor.
 * This collects rank -> path for use when writing phase2_mock_mapping.yaml.
 *
 * For world_size > 1, every rank participates in the exchange and the final barrier even if this
 * rank's env var is unset (empty path), so rank 0 cannot block on recv while another rank returns early.
 *
 * @param distributed_context MPI distributed context
 * @param mpi_rank_to_path Output map (populated only on rank 0): MPI rank -> absolute path (non-empty paths only)
 */
void gather_mock_cluster_desc_paths(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& distributed_context,
    std::map<int, std::string>& mpi_rank_to_path) {
    using namespace tt::tt_metal::distributed::multihost;
    constexpr int root_rank = 0;
    constexpr Tag k_mock_path_size_tag{100};
    constexpr Tag k_mock_path_payload_tag{101};
    auto my_rank = *distributed_context->rank();
    auto world_size = *distributed_context->size();

    std::string my_path;
    const char* my_path_env = std::getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
    if (my_path_env && std::strlen(my_path_env) > 0) {
        my_path.assign(my_path_env);
        std::error_code ec;
        auto resolved = std::filesystem::absolute(std::filesystem::path(my_path), ec);
        if (!ec) {
            my_path = resolved.string();
        }
    }

    if (world_size == 1) {
        if (!my_path.empty()) {
            mpi_rank_to_path[0] = std::move(my_path);
        }
        return;
    }

    if (my_rank == root_rank) {
        if (!my_path.empty()) {
            mpi_rank_to_path[root_rank] = my_path;
        }
        for (int r = 1; r < static_cast<int>(world_size); ++r) {
            std::size_t path_size = 0;
            distributed_context->recv(
                ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&path_size), sizeof(path_size)),
                Rank{r},
                k_mock_path_size_tag);
            std::vector<char> path_buf(path_size);
            if (path_size > 0) {
                distributed_context->recv(
                    ttsl::as_writable_bytes(
                        ttsl::Span<uint8_t>(reinterpret_cast<uint8_t*>(path_buf.data()), path_buf.size())),
                    Rank{r},
                    k_mock_path_payload_tag);
            }
            std::string path(path_buf.begin(), path_buf.end());
            if (!path.empty()) {
                mpi_rank_to_path[r] = std::move(path);
            }
        }
    } else {
        std::size_t path_size = my_path.size();
        distributed_context->send(
            ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(&path_size), sizeof(path_size)),
            Rank{root_rank},
            k_mock_path_size_tag);
        if (path_size > 0) {
            std::vector<uint8_t> path_bytes(my_path.begin(), my_path.end());
            distributed_context->send(
                ttsl::as_writable_bytes(ttsl::Span<uint8_t>(path_bytes.data(), path_bytes.size())),
                Rank{root_rank},
                k_mock_path_payload_tag);
        }
    }
    distributed_context->barrier();
}

struct ProgramArgs {
    std::string mesh_graph_descriptor_path;
    std::optional<std::string> physical_grouping_descriptor_path;
    std::optional<std::string> output_dir;
};

/**
 * @brief Parse command line arguments
 */
ProgramArgs parse_arguments(int argc, char** argv) {
    cxxopts::Options options(
        "generate_rank_bindings",
        "Generate rank bindings YAML file from Physical System Descriptor (PSD) discovery and topology mapping.\n"
        "Requires a Metal build with Open MPI (USE_MPI) enabled; run under an MPI launcher (e.g. mpirun, srun).\n"
        "Single-process runs are allowed (e.g. mpirun -np 1) when mapping a single-rank allocation.\n\n"
        "The Mesh Graph Descriptor (MGD) specifies the logical mesh topology.\n"
        "The Physical Grouping Descriptor (PGD) is optional and will be searched using fallback logic if not "
        "provided.");

    options.add_options()(
        "m,mesh-graph-descriptor",
        "Path to Mesh Graph Descriptor file (.textproto) - REQUIRED",
        cxxopts::value<std::string>())(
        "p,physical-grouping-descriptor",
        "Path to Physical Grouping Descriptor file (.textproto) - OPTIONAL",
        cxxopts::value<std::string>())(
        "o,output-dir",
        "Output directory for rank_bindings.yaml, rankfile, etc. (default: generated/ttrun)",
        cxxopts::value<std::string>())("h,help", "Print usage information");

    try {
        const auto result = options.parse(argc, argv);

        if (result.contains("help") || argc == 1) {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (!result.contains("mesh-graph-descriptor")) {
            throw std::invalid_argument("--mesh-graph-descriptor (-m) is required");
        }

        ProgramArgs args;
        args.mesh_graph_descriptor_path = result["mesh-graph-descriptor"].as<std::string>();

        if (result.contains("physical-grouping-descriptor")) {
            args.physical_grouping_descriptor_path = result["physical-grouping-descriptor"].as<std::string>();
        }
        if (result.contains("output-dir")) {
            args.output_dir = result["output-dir"].as<std::string>();
        }

        return args;

    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        exit(1);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    // Parse arguments first (before MPI initialization)
    ProgramArgs args = parse_arguments(argc, argv);

    tt::tt_metal::distributed::multihost::DistributedContext::create(argc, argv);

    const auto& context = tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();

    try {
        log_info(tt::LogFabric, "Generating rank bindings...");

        // Stage: Run PSD discovery
        log_info(tt::LogFabric, "Stage: Running Physical System Descriptor discovery...");
        PhysicalSystemDescriptor psd = run_psd_discovery();
        log_info(tt::LogFabric, "PSD discovery complete");

        // Gather mock cluster descriptor paths from all ranks (when TT_METAL_MOCK_CLUSTER_DESC_PATH is set)
        std::map<int, std::string> mpi_rank_to_cluster_desc_path;
        gather_mock_cluster_desc_paths(context, mpi_rank_to_cluster_desc_path);

        // Stage: Load Mesh Graph Descriptor
        std::filesystem::path mgd_path(args.mesh_graph_descriptor_path);
        log_info(tt::LogFabric, "Stage: Loading Mesh Graph Descriptor from: {}", mgd_path.string());
        if (!std::filesystem::exists(mgd_path) || !std::filesystem::is_regular_file(mgd_path)) {
            throw std::runtime_error("Mesh Graph Descriptor file does not exist: " + mgd_path.string());
        }
        MeshGraphDescriptor mgd(mgd_path);
        log_info(tt::LogFabric, "Mesh Graph Descriptor loaded");

        PhysicalGroupingDescriptor pgd = find_and_load_physical_grouping_descriptor(
            args.physical_grouping_descriptor_path.has_value()
                ? std::optional<std::filesystem::path>(*args.physical_grouping_descriptor_path)
                : std::nullopt,
            &psd);

        // Get current rank - only rank 0 performs topology mapping and file generation
        auto current_rank = *context->rank();
        if (current_rank == 0) {
            // Stage: Run topology mapping
            log_info(tt::LogFabric, "Stage: Running topology mapping...");
            TopologyMappingResult mapping_result = run_topology_mapping(psd, pgd, mgd, mgd_path);

            if (!mapping_result.success) {
                log_error(tt::LogFabric, "Topology mapping failed: {}", mapping_result.error_message);
                return 1;
            }
            log_info(tt::LogFabric, "Topology mapping complete");

            // Stage: Extract rank bindings
            log_info(tt::LogFabric, "Stage: Extracting rank bindings...");
            // Create MeshGraph for getting host ranks
            auto& context = tt::tt_metal::MetalContext::instance();
            const auto& cluster = context.get_cluster();
            MeshGraph mesh_graph(cluster, mgd_path.string());
            std::vector<RankBindingConfig> rank_bindings = extract_rank_bindings(psd, mapping_result, mesh_graph);
            log_info(tt::LogFabric, "Extracted {} rank binding(s)", rank_bindings.size());

            // Stage: Write YAML file
            log_info(tt::LogFabric, "Stage: Writing rank bindings to YAML...");

            std::filesystem::path output_dir =
                args.output_dir.has_value() ? std::filesystem::path(*args.output_dir) : "generated/ttrun";
            std::filesystem::create_directories(output_dir);

            std::filesystem::path output_file = output_dir / "rank_bindings.yaml";
            write_rank_bindings_yaml(rank_bindings, args.mesh_graph_descriptor_path, output_file.string());
            log_info(tt::LogFabric, "Successfully wrote: {}", output_file.string());

            std::filesystem::path rankfile_path = output_dir / "rankfile";
            const bool mock_cluster_rankfile = !mpi_rank_to_cluster_desc_path.empty();
            write_rankfile(rank_bindings, rankfile_path.string(), mock_cluster_rankfile);
            log_info(tt::LogFabric, "Successfully wrote: {}", rankfile_path.string());

            if (!mpi_rank_to_cluster_desc_path.empty()) {
                std::filesystem::path phase2_mock_path = output_dir / "phase2_mock_mapping.yaml";
                write_phase2_mock_mapping_yaml(rank_bindings, mpi_rank_to_cluster_desc_path, phase2_mock_path.string());
                log_info(
                    tt::LogFabric,
                    "Successfully wrote: {} (cluster descriptors used during allocation)",
                    phase2_mock_path.string());
            }

            // Flush all output files to storage before signaling peers via barrier.
            // std::ofstream::close() only drains the C++ stream buffer to the OS page cache.
            // Without fsync(), NFS peers (and local readers) may see stale or absent files
            // even after generate_rank_bindings exits.  We fsync each file and its parent
            // directory so that both data and directory entries are durable before we call
            // barrier() below — making the barrier the authoritative "writes are visible"
            // signal and allowing ttrun.py to skip any blind sleep after this subprocess.
            auto fsync_path = [](const std::filesystem::path& p) noexcept {
                int fd = ::open(p.c_str(), O_RDONLY);
                if (fd >= 0) {
                    ::fsync(fd);
                    ::close(fd);
                }
                int dir_fd = ::open(p.parent_path().c_str(), O_RDONLY | O_DIRECTORY);
                if (dir_fd >= 0) {
                    ::fsync(dir_fd);
                    ::close(dir_fd);
                }
            };
            fsync_path(output_file);
            fsync_path(rankfile_path);
            if (!mpi_rank_to_cluster_desc_path.empty()) {
                fsync_path(output_dir / "phase2_mock_mapping.yaml");
            }
            log_info(tt::LogFabric, "Fsynced output files; barrier will signal peers that writes are visible.");

            log_info(tt::LogFabric, "Rank bindings generation complete!");
        } else {
            log_info(
                tt::LogFabric,
                "Rank {}: Skipping topology mapping and file generation (only rank 0 performs these operations)",
                current_rank);
        }

        // Synchronize all ranks before exiting
        context->barrier();

    } catch (const std::exception& e) {
        log_error(tt::LogFabric, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
