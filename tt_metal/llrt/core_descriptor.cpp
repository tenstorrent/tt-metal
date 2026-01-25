// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core_descriptor.hpp"

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <bitset>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include "common/core_coord.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "common/tt_backend_api_types.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/simulation/simulation_chip.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/xy_pair.hpp>
#include <llrt/tt_cluster.hpp>

namespace tt {

using tt_metal::RelativeCoreCoord;

// Convert "x,y" to YAML::Node sequence [x, y]
inline YAML::Node string_to_yaml_node(const std::string& input) {
    // Format as YAML sequence: "[x, y]"
    std::string yaml_seq = "[" + input + "]";
    return YAML::Load(yaml_seq);
}

inline std::string get_core_descriptor_file(
    const tt::ARCH& arch, const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) {
    // Ability to skip this runtime opt, since trimmed SOC desc limits which DRAM channels are available.
    std::string core_desc_dir = tt_metal::MetalContext::instance().rtoptions().get_root_dir();
    if (core_desc_dir.back() != '/') {
        core_desc_dir += "/";
    }
    core_desc_dir += "tt_metal/core_descriptors/";

    bool use_small_core_desc_yaml = false; // override to a different core descriptor for small RTL sims
    if (tt_metal::MetalContext::instance().rtoptions().get_simulator_enabled()) {
        auto soc_desc = tt::umd::SimulationChip::get_soc_descriptor_path_from_simulator_path(
            tt_metal::MetalContext::instance().rtoptions().get_simulator_path());
        tt_xy_pair grid_size = tt::umd::SocDescriptor::get_grid_size_from_soc_descriptor_path(soc_desc);
        if (grid_size.y <= 2) {  // these SOC descriptors declare a 2x2 grid
            use_small_core_desc_yaml = true;
        }
    }
    if (use_small_core_desc_yaml) {
        switch (arch) {
            default:
                throw std::runtime_error(
                    "Invalid arch not supported");  // will be overwritten in tt_global_state constructor
            case tt::ARCH::WORMHOLE_B0: return core_desc_dir + "wormhole_b0_versim_1x1_arch.yaml";
            case tt::ARCH::BLACKHOLE: return core_desc_dir + "blackhole_simulation_1x2_arch.yaml";
            case tt::ARCH::QUASAR: return core_desc_dir + "quasar_simulation_1x3_arch.yaml";
        };
    } else {
        // Check if fabric tensix is enabled based on fabric tensix config
        tt_fabric::FabricTensixConfig fabric_tensix_config =
            tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
        bool use_fabric_tensix = (fabric_tensix_config != tt_fabric::FabricTensixConfig::DISABLED);

        auto core_type = get_core_type_from_config(dispatch_core_config);
        switch (arch) {
            default:
                throw std::runtime_error(
                    "Invalid arch not supported");  // will be overwritten in tt_global_state constructor
            case tt::ARCH::WORMHOLE_B0:
                if (core_type == CoreType::ETH) {
                    return core_desc_dir + "wormhole_b0_80_arch_eth_dispatch.yaml";
                } else if (use_fabric_tensix) {
                    return core_desc_dir + "wormhole_b0_80_arch_fabric_mux.yaml";
                } else {
                    return core_desc_dir + "wormhole_b0_80_arch.yaml";
                }
            case tt::ARCH::BLACKHOLE:
                if (core_type == CoreType::ETH) {
                    return core_desc_dir + "blackhole_140_arch_eth_dispatch.yaml";
                } else if (use_fabric_tensix) {
                    return core_desc_dir + "blackhole_140_arch_fabric_mux.yaml";
                } else {
                    return core_desc_dir + "blackhole_140_arch.yaml";
                }
            case tt::ARCH::QUASAR: return core_desc_dir + "quasar_simulation_1x3_arch.yaml";
        };
    }
    return "";
}

const core_descriptor_t& get_core_descriptor_config(
    ChipId device_id, const uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    // {arch : {product : {dispatch core axis: {fabric tensix config: {num hardware command queues : config}}}}}
    static std::unordered_map<
        ARCH,
        std::unordered_map<
            std::string,
            std::unordered_map<
                tt_metal::DispatchCoreConfig,
                std::unordered_map<tt_fabric::FabricTensixConfig, std::unordered_map<uint8_t, core_descriptor_t>>>>>
        config_by_arch;

    ARCH arch = tt::tt_metal::MetalContext::instance().get_cluster().arch();
    uint32_t harvesting_mask = tt::tt_metal::MetalContext::instance().get_cluster().get_harvesting_mask(device_id);
    std::bitset<32> mask_bitset(harvesting_mask);
    uint32_t num_harvested_on_axis = mask_bitset.count();

    if (num_harvested_on_axis > 2) {
        TT_THROW(
            "At most two rows or cols can be harvested, but detected {} along harvested axis", num_harvested_on_axis);
    }

    std::string product_name = get_product_name(arch, num_harvested_on_axis);
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(device_id) == BoardType::N150) {
            // some Galaxy machines are setup with N150s that have 0 harvested rows.
            // get_product_name ( ) returns those chips as galaxy. Override that to nebula_x1.
            product_name = "nebula_x1";
        } else {
            TT_ASSERT(
                tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(device_id) == BoardType::GALAXY,
                "Invalid Board Type in Galaxy Cluster. Only GALAXY and N150 are supported.");
        }
    }

    tt_fabric::FabricTensixConfig fabric_tensix_config =
        tt::tt_metal::MetalContext::instance().get_fabric_tensix_config();
    std::unordered_map<uint8_t, core_descriptor_t>& config_by_num_cqs =
        config_by_arch[arch][product_name][dispatch_core_config][fabric_tensix_config];
    if (config_by_num_cqs.contains(num_hw_cqs)) {
        return config_by_num_cqs.at(num_hw_cqs);
    }

    YAML::Node core_descriptor_yaml = YAML::LoadFile(get_core_descriptor_file(arch, dispatch_core_config));
    YAML::Node desc_yaml =
        core_descriptor_yaml[product_name]
                            [(dispatch_core_config.get_dispatch_core_axis() == tt_metal::DispatchCoreAxis::ROW) ? "row"
                                                                                                                : "col"]
                            [std::to_string(num_hw_cqs)];

    auto compute_with_storage_start = desc_yaml["compute_with_storage_grid_range"]["start"];
    auto compute_with_storage_end = desc_yaml["compute_with_storage_grid_range"]["end"];
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster() and product_name == "nebula_x1") {
        compute_with_storage_start = desc_yaml["tg_compute_with_storage_grid_range"]["start"];
        compute_with_storage_end = desc_yaml["tg_compute_with_storage_grid_range"]["end"];
    }
    TT_ASSERT(compute_with_storage_start.IsSequence() and compute_with_storage_end.IsSequence());
    TT_ASSERT(compute_with_storage_end[0].as<size_t>() >= compute_with_storage_start[0].as<size_t>());
    TT_ASSERT(compute_with_storage_end[1].as<size_t>() >= compute_with_storage_start[1].as<size_t>());
    // // Adjusts the core grid configuration based on the value of the environment variable
    if (tt_metal::MetalContext::instance().rtoptions().is_core_grid_override_todeprecate()) {
        auto compute_with_storage_end_override =
            string_to_yaml_node(tt_metal::MetalContext::instance().rtoptions().get_core_grid_override_todeprecate());
        TT_FATAL(
            compute_with_storage_end_override.IsSequence(),
            "compute_with_storage_end_override must be a YAML sequence");
        TT_FATAL(
            (compute_with_storage_end[0].as<int>() >= compute_with_storage_end_override[0].as<int>()),
            "compute_with_storage_end[0]= {} should be >= compute_with_storage_end_override[0]= {}",
            compute_with_storage_end[0].as<int>(),
            compute_with_storage_end_override[0].as<int>());
        TT_FATAL(
            compute_with_storage_end[1].as<int>() >= compute_with_storage_end_override[1].as<int>(),
            "compute_with_storage_end[1]= {} should be >= compute_with_storage_end_override[1]= {}",
            compute_with_storage_end[1].as<int>(),
            compute_with_storage_end_override[1].as<size_t>());
        TT_FATAL(
            compute_with_storage_end_override[0].as<int>() >= compute_with_storage_start[0].as<int>(),
            "compute_with_storage_end_override[0]= {} should be >= compute_with_storage_start[0]= {}",
            compute_with_storage_end_override[0].as<int>(),
            compute_with_storage_start[0].as<int>());
        TT_FATAL(
            compute_with_storage_end_override[1].as<int>() >= compute_with_storage_start[1].as<int>(),
            "compute_with_storage_end_override[1]= {} should be >= compute_with_storage_start[1]= {}",
            compute_with_storage_end_override[1].as<int>(),
            compute_with_storage_start[1].as<int>());
        compute_with_storage_end = compute_with_storage_end_override;
        log_warning(
            tt::LogDevice,
            "Overrided compute_with_storage_end [x, y]=[{}, {}]",
            compute_with_storage_end[0].as<std::string>(),
            compute_with_storage_end[1].as<std::string>());
    }
    CoreCoord compute_grid_size(
        (compute_with_storage_end[0].as<size_t>() - compute_with_storage_start[0].as<size_t>()) + 1,
        (compute_with_storage_end[1].as<size_t>() - compute_with_storage_start[1].as<size_t>()) + 1);

    std::vector<RelativeCoreCoord> compute_cores;
    for (auto x = 0; x < compute_grid_size.x; x++) {
        for (auto y = 0; y < compute_grid_size.y; y++) {
            const RelativeCoreCoord relative_coord{.x = x, .y = y};
            compute_cores.push_back(relative_coord);
        }
    }

    std::vector<RelativeCoreCoord> dispatch_cores;
    const auto* dispatch_cores_string = "dispatch_cores";
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster() and product_name == "nebula_x1") {
        dispatch_cores_string = "tg_dispatch_cores";
    }

    CoreCoord grid_size =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    // For mock devices, control plane doesn't exist, use empty set
    std::unordered_set<CoreCoord> logical_active_eth_cores;
    if (tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() != tt::TargetDevice::Mock) {
        logical_active_eth_cores =
            tt::tt_metal::MetalContext::instance().get_control_plane().get_active_ethernet_cores(device_id);
    }

    for (const auto& core_node : desc_yaml[dispatch_cores_string]) {
        RelativeCoreCoord coord = {};
        if (core_node.IsSequence()) {
            // Logical coord
            coord = RelativeCoreCoord({.x = core_node[0].as<int>(), .y = core_node[1].as<int>()});
            if (get_core_type_from_config(dispatch_core_config) == CoreType::ETH) {
                auto logical_coord = get_core_coord_from_relative(coord, grid_size);
                if (logical_active_eth_cores.contains(logical_coord)) {
                    continue;
                }
            }
        } else {
            TT_THROW("Only logical relative coords supported for dispatch_cores cores");
        }
        dispatch_cores.push_back(coord);
    }
    TT_ASSERT(
        !dispatch_cores.empty() || tt_metal::MetalContext::instance().rtoptions().get_simulator_enabled(),
        "Dispatch cores size must be positive");

    // Parse fabric_mux_cores
    std::vector<RelativeCoreCoord> fabric_mux_cores;
    if (desc_yaml["fabric_mux_cores"]) {
        for (const auto& core_node : desc_yaml["fabric_mux_cores"]) {
            RelativeCoreCoord coord = {};
            if (core_node.IsSequence()) {
                coord = RelativeCoreCoord({.x = core_node[0].as<int>(), .y = core_node[1].as<int>()});
            } else {
                TT_THROW("Only logical relative coords supported for fabric_mux_cores");
            }
            fabric_mux_cores.push_back(coord);
        }
    }

    std::vector<CoreCoord> logical_compute_cores;
    logical_compute_cores.reserve(compute_cores.size());
    std::transform(
        compute_cores.cbegin(),
        compute_cores.cend(),
        std::back_inserter(logical_compute_cores),
        [&grid_size](RelativeCoreCoord rel_coord) { return get_core_coord_from_relative(rel_coord, grid_size); });

    std::vector<CoreCoord> logical_dispatch_cores;
    logical_dispatch_cores.reserve(dispatch_cores.size());
    std::transform(
        dispatch_cores.cbegin(),
        dispatch_cores.cend(),
        std::back_inserter(logical_dispatch_cores),
        [&grid_size](RelativeCoreCoord rel_coord) { return get_core_coord_from_relative(rel_coord, grid_size); });

    // Convert fabric mux cores to logical coordinates
    std::vector<CoreCoord> logical_fabric_mux_cores;
    logical_fabric_mux_cores.reserve(fabric_mux_cores.size());
    std::transform(
        fabric_mux_cores.cbegin(),
        fabric_mux_cores.cend(),
        std::back_inserter(logical_fabric_mux_cores),
        [&grid_size](RelativeCoreCoord rel_coord) { return get_core_coord_from_relative(rel_coord, grid_size); });

    auto [it, _] = config_by_num_cqs.emplace(std::make_pair(
        num_hw_cqs,
        core_descriptor_t{
            .compute_grid_size = compute_grid_size,
            .relative_compute_cores = std::move(compute_cores),
            .relative_dispatch_cores = std::move(dispatch_cores),
            .relative_fabric_mux_cores = std::move(fabric_mux_cores),
            .logical_compute_cores = std::move(logical_compute_cores),
            .logical_dispatch_cores = std::move(logical_dispatch_cores),
            .logical_fabric_mux_cores = std::move(logical_fabric_mux_cores),
        }));
    return it->second;
}

const std::tuple<uint32_t, CoreRange>& get_physical_worker_grid_config(
    ChipId device_id, uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    // Get logical compute grid dimensions and num workers
    static std::unordered_map<uint32_t, std::tuple<uint32_t, CoreRange>> physical_grid_config_cache = {};
    // Unique hash generated based on the config that's being queried
    uint32_t config_hash = ((uint8_t)(get_core_type_from_config(dispatch_core_config))) |
                           ((uint8_t)(dispatch_core_config.get_dispatch_core_axis()) << 4) | (num_hw_cqs << 8) |
                           (device_id << 16);
    if (!physical_grid_config_cache.contains(config_hash)) {
        auto worker_grid = tt::get_compute_grid_size(device_id, num_hw_cqs, dispatch_core_config);
        std::size_t tensix_num_worker_cols = worker_grid.x;
        std::size_t tensix_num_worker_rows = worker_grid.y;
        uint32_t tensix_num_worker_cores = tensix_num_worker_cols * tensix_num_worker_rows;
        const metal_SocDescriptor& soc_desc =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);
        // Get physical compute grid range based on SOC Desc and Logical Coords
        // Logical Worker Coords start at 0,0
        CoreCoord tensix_worker_start_phys = soc_desc.get_physical_tensix_core_from_logical(CoreCoord(0, 0));
        CoreCoord tensix_worker_end_phys = soc_desc.get_physical_tensix_core_from_logical(
            CoreCoord(tensix_num_worker_cols - 1, tensix_num_worker_rows - 1));
        CoreRange tensix_worker_physical_grid = CoreRange(tensix_worker_start_phys, tensix_worker_end_phys);
        physical_grid_config_cache.insert(
            {config_hash, std::make_tuple(tensix_num_worker_cores, tensix_worker_physical_grid)});
    }
    return physical_grid_config_cache.at(config_hash);
}

}  // namespace tt
