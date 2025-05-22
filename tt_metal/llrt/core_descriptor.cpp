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

#include "assert.hpp"
#include "hal.hpp"
#include "hal_types.hpp"
#include "metal_soc_descriptor.h"
#include "tt_backend_api_types.hpp"
#include "impl/context/metal_context.hpp"
#include <umd/device/tt_core_coordinates.h>
#include <umd/device/types/arch.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <umd/device/types/xy_pair.h>
#include "utils.hpp"

namespace tt {

inline std::string get_core_descriptor_file(
    const tt::ARCH& arch, const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) {
    // Ability to skip this runtime opt, since trimmed SOC desc limits which DRAM channels are available.
    string core_desc_dir;
    if (getenv("TT_METAL_HOME")) {
        core_desc_dir = getenv("TT_METAL_HOME");
    } else {
        core_desc_dir = "./";
    }
    if (core_desc_dir.back() != '/') {
        core_desc_dir += "/";
    }
    core_desc_dir += "tt_metal/core_descriptors/";

    bool targeting_sim = tt_metal::MetalContext::instance().rtoptions().get_simulator_enabled();
    if (targeting_sim) {
        switch (arch) {
            default:
                throw std::runtime_error(
                    "Invalid arch not supported");  // will be overwritten in tt_global_state constructor
            case tt::ARCH::WORMHOLE_B0: return core_desc_dir + "wormhole_b0_versim_1x1_arch.yaml";
            case tt::ARCH::BLACKHOLE: return core_desc_dir + "blackhole_simulation_1x2_arch.yaml";
            case tt::ARCH::QUASAR: TT_THROW("No core descriptor for Quasar"); break;
        };
    } else {
        switch (arch) {
            default:
                throw std::runtime_error(
                    "Invalid arch not supported");  // will be overwritten in tt_global_state constructor
            case tt::ARCH::WORMHOLE_B0:
                return core_desc_dir + (dispatch_core_config.get_core_type() == CoreType::ETH
                                            ? "wormhole_b0_80_arch_eth_dispatch.yaml"
                                            : "wormhole_b0_80_arch.yaml");
            case tt::ARCH::BLACKHOLE:
                return core_desc_dir + (dispatch_core_config.get_core_type() == CoreType::ETH
                                            ? "blackhole_140_arch_eth_dispatch.yaml"
                                            : "blackhole_140_arch.yaml");
            case tt::ARCH::QUASAR: TT_THROW("No core descriptor for Quasar"); break;
        };
    }
    return "";
}

const core_descriptor_t& get_core_descriptor_config(
    chip_id_t device_id, const uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    // {arch : {product : {dispatch core axis: {num hardware command queues : config}}}}
    static std::unordered_map<
        ARCH,
        std::unordered_map<
            std::string,
            std::unordered_map<tt_metal::DispatchCoreConfig, std::unordered_map<uint8_t, core_descriptor_t>>>>
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

    std::unordered_map<uint8_t, core_descriptor_t>& config_by_num_cqs =
        config_by_arch[arch][product_name][dispatch_core_config];
    if (config_by_num_cqs.count(num_hw_cqs)) {
        return config_by_num_cqs.at(num_hw_cqs);
    }

    YAML::Node core_descriptor_yaml = YAML::LoadFile(get_core_descriptor_file(arch, dispatch_core_config));
    YAML::Node desc_yaml =
        core_descriptor_yaml[product_name]
                            [(dispatch_core_config.get_dispatch_core_axis() == tt_metal::DispatchCoreAxis::ROW) ? "row"
                                                                                                                : "col"]
                            [std::to_string(num_hw_cqs)];

    // Parse the yaml into core_descriptor_t
    std::vector<RelativeCoreCoord> storage_cores;
    for (const auto& core_node : desc_yaml["storage_cores"]) {
        RelativeCoreCoord coord = {};
        if (core_node.IsSequence()) {
            // Logical coord
            coord = RelativeCoreCoord({.x = core_node[0].as<int>(), .y = core_node[1].as<int>()});
        } else {
            TT_THROW("Only logical relative coords supported for storage_cores cores");
        }
        storage_cores.push_back(coord);
    }
    std::optional<uint32_t> storage_core_bank_size = std::nullopt;
    if (not storage_cores.empty()) {
        try {
            storage_core_bank_size = desc_yaml["storage_core_bank_size"].as<uint32_t>();
        } catch (std::runtime_error& ex) {
            TT_THROW(
                "Core descriptor yaml for {} needs to specify storage_core_bank_size since there are {} storage cores!",
                get_string_lowercase(arch),
                storage_cores.size());
        }
    }

    auto compute_with_storage_start = desc_yaml["compute_with_storage_grid_range"]["start"];
    auto compute_with_storage_end = desc_yaml["compute_with_storage_grid_range"]["end"];
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster() and product_name == "nebula_x1") {
        compute_with_storage_start = desc_yaml["tg_compute_with_storage_grid_range"]["start"];
        compute_with_storage_end = desc_yaml["tg_compute_with_storage_grid_range"]["end"];
    }
    TT_ASSERT(compute_with_storage_start.IsSequence() and compute_with_storage_end.IsSequence());
    TT_ASSERT(compute_with_storage_end[0].as<size_t>() >= compute_with_storage_start[0].as<size_t>());
    TT_ASSERT(compute_with_storage_end[1].as<size_t>() >= compute_with_storage_start[1].as<size_t>());
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
    auto dispatch_cores_string = "dispatch_cores";
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster() and product_name == "nebula_x1") {
        dispatch_cores_string = "tg_dispatch_cores";
    }

    CoreCoord grid_size =
        tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id).get_grid_size(CoreType::TENSIX);
    auto logical_active_eth_cores =
        tt::tt_metal::MetalContext::instance().get_cluster().get_active_ethernet_cores(device_id);

    for (const auto& core_node : desc_yaml[dispatch_cores_string]) {
        RelativeCoreCoord coord = {};
        if (core_node.IsSequence()) {
            // Logical coord
            coord = RelativeCoreCoord({.x = core_node[0].as<int>(), .y = core_node[1].as<int>()});
            if (dispatch_core_config.get_core_type() == CoreType::ETH) {
                auto logical_coord = get_core_coord_from_relative(coord, grid_size);
                if (logical_active_eth_cores.find(logical_coord) != logical_active_eth_cores.end()) {
                    continue;
                }
            }
        } else {
            TT_THROW("Only logical relative coords supported for dispatch_cores cores");
        }
        dispatch_cores.push_back(coord);
    }
    TT_ASSERT(
        dispatch_cores.size() || tt_metal::MetalContext::instance().rtoptions().get_simulator_enabled(),
        "Dispatch cores size must be positive");

    std::vector<CoreCoord> logical_compute_cores;
    logical_compute_cores.reserve(compute_cores.size());
    std::transform(
        compute_cores.cbegin(),
        compute_cores.cend(),
        std::back_inserter(logical_compute_cores),
        [&grid_size](RelativeCoreCoord rel_coord) { return get_core_coord_from_relative(rel_coord, grid_size); });

    std::vector<CoreCoord> logical_storage_cores;
    logical_storage_cores.reserve(storage_cores.size());
    std::transform(
        storage_cores.cbegin(),
        storage_cores.cend(),
        std::back_inserter(logical_storage_cores),
        [&grid_size](RelativeCoreCoord rel_coord) { return get_core_coord_from_relative(rel_coord, grid_size); });

    std::vector<CoreCoord> logical_dispatch_cores;
    logical_dispatch_cores.reserve(dispatch_cores.size());
    std::transform(
        dispatch_cores.cbegin(),
        dispatch_cores.cend(),
        std::back_inserter(logical_dispatch_cores),
        [&grid_size](RelativeCoreCoord rel_coord) { return get_core_coord_from_relative(rel_coord, grid_size); });

    auto [it, _] = config_by_num_cqs.emplace(std::make_pair(
        num_hw_cqs,
        core_descriptor_t{
            .compute_grid_size = std::move(compute_grid_size),
            .relative_compute_cores = std::move(compute_cores),
            .relative_storage_cores = std::move(storage_cores),
            .storage_core_bank_size = std::move(storage_core_bank_size),
            .relative_dispatch_cores = std::move(dispatch_cores),
            .logical_compute_cores = std::move(logical_compute_cores),
            .logical_storage_cores = std::move(logical_storage_cores),
            .logical_dispatch_cores = std::move(logical_dispatch_cores),
        }));
    return it->second;
}

const std::tuple<uint32_t, CoreRange>& get_physical_worker_grid_config(
    chip_id_t device_id, uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    // Get logical compute grid dimensions and num workers
    static std::unordered_map<uint32_t, std::tuple<uint32_t, CoreRange>> physical_grid_config_cache = {};
    // Unique hash generated based on the config that's being queried
    uint32_t config_hash = ((uint8_t)(dispatch_core_config.get_core_type())) |
                           ((uint8_t)(dispatch_core_config.get_dispatch_core_axis()) << 4) | (num_hw_cqs << 8) |
                           (device_id << 16);
    if (physical_grid_config_cache.find(config_hash) == physical_grid_config_cache.end()) {
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

std::optional<uint32_t> get_storage_core_bank_size(
    chip_id_t device_id, const uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    const core_descriptor_t& core_desc = get_core_descriptor_config(device_id, num_hw_cqs, dispatch_core_config);
    const metal_SocDescriptor& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(device_id);
    if (core_desc.storage_core_bank_size.has_value()) {
        TT_FATAL(
            core_desc.storage_core_bank_size.value() %
                    tt_metal::MetalContext::instance().hal().get_alignment(tt_metal::HalMemType::L1) ==
                0,
            "Storage core bank size must be {} B aligned",
            tt_metal::MetalContext::instance().hal().get_alignment(tt_metal::HalMemType::L1));
    }
    return core_desc.storage_core_bank_size;
}

}  // namespace tt
