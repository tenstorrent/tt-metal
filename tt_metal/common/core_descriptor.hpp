// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core_coord.h"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "yaml-cpp/yaml.h"

namespace tt {

struct core_descriptor_t {
    CoreCoord compute_grid_size;
    std::vector<RelativeCoreCoord> relative_compute_cores;
    std::vector<RelativeCoreCoord> relative_storage_cores;
    uint32_t storage_core_bank_size;
    std::vector<RelativeCoreCoord> relative_dispatch_cores;
    CoreType dispatch_core_type;
};

inline std::string get_core_descriptor_file(const tt::ARCH &arch) {

    // Ability to skip this runtime opt, since trimmed SOC desc limits which DRAM channels are available.
    string tt_metal_home;
    string wh_arch = "tt_metal/core_descriptors/wormhole_b0_80_arch.yaml";
    if (getenv("TT_METAL_HOME")) {
        tt_metal_home = getenv("TT_METAL_HOME");
    } else {
        tt_metal_home = "./";
    }
    if (tt_metal_home.back() != '/') {
        tt_metal_home += "/";
    }
    if (getenv("WH_ARCH_YAML")) {
        wh_arch = "tt_metal/core_descriptors/";
        wh_arch += getenv("WH_ARCH_YAML");
    }
    bool targeting_versim = false;
#ifndef TT_METAL_VERSIM_DISABLED
    targeting_versim = true;
#endif

    if (targeting_versim) {
        switch (arch) {
            case tt::ARCH::Invalid: throw std::runtime_error("Invalid arch not supported"); // will be overwritten in tt_global_state constructor
            case tt::ARCH::JAWBRIDGE: throw std::runtime_error("JAWBRIDGE arch not supported");
            case tt::ARCH::GRAYSKULL: return tt_metal_home + "tt_metal/core_descriptors/grayskull_versim_1x1_arch.yaml";
            case tt::ARCH::WORMHOLE: throw std::runtime_error("WORMHOLE arch not supported");
            case tt::ARCH::WORMHOLE_B0: return tt_metal_home + "tt_metal/core_descriptors/wormhole_b0_versim_1x1_arch.yaml";
            default: throw std::runtime_error("Unsupported device arch");
        };
    } else {
        switch (arch) {
            case tt::ARCH::Invalid: throw std::runtime_error("Invalid arch not supported"); // will be overwritten in tt_global_state constructor
            case tt::ARCH::JAWBRIDGE: throw std::runtime_error("JAWBRIDGE arch not supported");
            case tt::ARCH::GRAYSKULL: return tt_metal_home + "tt_metal/core_descriptors/grayskull_120_arch.yaml";
            case tt::ARCH::WORMHOLE: throw std::runtime_error("WORMHOLE arch not supported");
            case tt::ARCH::WORMHOLE_B0: return tt_metal_home + wh_arch;
            default: throw std::runtime_error("Unsupported device arch");
        };
    }
    return "";
}

inline const std::string get_product_name(tt::ARCH arch, uint32_t num_harvested_rows) {
    const static std::map<tt::ARCH, std::map<uint32_t, std::string>> product_name = {
        {tt::ARCH::GRAYSKULL, {{0, "E150"}, {2, "E75"}}},
        {tt::ARCH::WORMHOLE_B0, {{0, "galaxy"}, {1, "nebula_x1"}, {2, "nebula_x2"}}}};

    return product_name.at(arch).at(num_harvested_rows);
}

inline const core_descriptor_t &get_core_descriptor_config(chip_id_t device_id, const uint8_t num_hw_cqs) {
    // {arch : {product : {num hardware command queues : config}}}
    static std::unordered_map<ARCH, std::unordered_map<std::string, std::unordered_map<uint8_t, core_descriptor_t>>> config_by_arch;

    ARCH arch = tt::Cluster::instance().arch();
    uint32_t harvesting_mask = tt::Cluster::instance().get_harvested_rows(device_id);
    std::bitset<32> mask_bitset(harvesting_mask);
    uint32_t num_harvested_rows = mask_bitset.count();

    std::string product_name = get_product_name(arch, num_harvested_rows);

    if (num_harvested_rows > 2) {
        TT_THROW("At most two rows can be harvested, but detected {} harvested rows", num_harvested_rows);
    }
    if (num_harvested_rows == 1 and arch == tt::ARCH::GRAYSKULL) {
        TT_THROW("One row harvested Grayskull is not supported");
    }

    if (config_by_arch.count(arch) and config_by_arch.at(arch).count(product_name) and config_by_arch.at(arch).at(product_name).count(num_hw_cqs)) {
        return config_by_arch.at(arch).at(product_name).at(num_hw_cqs);
    }

    std::unordered_map<std::string, std::unordered_map<uint8_t, core_descriptor_t>> &config_by_product = config_by_arch[arch];
    std::unordered_map<uint8_t, core_descriptor_t> &config_by_num_cqs = config_by_product[product_name];

    YAML::Node core_descriptor_yaml = YAML::LoadFile(get_core_descriptor_file(arch));
    YAML::Node desc_yaml = core_descriptor_yaml[product_name][std::to_string(num_hw_cqs)];

    // Parse the yaml into core_descriptor_t
    uint32_t storage_core_bank_size = desc_yaml["l1_bank_size"].as<uint32_t>();
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

    auto compute_with_storage_start = desc_yaml["compute_with_storage_grid_range"]["start"];
    auto compute_with_storage_end = desc_yaml["compute_with_storage_grid_range"]["end"];
    TT_ASSERT(compute_with_storage_start.IsSequence() and compute_with_storage_end.IsSequence());
    TT_ASSERT(compute_with_storage_end[0].as<size_t>() >= compute_with_storage_start[0].as<size_t>());
    TT_ASSERT(compute_with_storage_end[1].as<size_t>() >= compute_with_storage_start[1].as<size_t>());
    CoreCoord compute_grid_size(
        (compute_with_storage_end[0].as<size_t>() - compute_with_storage_start[0].as<size_t>()) + 1,
        (compute_with_storage_end[1].as<size_t>() - compute_with_storage_start[1].as<size_t>()) + 1
    );

    std::vector<RelativeCoreCoord> compute_cores;
    for (auto x = 0; x < compute_grid_size.x; x++) {
        for (auto y = 0; y < compute_grid_size.y; y++) {
            const RelativeCoreCoord relative_coord{.x=x, .y=y};
            compute_cores.push_back(relative_coord);
        }
    }

    std::vector<RelativeCoreCoord> dispatch_cores;
    for (const auto& core_node : desc_yaml["dispatch_cores"]) {
        RelativeCoreCoord coord = {};
        if (core_node.IsSequence()) {
            // Logical coord
            coord = RelativeCoreCoord({.x = core_node[0].as<int>(), .y = core_node[1].as<int>()});
        } else {
            TT_THROW("Only logical relative coords supported for dispatch_cores cores");
        }
        dispatch_cores.push_back(coord);
    }
    TT_ASSERT(dispatch_cores.size(), "Dispatch cores size must be positive");

    auto dispatch_core_type = arch == tt::ARCH::GRAYSKULL ? "tensix" : desc_yaml["dispatch_core_type"].as<std::string>();
    CoreType core_type;
    if (dispatch_core_type == "tensix") {
        core_type = CoreType::WORKER;
    } else if (dispatch_core_type == "ethernet") {
        core_type = CoreType::ETH;
    } else {
        TT_THROW("Only tensix/ethernet cores supported for dispatch_core_type");
    }

    config_by_num_cqs[num_hw_cqs] = core_descriptor_t{
        .compute_grid_size = compute_grid_size,
        .relative_compute_cores = compute_cores,
        .relative_storage_cores = storage_cores,
        .storage_core_bank_size = storage_core_bank_size,
        .relative_dispatch_cores = dispatch_cores,
        .dispatch_core_type = core_type
    };
    return config_by_arch.at(arch).at(product_name).at(num_hw_cqs);
}

inline uint32_t get_storage_core_bank_size(chip_id_t device_id, const uint8_t num_hw_cqs) {
    const core_descriptor_t &core_desc = get_core_descriptor_config(device_id, num_hw_cqs);
    return core_desc.storage_core_bank_size;
}

inline const std::vector<CoreCoord> &get_logical_storage_cores(chip_id_t device_id, const uint8_t num_hw_cqs) {
    const core_descriptor_t &core_desc = get_core_descriptor_config(device_id, num_hw_cqs);
    static std::unordered_map<chip_id_t, std::vector<CoreCoord>> logical_storage_cores_by_device;
    if (logical_storage_cores_by_device.count(device_id)) {
        return logical_storage_cores_by_device.at(device_id);
    }
    CoreCoord grid_size = tt::Cluster::instance().get_soc_desc(device_id).worker_grid_size;
    std::vector<CoreCoord> &logical_storage_cores = logical_storage_cores_by_device[device_id];
    std::transform(core_desc.relative_storage_cores.cbegin(), core_desc.relative_storage_cores.cend(), std::back_inserter(logical_storage_cores),
                [&grid_size](RelativeCoreCoord rel_coord) { return get_core_coord_from_relative(rel_coord, grid_size); });
    return logical_storage_cores;
}

inline CoreCoord get_compute_grid_size(chip_id_t device_id, const uint8_t num_hw_cqs) {
    const core_descriptor_t &core_desc = get_core_descriptor_config(device_id, num_hw_cqs);
    return core_desc.compute_grid_size;
}

inline const std::vector<CoreCoord> &get_logical_compute_cores(chip_id_t device_id, const uint8_t num_hw_cqs) {
    const core_descriptor_t &core_desc = get_core_descriptor_config(device_id, num_hw_cqs);
    static std::unordered_map<chip_id_t, std::vector<CoreCoord>> logical_compute_cores_by_device;
    if (logical_compute_cores_by_device.count(device_id)) {
        return logical_compute_cores_by_device.at(device_id);
    }
    CoreCoord grid_size = tt::Cluster::instance().get_soc_desc(device_id).worker_grid_size;
    std::vector<CoreCoord> &logical_compute_cores = logical_compute_cores_by_device[device_id];
    std::transform(core_desc.relative_compute_cores.cbegin(), core_desc.relative_compute_cores.cend(), std::back_inserter(logical_compute_cores),
                [&grid_size](RelativeCoreCoord rel_coord) { return get_core_coord_from_relative(rel_coord, grid_size); });
    return logical_compute_cores;
}

inline const std::vector<CoreCoord> &get_logical_dispatch_cores(chip_id_t device_id, const uint8_t num_hw_cqs) {
    const core_descriptor_t &core_desc = get_core_descriptor_config(device_id, num_hw_cqs);
    static std::unordered_map<chip_id_t, std::vector<CoreCoord>> logical_dispatch_cores_by_device;
    if (logical_dispatch_cores_by_device.count(device_id)) {
        return logical_dispatch_cores_by_device.at(device_id);
    }
    CoreCoord grid_size = tt::Cluster::instance().get_soc_desc(device_id).worker_grid_size;
    std::vector<CoreCoord> &logical_dispatch_cores = logical_dispatch_cores_by_device[device_id];
    std::transform(core_desc.relative_dispatch_cores.cbegin(), core_desc.relative_dispatch_cores.cend(), std::back_inserter(logical_dispatch_cores),
                [&grid_size](RelativeCoreCoord rel_coord) { return get_core_coord_from_relative(rel_coord, grid_size); });
    return logical_dispatch_cores;
}

inline const CoreType get_dispatch_core_type(chip_id_t device_id, const uint8_t num_hw_cqs) {
    const core_descriptor_t &core_desc = get_core_descriptor_config(device_id, num_hw_cqs);
    static std::unordered_map<chip_id_t, CoreType> dispatch_core_type_by_device;
    if (dispatch_core_type_by_device.count(device_id)) {
        return dispatch_core_type_by_device.at(device_id);
    }
    dispatch_core_type_by_device[device_id] = core_desc.dispatch_core_type;
    return core_desc.dispatch_core_type;
}

/// @brief Get physical core coordinate from a logical location (device ID + core coordinate)
/// @param logical_location tt_cxy_pair describing chip and logical location core coordinate
/// @param core_type CoreType of core to translate
/// @return physical CoreCoord on the same chip as `logical_location`
inline CoreCoord get_physical_core_coordinate(const tt_cxy_pair &logical_location, const CoreType &core_type) {
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(logical_location.chip);
    return soc_desc.get_physical_core_from_logical_core(CoreCoord(logical_location.x, logical_location.y), core_type);
}

}   // namespace tt
