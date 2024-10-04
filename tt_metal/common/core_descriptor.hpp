// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core_coord.h"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "llrt/hal.hpp"

namespace tt {

struct core_descriptor_t {
    CoreCoord compute_grid_size;
    std::vector<RelativeCoreCoord> relative_compute_cores;
    std::vector<RelativeCoreCoord> relative_storage_cores;
    std::optional<uint32_t> storage_core_bank_size = std::nullopt;
    std::vector<RelativeCoreCoord> relative_dispatch_cores;
};

inline std::string get_core_descriptor_file(const tt::ARCH &arch, CoreType dispatch_core_type) {

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

    bool targeting_sim = std::getenv("TT_METAL_SIMULATOR_EN") != nullptr;
    if (targeting_sim) {
        switch (arch) {
            case tt::ARCH::Invalid: throw std::runtime_error("Invalid arch not supported"); // will be overwritten in tt_global_state constructor
            case tt::ARCH::GRAYSKULL: throw std::runtime_error("GRAYSKULL arch not supported for simulator");
            case tt::ARCH::WORMHOLE: throw std::runtime_error("WORMHOLE arch not supported for simulator");
            case tt::ARCH::WORMHOLE_B0: throw std::runtime_error("WORMHOLE_B0 arch not supported for simulator");
            case tt::ARCH::BLACKHOLE: return core_desc_dir + "blackhole_simulation_1x2_arch.yaml";
            default: throw std::runtime_error("Unsupported device arch");
        };
    } else {
        switch (arch) {
            case tt::ARCH::Invalid: throw std::runtime_error("Invalid arch not supported"); // will be overwritten in tt_global_state constructor
            case tt::ARCH::GRAYSKULL: return core_desc_dir + "grayskull_120_arch.yaml";
            case tt::ARCH::WORMHOLE: throw std::runtime_error("WORMHOLE arch not supported");
            case tt::ARCH::WORMHOLE_B0: return core_desc_dir + (dispatch_core_type == CoreType::ETH ? "wormhole_b0_80_arch_eth_dispatch.yaml" : "wormhole_b0_80_arch.yaml");
            case tt::ARCH::BLACKHOLE: return core_desc_dir + (dispatch_core_type == CoreType::ETH ? "blackhole_140_arch_eth_dispatch.yaml" : "blackhole_140_arch.yaml");
            default: throw std::runtime_error("Unsupported device arch");
        };
    }
    return "";
}

inline const std::string get_product_name(tt::ARCH arch, uint32_t num_harvested_rows) {
    const static std::map<tt::ARCH, std::map<uint32_t, std::string>> product_name = {
        {tt::ARCH::GRAYSKULL, {{0, "E150"}, {2, "E75"}}},
        {tt::ARCH::WORMHOLE_B0, {{0, "galaxy"}, {1, "nebula_x1"}, {2, "nebula_x2"}}},
        {tt::ARCH::BLACKHOLE, {{0, "blackhole"}}} // TODO (abhullar): revisit blackhole product names
    };

    return product_name.at(arch).at(num_harvested_rows);
}

const core_descriptor_t &get_core_descriptor_config(chip_id_t device_id, const uint8_t num_hw_cqs, CoreType dispatch_core_type);

const std::tuple<uint32_t, CoreRange>& get_physical_worker_grid_config(chip_id_t chip, uint8_t num_hw_cqs, CoreType dispatch_core_type);

inline std::optional<uint32_t> get_storage_core_bank_size(chip_id_t device_id, const uint8_t num_hw_cqs, CoreType dispatch_core_type) {
    const core_descriptor_t &core_desc = get_core_descriptor_config(device_id, num_hw_cqs, dispatch_core_type);
    const metal_SocDescriptor &soc_desc = tt::Cluster::instance().get_soc_desc(device_id);
    if (core_desc.storage_core_bank_size.has_value()) {
        TT_FATAL(core_desc.storage_core_bank_size.value() % tt_metal::hal.get_alignment(tt_metal::HalMemType::L1) == 0,
            "Storage core bank size must be {} B aligned", tt_metal::hal.get_alignment(tt_metal::HalMemType::L1));
    }
    return core_desc.storage_core_bank_size;
}

inline const std::vector<CoreCoord> &get_logical_storage_cores(chip_id_t device_id, const uint8_t num_hw_cqs, CoreType dispatch_core_type) {
    static std::unordered_map<chip_id_t, std::unordered_map<uint8_t, std::vector<CoreCoord>>> logical_storage_cores_by_device;
    static CoreType previous_dispatch_core_type;
    if (previous_dispatch_core_type != dispatch_core_type) {
        logical_storage_cores_by_device.clear();
        previous_dispatch_core_type = dispatch_core_type;
    }
    auto& logical_storage_cores_by_cq = logical_storage_cores_by_device[device_id];
    if (auto it = logical_storage_cores_by_cq.find(num_hw_cqs); it != logical_storage_cores_by_cq.end()) {
        return it->second;
    }
    CoreCoord grid_size = tt::Cluster::instance().get_soc_desc(device_id).worker_grid_size;
    std::vector<CoreCoord> &logical_storage_cores = logical_storage_cores_by_cq[num_hw_cqs];
    const core_descriptor_t &core_desc = get_core_descriptor_config(device_id, num_hw_cqs, dispatch_core_type);
    std::transform(core_desc.relative_storage_cores.cbegin(), core_desc.relative_storage_cores.cend(), std::back_inserter(logical_storage_cores),
                [&grid_size](RelativeCoreCoord rel_coord) { return get_core_coord_from_relative(rel_coord, grid_size); });
    return logical_storage_cores;
}

inline CoreCoord get_compute_grid_size(chip_id_t device_id, const uint8_t num_hw_cqs, CoreType dispatch_core_type) {
    const core_descriptor_t &core_desc = get_core_descriptor_config(device_id, num_hw_cqs, dispatch_core_type);
    return core_desc.compute_grid_size;
}

inline const std::vector<CoreCoord> &get_logical_compute_cores(chip_id_t device_id, const uint8_t num_hw_cqs, CoreType dispatch_core_type) {
    static std::unordered_map<chip_id_t, std::unordered_map<uint8_t, std::vector<CoreCoord>>> logical_compute_cores_by_device;
    static CoreType previous_dispatch_core_type;
    if (previous_dispatch_core_type != dispatch_core_type) {
        logical_compute_cores_by_device.clear();
        previous_dispatch_core_type = dispatch_core_type;
    }
    auto& logical_compute_cores_by_cq = logical_compute_cores_by_device[device_id];
    if (auto it = logical_compute_cores_by_cq.find(num_hw_cqs); it != logical_compute_cores_by_cq.end()) {
        return it->second;
    }
    CoreCoord grid_size = tt::Cluster::instance().get_soc_desc(device_id).worker_grid_size;
    std::vector<CoreCoord> &logical_compute_cores = logical_compute_cores_by_cq[num_hw_cqs];
    const core_descriptor_t &core_desc = get_core_descriptor_config(device_id, num_hw_cqs, dispatch_core_type);
    std::transform(core_desc.relative_compute_cores.cbegin(), core_desc.relative_compute_cores.cend(), std::back_inserter(logical_compute_cores),
                [&grid_size](RelativeCoreCoord rel_coord) { return get_core_coord_from_relative(rel_coord, grid_size); });
    return logical_compute_cores;
}

inline const std::vector<CoreCoord> &get_logical_dispatch_cores(
    chip_id_t device_id, const uint8_t num_hw_cqs, CoreType dispatch_core_type) {
    static std::unordered_map<chip_id_t, std::unordered_map<uint8_t, std::vector<CoreCoord>>>
        logical_dispatch_cores_by_device;
    static CoreType previous_dispatch_core_type;
    if (previous_dispatch_core_type != dispatch_core_type) {
        logical_dispatch_cores_by_device.clear();
        previous_dispatch_core_type = dispatch_core_type;
    }
    auto &logical_dispatch_cores_by_cq = logical_dispatch_cores_by_device[device_id];
    if (auto it = logical_dispatch_cores_by_cq.find(num_hw_cqs); it != logical_dispatch_cores_by_cq.end()) {
        return it->second;
    }
    CoreCoord grid_size = tt::Cluster::instance().get_soc_desc(device_id).worker_grid_size;
    std::vector<CoreCoord> &logical_dispatch_cores = logical_dispatch_cores_by_cq[num_hw_cqs];
    const core_descriptor_t &core_desc = get_core_descriptor_config(device_id, num_hw_cqs, dispatch_core_type);
    std::transform(
        core_desc.relative_dispatch_cores.cbegin(),
        core_desc.relative_dispatch_cores.cend(),
        std::back_inserter(logical_dispatch_cores),
        [&grid_size](RelativeCoreCoord rel_coord) { return get_core_coord_from_relative(rel_coord, grid_size); });
    return logical_dispatch_cores;
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
