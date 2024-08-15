// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core_descriptor.hpp"

#include "yaml-cpp/yaml.h"

namespace tt {

const core_descriptor_t &get_core_descriptor_config(chip_id_t device_id, const uint8_t num_hw_cqs, CoreType dispatch_core_type) {
    // {arch : {product : {num hardware command queues : config}}}
    static std::unordered_map<ARCH, std::unordered_map<std::string, std::unordered_map<uint8_t, core_descriptor_t>>> config_by_arch;
    // TODO: is there a better way to do this?
    static CoreType previous_dispatch_core_type;
    if (previous_dispatch_core_type != dispatch_core_type) {
        config_by_arch.clear();
        previous_dispatch_core_type = dispatch_core_type;
    }

    ARCH arch = tt::Cluster::instance().arch();
    uint32_t harvesting_mask = tt::Cluster::instance().get_harvested_rows(device_id);
    std::bitset<32> mask_bitset(harvesting_mask);
    uint32_t num_harvested_rows = mask_bitset.count();

    if (num_harvested_rows > 2) {
        TT_THROW("At most two rows can be harvested, but detected {} harvested rows", num_harvested_rows);
    }
    if (num_harvested_rows == 1 and arch == tt::ARCH::GRAYSKULL) {
        TT_THROW("One row harvested Grayskull is not supported");
    }

    std::string product_name = get_product_name(arch, num_harvested_rows);
    if (tt::Cluster::instance().is_galaxy_cluster()) {
        if (tt::Cluster::instance().get_board_type(device_id) == BoardType::N150) {
            //some Galaxy machines are setup with N150s that have 0 harvested rows.
            //get_product_name ( ) returns those chips as galaxy. Override that to nebula_x1.
            product_name = "nebula_x1";
        } else {
            TT_ASSERT(tt::Cluster::instance().get_board_type(device_id) == BoardType::GALAXY, "Invalid Board Type in Galaxy Cluster. Only GALAXY and N150 are supported.");
        }
    }

    if (config_by_arch.count(arch) and config_by_arch.at(arch).count(product_name) and config_by_arch.at(arch).at(product_name).count(num_hw_cqs)) {
        return config_by_arch.at(arch).at(product_name).at(num_hw_cqs);
    }

    std::unordered_map<std::string, std::unordered_map<uint8_t, core_descriptor_t>> &config_by_product = config_by_arch[arch];
    std::unordered_map<uint8_t, core_descriptor_t> &config_by_num_cqs = config_by_product[product_name];

    YAML::Node core_descriptor_yaml = YAML::LoadFile(get_core_descriptor_file(arch, dispatch_core_type));
    YAML::Node desc_yaml = core_descriptor_yaml[product_name][std::to_string(num_hw_cqs)];

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
        } catch (std::runtime_error &ex) {
            TT_THROW(
                "Core descriptor yaml for {} needs to specify storage_core_bank_size since there are {} storage cores!",
                get_string_lowercase(arch),
                storage_cores.size());
        }
    }

    auto compute_with_storage_start = desc_yaml["compute_with_storage_grid_range"]["start"];
    auto compute_with_storage_end = desc_yaml["compute_with_storage_grid_range"]["end"];
    if (tt::Cluster::instance().is_galaxy_cluster() and product_name == "nebula_x1") {
        compute_with_storage_start = desc_yaml["tg_compute_with_storage_grid_range"]["start"];
        compute_with_storage_end = desc_yaml["tg_compute_with_storage_grid_range"]["end"];
    }
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
    auto dispatch_cores_string = "dispatch_cores";
    if (tt::Cluster::instance().is_galaxy_cluster() and product_name == "nebula_x1") {
        dispatch_cores_string = "tg_dispatch_cores";
    }
    for (const auto& core_node : desc_yaml[dispatch_cores_string]) {
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

    config_by_num_cqs[num_hw_cqs] = core_descriptor_t{
        .compute_grid_size = compute_grid_size,
        .relative_compute_cores = compute_cores,
        .relative_storage_cores = storage_cores,
        .storage_core_bank_size = storage_core_bank_size,
        .relative_dispatch_cores = dispatch_cores,
    };
    return config_by_arch.at(arch).at(product_name).at(num_hw_cqs);
}

} // namespace tt
