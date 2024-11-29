// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core_descriptor.hpp"

#include "yaml-cpp/yaml.h"

namespace tt {

const core_descriptor_t& get_core_descriptor_config(
    chip_id_t device_id, const uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    // {arch : {product : {num hardware command queues : config}}}
    static std::unordered_map<ARCH, std::unordered_map<std::string, std::unordered_map<uint8_t, core_descriptor_t>>>
        config_by_arch;
    // TODO: is there a better way to do this?
    static CoreType previous_dispatch_core_type;
    if (previous_dispatch_core_type != dispatch_core_config.get_core_type()) {
        config_by_arch.clear();
        previous_dispatch_core_type = dispatch_core_config.get_core_type();
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
            // some Galaxy machines are setup with N150s that have 0 harvested rows.
            // get_product_name ( ) returns those chips as galaxy. Override that to nebula_x1.
            product_name = "nebula_x1";
        } else {
            TT_ASSERT(
                tt::Cluster::instance().get_board_type(device_id) == BoardType::GALAXY,
                "Invalid Board Type in Galaxy Cluster. Only GALAXY and N150 are supported.");
        }
    }

    if (config_by_arch.count(arch) and config_by_arch.at(arch).count(product_name) and
        config_by_arch.at(arch).at(product_name).count(num_hw_cqs)) {
        return config_by_arch.at(arch).at(product_name).at(num_hw_cqs);
    }

    std::unordered_map<std::string, std::unordered_map<uint8_t, core_descriptor_t>>& config_by_product =
        config_by_arch[arch];
    std::unordered_map<uint8_t, core_descriptor_t>& config_by_num_cqs = config_by_product[product_name];

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
    if (tt::Cluster::instance().is_galaxy_cluster() and product_name == "nebula_x1") {
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

const std::tuple<uint32_t, CoreRange>& get_physical_worker_grid_config(
    chip_id_t chip, uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    // Get logical compute grid dimensions and num workers
    static std::unordered_map<uint32_t, std::tuple<uint32_t, CoreRange>> physical_grid_config_cache = {};
    // Unique hash generated based on the config that's being queried
    uint32_t config_hash = ((uint8_t)(dispatch_core_config.get_core_type())) |
                           ((uint8_t)(dispatch_core_config.get_dispatch_core_axis()) << 4) | (num_hw_cqs << 8) |
                           (chip << 16);
    if (physical_grid_config_cache.find(config_hash) == physical_grid_config_cache.end()) {
        auto worker_grid = tt::get_compute_grid_size(chip, num_hw_cqs, dispatch_core_config);
        std::size_t tensix_num_worker_cols = worker_grid.x;
        std::size_t tensix_num_worker_rows = worker_grid.y;
        uint32_t tensix_num_worker_cores = tensix_num_worker_cols * tensix_num_worker_rows;
        const metal_SocDescriptor& soc_desc = tt::Cluster::instance().get_soc_desc(chip);
        // Get physical compute grid range based on SOC Desc and Logical Coords
        CoreCoord tensix_worker_start_phys = soc_desc.get_physical_core_from_logical_core(
            CoreCoord(0, 0), CoreType::WORKER);  // Logical Worker Coords start at 0,0
        CoreCoord tensix_worker_end_phys = soc_desc.get_physical_core_from_logical_core(
            CoreCoord(tensix_num_worker_cols - 1, tensix_num_worker_rows - 1), CoreType::WORKER);
        CoreRange tensix_worker_physical_grid = CoreRange(tensix_worker_start_phys, tensix_worker_end_phys);
        physical_grid_config_cache.insert(
            {config_hash, std::make_tuple(tensix_num_worker_cores, tensix_worker_physical_grid)});
    }
    return physical_grid_config_cache.at(config_hash);
}

void reassign_dram_interface_cores_for_grayskull(const std::vector<uint32_t>& non_worker_rows, std::vector<CoreCoord>& dram_interface_workers, uint32_t full_grid_size_y) {
    for (auto& coord : dram_interface_workers) {
        // if row is harvested, move core down by 1
        while (std::find(non_worker_rows.begin(), non_worker_rows.end(), coord.y) != non_worker_rows.end() and coord.y < (full_grid_size_y - 1)) {
            coord.y += 1;
        }
    }
}

std::vector<CoreCoord> reassign_dram_interface_cores_for_wormhole(const std::vector<uint32_t>& non_worker_rows, const std::vector<CoreCoord>& dram_interface_workers, uint32_t num_dram_banks, uint32_t max_worker_y_physical, uint32_t min_worker_y_physical) {
    std::vector<CoreCoord> dram_interface_workers_g1;
    std::vector<CoreCoord> dram_interface_workers_g2;
    std::vector<size_t> dram_interface_worker_y_coords_g1;
    std::vector<size_t> dram_interface_worker_y_coords_g2;

    dram_interface_workers_g1.reserve(num_dram_banks);
    dram_interface_worker_y_coords_g1.reserve(num_dram_banks);
    dram_interface_workers_g2.reserve(num_dram_banks);
    dram_interface_worker_y_coords_g2.reserve(num_dram_banks);

    // Separate Workers into 2 groups based on which DRAM column they are meant to interface with
    for (const auto& core : dram_interface_workers) {
        if (core.x == dram_interface_workers.front().x) {
            dram_interface_workers_g1.push_back(core);
        } else {
            dram_interface_workers_g2.push_back(core);
        }
    }

    // Track the indices of the workers inside each group
    std::vector<int> indices_g1(dram_interface_workers_g1.size());
    std::vector<int> indices_g2(dram_interface_workers_g2.size());
    std::iota(indices_g1.begin(), indices_g1.end(), 0);
    std::iota(indices_g2.begin(), indices_g2.end(), 0);

    // Sort workers and associated group indices based on y coord
    std::sort(indices_g1.begin(), indices_g1.end(), [&dram_interface_workers_g1](int i1, int i2) {
        return dram_interface_workers_g1[i1].y < dram_interface_workers_g1[i2].y;
    });
    std::sort(indices_g2.begin(), indices_g2.end(), [&dram_interface_workers_g2](int i1, int i2) {
        return dram_interface_workers_g2[i1].y < dram_interface_workers_g2[i2].y;
    });
    std::sort(dram_interface_workers_g1.begin(), dram_interface_workers_g1.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.y < b.y;
    });
    std::sort(dram_interface_workers_g2.begin(), dram_interface_workers_g2.end(), [](const CoreCoord& a, const CoreCoord& b) {
        return a.y < b.y;
    });
    // Place the bottom-most worker and associated index at the start of the group
    std::rotate(dram_interface_workers_g1.begin(), dram_interface_workers_g1.end() - 1, dram_interface_workers_g1.end());
    std::rotate(dram_interface_workers_g2.begin(), dram_interface_workers_g2.end() - 1, dram_interface_workers_g2.end());
    std::rotate(indices_g1.begin(), indices_g1.end() - 1, indices_g1.end());
    std::rotate(indices_g2.begin(), indices_g2.end() - 1, indices_g2.end());

    // Track the shuffled indices
    std::vector<int> indices_g1_realloc(dram_interface_workers_g1.size());
    std::vector<int> indices_g2_realloc(dram_interface_workers_g2.size());
    for (int new_index = 0; new_index < indices_g1.size(); ++new_index) {
        indices_g1_realloc[indices_g1[new_index]] = new_index;
    }
    for (int new_index = 0; new_index < indices_g2.size(); ++new_index) {
        indices_g2_realloc[indices_g2[new_index]] = new_index;
    }
    // Extract worker y coordinates per group
    for (auto core : dram_interface_workers_g1) {
        dram_interface_worker_y_coords_g1.push_back(core.y);
    }
    for (auto core : dram_interface_workers_g2) {
        dram_interface_worker_y_coords_g2.push_back(core.y);
    }
    uint32_t x_step = 3;
    // Helper function to shift harvested workers
    auto shift_group_based_on_harvesting = [&](std::vector<CoreCoord>& group, std::vector<size_t>& group_y, uint32_t x_step) {
        for (auto& coord : group) {
            auto y = coord.y;

            if (std::find(non_worker_rows.begin(), non_worker_rows.end(), y) != non_worker_rows.end() ||
                std::count(group_y.begin(), group_y.end(), y) >= 2) {
                auto shift_coord_based_on_harvesting = [&](int start, int end, int step) {
                    bool found_new_row = false;
                    for (int j = start; step > 0 ? j <= end : j >= end; j += step) {
                        if (std::find(non_worker_rows.begin(), non_worker_rows.end(), j) == non_worker_rows.end() &&
                            std::count(group_y.begin(), group_y.end(), j) == 0) {
                            coord.y = j;
                            coord.x += x_step;
                            x_step--;
                            found_new_row = true;
                            break;
                        }
                    }
                    if (not found_new_row) {
                        for (int j = start; step > 0 ? j <= end : j >= end; j += step) {
                            if (std::find(non_worker_rows.begin(), non_worker_rows.end(), j) == non_worker_rows.end()) {
                                coord.y = j;
                                coord.x += x_step;
                                x_step--;
                                found_new_row = true;
                                break;
                            }
                        }
                    }
                };

                if (y >= num_dram_banks - 1) {
                    shift_coord_based_on_harvesting(max_worker_y_physical, min_worker_y_physical, -1);
                } else {
                    shift_coord_based_on_harvesting(min_worker_y_physical, max_worker_y_physical, 1);
                }
            }
        }
    };
    // Shift harvested workers
    shift_group_based_on_harvesting(dram_interface_workers_g1, dram_interface_worker_y_coords_g1, x_step);
    shift_group_based_on_harvesting(dram_interface_workers_g2, dram_interface_worker_y_coords_g2, x_step);

    // Merge both groups based on original indices (maintain ordering by dram bank_id here)
    std::vector<CoreCoord> shifted_dram_interface_workers;
    shifted_dram_interface_workers.reserve(num_dram_banks);
    for (int i = 0; i < indices_g1_realloc.size(); ++i) {
        shifted_dram_interface_workers.push_back(dram_interface_workers_g1[indices_g1_realloc[i]]);
    }
    for (int i = 0; i < indices_g2_realloc.size(); ++i) {
        shifted_dram_interface_workers.push_back(dram_interface_workers_g2[indices_g2_realloc[i]]);
    }
    return shifted_dram_interface_workers;
}

void reassign_dram_interface_cores_for_blackhole(const std::vector<uint32_t>& harvested_cols, std::vector<CoreCoord>& dram_interface_workers, uint32_t full_grid_size_x) {
    for (auto& coord : dram_interface_workers) {
        // if col is harvested, move core right by 1
        while (std::find(harvested_cols.begin(), harvested_cols.end(), coord.x) != harvested_cols.end() and coord.x < (full_grid_size_x - 1)) {
            coord.x += 1;
        }
    }
}

std::vector<CoreCoord> reassign_cores_based_on_worker_grid_config(ARCH arch, const std::vector<CoreCoord>& dram_phy_coords, uint32_t full_grid_size_x, uint32_t full_grid_size_y, std::vector<uint32_t> worker_phy_x, std::vector<uint32_t> worker_phy_y, uint32_t num_dram_banks) {
    std::vector<uint32_t> non_worker_rows;
    std::vector<uint32_t> non_worker_cols;
    uint32_t max_worker_y_physical = 0;
    uint32_t min_worker_y_physical = std::numeric_limits<uint32_t>::max();

    if (arch == ARCH::GRAYSKULL or arch == ARCH::WORMHOLE_B0) {
        for (int y_coord = 0; y_coord < full_grid_size_y; ++y_coord) {
            if (std::find(worker_phy_y.begin(), worker_phy_y.end(), y_coord) == worker_phy_y.end()) {
                non_worker_rows.push_back(y_coord);
            }
            if (y_coord > max_worker_y_physical) {
                max_worker_y_physical = y_coord;
            }
            if (y_coord < min_worker_y_physical) {
                min_worker_y_physical = y_coord;
            }
        }
    }
    std::vector<CoreCoord> dram_interface_workers;
    for (int i = 0; i < num_dram_banks; ++i) {
        auto dram_core = dram_phy_coords[i];
        if (arch == ARCH::GRAYSKULL) {
            dram_interface_workers.push_back(CoreCoord(dram_core.x, dram_core.y + 1));
        } else if (arch == ARCH::WORMHOLE_B0 or arch == ARCH::BLACKHOLE) {
            dram_interface_workers.push_back(CoreCoord(dram_core.x + 1, dram_core.y));
        }
    }

    if (arch == ARCH::GRAYSKULL) {
        reassign_dram_interface_cores_for_grayskull(non_worker_rows, dram_interface_workers, full_grid_size_y);
        return dram_interface_workers;
    } else if (arch == ARCH::WORMHOLE_B0) {
        return reassign_dram_interface_cores_for_wormhole(non_worker_rows, dram_interface_workers, num_dram_banks, max_worker_y_physical, min_worker_y_physical);
    } else if (arch == ARCH::BLACKHOLE) {
        for (int x_coord = 0; x_coord < full_grid_size_x; ++x_coord) {
            if (std::find(worker_phy_x.begin(), worker_phy_x.end(), x_coord) == worker_phy_x.end()) {
                non_worker_cols.push_back(x_coord);
            }
        }
        reassign_dram_interface_cores_for_blackhole(non_worker_cols, dram_interface_workers, full_grid_size_x);
        return dram_interface_workers;
    }
    TT_THROW("Invalid Arch Name specified");
}

}  // namespace tt
