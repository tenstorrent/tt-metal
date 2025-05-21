// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>

#include "assert.hpp"
#include "core_assignment.hpp"
#include <umd/device/types/arch.h>
#include <umd/device/types/xy_pair.h>

namespace tt {
namespace tt_metal {

std::vector<CoreCoord> reassign_dram_interface_cores_for_wormhole(
    const std::vector<uint32_t>& non_worker_rows,
    const std::vector<CoreCoord>& dram_interface_workers,
    uint32_t num_dram_banks,
    uint32_t max_worker_y_physical,
    uint32_t min_worker_y_physical) {
    // Reassign optimally placed DRAM Interface worker cores based on harvesting for WH
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
    std::sort(
        dram_interface_workers_g1.begin(), dram_interface_workers_g1.end(), [](const CoreCoord& a, const CoreCoord& b) {
            return a.y < b.y;
        });
    std::sort(
        dram_interface_workers_g2.begin(), dram_interface_workers_g2.end(), [](const CoreCoord& a, const CoreCoord& b) {
            return a.y < b.y;
        });
    // Place the bottom-most worker and associated index at the start of the group
    std::rotate(
        dram_interface_workers_g1.begin(), dram_interface_workers_g1.end() - 1, dram_interface_workers_g1.end());
    std::rotate(
        dram_interface_workers_g2.begin(), dram_interface_workers_g2.end() - 1, dram_interface_workers_g2.end());
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
    auto shift_group_based_on_harvesting = [&](std::vector<CoreCoord>& group,
                                               std::vector<size_t>& group_y,
                                               uint32_t x_step) {
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

void reassign_dram_interface_cores_for_blackhole(
    const std::vector<uint32_t>& harvested_cols,
    std::vector<CoreCoord>& dram_interface_workers,
    uint32_t full_grid_size_x) {
    for (auto& coord : dram_interface_workers) {
        // if col is harvested, move core right by 1
        while (std::find(harvested_cols.begin(), harvested_cols.end(), coord.x) != harvested_cols.end() and
               coord.x < (full_grid_size_x - 1)) {
            coord.x += 1;
        }
    }
}

std::vector<CoreCoord> get_optimal_dram_to_physical_worker_assignment(
    ARCH arch,
    const std::vector<CoreCoord>& dram_phy_coords,
    uint32_t full_grid_size_x,
    uint32_t full_grid_size_y,
    std::vector<uint32_t> worker_phy_x,
    std::vector<uint32_t> worker_phy_y) {
    // Reassign optimally placed DRAM Interface worker cores based on harvesting for BH
    std::vector<uint32_t> non_worker_rows;
    std::vector<uint32_t> non_worker_cols;
    uint32_t max_worker_y_physical = 0;
    uint32_t min_worker_y_physical = std::numeric_limits<uint32_t>::max();
    // For WH, rows are harvested. Track them here.
    if (arch == ARCH::WORMHOLE_B0) {
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
    uint32_t num_dram_banks = dram_phy_coords.size();
    // Get the optimal dram -> worker configuration here.
    // For WH, worker cores are placed to the right of the DRAM Controller.
    for (int i = 0; i < num_dram_banks; ++i) {
        auto dram_core = dram_phy_coords[i];
        if (arch == ARCH::WORMHOLE_B0 or arch == ARCH::BLACKHOLE) {
            dram_interface_workers.push_back(CoreCoord(dram_core.x + 1, dram_core.y));
        }
    }

    if (arch == ARCH::WORMHOLE_B0) {
        // Reassign worker cores based on harvesting for WH.
        return reassign_dram_interface_cores_for_wormhole(
            non_worker_rows, dram_interface_workers, num_dram_banks, max_worker_y_physical, min_worker_y_physical);
    } else if (arch == ARCH::BLACKHOLE) {
        // Reassign worker cores based on harvesting for BH.
        // Need to account for column harvesting here.
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

}  // namespace tt_metal
}  // namespace tt
