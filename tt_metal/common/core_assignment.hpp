// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core_coord.hpp"
#include <tt_cluster.hpp>

namespace tt {
namespace tt_metal {
// Returns an ordered list of DRAM Bank ID to optimally placed worker cores. Placing DRAM reader or writer
// kernels on these worker cores will minimize NOC congestion and the number of NOC hops required to complete
// a DRAM read or write.
// Worker cores are derived based on architecture, harvesting configurations and DRAM Controller placement.
std::vector<CoreCoord> get_optimal_dram_to_physical_worker_assignment(
    ARCH arch,
    const std::vector<CoreCoord>& dram_phy_coords,
    uint32_t full_grid_size_x,
    uint32_t full_grid_size_y,
    std::vector<uint32_t> worker_phy_x,
    std::vector<uint32_t> worker_phy_y);

}  // namespace tt_metal
}  // namespace tt
