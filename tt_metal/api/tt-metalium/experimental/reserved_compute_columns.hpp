// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Experimental and subject to change: this header carries no API-stability guarantee.
namespace tt::tt_metal::experimental {

/// Narrows the compute grid every MeshDevice reports by `columns` trailing columns, so that operations which size
/// themselves from compute_with_storage_grid_size() stay off those columns, which a sub-device can then own (for
/// collectives running beside compute). Set it after loading such a sub-device manager and clear it (0) before
/// unloading it: device setup and worker resets read the grid too.
void SetReservedComputeColumns(uint32_t columns);
uint32_t GetReservedComputeColumns();

}  // namespace tt::tt_metal::experimental
