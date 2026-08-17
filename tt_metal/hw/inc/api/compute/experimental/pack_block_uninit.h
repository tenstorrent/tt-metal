// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Restore PackMode::Default packer MOP after pack_block_contiguous_* completes.
// pack_block_contiguous_init replaces the tile-pack MOP template.

#pragma once

#include "api/compute/common.h"

namespace ckernel {

ALWI void pack_block_contiguous_uninit() { PACK((_llk_pack_mop_config_<PackMode::Default>())); }

}  // namespace ckernel
