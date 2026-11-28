// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cpack_common.h"

// The W-rule arm of the pack sweeps: programs the base as llk_pack_w_take does, the W stride only when it changes.
inline void pack_sweep_w_reprogram(const std::uint32_t address, const std::uint32_t tile_stride)
{
    ckernel::packer::program_packer_destination(address);
    if (tile_stride != ckernel::packer::pack_ch1_w_stride())
    {
        ckernel::cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_1_Wstride_RMW>(tile_stride);
        ckernel::packer::pack_ch1_w_stride_set(tile_stride);
    }
}
