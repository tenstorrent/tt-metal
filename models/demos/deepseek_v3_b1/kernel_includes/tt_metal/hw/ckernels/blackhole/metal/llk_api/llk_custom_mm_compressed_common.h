// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_unpack_compressed.h"

namespace compressed {

/**
 * @brief Reconfig SrcA format for use inside custom_mm MOP loop.
 *
 * Uses direct cfg[] writes instead of cfg_reg_rmw_tensix to avoid
 * tensix instruction pipeline latency. Must be called AFTER
 * wait_for_next_context() when the unpacker is paused at semaphore.
 */
FORCE_INLINE void reconfig_custom_mm_srca(
    volatile uint* cfg, uint32_t fmt_idx, uint32_t reg0_base, uint32_t reg2_base) {
    uint32_t src_format = DATA_FORMATS[fmt_idx];
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] = reg2_base | src_format;
}

/** @brief Reconfig SrcA with pre-resolved DataFormat value (no lookup). */
FORCE_INLINE void reconfig_custom_mm_srca_raw(
    volatile uint* cfg, uint32_t src_format, uint32_t reg0_base, uint32_t reg2_base) {
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
    cfg[THCON_SEC0_REG2_Out_data_format_ADDR32] = reg2_base | src_format;
}

/** @brief Reconfig SrcA input format only (REG0). REG2 stays unchanged. */
FORCE_INLINE void reconfig_custom_mm_srca_input_only(volatile uint* cfg, uint32_t src_format, uint32_t reg0_base) {
    cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32] = reg0_base | src_format;
}

}  // namespace compressed
