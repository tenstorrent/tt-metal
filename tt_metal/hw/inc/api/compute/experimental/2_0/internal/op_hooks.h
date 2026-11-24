// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Quasar 2.0 op hooks: the Dest section size. Dest sync itself is the legacy tile_regs_acquire / commit / wait /
// release (api/compute/reg_api.h); ops that write Dest from UNPACK or the SFPU thread borrow the section from MATH
// inside the op (tile_regs_borrow / tile_regs_return).

#ifdef ARCH_QUASAR

#include <cstdint>

#include "api/compute/common_globals.h"
#include "ckernel_trisc_common.h"  // trisc::get_dest_max_tiles

namespace ckernel {
namespace experimental {
namespace detail {

// Dest tiles per section for the kernel's sync mode and Dest width.
template <bool is_fp32_dest_acc_en>
constexpr std::uint32_t dest_section_tiles() {
    return trisc::get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, trisc::DstTileShape::Tile32x32>();
}

}  // namespace detail
}  // namespace experimental
}  // namespace ckernel

#endif  // ARCH_QUASAR
