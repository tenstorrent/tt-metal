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
#include "ckernel_trisc_common.h"                                  // trisc::get_dest_max_tiles
#include "api/compute/experimental/2_0/internal/llk_descriptor.h"  // is_legal_tile_shape

namespace ckernel {
namespace experimental {
namespace detail {

// Full 32x32 tile (tilize / untilize: tiny tiles need the strided buffer descriptor, not wired up yet).
constexpr bool is_32x32(TensorShape s) {
    return is_legal_tile_shape(s) && s.total_num_faces() == 4 && s.face_r_dim == 16;
}

// Dest tiles per section for the kernel's sync mode and Dest width.
template <bool is_fp32_dest_acc_en>
constexpr std::uint32_t dest_section_tiles() {
    return trisc::get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, trisc::DstTileShape::Tile32x32>();
}

}  // namespace detail
}  // namespace experimental
}  // namespace ckernel

#endif  // ARCH_QUASAR
