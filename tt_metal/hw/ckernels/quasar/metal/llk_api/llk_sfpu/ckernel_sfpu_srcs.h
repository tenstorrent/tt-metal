// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// The *_srcs adapters address SrcS in SFPI index steps; their slot arithmetic assumes one step
// is one SFPU pass.
static_assert(sfpi::SFP_SRCSREG_STRIDE == ckernel::math::SFP_ROWS, "one sfpi index step must be one SFPU op");

/**
 * @brief Geometry and slot offsets (in SFPI steps) of one floating-point SrcS slice.
 *
 * Single source for every *_srcs adapter. Must match the unpack/pack base addresses in
 * llk_sfpu_srcs_api.h: in0 at the slice base, in1 at + rows, result at + 2 * rows.
 */
template <sfpi::DataLayout LAYOUT>
struct SrcsLayout {
    static_assert(
        LAYOUT == sfpi::DataLayout::F16a || LAYOUT == sfpi::DataLayout::F16b || LAYOUT == sfpi::DataLayout::F32,
        "SrcS SFPI adapters support F16a, F16b and F32 layouts");

    static constexpr sfpi::DataLayout layout = LAYOUT;
    static constexpr int rows = static_cast<int>(trisc::srcs_dims::ydim(LAYOUT == sfpi::DataLayout::F32));
    static_assert(
        rows > 0 && rows % static_cast<int>(ckernel::math::SFP_ROWS) == 0, "SrcS slice must contain whole SFPU passes");

    static constexpr int ops = rows / static_cast<int>(ckernel::math::SFP_ROWS);
    static constexpr int in0 = 0;
    static constexpr int in1 = ops;
    static constexpr int out = 2 * ops;
};

/**
 * @brief Resolve a runtime SrcS register format to a SrcsLayout once, outside the tile loop.
 *
 * Calls op(SrcsLayout<layout>{}). Pass the register format (unpack_S_dst / pack_S_src), not the
 * L1 format; MX inputs use their unpacked register format.
 */
template <class Op>
sfpi_inline void dispatch_sfpu_srcs_format(const DataFormat format, Op&& op) {
    switch (format) {
        case DataFormat::Float32: op(SrcsLayout<sfpi::DataLayout::F32>{}); break;
        case DataFormat::Float16: op(SrcsLayout<sfpi::DataLayout::F16a>{}); break;
        case DataFormat::Float16_b: op(SrcsLayout<sfpi::DataLayout::F16b>{}); break;
        default: LLK_ASSERT(false, "Unsupported floating-point SrcS storage format"); break;
    }
}

}  // namespace ckernel::sfpu
