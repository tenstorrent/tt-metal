// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Compile-time layout and geometry of a standard floating-point SrcS slice.
template <sfpi::DataLayout LAYOUT>
struct SrcsLayout {
    static constexpr sfpi::DataLayout layout = LAYOUT;
    static constexpr int rows = static_cast<int>(trisc::srcs_dims::ydim(LAYOUT == sfpi::DataLayout::F32));
};

/**
 * @brief Resolve a runtime SrcS storage format once, before running a typed SFPU pipeline.
 *
 * Calls op(SrcsLayout<layout>{}); the callable chooses the operation. Input and output
 * formats can be dispatched independently. This does not configure hardware or signal
 * completion. Use the register formats (unpack_S_dst / pack_S_src), not the L1 formats.
 * Supports Float16, Float16_b and Float32 storage. MX L1 inputs must use their unpacked
 * register format here. Callers using an implied SFPI layout can use typed adapters directly.
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
