// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_operand.h"
#include "sfpu_issue.h"

namespace ckernel::sfpu {

static_assert(sfpi::SFP_SRCSREG_STRIDE == ckernel::math::SFP_ROWS, "one sfpi index step must be one SFPU op");

/**
 * @brief Geometry and slot offsets (in SFPI steps) of one floating-point SrcS slice.
 *
 * Must match the unpack/pack base addresses in llk_sfpu_srcs_api.h: in0 at the slice base,
 * in1 at + rows, result at + 2 * rows.
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

    // SFPLOAD/SFPSTORE format code for this layout, for raw-instruction and SFPLOADMACRO paths.
    static constexpr std::uint32_t sfpmem = LAYOUT == sfpi::DataLayout::F32    ? p_sfpu::sfpmem::FP32
                                            : LAYOUT == sfpi::DataLayout::F16a ? p_sfpu::sfpmem::FP16A
                                                                               : p_sfpu::sfpmem::FP16B;
};

/**
 * @brief One unary SFPU op over one SrcS slice, run by llk_sfpu_srcs_unary<Op>.
 *
 * MATH is the math policy (static vFloat apply(vFloat)); ISSUE selects how it is issued. Each
 * supported (MATH, ISSUE) pair is a specialization exposing:
 *   - init(): one-time setup of state the op owns (macros, replay buffer);
 *   - run_slice(): reads the in0 slot and writes the out slot of the current slice
 *     (@ref SrcsLayout);
 *   - hw_clears_valids: true when run_slice() hands the SrcS banks back itself, so the caller
 *     must not clear the valids again.
 * The Sfpi specialization below serves every MATH; LoadMacro versions are defined per op.
 *
 * @tparam MATH: Math policy, e.g. @ref ExpHwLut.
 * @tparam LAYOUT: Load and store layout, values = <F16a/F16b/F32>; unpack destination and pack
 *         source formats must match.
 * @tparam ISSUE: Issue mechanism, values = <Sfpi/LoadMacro>; resolve it with
 *         @ref resolve_sfpu_issue.
 * @note Call @ref llk_sfpu_srcs_unary_init with this type before @ref llk_sfpu_srcs_unary.
 */
template <class MATH, sfpi::DataLayout LAYOUT, SfpuIssue ISSUE>
struct SrcsUnaryOp {
    static_assert(sizeof(MATH) == 0, "This SFPU op has no SrcS implementation for the requested SfpuIssue");
};

template <class MATH, sfpi::DataLayout LAYOUT>
struct SrcsUnaryOp<MATH, LAYOUT, SfpuIssue::Sfpi> {
    static constexpr bool hw_clears_valids = false;

    static void init() {}

    sfpi_inline static void run_slice() {
        using Layout = SrcsLayout<LAYOUT>;
        using Operand = SfpuOperand<SfpuReg::SrcS, SfpiFormat<LAYOUT, sfpi::vFloat>>;
        const Operand input{Layout::in0};
        const Operand output{Layout::out};
#pragma GCC unroll 8
        for (int d = 0; d < Layout::ops; d++) {
            output.store(d, MATH::apply(input.load(d)));
        }
    }
};

/**
 * @brief Resolve a runtime SrcS register format to a SrcsLayout once, outside the tile loop.
 *
 * Pass the register format (unpack_S_dst / pack_S_src), not the L1 format; MX inputs use their
 * unpacked register format.
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
