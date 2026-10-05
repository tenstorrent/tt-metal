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
 * @brief CRTP base for unary SrcS SFPU op types; provides init() and run().
 *
 * The SrcS counterpart of the Dest op-class base: an op supplies calculate() (one SrcS slice) and,
 * when it owns state, init_op() and hw_clears_valids; the base adds the SrcS pipeline set-up and
 * the tile/slice walk. init() and run() are defined in llk_sfpu_srcs_api.h next to the SrcS
 * pipeline, so SFPU kernel headers do not pull the unpack/pack code into other builds.
 *
 * @tparam Op: The derived op type.
 * @note Include llk_sfpu_srcs_api.h; call init() once, then run() for the tiles.
 */
template <class Op>
struct SfpuSrcsUnaryOp {
    /// calculate() leaves the SrcS valids to the base. Set to true in ops whose last instruction
    /// hands the banks back itself, so the base does not clear them again.
    static constexpr bool hw_clears_valids = false;

    /// Default per-op init: no state beyond the SrcS pipeline set-up.
    static void init_op() {}

    template <std::uint8_t INSTRN_COUNT = 1>
    static void init(
        const std::uint32_t l1_in_addr_16B,
        const DataFormat unpack_S_src_format,
        const DataFormat unpack_S_dst_format,
        const std::uint32_t l1_out_addr_16B,
        const DataFormat pack_S_src_format,
        const DataFormat pack_S_dst_format,
        const bool implied_math_format);

    template <std::uint8_t INSTRN_COUNT = 1>
    static void run(const std::uint32_t num_tiles, const DataFormat unpack_S_dst_format);
};

/**
 * @brief Unary SrcS op for math policy MATH, issued per ISSUE; see @ref SfpuSrcsUnaryOp.
 *
 * The Sfpi specialization below serves every MATH (static vFloat apply(vFloat)); LoadMacro
 * versions are defined per op, next to the op's kernel. Any other pair fails to compile.
 *
 * @tparam MATH: Math policy, e.g. @ref ExpHwLut.
 * @tparam LAYOUT: Load and store layout, values = <F16a/F16b/F32>; unpack destination and pack
 *         source formats must match.
 * @tparam ISSUE: Issue mechanism, values = <Sfpi/LoadMacro>; resolve it with
 *         @ref resolve_sfpu_issue.
 */
template <class MATH, sfpi::DataLayout LAYOUT, SfpuIssue ISSUE>
struct SrcsUnary {
    static_assert(sizeof(MATH) == 0, "This SFPU op has no SrcS implementation for the requested SfpuIssue");
};

template <class MATH, sfpi::DataLayout LAYOUT>
struct SrcsUnary<MATH, LAYOUT, SfpuIssue::Sfpi> : SfpuSrcsUnaryOp<SrcsUnary<MATH, LAYOUT, SfpuIssue::Sfpi>> {
    // Reads the in0 slot and writes the out slot of the current slice (@ref SrcsLayout).
    sfpi_inline static void calculate() {
        using Layout = SrcsLayout<LAYOUT>;
        using Operand = SfpuOperand<SfpuReg::SrcS, SfpiFormat<LAYOUT, sfpi::vFloat>>;
        calculate_unary_operands<MATH, Layout::ops>(Operand{Layout::in0}, Operand{Layout::out});
    }
};

/**
 * @brief CRTP base for binary SrcS SFPU op types; provides init() and run().
 *
 * Like @ref SfpuSrcsUnaryOp, for ops with two inputs sharing formats: init() configures the binary
 * SrcS pipeline (two UNP_S table rows, see llk_sfpu_srcs_binary_init) then Op::init_op(); run()
 * unpacks one slice of each input per slice and calls Op::calculate(). Defined in
 * llk_sfpu_srcs_api.h.
 *
 * @tparam Op: The derived op type.
 * @note Include llk_sfpu_srcs_api.h; call init() once, then run() for the tiles.
 */
template <class Op>
struct SfpuSrcsBinaryOp {
    static constexpr bool hw_clears_valids = false;

    static void init_op() {}

    template <std::uint8_t INSTRN_COUNT = 1>
    static void init(
        const std::uint32_t l1_in0_addr_16B,
        const std::uint32_t l1_in1_addr_16B,
        const DataFormat unpack_S_src_format,
        const DataFormat unpack_S_dst_format,
        const std::uint32_t l1_out_addr_16B,
        const DataFormat pack_S_src_format,
        const DataFormat pack_S_dst_format,
        const bool implied_math_format);

    template <std::uint8_t INSTRN_COUNT = 1>
    static void run(const std::uint32_t num_tiles, const DataFormat unpack_S_dst_format);
};

/**
 * @brief Binary SrcS op for math policy MATH (static apply(a, b)), issued per ISSUE; see @ref SfpuSrcsBinaryOp.
 *
 * Only the Sfpi specialization exists today; any other pair fails to compile.
 *
 * @tparam MATH: Math policy, e.g. AddMath.
 * @tparam LAYOUT: Load and store layout, values = <F16a/F16b/F32>; unpack destination and pack
 *         source formats must match.
 * @tparam ISSUE: Issue mechanism, values = <Sfpi>; resolve it with @ref resolve_sfpu_issue.
 */
template <class MATH, sfpi::DataLayout LAYOUT, SfpuIssue ISSUE>
struct SrcsBinary {
    static_assert(sizeof(MATH) == 0, "This SFPU op has no SrcS implementation for the requested SfpuIssue");
};

template <class MATH, sfpi::DataLayout LAYOUT>
struct SrcsBinary<MATH, LAYOUT, SfpuIssue::Sfpi> : SfpuSrcsBinaryOp<SrcsBinary<MATH, LAYOUT, SfpuIssue::Sfpi>> {
    // Reads the in0 and in1 slots and writes the out slot of the current slice (@ref SrcsLayout).
    sfpi_inline static void calculate() {
        using Layout = SrcsLayout<LAYOUT>;
        using Operand = SfpuOperand<SfpuReg::SrcS, SfpiFormat<LAYOUT, sfpi::vFloat>>;
        calculate_binary_operands<MATH, Layout::ops>(Operand{Layout::in0}, Operand{Layout::in1}, Operand{Layout::out});
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
