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

/// Slot offsets (SFPI steps) of one SrcS slice; must match the base addresses in llk_sfpu_srcs_api.h.
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

    // SrcS register format (unpack destination = pack source) implied by the layout.
    static constexpr DataFormat format = LAYOUT == sfpi::DataLayout::F32    ? DataFormat::Float32
                                         : LAYOUT == sfpi::DataLayout::F16a ? DataFormat::Float16
                                                                            : DataFormat::Float16_b;

    // SFPLOAD/SFPSTORE format code, for the SFPLOADMACRO path.
    static constexpr std::uint32_t sfpmem = LAYOUT == sfpi::DataLayout::F32    ? p_sfpu::sfpmem::FP32
                                            : LAYOUT == sfpi::DataLayout::F16a ? p_sfpu::sfpmem::FP16A
                                                                               : p_sfpu::sfpmem::FP16B;
};

/// CRTP base for unary SrcS ops: Op supplies calculate(); init()/run() live in llk_sfpu_srcs_api.h.
template <typename Op>
struct SfpuSrcsUnaryOp {
    // True when the op's own instructions release the SrcS banks (SFPLOADMACRO).
    static constexpr bool hw_clears_valids = false;

    static void init_op() {}

    // Takes only the L1 formats; the SrcS register format comes from Op::Layout.
    template <std::uint8_t INSTRN_COUNT = 1>
    static void init(
        const std::uint32_t l1_in_addr_16B,
        const DataFormat l1_in_format,
        const std::uint32_t l1_out_addr_16B,
        const DataFormat l1_out_format,
        const bool implied_math_format);

    template <std::uint8_t INSTRN_COUNT = 1>
    static void run(const std::uint32_t num_tiles);
};

/// Unary SrcS op. Sfpi serves every MATH; LoadMacro versions are specialized next to the op.
template <typename MATH, sfpi::DataLayout LAYOUT, SfpuIssue ISSUE>
struct SrcsUnary {
    static_assert(sizeof(MATH) == 0, "This SFPU op has no SrcS implementation for the requested SfpuIssue");
};

template <typename MATH, sfpi::DataLayout LAYOUT>
struct SrcsUnary<MATH, LAYOUT, SfpuIssue::Sfpi> : SfpuSrcsUnaryOp<SrcsUnary<MATH, LAYOUT, SfpuIssue::Sfpi>> {
    using Layout = SrcsLayout<LAYOUT>;

    sfpi_inline static void calculate() {
        using Operand = SfpuOperand<SfpuReg::SrcS, SfpiFormat<LAYOUT, sfpi::vFloat>>;
        calculate_unary_operands<MATH, Layout::ops>(Operand{Layout::in0}, Operand{Layout::out});
    }
};

/// CRTP base for binary SrcS ops; both inputs share formats. init()/run() live in llk_sfpu_srcs_api.h.
template <typename Op>
struct SfpuSrcsBinaryOp {
    static constexpr bool hw_clears_valids = false;

    static void init_op() {}

    template <std::uint8_t INSTRN_COUNT = 1>
    static void init(
        const std::uint32_t l1_in0_addr_16B,
        const std::uint32_t l1_in1_addr_16B,
        const DataFormat l1_in_format,
        const std::uint32_t l1_out_addr_16B,
        const DataFormat l1_out_format,
        const bool implied_math_format);

    template <std::uint8_t INSTRN_COUNT = 1>
    static void run(const std::uint32_t num_tiles);
};

/// Binary SrcS op. Only Sfpi exists.
template <typename MATH, sfpi::DataLayout LAYOUT, SfpuIssue ISSUE>
struct SrcsBinary {
    static_assert(sizeof(MATH) == 0, "This SFPU op has no SrcS implementation for the requested SfpuIssue");
};

template <typename MATH, sfpi::DataLayout LAYOUT>
struct SrcsBinary<MATH, LAYOUT, SfpuIssue::Sfpi> : SfpuSrcsBinaryOp<SrcsBinary<MATH, LAYOUT, SfpuIssue::Sfpi>> {
    using Layout = SrcsLayout<LAYOUT>;

    sfpi_inline static void calculate() {
        using Operand = SfpuOperand<SfpuReg::SrcS, SfpiFormat<LAYOUT, sfpi::vFloat>>;
        calculate_binary_operands<MATH, Layout::ops>(Operand{Layout::in0}, Operand{Layout::in1}, Operand{Layout::out});
    }
};

/// Map the SrcS register format (not the L1 format) to a SrcsLayout, once outside the tile loop.
template <typename Op>
sfpi_inline void dispatch_sfpu_srcs_format(const DataFormat format, Op&& op) {
    switch (format) {
        case DataFormat::Float32: op(SrcsLayout<sfpi::DataLayout::F32>{}); break;
        case DataFormat::Float16: op(SrcsLayout<sfpi::DataLayout::F16a>{}); break;
        case DataFormat::Float16_b: op(SrcsLayout<sfpi::DataLayout::F16b>{}); break;
        default: LLK_ASSERT(false, "Unsupported floating-point SrcS storage format"); break;
    }
}

}  // namespace ckernel::sfpu
