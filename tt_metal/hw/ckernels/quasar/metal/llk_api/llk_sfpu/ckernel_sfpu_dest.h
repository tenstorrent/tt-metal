// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_sfpu_operand.h"
#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"

// Dest op types: the Dest twin of ckernel_sfpu_srcs.h. An op supplies calculate() and optionally
// init_op(); the base supplies init() and run(). The macro-path free functions (calculate_exponential, ...)
// stay and share one body with these types.

namespace ckernel {
namespace sfpu {

namespace detail {

// Dest bounds check; only metal builds define the kernel-wide sync mode.
template <trisc::DstTileShape SLOT>
inline void check_dest_index([[maybe_unused]] const std::uint32_t dst_index) {
#ifdef DST_SYNC_MODE
    LLK_ASSERT(
        (dst_index < trisc::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, SLOT>()),
        "dst_index exceeds max dest tiles");
#endif
}

}  // namespace detail

/** @brief Explicit F32 on a 32-bit Dest, HW-implied on a 16-bit one: all an entry without a CB (exp_tile) can know. */
template <bool EN_32BIT_DEST>
constexpr sfpi::DataLayout dest_float_layout = EN_32BIT_DEST ? sfpi::DataLayout::F32 : sfpi::DataLayout::Default;

/** @brief CRTP base for unary Dest ops; Op supplies calculate() and optionally init_op(). */
template <typename Op>
struct SfpuDestUnaryOp {
    static void init_op() {}

    static void init() {
        _llk_math_eltwise_sfpu_init_();
        Op::init_op();
    }

    /** @brief Run Op on one Dest tile; vector_mode selects the faces, as the SFPU_UNARY_CALL macro does. */
    template <trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
    static void run(const std::uint32_t dst_index, const VectorMode vector_mode = VectorMode::RC) {
        detail::check_dest_index<SLOT>(dst_index);
        _llk_math_eltwise_unary_sfpu_params_<SLOT>(Op::calculate, dst_index, vector_mode);
    }
};

/** @brief CRTP base for binary Dest ops; Op supplies calculate(in0, in1, out) and optionally init_op(). */
template <typename Op>
struct SfpuDestBinaryOp {
    static void init_op() {}

    static void init() {
        _llk_math_eltwise_sfpu_init_();
        Op::init_op();
    }

    template <trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
    static void run(
        const std::uint32_t dst_index_in0,
        const std::uint32_t dst_index_in1,
        const std::uint32_t dst_index_out,
        const VectorMode vector_mode = VectorMode::RC) {
        detail::check_dest_index<SLOT>(dst_index_in0);
        detail::check_dest_index<SLOT>(dst_index_in1);
        detail::check_dest_index<SLOT>(dst_index_out);
        _llk_math_eltwise_binary_sfpu_params_<SLOT>(
            Op::calculate, dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
};

/**
 * @brief Generic unary Dest op: ITERATIONS SFPI passes of MATH over one face, in place.
 *
 * @tparam LAYOUT: SFPLOAD/SFPSTORE format; Default lets the HW derive it from the ALU config.
 */
template <typename MATH, sfpi::DataLayout LAYOUT = sfpi::DataLayout::Default, int ITERATIONS = SFPU_ITERATIONS>
struct DestUnary : SfpuDestUnaryOp<DestUnary<MATH, LAYOUT, ITERATIONS>> {
    static void calculate() {
        using Operand = SfpuOperand<SfpuReg::Dest, SfpiFormat<LAYOUT, typename MATH::value_type>>;
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            calculate_unary_operands<MATH, 1 /*ITERATIONS*/>(Operand{}, Operand{});
            sfpi::dst_reg++;
        }
    }
};

/**
 * @brief Generic binary Dest op: out = MATH(in0, in1) over one face of each tile.
 *
 * @tparam OUT: Store policy of the output (rounding lives here, see DestBf16RneFormat).
 * @tparam TILE_SHAPE: Dest tile shape the indices are measured in; pass the same shape to run<>().
 */
template <
    typename MATH,
    sfpi::DataLayout LAYOUT = sfpi::DataLayout::Default,
    typename OUT = SfpiFormat<LAYOUT, typename MATH::value_type>,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
struct DestBinary : SfpuDestBinaryOp<DestBinary<MATH, LAYOUT, OUT, ITERATIONS, TILE_SHAPE>> {
    static void calculate(
        const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
        using Input = SfpuOperand<SfpuReg::Dest, SfpiFormat<LAYOUT, typename MATH::value_type>>;
        using Output = SfpuOperand<SfpuReg::Dest, OUT>;
        constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            calculate_binary_operands<MATH, 1 /*ITERATIONS*/>(
                Input{static_cast<int>(dst_index_in0 * dst_tile_size_sfpi)},
                Input{static_cast<int>(dst_index_in1 * dst_tile_size_sfpi)},
                Output{static_cast<int>(dst_index_out * dst_tile_size_sfpi)});
            sfpi::dst_reg++;
        }
    }
};

/** @brief A DataLayout as a type, so a lambda can take it and use it as a template argument. */
template <sfpi::DataLayout L>
using layout_c = std::integral_constant<sfpi::DataLayout, L>;

/**
 * @brief Call fn(layout_c<L>{}) with the explicit SFPLOAD/SFPSTORE layout of a float Dest; once per tile loop.
 *
 * The harness' Dest family (sfpu_src / math format) stays Float16_b under dest_acc, so EN_32BIT_DEST decides
 * first. Only a 16-bit Dest looks at the family: F16a, F16b, else HW-implied.
 */
template <bool EN_32BIT_DEST, typename Fn>
inline void dispatch_dest_float_layout(const DataFormat family, Fn&& fn) {
    if constexpr (EN_32BIT_DEST) {
        fn(layout_c<sfpi::DataLayout::F32>{});
    } else {
        switch (family) {
            case DataFormat::Float16: fn(layout_c<sfpi::DataLayout::F16a>{}); break;
            case DataFormat::Float16_b: fn(layout_c<sfpi::DataLayout::F16b>{}); break;
            default: fn(layout_c<sfpi::DataLayout::Default>{}); break;
        }
    }
}

}  // namespace sfpu
}  // namespace ckernel
