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
// init_op(); the base supplies init() and run(). Free functions (calculate_exponential, ...) stay
// for the macro path and forward to, or are forwarded to by, these types.

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

/** @brief SFPLOAD/SFPSTORE layout for a Dest holding @p format; Default (HW-implied) if it has no explicit mode. */
constexpr sfpi::DataLayout dest_layout(const DataFormat format) {
    return format == DataFormat::Float16                                   ? sfpi::DataLayout::F16a
           : format == DataFormat::Float16_b                               ? sfpi::DataLayout::F16b
           : (format == DataFormat::Float32 || format == DataFormat::Tf32) ? sfpi::DataLayout::F32
           : format == DataFormat::Int32                                   ? sfpi::DataLayout::I32
                                                                           : sfpi::DataLayout::Default;
}

/** @brief Explicit 32-bit float layout when the Dest is 32-bit, else HW-implied; for the fixed compute-API entries. */
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
    using value_type = typename MATH::value_type;
    static constexpr sfpi::DataLayout layout = LAYOUT;

    static void calculate() {
        using Operand = SfpuOperand<SfpuReg::Dest, SfpiFormat<LAYOUT, value_type>>;
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
    using value_type = typename MATH::value_type;
    static constexpr sfpi::DataLayout layout = LAYOUT;

    static void calculate(
        const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
        using Input = SfpuOperand<SfpuReg::Dest, SfpiFormat<LAYOUT, value_type>>;
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

/** @brief True when OP<LAYOUT>'s math computes on the vector type LAYOUT loads into (I32 -> vInt, else vFloat). */
template <template <sfpi::DataLayout> class OP, sfpi::DataLayout LAYOUT>
constexpr bool dest_op_supports = std::is_same_v<
    typename OP<LAYOUT>::value_type,
    std::conditional_t<LAYOUT == sfpi::DataLayout::I32, sfpi::vInt, sfpi::vFloat>>;

/**
 * @brief Call fn(OP<LAYOUT>{}) for the explicit LAYOUT of the Dest, once outside the tile loop.
 *
 * A 32-bit Dest holds Float32 (or Int32 for an integer family), whatever the 16-bit family of the inputs was, so
 * EN_32BIT_DEST decides first and @p dest_family only matters for a 16-bit Dest. Only the layouts OP supports are
 * compiled in (@ref dest_op_supports); a 16-bit family with no explicit mode runs the HW-implied layout.
 *
 * @tparam EN_32BIT_DEST: Dest is 32-bit.
 * @param dest_family: Dest format family of the inputs (the harness' sfpu_src / math format).
 * @note Test-harness helper: formats are runtime values there. Kernels with compile-time formats use OP<LAYOUT>
 * directly.
 */
template <template <sfpi::DataLayout> class OP, bool EN_32BIT_DEST, typename Fn>
inline void dispatch_sfpu_dest_op(const DataFormat dest_family, Fn&& fn) {
    bool handled = false;
    const auto run_as = [&](auto op) {
        fn(op);
        handled = true;
    };
    if constexpr (EN_32BIT_DEST) {
        if constexpr (dest_op_supports<OP, sfpi::DataLayout::I32>) {
            if (dest_family == DataFormat::Int32) {
                run_as(OP<sfpi::DataLayout::I32>{});
            }
        }
        if constexpr (dest_op_supports<OP, sfpi::DataLayout::F32>) {
            if (dest_family != DataFormat::Int32) {
                run_as(OP<sfpi::DataLayout::F32>{});
            }
        }
    } else {
        const sfpi::DataLayout layout = dest_layout(dest_family);
        const auto try_layout = [&](auto op) {
            if (layout == decltype(op)::layout) {
                run_as(op);
            }
        };
        if constexpr (dest_op_supports<OP, sfpi::DataLayout::F16a>) {
            try_layout(OP<sfpi::DataLayout::F16a>{});
        }
        if constexpr (dest_op_supports<OP, sfpi::DataLayout::F16b>) {
            try_layout(OP<sfpi::DataLayout::F16b>{});
        }
        if constexpr (dest_op_supports<OP, sfpi::DataLayout::Default>) {
            if (layout != sfpi::DataLayout::F16a && layout != sfpi::DataLayout::F16b) {
                run_as(OP<sfpi::DataLayout::Default>{});
            }
        }
    }
    LLK_ASSERT(handled, "Dest format not supported by this op");
}

}  // namespace sfpu
}  // namespace ckernel
