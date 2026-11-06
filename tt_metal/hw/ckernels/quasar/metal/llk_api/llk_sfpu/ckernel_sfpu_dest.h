// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_operand.h"
#include "ckernel_trisc_common.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"
#include "sfpu_impl.h"

// Dest op types, the Dest twin of ckernel_sfpu_srcs.h: an op supplies calculate() and optionally init_op().

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

/// Quasar SFPU vector modes; the check the SFPU_*_CALL macros made before an op ran.
inline void check_vector_mode([[maybe_unused]] const VectorMode vector_mode) {
    LLK_ASSERT(
        vector_mode == VectorMode::R || vector_mode == VectorMode::C || vector_mode == VectorMode::RC ||
            vector_mode == VectorMode::None || vector_mode == VectorMode::RC_custom,
        "Quasar SFPU supports vector modes R, C, RC, None, RC_custom");
}

}  // namespace detail

/**
 * @brief SFPLOAD/SFPSTORE layout of a Dest holding tiles of @p FAMILY. A 32-bit Dest is F32 whatever the family;
 *        Int32 is always explicit I32; a 16-bit Dest follows the family. There is no implied fallback.
 */
template <DataFormat FAMILY, bool EN_32BIT_DEST>
constexpr sfpi::DataLayout dest_layout() {
    if constexpr (FAMILY == DataFormat::Int32) {
        static_assert(EN_32BIT_DEST, "Int32 tiles need a 32-bit Dest");
        return sfpi::DataLayout::I32;
    } else if constexpr (EN_32BIT_DEST) {
        return sfpi::DataLayout::F32;
    } else if constexpr (FAMILY == DataFormat::Float16) {
        return sfpi::DataLayout::F16a;
    } else {
        static_assert(FAMILY == DataFormat::Float16_b, "a 16-bit Dest holds Float16 or Float16_b tiles");
        return sfpi::DataLayout::F16b;
    }
}

/** @brief CRTP base for unary Dest ops; Op supplies calculate() and optionally init_op(). */
template <typename Op>
struct SfpuDestUnaryOp {
    static void init_op() {}

    static void init() {
        _llk_math_eltwise_sfpu_init_();
        Op::init_op();
    }

    /** @brief Run Op on one Dest tile; vector_mode selects the faces. */
    template <trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
    static void run(const std::uint32_t dst_index, const VectorMode vector_mode = VectorMode::RC) {
        detail::check_dest_index<SLOT>(dst_index);
        detail::check_vector_mode(vector_mode);
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

    template <trisc::DstTileShape SLOT = Op::tile_shape>
    static void run(
        const std::uint32_t dst_index_in0,
        const std::uint32_t dst_index_in1,
        const std::uint32_t dst_index_out,
        const VectorMode vector_mode = VectorMode::RC) {
        detail::check_dest_index<SLOT>(dst_index_in0);
        detail::check_dest_index<SLOT>(dst_index_in1);
        detail::check_dest_index<SLOT>(dst_index_out);
        detail::check_vector_mode(vector_mode);
        _llk_math_eltwise_binary_sfpu_params_<SLOT>(
            Op::calculate, dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
};

/**
 * @brief Generic unary Dest op: ITERATIONS SFPI passes of MATH over one face, in place.
 *
 * Sfpi serves every MATH; LoadMacro versions are specialized next to the op.
 *
 * @tparam LAYOUT: SFPLOAD/SFPSTORE format; the caller names it, there is no default.
 * @tparam IMPL: Sfpi or LoadMacro, see @ref resolve_sfpu_impl.
 */
template <typename MATH, sfpi::DataLayout LAYOUT, int ITERATIONS = SFPU_ITERATIONS, SfpuImpl IMPL = SfpuImpl::Sfpi>
struct DestUnary {
    static_assert(sizeof(MATH) == 0, "This SFPU op has no Dest implementation for the requested SfpuImpl");
};

template <typename MATH, sfpi::DataLayout LAYOUT, int ITERATIONS>
struct DestUnary<MATH, LAYOUT, ITERATIONS, SfpuImpl::Sfpi>
    : SfpuDestUnaryOp<DestUnary<MATH, LAYOUT, ITERATIONS, SfpuImpl::Sfpi>> {
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
 * Sfpi serves every MATH; LoadMacro versions are specialized next to the op.
 *
 * @tparam LAYOUT: SFPLOAD/SFPSTORE format; the caller names it, there is no default.
 * @tparam OUT: Store policy of the output (rounding lives here, see DestBf16RneFormat).
 * @tparam TILE_SHAPE: Dest tile shape the indices are measured in; run<>() defaults to it.
 * @tparam IMPL: Sfpi or LoadMacro, see @ref resolve_sfpu_impl.
 */
template <
    typename MATH,
    sfpi::DataLayout LAYOUT,
    typename OUT = SfpiFormat<LAYOUT, typename MATH::value_type>,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32,
    SfpuImpl IMPL = SfpuImpl::Sfpi>
struct DestBinary {
    static_assert(sizeof(MATH) == 0, "This SFPU op has no Dest implementation for the requested SfpuImpl");
};

template <typename MATH, sfpi::DataLayout LAYOUT, typename OUT, int ITERATIONS, trisc::DstTileShape TILE_SHAPE>
struct DestBinary<MATH, LAYOUT, OUT, ITERATIONS, TILE_SHAPE, SfpuImpl::Sfpi>
    : SfpuDestBinaryOp<DestBinary<MATH, LAYOUT, OUT, ITERATIONS, TILE_SHAPE, SfpuImpl::Sfpi>> {
    static constexpr trisc::DstTileShape tile_shape = TILE_SHAPE;

    static void calculate(
        const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
        using Input = SfpuOperand<SfpuReg::Dest, SfpiFormat<LAYOUT, typename MATH::value_type>>;
        using Output = SfpuOperand<SfpuReg::Dest, OUT>;
        // sfpi dst_reg indexes in SFP_ROWS (2) units, so the tile stride is dest-rows / 2
        // (32 for Tile32x32; same as BH/WH 64/SFP_DESTREG_STRIDE).
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

}  // namespace sfpu
}  // namespace ckernel
