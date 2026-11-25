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
#include "llk_math_common.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpi.h"
#include "sfpu_impl.h"

// Dest op type, the Dest twin of ckernel_sfpu_srcs.h: SfpuDestOp<MATH, LAYOUT> runs any math policy on Dest tiles.
// An op with a tuned loop or a LoadMacro version specializes SfpuDestOp next to its math.

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

// True when @p format is in a DataFormat bit mask (the masks of llk_math_common.h); formats past bit 63 are in none.
constexpr bool format_in(const std::uint64_t format_mask, const DataFormat format) {
    const std::uint32_t bit = static_cast<std::uint32_t>(format);
    return bit < 64 && ((format_mask >> bit) & 1U) != 0;
}

}  // namespace detail

/**
 * @brief SFPLOAD/SFPSTORE layout of Dest tiles that kept the register format @p FAMILY of their DFB: copied or
 *        unpacked to Dest, not computed by an FPU math op (that is @ref fpu_dest_layout).
 *
 * A 32-bit Dest is I32 for Int32 and F32 for every float family; a 16-bit Dest follows the family's exponent
 * width (F16a for Float16 and MxFp4_2x_A, F16b for Float16_b and MxFp4_2x_B). There is no implied fallback: a
 * format with no layout is a compile error.
 *
 * @tparam FAMILY: Register format of the DFB the tiles came from.
 * @tparam EN_32BIT_DEST: Dest is in 32-bit mode (Float32/Int32).
 */
template <DataFormat FAMILY, bool EN_32BIT_DEST>
constexpr sfpi::DataLayout dest_layout() {
    if constexpr (FAMILY == DataFormat::Int32) {
        static_assert(EN_32BIT_DEST, "Int32 tiles need a 32-bit Dest");
        return sfpi::DataLayout::I32;
    } else if constexpr (EN_32BIT_DEST) {
        static_assert(
            detail::format_in(kFp32DestCompatibleMask, FAMILY),
            "a 32-bit Dest holds Float32 only for a float family; this format has no SFPU layout on Dest");
        return sfpi::DataLayout::F32;
    } else if constexpr (FAMILY == DataFormat::Float16 || FAMILY == DataFormat::MxFp4_2x_A) {
        return sfpi::DataLayout::F16a;
    } else {
        static_assert(
            FAMILY == DataFormat::Float16_b || FAMILY == DataFormat::MxFp4_2x_B,
            "a 16-bit Dest holds Float16 or Float16_b tiles; this format has no SFPU layout on Dest");
        return sfpi::DataLayout::F16b;
    }
}

/**
 * @brief SFPLOAD/SFPSTORE layout of the Dest tiles an FPU math op leaves, from the register formats of its SrcA
 *        and SrcB.
 *
 * Mirrors the ALU format rule (_build_alu_words_ in llk_math_common.h): a 32-bit Dest holds Int32 when both
 * sources are Int32-compatible and Float32 otherwise. A 16-bit Dest takes the exponent width the sources share.
 *
 * @tparam SRCA: Register format of the FPU op's SrcA.
 * @tparam SRCB: Register format of the FPU op's SrcB.
 * @tparam EN_32BIT_DEST: Dest is in 32-bit mode (Float32/Int32).
 */
template <DataFormat SRCA, DataFormat SRCB, bool EN_32BIT_DEST>
constexpr sfpi::DataLayout fpu_dest_layout() {
    constexpr bool int_math =
        detail::format_in(kInt32DestCompatibleMask, SRCA) && detail::format_in(kInt32DestCompatibleMask, SRCB);
    if constexpr (EN_32BIT_DEST && int_math) {
        return sfpi::DataLayout::I32;
    } else if constexpr (EN_32BIT_DEST) {
        static_assert(
            detail::format_in(kFp32DestCompatibleMask, SRCA) && detail::format_in(kFp32DestCompatibleMask, SRCB),
            "these FPU source formats do not leave Float32 or Int32 in a 32-bit Dest");
        return sfpi::DataLayout::F32;
    } else {
        static_assert(
            dest_layout<SRCA, false /*EN_32BIT_DEST*/>() == dest_layout<SRCB, false /*EN_32BIT_DEST*/>(),
            "on a 16-bit Dest the FPU sources must share one exponent width: both Float16 or both Float16_b family");
        return dest_layout<SRCA, false /*EN_32BIT_DEST*/>();
    }
}

/**
 * @brief CRTP base of a Dest op with ARITY inputs: init() and run(). Op supplies calculate(), tile_shape and
 *        optionally init_op().
 *
 * The walk differs per number of inputs: one input runs in place on a tile, two inputs take three tile indices.
 *
 * @tparam Op: The op type deriving from this base.
 * @tparam ARITY: Number of inputs, values = <1/2>
 */
template <typename Op, int ARITY>
struct SfpuDestOpBase {
    static_assert(sizeof(Op) == 0, "Dest has no walk for an SFPU op with this many inputs");
};

template <typename Op>
struct SfpuDestOpBase<Op, 1> {
    static void init_op() {}

    static void init() {
        _llk_math_eltwise_sfpu_init_();
        Op::init_op();
    }

    /** @brief Run Op on one Dest tile, in place; vector_mode selects the faces. */
    template <trisc::DstTileShape SLOT>
    static void run(const DstIndex dst_index, const VectorMode vector_mode = VectorMode::RC) {
        const std::uint32_t index = static_cast<std::uint32_t>(dst_index);
        detail::check_dest_index<SLOT>(index);
        detail::check_vector_mode(vector_mode);
        _llk_math_eltwise_unary_sfpu_params_<SLOT>(Op::calculate, index, vector_mode);
    }

    /** @brief run<SLOT>() with the tile shape of Op. */
    static void run(const DstIndex dst_index, const VectorMode vector_mode = VectorMode::RC) {
        run<Op::tile_shape>(dst_index, vector_mode);
    }
};

template <typename Op>
struct SfpuDestOpBase<Op, 2> {
    static void init_op() {}

    static void init() {
        _llk_math_eltwise_sfpu_init_();
        Op::init_op();
    }

    /** @brief Run Op on two Dest tiles into a third (the output may be one of the inputs). */
    template <trisc::DstTileShape SLOT>
    static void run(
        const DstIndex dst_index_in0,
        const DstIndex dst_index_in1,
        const DstIndex dst_index_out,
        const VectorMode vector_mode = VectorMode::RC) {
        const std::uint32_t in0 = static_cast<std::uint32_t>(dst_index_in0);
        const std::uint32_t in1 = static_cast<std::uint32_t>(dst_index_in1);
        const std::uint32_t out = static_cast<std::uint32_t>(dst_index_out);
        detail::check_dest_index<SLOT>(in0);
        detail::check_dest_index<SLOT>(in1);
        detail::check_dest_index<SLOT>(out);
        detail::check_vector_mode(vector_mode);
        _llk_math_eltwise_binary_sfpu_params_<SLOT>(Op::calculate, in0, in1, out, vector_mode);
    }

    /** @brief run<SLOT>() with the tile shape of Op. */
    static void run(
        const DstIndex dst_index_in0,
        const DstIndex dst_index_in1,
        const DstIndex dst_index_out,
        const VectorMode vector_mode = VectorMode::RC) {
        run<Op::tile_shape>(dst_index_in0, dst_index_in1, dst_index_out, vector_mode);
    }
};

namespace detail {

// Row loop of the generic Dest op over one face, per number of inputs.
template <
    int ARITY,
    typename MATH,
    sfpi::DataLayout LAYOUT,
    sfpi::DataLayout OUT_LAYOUT,
    int ITERATIONS,
    trisc::DstTileShape TILE_SHAPE>
struct DestLoop;

template <
    typename MATH,
    sfpi::DataLayout LAYOUT,
    sfpi::DataLayout OUT_LAYOUT,
    int ITERATIONS,
    trisc::DstTileShape TILE_SHAPE>
struct DestLoop<1, MATH, LAYOUT, OUT_LAYOUT, ITERATIONS, TILE_SHAPE> {
    static void calculate() {
        using Input = SfpuOperand<SfpuReg::Dest, SfpiFormat<LAYOUT, layout_value_t<LAYOUT>>>;
        using Output = SfpuOperand<SfpuReg::Dest, SfpiFormat<OUT_LAYOUT, layout_value_t<OUT_LAYOUT>>>;
        const auto setup = math_setup<MATH>();
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            calculate_operands<MATH, 1 /*ITERATIONS*/>(setup, Output{}, Input{});
            sfpi::dst_reg++;
        }
    }
};

template <
    typename MATH,
    sfpi::DataLayout LAYOUT,
    sfpi::DataLayout OUT_LAYOUT,
    int ITERATIONS,
    trisc::DstTileShape TILE_SHAPE>
struct DestLoop<2, MATH, LAYOUT, OUT_LAYOUT, ITERATIONS, TILE_SHAPE> {
    static void calculate(
        const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
        using Input = SfpuOperand<SfpuReg::Dest, SfpiFormat<LAYOUT, layout_value_t<LAYOUT>>>;
        using Output = SfpuOperand<SfpuReg::Dest, SfpiFormat<OUT_LAYOUT, layout_value_t<OUT_LAYOUT>>>;
        // sfpi dst_reg indexes in SFP_ROWS (2) units, so the tile stride is dest-rows / 2
        // (32 for Tile32x32; same as BH/WH 64/SFP_DESTREG_STRIDE).
        constexpr std::uint32_t dst_tile_size_sfpi = 1U << (trisc::get_dest_tile_size_log2(TILE_SHAPE) - 1);
        // Two rows per iteration when the math is one instruction and the row count is even (see
        // calculate_binary_operands_two_rows); any other math keeps one row per iteration.
        constexpr bool two_rows = ITERATIONS % 2 == 0 && is_single_instruction_math<MATH>;
        if constexpr (two_rows) {
            // The second row's offset is formed before the cast to int: with an unsigned offset GCC reuses the
            // first row's address computation, with a signed one it redoes it for every runtime Dest index.
            const std::uint32_t off_in0 = dst_index_in0 * dst_tile_size_sfpi;
            const std::uint32_t off_in1 = dst_index_in1 * dst_tile_size_sfpi;
            const std::uint32_t off_out = dst_index_out * dst_tile_size_sfpi;
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d += 2) {
                calculate_binary_operands_two_rows<MATH>(
                    Input{static_cast<int>(off_in0)},
                    Input{static_cast<int>(off_in1)},
                    Output{static_cast<int>(off_out)},
                    Input{static_cast<int>(off_in0 + 1)},
                    Input{static_cast<int>(off_in1 + 1)},
                    Output{static_cast<int>(off_out + 1)});
                sfpi::dst_reg += 2;
            }
        } else {
            const auto setup = math_setup<MATH>();
#pragma GCC unroll 8
            for (int d = 0; d < ITERATIONS; d++) {
                calculate_operands<MATH, 1 /*ITERATIONS*/>(
                    setup,
                    Output{static_cast<int>(dst_index_out * dst_tile_size_sfpi)},
                    Input{static_cast<int>(dst_index_in0 * dst_tile_size_sfpi)},
                    Input{static_cast<int>(dst_index_in1 * dst_tile_size_sfpi)});
                sfpi::dst_reg++;
            }
        }
    }
};

}  // namespace detail

/**
 * @brief Dest op type: out = MATH(inputs...) over one face of each tile, for any math policy.
 *
 * The number of inputs is read from MATH::apply. One input runs in place: calculate(), run(dst_index). Two inputs
 * take tile indices: calculate(in0, in1, out), run(in0, in1, out). Sfpi serves every MATH; an op specializes this
 * type next to its math for a tuned loop or a LoadMacro version.
 *
 * @tparam MATH: Math policy; MATH::value_type is the vector type its inputs load into.
 * @tparam LAYOUT: SFPLOAD format of the inputs.
 * @tparam OUT_LAYOUT: SFPSTORE format of the output; differs from LAYOUT only for a math that changes the type.
 * @tparam ITERATIONS: Number of SFPI rows per face.
 * @tparam TILE_SHAPE: Dest tile shape the indices are measured in; run() defaults to it.
 * @tparam IMPL: Sfpi or LoadMacro, see @ref resolve_sfpu_impl.
 * @note Call init() once before the first run() or calculate() of the op.
 */
template <
    typename MATH,
    sfpi::DataLayout LAYOUT,
    sfpi::DataLayout OUT_LAYOUT = LAYOUT,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32,
    SfpuImpl IMPL = SfpuImpl::Sfpi>
struct SfpuDestOp {
    static_assert(sizeof(MATH) == 0, "This SFPU op has no Dest implementation for the requested SfpuImpl");
};

template <
    typename MATH,
    sfpi::DataLayout LAYOUT,
    sfpi::DataLayout OUT_LAYOUT,
    int ITERATIONS,
    trisc::DstTileShape TILE_SHAPE>
struct SfpuDestOp<MATH, LAYOUT, OUT_LAYOUT, ITERATIONS, TILE_SHAPE, SfpuImpl::Sfpi>
    : SfpuDestOpBase<SfpuDestOp<MATH, LAYOUT, OUT_LAYOUT, ITERATIONS, TILE_SHAPE, SfpuImpl::Sfpi>, math_arity<MATH>>,
      detail::DestLoop<math_arity<MATH>, MATH, LAYOUT, OUT_LAYOUT, ITERATIONS, TILE_SHAPE> {
    static_assert(
        std::is_same_v<typename MATH::value_type, layout_value_t<LAYOUT>>,
        "this SFPU op does not take tiles of this Dest layout");

    static constexpr trisc::DstTileShape tile_shape = TILE_SHAPE;
};

/**
 * @brief The Dest op type of an op tag on a layout.
 *
 * @tparam OP: Op tag (ckernel::op).
 * @tparam LAYOUT: SFPLOAD/SFPSTORE format of the tiles, see @ref dest_layout and @ref fpu_dest_layout.
 * @tparam ITERATIONS: Number of SFPI rows per face.
 * @tparam TILE_SHAPE: Dest tile shape the indices are measured in.
 * @tparam INT_ENCODING: Encoding of I32 tiles: SignMagnitude when an FPU op computed them, TwosComplement for
 *         native Int32 tiles copied or unpacked to Dest.
 */
template <
    typename OP,
    sfpi::DataLayout LAYOUT,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32,
    IntEncoding INT_ENCODING = IntEncoding::SignMagnitude>
using dest_op_t = SfpuDestOp<math_for_t<OP, LAYOUT, INT_ENCODING>, LAYOUT, LAYOUT, ITERATIONS, TILE_SHAPE>;

}  // namespace sfpu
}  // namespace ckernel
