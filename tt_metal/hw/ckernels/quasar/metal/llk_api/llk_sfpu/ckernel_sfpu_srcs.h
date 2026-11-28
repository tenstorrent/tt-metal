// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "ckernel_instr_params.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "ckernel_sfpu_operand.h"
#include "sfpu_impl.h"

// SrcS op type, the SrcS twin of ckernel_sfpu_dest.h: SfpuSrcsOp<MATH, LAYOUT> runs any math policy on SrcS slices.
// An op with a LoadMacro version specializes SfpuSrcsOp next to its math.

namespace ckernel::sfpu {

static_assert(sfpi::SFP_SRCSREG_STRIDE == ckernel::math::SFP_ROWS, "one sfpi index step must be one SFPU op");

/**
 * @brief Slot offsets (SFPI steps) and formats of one SrcS slice.
 *
 * @note Slot order must match the SrcS slice convention in llk_sfpu_srcs_api.h: in0 slice 0, in1 slice 1, out slice 2.
 */
template <sfpi::DataLayout LAYOUT>
struct SrcsLayout {
    static_assert(
        LAYOUT == sfpi::DataLayout::F16a || LAYOUT == sfpi::DataLayout::F16b || LAYOUT == sfpi::DataLayout::F32 ||
            LAYOUT == sfpi::DataLayout::I32,
        "SrcS SFPI adapters support F16a, F16b, F32 and I32 layouts");

    static constexpr bool is_32bit = LAYOUT == sfpi::DataLayout::F32 || LAYOUT == sfpi::DataLayout::I32;
    static constexpr int rows = static_cast<int>(trisc::srcs_dims::ydim(is_32bit));
    static_assert(
        rows > 0 && rows % static_cast<int>(ckernel::math::SFP_ROWS) == 0, "SrcS slice must contain whole SFPU passes");

    static constexpr int ops = rows / static_cast<int>(ckernel::math::SFP_ROWS);
    static constexpr int slices = static_cast<int>(trisc::srcs_dims::slice_count(is_32bit));  // of one 32x32 tile
    static constexpr int inputs = 2;  // input slices ahead of the output slice
    static constexpr int out = inputs * ops;

    /** @brief Slot of input @p index: inputs fill the slices in order. */
    static constexpr int input(const int index) { return index * ops; }

    // SrcS register format (unpack destination = pack source) implied by the layout.
    static constexpr DataFormat format = LAYOUT == sfpi::DataLayout::F32    ? DataFormat::Float32
                                         : LAYOUT == sfpi::DataLayout::I32  ? DataFormat::Int32
                                         : LAYOUT == sfpi::DataLayout::F16a ? DataFormat::Float16
                                                                            : DataFormat::Float16_b;

    // SFPLOAD/SFPSTORE format code, for the SFPLOADMACRO path.
    static constexpr std::uint32_t sfpmem = LAYOUT == sfpi::DataLayout::F32    ? p_sfpu::sfpmem::FP32
                                            : LAYOUT == sfpi::DataLayout::I32  ? p_sfpu::sfpmem::INT32
                                            : LAYOUT == sfpi::DataLayout::F16a ? p_sfpu::sfpmem::FP16A
                                                                               : p_sfpu::sfpmem::FP16B;
};

/**
 * @brief CRTP base of a SrcS op: the SFPU-thread pipeline that unpacks L1_INPUTS inputs from L1 into SrcS, runs the
 *        op and packs the result to L1. Op supplies Layout, calculate() and optionally init_op() and
 *        hw_clears_valids.
 *
 * @tparam Op: The op type deriving from this base.
 * @tparam L1_INPUTS: Number of inputs unpacked from L1, values = <1/2>
 * @note The functions are defined in llk_sfpu_srcs_api.h; include it to use them. Call init() once, then unpack()
 *       and pack() to set the L1 buffers and run() for the tiles in them.
 */
template <typename Op, int L1_INPUTS>
struct SfpuSrcsOpBase {
    static_assert(sizeof(Op) == 0, "SrcS has no pipeline for an SFPU op with this many L1 inputs");
};

/** @brief What every SrcS pipeline shares. */
template <typename Op>
struct SfpuSrcsOpCommon {
    // True when the op's own instructions release the SrcS banks (SFPLOADMACRO).
    static constexpr bool hw_clears_valids = false;

    static void init_op() {}

    // Set the L1 buffer packer 1 packs the result to.
    static void pack(const std::uint32_t l1_addr_16B, const DataFormat l1_format);
};

template <typename Op>
struct SfpuSrcsOpBase<Op, 1> : SfpuSrcsOpCommon<Op> {
    // Set up unpacker 2, packer 1, the SFPU and the op. Once.
    static void init();

    // init(), unpack() and pack() in one call.
    static void init(
        const std::uint32_t l1_in_addr_16B,
        const DataFormat l1_in_format,
        const std::uint32_t l1_out_addr_16B,
        const DataFormat l1_out_format);

    // Set the L1 buffer unpacker 2 unpacks the input from.
    static void unpack(const std::uint32_t l1_addr_16B, const DataFormat l1_format);

    // Unpack, run the op and pack num_tiles consecutive tiles of the buffers unpack() and pack() set.
    static void run(const std::uint32_t num_tiles = 1);
};

// Both inputs share the L1 format.
template <typename Op>
struct SfpuSrcsOpBase<Op, 2> : SfpuSrcsOpCommon<Op> {
    static void init();

    static void init(
        const std::uint32_t l1_in0_addr_16B,
        const std::uint32_t l1_in1_addr_16B,
        const DataFormat l1_in_format,
        const std::uint32_t l1_out_addr_16B,
        const DataFormat l1_out_format);

    static void unpack(
        const std::uint32_t l1_in0_addr_16B, const std::uint32_t l1_in1_addr_16B, const DataFormat l1_format);

    static void run(const std::uint32_t num_tiles = 1);
};

/**
 * @brief SrcS op type: out = MATH(inputs...) over one slice, for any math policy. An op specializes this type next
 *        to its math for a LoadMacro version and declares it with @ref srcs_has_loadmacro.
 *
 * @tparam MATH: Math policy; MATH::value_type is the vector type its inputs load into.
 * @tparam LAYOUT: SFPLOAD/SFPSTORE format of the slices.
 * @tparam IMPL: Sfpi or LoadMacro, see @ref resolve_sfpu_impl.
 * @note Call init() once before the first tile.
 */
template <typename MATH, sfpi::DataLayout LAYOUT, SfpuImpl IMPL = SfpuImpl::Sfpi>
struct SfpuSrcsOp {
    static_assert(sizeof(MATH) == 0, "This SFPU op has no SrcS implementation for the requested SfpuImpl");
};

template <typename MATH, sfpi::DataLayout LAYOUT>
struct SfpuSrcsOp<MATH, LAYOUT, SfpuImpl::Sfpi>
    : SfpuSrcsOpBase<SfpuSrcsOp<MATH, LAYOUT, SfpuImpl::Sfpi>, math_arity<MATH>> {
    using Layout = SrcsLayout<LAYOUT>;
    static_assert(
        std::is_same_v<typename MATH::value_type, layout_value_t<LAYOUT>>,
        "this SFPU op does not take tiles of this SrcS layout");
    static_assert(math_arity<MATH> <= Layout::inputs, "SrcS has input slices for two operands");

    sfpi_inline static void calculate() { calculate(std::make_index_sequence<math_arity<MATH>>{}); }

private:
    template <std::size_t... INPUT>
    sfpi_inline static void calculate(std::index_sequence<INPUT...>) {
        using Operand = SfpuOperand<SfpuReg::SrcS, SfpiFormat<LAYOUT, layout_value_t<LAYOUT>>>;
        calculate_operands<MATH, Layout::ops>(
            math_setup<MATH>(), Operand{Layout::out}, Operand{Layout::input(static_cast<int>(INPUT))}...);
    }
};

/** @brief True for a MATH that has a LoadMacro SfpuSrcsOp specialization; set next to that specialization. */
template <typename MATH>
inline constexpr bool srcs_has_loadmacro = false;

/**
 * @brief The SrcS op type of an op tag on a layout. I32 slices are sign-magnitude (@ref IntEncoding).
 *
 * @tparam OP: Op tag (ckernel::op).
 * @tparam LAYOUT: SFPLOAD/SFPSTORE format of the slices, see @ref sfpu_layout.
 * @tparam IMPL: Sfpi or LoadMacro, see @ref resolve_sfpu_impl.
 */
template <typename OP, sfpi::DataLayout LAYOUT, SfpuImpl IMPL = SfpuImpl::Sfpi>
using srcs_op_t =
    SfpuSrcsOp<math_for_t<OP, LAYOUT>, LAYOUT, resolve_sfpu_impl<IMPL, srcs_has_loadmacro<math_for_t<OP, LAYOUT>>>()>;

/**
 * @brief SrcS op type whose first input is a Dest tile: out = MATH(dest, in), in from L1, out to L1.
 *
 * @tparam MATH: Two-input float math policy.
 * @tparam LAYOUT: SFPLOAD/SFPSTORE format of the SrcS slices.
 * @tparam DEST_LAYOUT: SFPLOAD format of the Dest tile.
 * @note The SFPU thread must hold the Dest loan (_llk_tile_regs_borrow_) around run().
 */
template <typename MATH, sfpi::DataLayout LAYOUT, sfpi::DataLayout DEST_LAYOUT>
struct SfpuSrcsDestOp : SfpuSrcsOpBase<SfpuSrcsDestOp<MATH, LAYOUT, DEST_LAYOUT>, 1 /*L1_INPUTS*/> {
    using Layout = SrcsLayout<LAYOUT>;
    static_assert(math_arity<MATH> == 2, "this SrcS op takes one Dest tile and one input from L1");
    static_assert(
        std::is_same_v<typename MATH::value_type, sfpi::vFloat> &&
            std::is_same_v<layout_value_t<LAYOUT>, sfpi::vFloat> &&
            std::is_same_v<layout_value_t<DEST_LAYOUT>, sfpi::vFloat>,
        "a Dest input is supported for float ops: Int32 is encoded differently in Dest and SrcS");

    // One slice; dest_step is the SFPI step of its first row in the lent Dest section.
    sfpi_inline static void calculate(const int dest_step) {
        using Srcs = SfpuOperand<SfpuReg::SrcS, SfpiFormat<LAYOUT, sfpi::vFloat>>;
        using Dest = SfpuOperand<SfpuReg::Dest, SfpiFormat<DEST_LAYOUT, sfpi::vFloat>>;
        calculate_operands<MATH, Layout::ops>(
            math_setup<MATH>(), Srcs{Layout::out}, Dest{dest_step}, Srcs{Layout::input(0)});
    }

    // Run the op on Dest tile dst_index and one tile of the buffers unpack() and pack() set. DST_SYNC and DST_ACCUM
    // (true for a 32-bit Dest) bound the Dest index.
    template <DstSync DST_SYNC, bool DST_ACCUM>
    static void run(const DstIndex dst_index);
};

/** @brief The @ref SfpuSrcsDestOp of an op tag. */
template <typename OP, sfpi::DataLayout LAYOUT, sfpi::DataLayout DEST_LAYOUT>
using srcs_dest_op_t = SfpuSrcsDestOp<math_for_t<OP, LAYOUT>, LAYOUT, DEST_LAYOUT>;

}  // namespace ckernel::sfpu
