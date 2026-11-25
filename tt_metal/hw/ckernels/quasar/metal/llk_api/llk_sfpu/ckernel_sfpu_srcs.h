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
 * @brief CRTP base of a SrcS op with ARITY inputs: init() and run(), or init() in two steps as configure() and
 *        set_l1(). Op supplies Layout, calculate() and optionally init_op() and hw_clears_valids.
 *
 * The pipeline differs per number of inputs (see llk_sfpu_srcs_api.h): one input is unpacked by the auto-loop,
 * two inputs by a per-slice MOP.
 *
 * @tparam Op: The op type deriving from this base.
 * @tparam ARITY: Number of inputs, values = <1/2>
 * @note The functions are defined in llk_sfpu_srcs_api.h; include it to use them.
 */
template <typename Op, int ARITY>
struct SfpuSrcsOpBase {
    static_assert(sizeof(Op) == 0, "SrcS has no pipeline for an SFPU op with this many inputs");
};

template <typename Op>
struct SfpuSrcsOpBase<Op, 1> {
    // True when the op's own instructions release the SrcS banks (SFPLOADMACRO).
    static constexpr bool hw_clears_valids = false;

    static void init_op() {}

    static void configure();

    // Takes only the L1 formats; the SrcS register format comes from Op::Layout.
    static void set_l1(
        const std::uint32_t l1_in_addr_16B,
        const DataFormat l1_in_format,
        const std::uint32_t l1_out_addr_16B,
        const DataFormat l1_out_format);

    static void init(
        const std::uint32_t l1_in_addr_16B,
        const DataFormat l1_in_format,
        const std::uint32_t l1_out_addr_16B,
        const DataFormat l1_out_format);

    static void run(const std::uint32_t num_tiles);
};

// Both inputs share formats.
template <typename Op>
struct SfpuSrcsOpBase<Op, 2> {
    // True when the op's own instructions release the SrcS banks (SFPLOADMACRO).
    static constexpr bool hw_clears_valids = false;

    static void init_op() {}

    static void configure();

    // Takes only the L1 formats; the SrcS register format comes from Op::Layout.
    static void set_l1(
        const std::uint32_t l1_in0_addr_16B,
        const std::uint32_t l1_in1_addr_16B,
        const DataFormat l1_in_format,
        const std::uint32_t l1_out_addr_16B,
        const DataFormat l1_out_format);

    static void init(
        const std::uint32_t l1_in0_addr_16B,
        const std::uint32_t l1_in1_addr_16B,
        const DataFormat l1_in_format,
        const std::uint32_t l1_out_addr_16B,
        const DataFormat l1_out_format);

    static void run(const std::uint32_t num_tiles);
};

/**
 * @brief SrcS op type: out = MATH(inputs...) over one slice, for any math policy.
 *
 * The number of inputs is read from MATH::apply. Sfpi serves every MATH; an op specializes this type next to its
 * math for a LoadMacro version (and declares it with @ref srcs_has_loadmacro).
 *
 * @tparam MATH: Math policy; MATH::value_type is the vector type its inputs load into.
 * @tparam LAYOUT: SFPLOAD/SFPSTORE format of the slices.
 * @tparam IMPL: Sfpi or LoadMacro, see @ref resolve_sfpu_impl.
 * @note Call init() once before run().
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
 * @brief The SrcS op type of an op tag on a layout.
 *
 * Requesting LoadMacro for a math without one fails to compile; under DISABLE_SFPLOADMACRO (ttsim) every request
 * resolves to Sfpi. I32 slices are sign-magnitude (@ref IntEncoding), which is how SrcS holds Int32.
 *
 * @tparam OP: Op tag (ckernel::op).
 * @tparam LAYOUT: SFPLOAD/SFPSTORE format of the slices, see @ref sfpu_layout.
 * @tparam IMPL: Sfpi or LoadMacro.
 */
template <typename OP, sfpi::DataLayout LAYOUT, SfpuImpl IMPL = SfpuImpl::Sfpi>
using srcs_op_t =
    SfpuSrcsOp<math_for_t<OP, LAYOUT>, LAYOUT, resolve_sfpu_impl<IMPL, srcs_has_loadmacro<math_for_t<OP, LAYOUT>>>()>;

}  // namespace ckernel::sfpu
