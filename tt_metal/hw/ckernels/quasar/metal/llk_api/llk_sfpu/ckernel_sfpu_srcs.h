// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_instr_params.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "sfpi.h"
#include "ckernel_sfpu_operand.h"
#include "sfpu_impl.h"

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
    static constexpr int in0 = 0;
    static constexpr int in1 = ops;
    static constexpr int out = 2 * ops;

    // SrcS register format (unpack destination = pack source) implied by the layout.
    static constexpr DataFormat format = LAYOUT == sfpi::DataLayout::F32    ? DataFormat::Float32
                                         : LAYOUT == sfpi::DataLayout::I32  ? DataFormat::Int32
                                         : LAYOUT == sfpi::DataLayout::F16a ? DataFormat::Float16
                                                                            : DataFormat::Float16_b;

    // SFPI vector type the layout loads into.
    using value_type = std::conditional_t<LAYOUT == sfpi::DataLayout::I32, sfpi::vInt, sfpi::vFloat>;

    // SFPLOAD/SFPSTORE format code, for the SFPLOADMACRO path.
    static constexpr std::uint32_t sfpmem = LAYOUT == sfpi::DataLayout::F32    ? p_sfpu::sfpmem::FP32
                                            : LAYOUT == sfpi::DataLayout::I32  ? p_sfpu::sfpmem::INT32
                                            : LAYOUT == sfpi::DataLayout::F16a ? p_sfpu::sfpmem::FP16A
                                                                               : p_sfpu::sfpmem::FP16B;
};

/**
 * @brief CRTP base for unary SrcS ops; Op supplies calculate().
 *
 * @note init() and run() are defined in llk_sfpu_srcs_api.h; include it to use them.
 */
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
        const DataFormat l1_out_format);

    template <std::uint8_t INSTRN_COUNT = 1>
    static void run(const std::uint32_t num_tiles);
};

/** @brief Unary SrcS op. Sfpi serves every MATH; LoadMacro versions are specialized next to the op. */
template <typename MATH, sfpi::DataLayout LAYOUT, SfpuImpl IMPL>
struct SrcsUnary {
    static_assert(sizeof(MATH) == 0, "This SFPU op has no SrcS implementation for the requested SfpuImpl");
};

template <typename MATH, sfpi::DataLayout LAYOUT>
struct SrcsUnary<MATH, LAYOUT, SfpuImpl::Sfpi> : SfpuSrcsUnaryOp<SrcsUnary<MATH, LAYOUT, SfpuImpl::Sfpi>> {
    using Layout = SrcsLayout<LAYOUT>;

    sfpi_inline static void calculate() {
        using Operand = SfpuOperand<SfpuReg::SrcS, SfpiFormat<LAYOUT, typename Layout::value_type>>;
        calculate_unary_operands<MATH, Layout::ops>(Operand{Layout::in0}, Operand{Layout::out});
    }
};

/**
 * @brief CRTP base for binary SrcS ops; both inputs share formats.
 *
 * @note init() and run() are defined in llk_sfpu_srcs_api.h; include it to use them.
 */
template <typename Op>
struct SfpuSrcsBinaryOp {
    // True when the op's own instructions release the SrcS banks (SFPLOADMACRO).
    static constexpr bool hw_clears_valids = false;

    static void init_op() {}

    // Takes only the L1 formats; the SrcS register format comes from Op::Layout.
    template <std::uint8_t INSTRN_COUNT = 1>
    static void init(
        const std::uint32_t l1_in0_addr_16B,
        const std::uint32_t l1_in1_addr_16B,
        const DataFormat l1_in_format,
        const std::uint32_t l1_out_addr_16B,
        const DataFormat l1_out_format);

    template <std::uint8_t INSTRN_COUNT = 1>
    static void run(const std::uint32_t num_tiles);
};

/** @brief Binary SrcS op; only Sfpi exists. */
template <typename MATH, sfpi::DataLayout LAYOUT, SfpuImpl IMPL>
struct SrcsBinary {
    static_assert(sizeof(MATH) == 0, "This SFPU op has no SrcS implementation for the requested SfpuImpl");
};

template <typename MATH, sfpi::DataLayout LAYOUT>
struct SrcsBinary<MATH, LAYOUT, SfpuImpl::Sfpi> : SfpuSrcsBinaryOp<SrcsBinary<MATH, LAYOUT, SfpuImpl::Sfpi>> {
    using Layout = SrcsLayout<LAYOUT>;

    sfpi_inline static void calculate() {
        using Operand = SfpuOperand<SfpuReg::SrcS, SfpiFormat<LAYOUT, typename Layout::value_type>>;
        calculate_binary_operands<MATH, Layout::ops>(Operand{Layout::in0}, Operand{Layout::in1}, Operand{Layout::out});
    }
};

namespace detail {

// MATH of a SrcS op shape.
template <typename Op>
struct srcs_op_math;
template <typename MATH, sfpi::DataLayout LAYOUT, SfpuImpl IMPL>
struct srcs_op_math<SrcsUnary<MATH, LAYOUT, IMPL>> {
    using type = MATH;
};
template <typename MATH, sfpi::DataLayout LAYOUT, SfpuImpl IMPL>
struct srcs_op_math<SrcsBinary<MATH, LAYOUT, IMPL>> {
    using type = MATH;
};

// Vector type OP<LAYOUT>'s math computes on (MATH::value_type).
template <template <sfpi::DataLayout> class OP, sfpi::DataLayout LAYOUT>
using srcs_op_value_t = typename srcs_op_math<OP<LAYOUT>>::type::value_type;

}  // namespace detail

/**
 * @brief True when OP<LAYOUT>'s math computes on the vector type LAYOUT loads into.
 *
 * Only names OP<LAYOUT>; its calculate() is never instantiated, so a float op may be queried for I32.
 */
template <template <sfpi::DataLayout> class OP, sfpi::DataLayout LAYOUT>
constexpr bool srcs_op_supports =
    std::is_same_v<detail::srcs_op_value_t<OP, LAYOUT>, typename SrcsLayout<LAYOUT>::value_type>;

/**
 * @brief Call fn(OP<LAYOUT>{}) for the LAYOUT matching the runtime SrcS format, once outside the tile loop.
 *
 * Only the layouts OP supports are compiled in (@ref srcs_op_supports). The SrcS register format is the
 * unpack destination and the pack source; both are passed so a mismatch is caught here.
 *
 * @note Test-harness helper: formats are runtime values there. Kernels with compile-time formats use OP<LAYOUT>
 * directly.
 */
template <template <sfpi::DataLayout> class OP, typename Fn>
sfpi_inline void dispatch_sfpu_srcs_op(
    const DataFormat unpack_S_dst_format, const DataFormat pack_S_src_format, Fn&& fn) {
    LLK_ASSERT(
        unpack_S_dst_format == pack_S_src_format,
        "SrcS ops require matching unpack destination and pack source formats");
    bool handled = false;
    const auto try_layout = [&](auto op) {
        using Op = decltype(op);
        if (unpack_S_dst_format == Op::Layout::format) {
            fn(op);
            handled = true;
        }
    };
    if constexpr (srcs_op_supports<OP, sfpi::DataLayout::F16a>) {
        try_layout(OP<sfpi::DataLayout::F16a>{});
    }
    if constexpr (srcs_op_supports<OP, sfpi::DataLayout::F16b>) {
        try_layout(OP<sfpi::DataLayout::F16b>{});
    }
    if constexpr (srcs_op_supports<OP, sfpi::DataLayout::F32>) {
        try_layout(OP<sfpi::DataLayout::F32>{});
    }
    if constexpr (srcs_op_supports<OP, sfpi::DataLayout::I32>) {
        try_layout(OP<sfpi::DataLayout::I32>{});
    }
    LLK_ASSERT(handled, "SrcS format not supported by this op");
}

}  // namespace ckernel::sfpu
