// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <utility>

#include "sfpi.h"
#include "sfpu_reg.h"

namespace ckernel::sfpu {

namespace detail {

inline constexpr sfpi::SrcSView<0> srcs_reg{};

template <SfpuReg REG>
sfpi_inline auto sfpu_operand_access(int index) {
    static_assert(REG == SfpuReg::Dest || REG == SfpuReg::SrcS, "Unsupported SFPU register space");
    if constexpr (REG == SfpuReg::Dest) {
        return sfpi::dst_reg[index];
    } else {
        // SFPI's SrcS builtins select the register file (SFP_SRCSREG_BASE is 0); never add
        // SFPU_SRCS_BASE_ADDR here.
        return srcs_reg[index];
    }
}

// MATH::single_instruction: apply() is one SFPU instruction. False when the math policy does not declare it.
template <typename MATH, typename = void>
inline constexpr bool is_single_instruction_math = false;
template <typename MATH>
inline constexpr bool is_single_instruction_math<MATH, std::void_t<decltype(MATH::single_instruction)>> =
    MATH::single_instruction;

}  // namespace detail

/**
 * @brief Load/store policy between a register LAYOUT and a VALUE vector type.
 *
 * @tparam STORE_ADDR_MODE: NOINC (default) or a Dest ADDR_MOD_*; SrcS stores must use NOINC.
 */
template <sfpi::DataLayout LAYOUT, typename VALUE, int STORE_ADDR_MODE = sfpi::SFPSTORE_ADDR_MODE_NOINC>
struct SfpiFormat {
    using value_type = VALUE;
    static constexpr sfpi::DataLayout layout = LAYOUT;
    // store() is one SFPSTORE that leaves the register address alone (see calculate_binary_operands_two_rows).
    static constexpr bool plain_store = STORE_ADDR_MODE == sfpi::SFPSTORE_ADDR_MODE_NOINC;

    template <SfpuReg REG>
    sfpi_inline static value_type load(int index) {
        return detail::sfpu_operand_access<REG>(index).template mode<LAYOUT>();
    }

    template <SfpuReg REG>
    sfpi_inline static void store(int index, value_type value) {
        static_assert(
            REG == SfpuReg::Dest || STORE_ADDR_MODE == sfpi::SFPSTORE_ADDR_MODE_NOINC,
            "SrcS requires no-increment addressing");
        detail::sfpu_operand_access<REG>(index).template mode<LAYOUT>(STORE_ADDR_MODE) = value;
    }
};

/**
 * @brief One SFPU operand: register file, format and base offset.
 *
 * Indices are SFPI steps (SFP_ROWS rows each), not tile indices.
 */
template <SfpuReg REG, typename FORMAT>
class SfpuOperand {
    static_assert(REG == SfpuReg::Dest || REG == SfpuReg::SrcS, "Unsupported SFPU register space");

public:
    using value_type = typename FORMAT::value_type;

    sfpi_inline constexpr explicit SfpuOperand(int base_offset = 0) : base_offset_(base_offset) {}

    sfpi_inline value_type load(int index) const { return FORMAT::template load<REG>(base_offset_ + index); }

    sfpi_inline void store(int index, value_type value) const {
        FORMAT::template store<REG>(base_offset_ + index, value);
    }

private:
    int base_offset_;
};

/**
 * @brief output[d] = MATH::apply(input[d]) for d < ITERATIONS, on Dest or SrcS operands.
 *
 * @tparam MATH: Math policy with a static apply().
 */
template <typename MATH, int ITERATIONS, typename Input, typename Output>
sfpi_inline void calculate_unary_operands(const Input& input, const Output& output) {
    static_assert(ITERATIONS > 0, "A unary SFPU op requires at least one SFPI access");
    static_assert(
        std::is_same_v<decltype(MATH::apply(std::declval<typename Input::value_type>())), typename Output::value_type>,
        "MATH::apply must map the input operand's value type to the output operand's value type");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        output.store(d, MATH::apply(input.load(d)));
    }
}

/**
 * @brief output[d] = MATH::apply(input0[d], input1[d]) for d < ITERATIONS, on Dest or SrcS operands.
 *
 * @tparam MATH: Math policy with a static apply().
 */
template <typename MATH, int ITERATIONS, typename Input0, typename Input1, typename Output>
sfpi_inline void calculate_binary_operands(const Input0& input0, const Input1& input1, const Output& output) {
    static_assert(ITERATIONS > 0, "A binary SFPU op requires at least one SFPI access");
    static_assert(
        std::is_same_v<
            decltype(MATH::apply(
                std::declval<typename Input0::value_type>(), std::declval<typename Input1::value_type>())),
            typename Output::value_type>,
        "MATH::apply must map the input operands' value types to the output operand's value type");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        output.store(d, MATH::apply(input0.load(d), input1.load(d)));
    }
}

/**
 * @brief Two rows together: a_output[0] = MATH::apply(a_input0[0], a_input1[0]) and the same for the b operands,
 *        issued as load x4, op x2, store x2.
 *
 * One row at a time is load, load, op, store, and the SFPU stalls on the op -> store dependency of every row.
 * Issuing two rows together lets the second row's op fill that stall. Same instructions on the same operands as
 * two @ref calculate_binary_operands calls, only the issue order differs, so results are bit-identical.
 *
 * @tparam MATH: Math policy whose apply() is one SFPU instruction (MATH::single_instruction).
 * @note The output format must be a plain store (FORMAT::plain_store): no software rounding, no address increment.
 */
template <typename MATH, typename Input0, typename Input1, typename Output>
sfpi_inline void calculate_binary_operands_two_rows(
    const Input0& a_input0,
    const Input1& a_input1,
    const Output& a_output,
    const Input0& b_input0,
    const Input1& b_input1,
    const Output& b_output) {
    static_assert(detail::is_single_instruction_math<MATH>, "two rows together need a single-instruction MATH");
    static_assert(
        std::is_same_v<
            decltype(MATH::apply(
                std::declval<typename Input0::value_type>(), std::declval<typename Input1::value_type>())),
            typename Output::value_type>,
        "MATH::apply must map the input operands' value types to the output operand's value type");
    const typename Input0::value_type a_in0 = a_input0.load(0);
    const typename Input1::value_type a_in1 = a_input1.load(0);
    const typename Input0::value_type b_in0 = b_input0.load(0);
    const typename Input1::value_type b_in1 = b_input1.load(0);
    const typename Output::value_type a_res = MATH::apply(a_in0, a_in1);
    const typename Output::value_type b_res = MATH::apply(b_in0, b_in1);
    a_output.store(0, a_res);
    b_output.store(0, b_res);
}

}  // namespace ckernel::sfpu
