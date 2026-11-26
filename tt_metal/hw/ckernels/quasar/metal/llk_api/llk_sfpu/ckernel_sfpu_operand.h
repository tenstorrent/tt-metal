// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include "llk_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_ops.h"
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

template <typename FUNCTION>
struct function_arity;
template <typename RESULT, typename... ARGS>
struct function_arity<RESULT (*)(ARGS...)> : std::integral_constant<int, sizeof...(ARGS)> {};

// MATH::setup(): the values a math policy keeps for a whole row loop. False when the policy has none.
template <typename MATH, typename = void>
inline constexpr bool has_setup = false;
template <typename MATH>
inline constexpr bool has_setup<MATH, std::void_t<decltype(&MATH::setup)>> = true;

struct NoSetup {};

// MATH::apply on one row; a policy with setup() takes its value as the first argument.
template <typename MATH, typename Setup, typename... Values>
sfpi_inline auto math_apply([[maybe_unused]] const Setup& setup, Values... values) {
    if constexpr (has_setup<MATH>) {
        return MATH::apply(setup, values...);
    } else {
        return MATH::apply(values...);
    }
}

}  // namespace detail

/** @brief SFPI vector type a register @p LAYOUT loads into and stores from. */
template <sfpi::DataLayout LAYOUT>
using layout_value_t = std::conditional_t<LAYOUT == sfpi::DataLayout::I32, sfpi::vInt, sfpi::vFloat>;

/**
 * @brief Encoding of I32 tiles. SignMagnitude: SrcS, and Dest tiles an FPU op computed. TwosComplement: native
 *        Int32 tiles copied or unpacked to Dest.
 */
enum class IntEncoding : std::uint8_t {
    SignMagnitude,
    TwosComplement,
};

/**
 * @brief SFPLOAD/SFPSTORE layout of a register holding tiles of a format; any other format is a compile error.
 *
 * @tparam FMT: Format the register holds, values = <Float16/Float16_b/Float32/Int32>
 */
template <DataFormat FMT>
constexpr sfpi::DataLayout sfpu_layout() {
    if constexpr (FMT == DataFormat::Int32) {
        return sfpi::DataLayout::I32;
    } else if constexpr (FMT == DataFormat::Float32) {
        return sfpi::DataLayout::F32;
    } else if constexpr (FMT == DataFormat::Float16) {
        return sfpi::DataLayout::F16a;
    } else {
        static_assert(FMT == DataFormat::Float16_b, "SFPU layouts exist for Float16, Float16_b, Float32 and Int32");
        return sfpi::DataLayout::F16b;
    }
}

/**
 * @brief The math policy of an op tag on a register layout, as math_for<OP, LAYOUT>::type. Specialized once per
 *        op, next to its math (ckernel_sfpu_<op>.h), for Dest and SrcS alike.
 *
 * @tparam OP: Op tag (ckernel::op).
 * @tparam LAYOUT: SFPLOAD/SFPSTORE format of the tiles.
 * @tparam INT_ENCODING: How I32 tiles encode negative values, see @ref IntEncoding; float layouts ignore it.
 */
template <typename OP, sfpi::DataLayout LAYOUT, IntEncoding INT_ENCODING = IntEncoding::SignMagnitude>
struct math_for;
template <typename OP, sfpi::DataLayout LAYOUT, IntEncoding INT_ENCODING = IntEncoding::SignMagnitude>
using math_for_t = typename math_for<OP, LAYOUT, INT_ENCODING>::type;

/** @brief True when the math of @p OP on @p LAYOUT computes on the vector type LAYOUT loads into. */
template <typename OP, sfpi::DataLayout LAYOUT>
inline constexpr bool op_takes_layout =
    std::is_same_v<typename math_for_t<OP, LAYOUT>::value_type, layout_value_t<LAYOUT>>;

/** @brief Number of inputs of a math policy: the arguments MATH::apply takes, without the setup value. */
template <typename MATH>
inline constexpr int math_arity =
    detail::function_arity<decltype(&MATH::apply)>::value - (detail::has_setup<MATH> ? 1 : 0);

/**
 * @brief The value a math policy keeps for a row loop: MATH::setup() when the policy has one, else an empty value.
 *
 * @note Take it once before the rows; apply() gets it as its first argument.
 */
template <typename MATH>
sfpi_inline auto math_setup() {
    if constexpr (detail::has_setup<MATH>) {
        return MATH::setup();
    } else {
        return detail::NoSetup{};
    }
}

/**
 * @brief Load/store policy between a register LAYOUT and a VALUE vector type.
 *
 * @tparam STORE_ADDR_MODE: NOINC (default) or a Dest ADDR_MOD_*; SrcS stores must use NOINC.
 */
template <sfpi::DataLayout LAYOUT, typename VALUE, int STORE_ADDR_MODE = sfpi::SFPSTORE_ADDR_MODE_NOINC>
struct SfpiFormat {
    using value_type = VALUE;
    static constexpr sfpi::DataLayout layout = LAYOUT;

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
    static constexpr SfpuReg reg = REG;

    sfpi_inline constexpr explicit SfpuOperand(int base_offset = 0) : base_offset_(base_offset) {}

    sfpi_inline value_type load(int index) const { return FORMAT::template load<REG>(base_offset_ + index); }

    sfpi_inline void store(int index, value_type value) const {
        FORMAT::template store<REG>(base_offset_ + index, value);
    }

private:
    int base_offset_;
};

/**
 * @brief output[d] = MATH::apply(inputs[d]...) for d < ITERATIONS, on Dest or SrcS operands, for any number of
 *        inputs.
 *
 * @tparam MATH: Math policy with a static apply() taking one argument per input.
 * @tparam ITERATIONS: Number of SFPI steps to walk.
 * @param setup: The value of @ref math_setup for MATH, taken by the caller before its rows.
 * @param output: Operand the results are stored to.
 * @param inputs: One operand per MATH::apply argument, in argument order.
 */
template <typename MATH, int ITERATIONS, typename Setup, typename Output, typename... Inputs>
sfpi_inline void calculate_operands(const Setup& setup, const Output& output, const Inputs&... inputs) {
    static_assert(ITERATIONS > 0, "An SFPU op requires at least one SFPI access");
    static_assert(std::is_same_v<Setup, decltype(math_setup<MATH>())>, "setup must be the value of math_setup<MATH>()");
    static_assert(sizeof...(Inputs) == math_arity<MATH>, "MATH::apply takes one argument per input operand");
    static_assert(
        ((Inputs::reg == Output::reg) && ...),
        "An SFPU op reads and writes one register file; no pipeline delivers a mixed set");
    static_assert(
        std::is_same_v<
            decltype(detail::math_apply<MATH>(setup, std::declval<typename Inputs::value_type>()...)),
            typename Output::value_type>,
        "MATH::apply must map the input operands' value types to the output operand's value type");
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        output.store(d, detail::math_apply<MATH>(setup, inputs.load(d)...));
    }
}

/**
 * @brief Two rows per call, issued as load x4, op x2, store x2, so the second row's op fills the op -> store stall.
 *        Bit-identical to two @ref calculate_operands calls; MATH must be one instruction, the store plain.
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
        Input0::reg == Output::reg && Input1::reg == Output::reg,
        "An SFPU op reads and writes one register file; no pipeline delivers a mixed set");
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
