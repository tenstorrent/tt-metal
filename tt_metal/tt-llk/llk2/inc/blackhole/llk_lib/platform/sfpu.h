// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_lib/platform/issue.h"

// HAL gap: the Blackhole HAL has no vector unit (SFPU) file. These are the SFPU instructions llk-lib issues as
// encoded words (the fast exp's SFPLOADMACRO programming and the lane control reset), typed the way the HAL types
// its own; the rest of the SFPU's instruction set is reached through SFPI and the TTI_SFP* macros in the vec bodies.
namespace llk2::platform::sfpu
{

/**
 * @brief Name one vector register (LREG).
 *
 * L0-L7 are general purpose. L11-L14 are the programmable constants, written through the
 * SFPU configuration instruction and read-only to every other instruction.
 */
enum class Lreg : std::uint8_t
{
    L0  = 0,
    L1  = 1,
    L2  = 2,
    L3  = 3,
    L4  = 4,
    L5  = 5,
    L6  = 6,
    L7  = 7,
    L11 = 11,
    L12 = 12,
    L13 = 13,
    L14 = 14
};

/**
 * @brief Describe one lanewise shift by a per-lane amount (SFPSHFT2, its shift-by-register form).
 *
 * Each lane of destination becomes the same lane of value shifted left by the same lane of
 * amount. Only positive amounts are in use (the fast exp shifts by 15); what a negative
 * amount does is not stated here.
 */
struct ShiftByRegister
{
    Lreg value;
    Lreg amount;
    Lreg destination;
};

/**
 * @brief Encode a lanewise shift by a per-lane amount without issuing it.
 *
 * @tparam Operation: Registers the shift reads and writes.
 */
template <ShiftByRegister Operation>
constexpr std::uint32_t shift_by_register_operation()
{
    return static_cast<std::uint32_t>(TT_OP_SFPSHFT2(
        static_cast<std::uint32_t>(Operation.value),
        static_cast<std::uint32_t>(Operation.amount),
        static_cast<std::uint32_t>(Operation.destination),
        5 /*instr_mod1: shift by the amount in lreg_src_c*/));
}

/**
 * @brief Shift every lane of a vector register by the amount in the same lane of another (SFPSHFT2).
 *
 * @tparam Operation: Registers the shift reads and writes.
 */
template <ShiftByRegister Operation>
inline __attribute__((always_inline)) void shift_by_register()
{
    ::llk2::platform::issue<(shift_by_register_operation<Operation>())>();
}

/**
 * @brief Select the half of a vector register a 16-bit immediate load writes.
 *
 * The other half of every lane keeps its value, so a Lower load followed by an Upper load
 * builds a 32-bit pattern.
 */
enum class ImmediateHalf : std::uint8_t
{
    Upper = 0x8,
    Lower = 0xA
};

/**
 * @brief Describe one 16-bit immediate load into every lane of a vector register (SFPLOADI).
 */
struct LoadImmediate
{
    Lreg destination;
    ImmediateHalf half;
    std::uint16_t value;
};

/**
 * @brief Encode a 16-bit immediate load without issuing it.
 *
 * @tparam Operation: Register, half and value of the load.
 */
template <LoadImmediate Operation>
constexpr std::uint32_t load_immediate_operation()
{
    return static_cast<std::uint32_t>(TT_OP_SFPLOADI(
        static_cast<std::uint32_t>(Operation.destination), static_cast<std::uint32_t>(Operation.half), static_cast<std::uint32_t>(Operation.value)));
}

/**
 * @brief Load a 16-bit immediate into one half of every lane of a vector register (SFPLOADI).
 *
 * @tparam Operation: Register, half and value of the load.
 */
template <LoadImmediate Operation>
inline __attribute__((always_inline)) void load_immediate()
{
    ::llk2::platform::issue<(load_immediate_operation<Operation>())>();
}

/**
 * @brief Load a 32-bit pattern into every lane of a vector register: its lower half, then its upper half.
 *
 * @tparam Destination: Register to load.
 * @tparam Bits: Pattern each lane receives.
 */
template <Lreg Destination, std::uint32_t Bits>
inline __attribute__((always_inline)) void load_immediate32()
{
    load_immediate<LoadImmediate {.destination = Destination, .half = ImmediateHalf::Lower, .value = static_cast<std::uint16_t>(Bits & 0xFFFFu)}>();
    load_immediate<LoadImmediate {.destination = Destination, .half = ImmediateHalf::Upper, .value = static_cast<std::uint16_t>(Bits >> 16)}>();
}

/**
 * @brief Name the vector unit state the configuration instruction (SFPCONFIG) writes.
 *
 * MacroInstruction0-3 are the four programmable macro instruction registers (macro
 * instructions 4-7; 0-3 are fixed). MacroSequence0-3 each hold one LOADMACRO schedule.
 * MacroControl holds the per-lane LOADMACRO control bits. Constant11-14 are the programmable
 * constant registers L11-L14. LaneControl holds the per-lane control bits of the unit.
 */
enum class ConfigTarget : std::uint8_t
{
    MacroInstruction0 = 0,
    MacroInstruction1 = 1,
    MacroInstruction2 = 2,
    MacroInstruction3 = 3,
    MacroSequence0    = 4,
    MacroSequence1    = 5,
    MacroSequence2    = 6,
    MacroSequence3    = 7,
    MacroControl      = 8,
    Constant11        = 11,
    Constant12        = 12,
    Constant13        = 13,
    Constant14        = 14,
    LaneControl       = 15
};

/**
 * @brief Select where the configuration instruction takes the value it writes.
 */
enum class ConfigSource : std::uint8_t
{
    L0        = 0x0,
    Immediate = 0x1
};

/**
 * @brief Describe one write of vector unit state (SFPCONFIG).
 *
 * With ConfigSource::L0 each lane's value comes from the first eight lanes of vector register
 * L0 (lane modulo 8) and immediate is unused; with ConfigSource::Immediate it is immediate.
 */
struct Configure
{
    ConfigTarget target;
    ConfigSource source     = ConfigSource::L0;
    std::uint16_t immediate = 0;
};

/**
 * @brief Encode a write of vector unit state without issuing it.
 *
 * @tparam Operation: Target, source and immediate of the write.
 */
template <Configure Operation>
constexpr std::uint32_t configure_operation()
{
    return static_cast<std::uint32_t>(TT_OP_SFPCONFIG(
        static_cast<std::uint32_t>(Operation.immediate), static_cast<std::uint32_t>(Operation.target), static_cast<std::uint32_t>(Operation.source)));
}

/**
 * @brief Write vector unit state: a programmable constant, a macro register or the lane control (SFPCONFIG).
 *
 * @tparam Operation: Target, source and immediate of the write.
 */
template <Configure Operation>
inline __attribute__((always_inline)) void configure()
{
    ::llk2::platform::issue<(configure_operation<Operation>())>();
}

/**
 * @brief Describe one lanewise multiply-add: destination = multiplicand * multiplier + addend (SFPMAD).
 */
struct MultiplyAdd
{
    Lreg multiplicand;
    Lreg multiplier;
    Lreg addend;
    Lreg destination;
};

/**
 * @brief Encode a lanewise multiply-add without issuing it.
 *
 * @tparam Operation: Registers the multiply-add reads and writes.
 */
template <MultiplyAdd Operation>
constexpr std::uint32_t multiply_add_operation()
{
    return static_cast<std::uint32_t>(TT_OP_SFPMAD(
        static_cast<std::uint32_t>(Operation.multiplicand),
        static_cast<std::uint32_t>(Operation.multiplier),
        static_cast<std::uint32_t>(Operation.addend),
        static_cast<std::uint32_t>(Operation.destination),
        0 /*instr_mod1: registers as named*/));
}

/**
 * @brief Select the integer a @ref RoundToInteger operation produces from an FP32 lane.
 */
enum class RoundTarget : std::uint8_t
{
    SignMagnitude16 = 0x7
};

/**
 * @brief Describe one lanewise FP32 to integer conversion, round to nearest, no descale (SFP_STOCH_RND).
 *
 * SignMagnitude16 keeps the sign in bit 31 and saturates the magnitude at 32767.
 */
struct RoundToInteger
{
    Lreg value;
    Lreg destination;
    RoundTarget target;
};

/**
 * @brief Encode a lanewise FP32 to integer conversion without issuing it.
 *
 * @tparam Operation: Registers and integer form of the conversion.
 */
template <RoundToInteger Operation>
constexpr std::uint32_t round_to_integer_operation()
{
    return static_cast<std::uint32_t>(TT_OP_SFP_STOCH_RND(
        0 /*rnd_mode: nearest*/,
        0 /*imm8_math: no descale*/,
        0 /*lreg_src_b: unused*/,
        static_cast<std::uint32_t>(Operation.value),
        static_cast<std::uint32_t>(Operation.destination),
        static_cast<std::uint32_t>(Operation.target)));
}

/**
 * @brief Describe one lanewise sign replacement (SFPSETSGN, its sign-from-register form).
 *
 * Each lane of sign_and_destination keeps its own sign and takes the exponent and mantissa
 * of the same lane of magnitude.
 */
struct SetSign
{
    Lreg magnitude;
    Lreg sign_and_destination;
};

/**
 * @brief Encode a lanewise sign replacement without issuing it.
 *
 * @tparam Operation: Registers the sign replacement reads and writes.
 */
template <SetSign Operation>
constexpr std::uint32_t set_sign_operation()
{
    return static_cast<std::uint32_t>(TT_OP_SFPSETSGN(
        0 /*imm12_math: unused*/,
        static_cast<std::uint32_t>(Operation.magnitude),
        static_cast<std::uint32_t>(Operation.sign_and_destination),
        0 /*instr_mod1: sign from the destination register*/));
}

/**
 * @brief Name one programmable macro instruction register by the macro instruction it provides.
 *
 * A LOADMACRO schedule selects macro instructions 0-7; 0-3 are fixed (3 is the store) and
 * 4-7 are these registers.
 */
enum class MacroInstruction : std::uint8_t
{
    M4 = 12,
    M5 = 13,
    M6 = 14,
    M7 = 15
};

/**
 * @brief Install an encoded vector operation in a programmable macro instruction register.
 *
 * The operation is issued with its destination register field replaced by the register's
 * install code; the vector unit then stores the instruction instead of executing it. When a
 * LOADMACRO schedule runs it, the schedule supplies the destination (the loaded register or
 * the staging register) and optionally the loaded register as its VB operand; the operation's
 * other fields apply as installed.
 *
 * @tparam Register: Macro instruction register to program.
 * @tparam Operation: Encoded vector operation whose destination register field is bits 7:4,
 *         from one of this file's *_operation encoders.
 */
template <MacroInstruction Register, std::uint32_t Operation>
inline __attribute__((always_inline)) void install_macro_instruction()
{
    constexpr std::uint32_t destination_field = 0xFu << 4;
    ::llk2::platform::issue<((Operation & ~destination_field) | (static_cast<std::uint32_t>(Register) << 4))>();
}

/**
 * @brief Name one LOADMACRO schedule register.
 */
enum class MacroSequence : std::uint8_t
{
    S0 = 0,
    S1 = 1,
    S2 = 2,
    S3 = 3
};

/**
 * @brief Describe one LOADMACRO: a load from Dest that also starts a schedule of macro instructions.
 *
 * Loads 32 datums from Dest (the row counter plus dest_row_offset, in Dest's format) into
 * destination, which must be L0-L3, and runs the schedule in sequence on it: up to one macro
 * instruction on each of the simple, multiply-add, round and store units, each after its
 * programmed delay. address_modifier steps the Dest row counter after the load.
 */
struct LoadMacro
{
    Lreg destination;
    MacroSequence sequence;
    std::uint8_t address_modifier = 0;
    std::uint16_t dest_row_offset = 0;
};

/**
 * @brief Encode a LOADMACRO without issuing it.
 *
 * @tparam Operation: Register, schedule and Dest addressing of the load.
 */
template <LoadMacro Operation>
constexpr std::uint32_t load_macro_operation()
{
    return static_cast<std::uint32_t>(TT_OP_SFPLOADMACRO(
        (static_cast<std::uint32_t>(Operation.sequence) << 2) | static_cast<std::uint32_t>(Operation.destination),
        0 /*instr_mod0: Dest's format*/,
        static_cast<std::uint32_t>(Operation.address_modifier),
        static_cast<std::uint32_t>(Operation.dest_row_offset)));
}

/**
 * @brief Load from Dest and start a schedule of macro instructions on the loaded register (SFPLOADMACRO).
 *
 * @tparam Operation: Register, schedule and Dest addressing of the load.
 * @note Program the schedule (@ref configure, ConfigTarget::MacroSequence0-3) and the macro
 *       instructions it selects (@ref install_macro_instruction) first.
 */
template <LoadMacro Operation>
inline __attribute__((always_inline)) void load_macro()
{
    static_assert(static_cast<std::uint32_t>(Operation.destination) < 4, "LOADMACRO loads L0-L3");
    ::llk2::platform::issue<(load_macro_operation<Operation>())>();
}

} // namespace llk2::platform::sfpu
