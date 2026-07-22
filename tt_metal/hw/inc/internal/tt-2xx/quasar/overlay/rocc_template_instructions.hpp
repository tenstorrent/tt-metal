// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
// Version: FFN1.3.0

/**
 * @file rocc_template_instructions.hpp
 * @brief Compile-time helpers for emitting Quasar RoCC instructions.
 *
 * RoCC opcodes and function fields are template arguments so every emitted
 * instruction is a compile-time constant. The operand helpers use the same
 * fixed RISC-V registers as the legacy macros in xcustom_test.hpp.
 */

#pragma once

#include <cstdint>

#include "xcustom_test.hpp"

namespace overlay::rocc {

constexpr std::uint32_t kCustom0Opcode = 0b0001011;
constexpr std::uint32_t kCustom1Opcode = 0b0101011;
constexpr std::uint32_t kDestinationRegister = 10;
constexpr std::uint32_t kSourceRegister1 = 11;
constexpr std::uint32_t kSourceRegister2 = 12;

/** Encode the operand flags, fixed registers, and function into one custom instruction. */
constexpr std::uint32_t encode_instruction(
    std::uint32_t opcode,
    bool has_destination,
    bool has_source_1,
    bool has_source_2,
    std::uint32_t destination_register,
    std::uint32_t source_register_1,
    std::uint32_t source_register_2,
    std::uint32_t function) {
    return opcode | (destination_register << 7) | (static_cast<std::uint32_t>(has_source_2) << 12) |
           (static_cast<std::uint32_t>(has_source_1) << 13) | (static_cast<std::uint32_t>(has_destination) << 14) |
           (source_register_1 << 15) | (source_register_2 << 20) | ((function & 0x7f) << 25);
}

static_assert(encode_instruction(kCustom0Opcode, false, false, false, 0, 0, 0, 0) == CUSTOM_0);
static_assert(encode_instruction(kCustom1Opcode, false, false, false, 0, 0, 0, 0) == CUSTOM_1);
static_assert(encode_instruction(kCustom0Opcode, false, false, false, 0, 0, 0, 128) == CUSTOM_0);
static_assert(encode_instruction(kCustom0Opcode, false, false, false, 0, 0, 0, 59) == 0x7600000bU);
static_assert(encode_instruction(kCustom0Opcode, false, true, false, 0, kSourceRegister1, 0, 104) == 0xd005a00bU);
static_assert(
    encode_instruction(kCustom0Opcode, false, true, true, 0, kSourceRegister1, kSourceRegister2, 55) == 0x6ec5b00bU);
static_assert(
    encode_instruction(kCustom0Opcode, true, true, false, kDestinationRegister, kSourceRegister1, 0, 121) ==
    0xf205e50bU);
static_assert(encode_instruction(kCustom1Opcode, false, true, false, 0, kSourceRegister1, 0, 127) == 0xfe05a02bU);

/** Emit a custom instruction with no operands. */
template <std::uint32_t Opcode, std::uint32_t Function>
inline __attribute__((always_inline)) void instruction() {
    static_assert(Function < 128, "RoCC function must fit in the 7-bit function field");
    constexpr std::uint32_t instruction = encode_instruction(Opcode, false, false, false, 0, 0, 0, Function);
    asm volatile(".word %[instruction]\n\t" : : [instruction] "i"(instruction));
}

/** Emit a custom instruction with one source operand. */
template <std::uint32_t Opcode, std::uint32_t Function>
inline __attribute__((always_inline)) void instruction_s(std::uint64_t source) {
    static_assert(Function < 128, "RoCC function must fit in the 7-bit function field");
    register std::uint64_t source_register asm("x11") = source;
    constexpr std::uint32_t instruction =
        encode_instruction(Opcode, false, true, false, 0, kSourceRegister1, 0, Function);
    asm volatile(".word %[instruction]\n\t" : : [instruction] "i"(instruction), [source] "r"(source_register));
}

/** Emit a custom instruction with two source operands. */
template <std::uint32_t Opcode, std::uint32_t Function>
inline __attribute__((always_inline)) void instruction_ss(std::uint64_t source_1, std::uint64_t source_2) {
    static_assert(Function < 128, "RoCC function must fit in the 7-bit function field");
    register std::uint64_t source_register_1 asm("x11") = source_1;
    register std::uint64_t source_register_2 asm("x12") = source_2;
    constexpr std::uint32_t instruction =
        encode_instruction(Opcode, false, true, true, 0, kSourceRegister1, kSourceRegister2, Function);
    asm volatile(
        ".word %[instruction]\n\t"
        :
        : [instruction] "i"(instruction), [source_1] "r"(source_register_1), [source_2] "r"(source_register_2));
}

/** Emit a custom instruction with one destination operand. */
template <std::uint32_t Opcode, std::uint32_t Function>
inline __attribute__((always_inline)) std::uint64_t instruction_d() {
    static_assert(Function < 128, "RoCC function must fit in the 7-bit function field");
    register std::uint64_t destination_register asm("x10");
    constexpr std::uint32_t instruction =
        encode_instruction(Opcode, true, false, false, kDestinationRegister, 0, 0, Function);
    asm volatile(".word %[instruction]\n\t" : "=r"(destination_register) : [instruction] "i"(instruction));
    return destination_register;
}

/** Emit a custom instruction with one destination and one source operand. */
template <std::uint32_t Opcode, std::uint32_t Function>
inline __attribute__((always_inline)) std::uint64_t instruction_ds(std::uint64_t source) {
    static_assert(Function < 128, "RoCC function must fit in the 7-bit function field");
    register std::uint64_t destination_register asm("x10");
    register std::uint64_t source_register asm("x11") = source;
    constexpr std::uint32_t instruction =
        encode_instruction(Opcode, true, true, false, kDestinationRegister, kSourceRegister1, 0, Function);
    asm volatile(".word %[instruction]\n\t"
                 : "=r"(destination_register)
                 : [instruction] "i"(instruction), [source] "r"(source_register));
    return destination_register;
}

}  // namespace overlay::rocc
