// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Copyright 2018--2020 IBM
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Based on code by Schuyler Eldridge. Copyright (c) Boston University
// https://github.com/seldridge/rocket-rocc-examples/blob/master/src/main/c/rocc.h

#pragma once

#define RVTEST_XS_ENABLE                  \
    li a0, MSTATUS_XS &(MSTATUS_XS >> 1); \
    csrs mstatus, a0;

#define RVTEST_WITH_ROCC \
    .macro init;         \
    RVTEST_XS_ENABLE.endm

#define STR1(x) #x
#ifndef STR
#define STR(x) STR1(x)
#endif

#define CAT_(A, B) A##B
#define CAT(A, B) CAT_(A, B)

/** Assembly macro for creating "raw" Rocket Custom Coproessor (RoCC)
 * assembly language instructions that will return data in rd. These
 * are to be used only in assembly language programs (not C/C++).
 *
 * Example:
 *
 * Consider the following macro consisting of a CUSTOM_0 instruction
 * with func7 "42" that is doing some operation of "a0 = op(a1, a2)":
 *
 *     ROCC_INSTRUCTION_RAW_R_R_R(0, a0, a1, a2, 42)
 *
 * This will produce the following pseudo assembly language
 * instruction:
 *
 *     .insn r CUSTOM_0, 7, 42, a0, a1, a2
 *
 * @param x the custom instruction number: 0, 1, 2, or 3
 * @param rd the destination register, e.g., a0 or x10
 * @param rs1 the first source register, e.g., a0 or x10
 * @param rs2 the second source register, e.g., a0 or x10
 * @param func7 the value of the func7 field
 * @return a raw .insn RoCC instruction
 */
#define ROCC_INSTRUCTION_RAW_R_R_R(x, rd, rs1, rs2, func7) .insn r CAT(CUSTOM_, x), 7, func7, rd, rs1, rs2

#define ROCC_INSTRUCTION_RAW_R_R_0(x, rd, rs1, func7) .insn r CAT(CUSTOM_, x), 6, func7, rd, rs1, x0

#define ROCC_INSTRUCTION_RAW_R_0_0(x, rd, func7) .insn r CAT(CUSTOM_, x), 4, func7, rd, x0, x0

#define ROCC_INSTRUCTION_RAW_0_R_R(x, rs1, rs2, func7) .insn r CAT(CUSTOM_, x), 3, func7, x0, rs1, rs2

#define ROCC_INSTRUCTION_RAW_0_R_0(x, rs1, func7) .insn r CAT(CUSTOM_, x), 2, func7, x0, rs1, x0

#define ROCC_INSTRUCTION_RAW_0_0_0(x, func7) .insn r CAT(CUSTOM_, x), 0, func7, x0, x0, x0

// C

#define EXTRACT(a, size, offset) (((~(~0 << size) << offset) & a) >> offset)

#define CUSTOMX_OPCODE(x) CUSTOM_##x
#define CUSTOM_0 0b0001011
#define CUSTOM_1 0b0101011
#define CUSTOM_2 0b1011011
#define CUSTOM_3 0b1111011

#define CUSTOMX(X, xd, xs1, xs2, rd, rs1, rs2, funct)                                                 \
    CUSTOMX_OPCODE(X) | (rd << (7)) | (xs2 << (7 + 5)) | (xs1 << (7 + 5 + 1)) | (xd << (7 + 5 + 2)) | \
        (rs1 << (7 + 5 + 3)) | (rs2 << (7 + 5 + 3 + 5)) | (EXTRACT(funct, 7, 0) << (7 + 5 + 3 + 5 + 5))

// Standard macro that passes rd, rs1, and rs2 via registers
#define ROCC_INSTRUCTION_DSS(X, rd, rs1, rs2, funct) ROCC_INSTRUCTION_R_R_R(X, rd, rs1, rs2, funct, 10, 11, 12)

#define ROCC_INSTRUCTION_DS(X, rd, rs1, funct) ROCC_INSTRUCTION_R_R_I(X, rd, rs1, 0, funct, 10, 11)

#define ROCC_INSTRUCTION_D(X, rd, funct) ROCC_INSTRUCTION_R_I_I(X, rd, 0, 0, funct, 10)

#define ROCC_INSTRUCTION_SS(X, rs1, rs2, funct) ROCC_INSTRUCTION_I_R_R(X, 0, rs1, rs2, funct, 11, 12)

#define ROCC_INSTRUCTION_S(X, rs1, funct) ROCC_INSTRUCTION_I_R_I(X, 0, rs1, 0, funct, 11)

#define ROCC_INSTRUCTION(X, funct) ROCC_INSTRUCTION_I_I_I(X, 0, 0, 0, funct)

// rd, rs1, and rs2 are data
// rd_n, rs_1, and rs2_n are the register numbers to use
#define ROCC_INSTRUCTION_R_R_R(X, rd, rs1, rs2, funct, rd_n, rs1_n, rs2_n)               \
    {                                                                                    \
        register uint64_t rd_ asm("x" #rd_n);                                            \
        register uint64_t rs1_ asm("x" #rs1_n) = (uint64_t)rs1;                          \
        register uint64_t rs2_ asm("x" #rs2_n) = (uint64_t)rs2;                          \
        asm volatile(".word " STR(CUSTOMX(X, 1, 1, 1, rd_n, rs1_n, rs2_n, funct)) "\n\t" \
                     : "=r"(rd_)                                                         \
                     : [_rs1] "r"(rs1_), [_rs2] "r"(rs2_));                              \
        rd = rd_;                                                                        \
    }

#define ROCC_INSTRUCTION_R_R_I(X, rd, rs1, rs2, funct, rd_n, rs1_n)                    \
    {                                                                                  \
        register uint64_t rd_ asm("x" #rd_n);                                          \
        register uint64_t rs1_ asm("x" #rs1_n) = (uint64_t)rs1;                        \
        asm volatile(".word " STR(CUSTOMX(X, 1, 1, 0, rd_n, rs1_n, rs2, funct)) "\n\t" \
                     : "=r"(rd_)                                                       \
                     : [_rs1] "r"(rs1_));                                              \
        rd = rd_;                                                                      \
    }

#define ROCC_INSTRUCTION_R_I_I(X, rd, rs1, rs2, funct, rd_n)                                       \
    {                                                                                              \
        register uint64_t rd_ asm("x" #rd_n);                                                      \
        asm volatile(".word " STR(CUSTOMX(X, 1, 0, 0, rd_n, rs1, rs2, funct)) "\n\t" : "=r"(rd_)); \
        rd = rd_;                                                                                  \
    }

#define ROCC_INSTRUCTION_I_R_R(X, rd, rs1, rs2, funct, rs1_n, rs2_n)                                                 \
    {                                                                                                                \
        register uint64_t rs1_ asm("x" #rs1_n) = (uint64_t)rs1;                                                      \
        register uint64_t rs2_ asm("x" #rs2_n) = (uint64_t)rs2;                                                      \
        asm volatile(                                                                                                \
            ".word " STR(CUSTOMX(X, 0, 1, 1, rd, rs1_n, rs2_n, funct)) "\n\t" ::[_rs1] "r"(rs1_), [_rs2] "r"(rs2_)); \
    }

#define ROCC_INSTRUCTION_I_R_I(X, rd, rs1, rs2, funct, rs1_n)                                             \
    {                                                                                                     \
        register uint64_t rs1_ asm("x" #rs1_n) = (uint64_t)rs1;                                           \
        asm volatile(".word " STR(CUSTOMX(X, 0, 1, 0, rd, rs1_n, rs2, funct)) "\n\t" ::[_rs1] "r"(rs1_)); \
    }

#define ROCC_INSTRUCTION_I_I_I(X, rd, rs1, rs2, funct)                               \
    {                                                                                \
        asm volatile(".word " STR(CUSTOMX(X, 0, 0, 0, rd, rs1, rs2, funct)) "\n\t"); \
    }
