// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file contains common timing perturbation utilities for debugging

#if defined(COMPILE_FOR_TRISC) && defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)

namespace tt::compute::common {

/**
 * @brief Helper to inline variable number of nops/TTI_NOPs
 *
 * @tparam num_nops: number of nops to insert
 * @tparam is_riscv_nop: 0 for Tensix nop, 1 for RISC-V nop
 * @tparam use_loop: 0 for inline assembly, 1 for loop-based nops
 */
template <const int num_nops, const int is_riscv_nop, const int use_loop = 0>
inline void add_nops() {
    if constexpr (use_loop) {
        for (int i = 0; i < num_nops; i++) {
            if constexpr (is_riscv_nop) {
                asm volatile("nop");
            } else {
                TTI_NOP;
            }
        }
    } else {
        if constexpr (is_riscv_nop) {
            asm volatile(
                ".rept %0\n\t"
                "nop\n\t"
                ".endr\n\t"
                :
                : "i"(num_nops));
        } else {
            asm volatile(
                ".rept %0\n\t"
                ".ttinsn %1\n\t"
                ".endr\n\t"
                :
                : "i"(num_nops), "i"(TT_OP_NOP));
        }
    }
}

/**
 * @brief Uses template parameters to insert variable compute nops for timing perturbation
 *
 * @tparam num_unpack_nops: number of unpack nops to insert
 * @tparam num_math_nops: number of math nops to insert
 * @tparam num_pack_nops: number of pack nops to insert
 * @tparam is_riscv_nop: 0 for Tensix nop, 1 for RISC-V nop
 * @tparam use_loop: 0 for inline assembly, 1 for loop-based nops
 */
template <
    const int num_unpack_nops,
    const int num_math_nops,
    const int num_pack_nops,
    const int is_riscv_nop,
    const int use_loop = 0>
inline void add_compute_nops() {
    DPRINT << "is_riscv_nop: " << (uint32_t)is_riscv_nop << " Unpack NOPs: " << (uint32_t)num_unpack_nops
           << " Math NOPs: " << (uint32_t)num_math_nops << " Pack NOPs: " << (uint32_t)num_pack_nops << ENDL();
    if constexpr (num_unpack_nops) {
        UNPACK((add_nops<num_unpack_nops, is_riscv_nop, use_loop>()));
    }

    if constexpr (num_math_nops) {
        MATH((add_nops<num_math_nops, is_riscv_nop, use_loop>()));
    }

    if constexpr (num_pack_nops) {
        PACK((add_nops<num_pack_nops, is_riscv_nop, use_loop>()));
    }
}
}  // namespace tt::compute::common

#endif  // defined(COMPILE_FOR_TRISC) && defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)
