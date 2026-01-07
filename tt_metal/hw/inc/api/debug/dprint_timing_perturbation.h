// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file contains common timing perturbation utilities for debugging

#if defined(COMPILE_FOR_TRISC) && defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)

namespace tt::compute::common {
constexpr int nop_riscv = 1;
constexpr int nop_accelerator = 0;

template <const int num_nops, const int nop_target>
inline void add_nops() {
    for (int i = 0; i < num_nops; i++) {
        if constexpr (nop_target) {
            asm volatile("nop");
        } else {
            TTI_NOP;
        }
    }
}

template <const int num_unpack_nops, const int num_math_nops, const int num_pack_nops>
inline void add_trisc_nops() {
    DPRINT << "RISCV Unpack NOPs: " << (uint32_t)num_unpack_nops << " Math NOPs: " << (uint32_t)num_math_nops
           << " Pack NOPs: " << (uint32_t)num_pack_nops << ENDL();
    if constexpr (num_unpack_nops) {
        UNPACK((add_nops<num_unpack_nops, nop_riscv>()));
    }

    if constexpr (num_math_nops) {
        MATH((add_nops<num_math_nops, nop_riscv>()));
    }

    if constexpr (num_pack_nops) {
        PACK((add_nops<num_pack_nops, nop_riscv>()));
    }
}

template <const int num_unpack_nops, const int num_math_nops, const int num_pack_nops>
inline void add_accel_nops() {
    DPRINT << "ACCEL Unpack NOPs: " << (uint32_t)num_unpack_nops << " Math NOPs: " << (uint32_t)num_math_nops
           << " Pack NOPs: " << (uint32_t)num_pack_nops << ENDL();
    if constexpr (num_unpack_nops) {
        UNPACK((add_nops<num_unpack_nops, nop_accelerator>()));
    }

    if constexpr (num_math_nops) {
        MATH((add_nops<num_math_nops, nop_accelerator>()));
    }

    if constexpr (num_pack_nops) {
        PACK((add_nops<num_pack_nops, nop_accelerator>()));
    }
}
}  // namespace tt::compute::common

#endif  // defined(COMPILE_FOR_TRISC) && defined(DEBUG_PRINT_ENABLED) && !defined(FORCE_DPRINT_OFF)
