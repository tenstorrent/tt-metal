// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * rt/x280_vec_m1.h -- RVV setup at LMUL=1 (VLEN=512, SEW=32: 64 B per vector
 * instruction) and the load of the v1 store pattern. v1 is hand-assigned
 * because the default -march=rv64gc build does not let the compiler allocate
 * vector registers.
 */
#ifndef X280_RT_VEC_M1_H
#define X280_RT_VEC_M1_H

#include <stdint.h>

/* Wrap a vector asm body so it assembles in non-V translation units too. */
#define X280_VEC_ASM(body) ".option push\n.option arch, +v\n" body ".option pop\n"

/* VLEN=512, SEW=32. */
#define X280_VLMAX_M1 16                      /* elements per vector op  */
#define X280_VEC_M1_BYTES (X280_VLMAX_M1 * 4) /* 64 B                    */

/* vl = VLMAX. */
static inline void x280_vec_setup_m1(void) {
    __asm__ volatile(X280_VEC_ASM("vsetvli zero, %0, e32, m1, ta, ma\n") : : "r"(X280_VLMAX_M1));
}

/* Load the 64 B store pattern into v1. Reads X280_VLMAX_M1 u32s. */
static inline void x280_vec_pattern_v1(const void* src) {
    __asm__ volatile(X280_VEC_ASM("vle32.v v1, (%0)\n") : : "r"(src) : "memory");
}

#endif /* X280_RT_VEC_M1_H */
