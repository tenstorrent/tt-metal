// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Minimal RISC-V Vector (Zve32f) smoke kernel for ComputeConfig::enable_trisc2_rvv.
//
// When the kernel opts into enable_trisc2_rvv, its TRISC2 (pack) translation unit is
// compiled with the Zve32f extension, so the compiler defines __riscv_vector and the
// vector block below is compiled in: a float vadd over an L1 scratch region, written
// with explicit intrinsics (auto-vectorization stays disabled by the toolchain flags).
//
// Without the opt-in (and on TRISC0/1 always), __riscv_vector is not defined and this
// compiles to an empty kernel — the knob-off binary must contain no vector instructions.
//
// Compile-only test collateral (tests/tt_metal/tt_metal/jit_build/test_trisc2_rvv.cpp);
// never dispatched to a device.

#include <cstdint>

#include "api/compute/common.h"

#if defined(TRISC_PACK) && defined(__riscv_vector)
#include <riscv_vector.h>
#endif

void kernel_main() {
#if defined(TRISC_PACK) && defined(__riscv_vector)
    constexpr uint32_t scratch_base = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);

    float* a = reinterpret_cast<float*>(scratch_base);
    float* b = reinterpret_cast<float*>(scratch_base + n * sizeof(float));
    float* c = reinterpret_cast<float*>(scratch_base + 2u * n * sizeof(float));

    for (uint32_t i = 0, vl = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m4(n - i);
        vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(b + i, vl);
        __riscv_vse32_v_f32m4(c + i, __riscv_vfadd_vv_f32m4(va, vb, vl), vl);
    }
#endif
}
