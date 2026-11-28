// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// The sticky special-value flags are one RISC load or store of RISCV_DEBUG_REG_FPU_STICKY_BITS, as production's.

#include <cstdint>

#include "hal/fpu.h"

namespace fpu = hal::fpu;

extern "C" __attribute__((noinline, used)) std::uint32_t read_special_value_flags()
{
    return fpu::MatrixUnit.special_value_flags();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_read_special_value_flags()
{
    return ckernel::reg_read(RISCV_DEBUG_REG_FPU_STICKY_BITS);
}

extern "C" __attribute__((noinline, used)) void clear_special_value_flags()
{
    fpu::MatrixUnit.clear_special_value_flags();
}

extern "C" __attribute__((noinline, used)) void reference_clear_special_value_flags()
{
    ckernel::reg_write(RISCV_DEBUG_REG_FPU_STICKY_BITS, 0);
}
