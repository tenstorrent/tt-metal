// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.

#include <cstdint>

#include "hal/mop.h"

namespace mop = hal::mop;

// Template 0 MopCfg words: 1 = flags (bit 0 InsnB, bit 1 InsnA1-3), 2 = InsnB,
// 3 = InsnA0, 4-6 = InsnA1-3, 7 = SkipA0, 8 = SkipB. Words of a disabled group
// are not written.
inline volatile std::uint32_t *reference_mop_cfg()
{
    return reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);
}

inline constexpr std::uint32_t OP_A0     = 0x10001000;
inline constexpr std::uint32_t OP_A1     = 0x10002000;
inline constexpr std::uint32_t OP_A2     = 0x10003000;
inline constexpr std::uint32_t OP_A3     = 0x10004000;
inline constexpr std::uint32_t OP_B      = 0x10005000;
inline constexpr std::uint32_t OP_SKIP_A = 0x10006000;
inline constexpr std::uint32_t OP_SKIP_B = 0x10007000;

extern "C" __attribute__((noinline, used)) void program_all_groups()
{
    mop::program(mop::foreach_mask_bit(mop::if_clear(OP_A0, OP_A1, OP_A2, OP_A3, OP_B), mop::otherwise(OP_SKIP_A, OP_SKIP_B)));
}

extern "C" __attribute__((noinline, used)) void reference_program_all_groups()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[1] = 3;
    mop_cfg[2] = OP_B;
    mop_cfg[3] = OP_A0;
    mop_cfg[4] = OP_A1;
    mop_cfg[5] = OP_A2;
    mop_cfg[6] = OP_A3;
    mop_cfg[7] = OP_SKIP_A;
    mop_cfg[8] = OP_SKIP_B;
}

// A set mask bit issues a NOP SkipA0 when otherwise is omitted.
extern "C" __attribute__((noinline, used)) void program_a0_only()
{
    mop::program(mop::foreach_mask_bit(mop::if_clear(OP_A0)));
}

extern "C" __attribute__((noinline, used)) void reference_program_a0_only()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[1] = 0;
    mop_cfg[3] = OP_A0;
    mop_cfg[7] = TT_OP_NOP;
}

extern "C" __attribute__((noinline, used)) void program_b_group()
{
    mop::program(mop::foreach_mask_bit(mop::if_clear(OP_A0, OP_B), mop::otherwise(OP_SKIP_A, OP_SKIP_B)));
}

extern "C" __attribute__((noinline, used)) void reference_program_b_group()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[1] = 1;
    mop_cfg[2] = OP_B;
    mop_cfg[3] = OP_A0;
    mop_cfg[7] = OP_SKIP_A;
    mop_cfg[8] = OP_SKIP_B;
}

extern "C" __attribute__((noinline, used)) void program_a123_group()
{
    mop::program(mop::foreach_mask_bit(mop::if_clear(OP_A0, OP_A1, OP_A2, OP_A3), mop::otherwise(OP_SKIP_A)));
}

extern "C" __attribute__((noinline, used)) void reference_program_a123_group()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[1] = 2;
    mop_cfg[3] = OP_A0;
    mop_cfg[4] = OP_A1;
    mop_cfg[5] = OP_A2;
    mop_cfg[6] = OP_A3;
    mop_cfg[7] = OP_SKIP_A;
}

// The argument count fixes the shape, so runtime operations keep the flags constant.
extern "C" __attribute__((noinline, used)) void program_runtime_operations(const std::uint32_t a0, const std::uint32_t mid, const std::uint32_t skip_a0)
{
    mop::program(mop::foreach_mask_bit(mop::if_clear(a0, mid, mid, mid), mop::otherwise(skip_a0)));
}

extern "C" __attribute__((noinline, used)) void reference_program_runtime_operations(
    const std::uint32_t a0, const std::uint32_t mid, const std::uint32_t skip_a0)
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[1] = 2;
    mop_cfg[3] = a0;
    mop_cfg[4] = mid;
    mop_cfg[5] = mid;
    mop_cfg[6] = mid;
    mop_cfg[7] = skip_a0;
}
