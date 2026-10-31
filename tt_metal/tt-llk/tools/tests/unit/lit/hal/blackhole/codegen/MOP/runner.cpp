// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// Execution uses TTI_*/TT_* references; MOP encoding uses TT_OP_* references.
// MOP operands are (template, count - 1, mask low 16 bits); MOP_CFG carries the
// upper 16 mask bits. Template 1 takes its counts from MopCfg and encodes zeros.

#include <cstdint>

#include "hal/mop.h"

namespace mop = hal::mop;

using Template0Runner = mop::Runner<mop::MopTemplate::Template0>;
using Template1Runner = mop::Runner<mop::MopTemplate::Template1>;

// A mask of at most 16 bits applied to at most 16 iterations needs no MOP_CFG.
extern "C" __attribute__((noinline, used)) void run_template0_short()
{
    Template0Runner::run<4, 0x5>();
}

extern "C" __attribute__((noinline, used)) void reference_run_template0_short()
{
    TTI_MOP(0 /*mop_type*/, 3 /*loop_count*/, 0x5 /*zmask_lo16_or_loop_count*/);
}

extern "C" __attribute__((noinline, used)) void run_template0_long_count()
{
    Template0Runner::run<128, 0x5>();
}

extern "C" __attribute__((noinline, used)) void reference_run_template0_long_count()
{
    TTI_MOP_CFG(0 /*zmask_hi16*/);
    TTI_MOP(0 /*mop_type*/, 127 /*loop_count*/, 0x5 /*zmask_lo16_or_loop_count*/);
}

extern "C" __attribute__((noinline, used)) void run_template0_wide_mask()
{
    Template0Runner::run<1, 0x12345>();
}

extern "C" __attribute__((noinline, used)) void reference_run_template0_wide_mask()
{
    TTI_MOP_CFG(0x1 /*zmask_hi16*/);
    TTI_MOP(0 /*mop_type*/, 0 /*loop_count*/, 0x2345 /*zmask_lo16_or_loop_count*/);
}

// A runtime count may exceed 16, so MOP_CFG is written even for a 16-bit mask.
extern "C" __attribute__((noinline, used)) void run_template0_runtime_count(const std::uint32_t count)
{
    Template0Runner::run<0x5>(count);
}

extern "C" __attribute__((noinline, used)) void reference_run_template0_runtime_count(const std::uint32_t count)
{
    TTI_MOP_CFG(0 /*zmask_hi16*/);
    TT_MOP(0 /*mop_type*/, count - 1, 0x5 /*zmask_lo16_or_loop_count*/);
}

extern "C" __attribute__((noinline, used)) void run_template0_runtime_mask(const std::uint32_t mask)
{
    Template0Runner::run_with_runtime_mask<4>(mask);
}

extern "C" __attribute__((noinline, used)) void reference_run_template0_runtime_mask(const std::uint32_t mask)
{
    TT_MOP_CFG(mask >> 16);
    TT_MOP(0 /*mop_type*/, 3 /*loop_count*/, mask & 0xffff);
}

extern "C" __attribute__((noinline, used)) void run_template0_runtime(const std::uint32_t count, const std::uint32_t mask)
{
    Template0Runner::run(count, mask);
}

extern "C" __attribute__((noinline, used)) void reference_run_template0_runtime(const std::uint32_t count, const std::uint32_t mask)
{
    // Hoisting the loop count avoids an incidental reassociation difference from the HAL body.
    const std::uint32_t loop_count = count - 1;
    TT_MOP_CFG(mask >> 16);
    TT_MOP(0 /*mop_type*/, loop_count, mask & 0xffff);
}

// With MaskHigh::Preset the run issues no MOP_CFG; configure_mask_high wrote it.
extern "C" __attribute__((noinline, used)) void run_template0_preset()
{
    Template0Runner::configure_mask_high<0x12345>();
    Template0Runner::run<32, 0x12345, mop::MaskHigh::Preset>();
}

extern "C" __attribute__((noinline, used)) void reference_run_template0_preset()
{
    TTI_MOP_CFG(0x1 /*zmask_hi16*/);
    TTI_MOP(0 /*mop_type*/, 31 /*loop_count*/, 0x2345 /*zmask_lo16_or_loop_count*/);
}

extern "C" __attribute__((noinline, used)) void run_template0_runtime_preset(const std::uint32_t count, const std::uint32_t mask)
{
    Template0Runner::run<mop::MaskHigh::Preset>(count, mask);
}

extern "C" __attribute__((noinline, used)) void reference_run_template0_runtime_preset(const std::uint32_t count, const std::uint32_t mask)
{
    TT_MOP(0 /*mop_type*/, count - 1, mask & 0xffff);
}

extern "C" __attribute__((noinline, used)) void configure_runtime_mask_high(const std::uint32_t mask)
{
    Template0Runner::configure_mask_high(mask);
}

extern "C" __attribute__((noinline, used)) void reference_configure_runtime_mask_high(const std::uint32_t mask)
{
    TT_MOP_CFG(mask >> 16);
}

extern "C" __attribute__((noinline, used)) std::uint32_t get_template0_operation()
{
    return Template0Runner {.count = 3, .mask = 0x5}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_get_template0_operation()
{
    return TT_OP_MOP(0 /*mop_type*/, 2 /*loop_count*/, 0x5 /*zmask_lo16_or_loop_count*/);
}

// With MaskHigh::Preset only the low 16 mask bits are encoded.
extern "C" __attribute__((noinline, used)) std::uint32_t get_template0_preset_operation()
{
    return Template0Runner {.count = 128, .mask = 0x12345, .mask_high = mop::MaskHigh::Preset}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_get_template0_preset_operation()
{
    return TT_OP_MOP(0 /*mop_type*/, 127 /*loop_count*/, 0x2345 /*zmask_lo16_or_loop_count*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t get_template0_runtime_operation(const std::uint32_t count, const std::uint32_t mask)
{
    return Template0Runner {.count = count, .mask = mask}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_get_template0_runtime_operation(const std::uint32_t count, const std::uint32_t mask)
{
    return TT_OP_MOP(0 /*mop_type*/, count - 1, mask & 0xffff);
}

extern "C" __attribute__((noinline, used)) void run_template1()
{
    Template1Runner::run();
}

extern "C" __attribute__((noinline, used)) void reference_run_template1()
{
    TTI_MOP(1 /*mop_type*/, 0 /*loop_count*/, 0 /*zmask_lo16_or_loop_count*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t get_template1_operation()
{
    return Template1Runner {}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_get_template1_operation()
{
    return TT_OP_MOP(1 /*mop_type*/, 0 /*loop_count*/, 0 /*zmask_lo16_or_loop_count*/);
}
