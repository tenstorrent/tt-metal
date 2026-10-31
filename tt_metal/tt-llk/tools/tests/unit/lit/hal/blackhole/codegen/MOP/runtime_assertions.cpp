// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -fno-ipa-icf -fno-reorder-blocks -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function with
// LLK_ASSERT enabled; references state each check as an LLK_ASSERT before the
// operation. Disable identical-code folding so both bodies are emitted
// independently, and block reordering so the layout of the assert branches does
// not depend on GCC's placement heuristics.

#include <cstdint>

#include "hal/mop.h"

namespace mop = hal::mop;

using Template0Runner = mop::Runner<mop::MopTemplate::Template0>;

inline volatile std::uint32_t *reference_mop_cfg()
{
    return reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);
}

inline constexpr std::uint32_t OP = 0x10001000;

// The argument count fixes the shape, so nothing is checked.
extern "C" __attribute__((noinline, used)) void template0_runtime_operations(const std::uint32_t a0, const std::uint32_t b)
{
    mop::program(mop::foreach_mask_bit(mop::if_clear(a0, b), mop::otherwise(a0, b)));
}

extern "C" __attribute__((noinline, used)) void reference_template0_runtime_operations(const std::uint32_t a0, const std::uint32_t b)
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[1] = 1;
    mop_cfg[2] = b;
    mop_cfg[3] = a0;
    mop_cfg[7] = a0;
    mop_cfg[8] = b;
}

// A start op rules out the count anomaly, leaving the 10-bit count checks.
extern "C" __attribute__((noinline, used)) void template1_runtime_counts(const std::uint32_t outer, const std::uint32_t inner)
{
    mop::program(mop::loop(outer, OP, mop::loop(inner, OP), OP));
}

extern "C" __attribute__((noinline, used)) void reference_template1_runtime_counts(const std::uint32_t outer, const std::uint32_t inner)
{
    LLK_ASSERT(outer < 1024, "Template 1 MOP loop count must fit in 10 bits");
    LLK_ASSERT(inner < 1024, "Template 1 MOP loop count must fit in 10 bits");
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = outer;
    mop_cfg[1] = inner;
    mop_cfg[2] = OP;
    mop_cfg[3] = OP;
    mop_cfg[4] = TT_OP_NOP;
    if (inner != 0)
    {
        mop_cfg[5] = OP;
        mop_cfg[6] = TT_OP_NOP;
        mop_cfg[7] = OP;
        mop_cfg[8] = OP;
    }
}

// Without a start op and with only EndOp0, an inner count of zero triggers the count anomaly.
extern "C" __attribute__((noinline, used)) void template1_runtime_inner_count(const std::uint32_t inner)
{
    mop::program(mop::loop(4, mop::loop(inner, OP), OP));
}

extern "C" __attribute__((noinline, used)) void reference_template1_runtime_inner_count(const std::uint32_t inner)
{
    LLK_ASSERT(inner < 1024, "Template 1 MOP loop count must fit in 10 bits");
    LLK_ASSERT(inner != 0, "Template 1 MOP configuration triggers the no-start/zero-inner/end-op outer count anomaly");
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = 4;
    mop_cfg[1] = inner;
    mop_cfg[2] = TT_OP_NOP;
    mop_cfg[3] = OP;
    mop_cfg[4] = TT_OP_NOP;
    if (inner != 0)
    {
        mop_cfg[5] = OP;
        mop_cfg[6] = TT_OP_NOP;
        mop_cfg[7] = OP;
        mop_cfg[8] = OP;
    }
}

// With EndOp1 set the count anomaly needs an outer count of 1.
extern "C" __attribute__((noinline, used)) void template1_runtime_second_end(const std::uint32_t outer)
{
    mop::program(mop::loop(outer, mop::loop(0, OP), OP, OP));
}

extern "C" __attribute__((noinline, used)) void reference_template1_runtime_second_end(const std::uint32_t outer)
{
    LLK_ASSERT(outer < 1024, "Template 1 MOP loop count must fit in 10 bits");
    LLK_ASSERT(outer != 1, "Template 1 MOP configuration triggers the no-start/zero-inner/end-op outer count anomaly");
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = outer;
    mop_cfg[1] = 0;
    mop_cfg[2] = TT_OP_NOP;
    mop_cfg[3] = OP;
    mop_cfg[4] = OP;
}

// A patch with a runtime inner count checks that the patched op can be issued.
extern "C" __attribute__((noinline, used)) void template1_runtime_patch(const std::uint32_t inner, const std::uint32_t operation)
{
    mop::program(mop::loop(1, OP, mop::loop(inner, OP)).patch(mop::last_inner, operation));
}

extern "C" __attribute__((noinline, used)) void reference_template1_runtime_patch(const std::uint32_t inner, const std::uint32_t operation)
{
    LLK_ASSERT(inner < 1024, "Template 1 MOP loop count must fit in 10 bits");
    LLK_ASSERT(inner != 0, "Template 1 MOP last-iteration ops are never issued with an empty inner loop");
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = 1;
    mop_cfg[1] = inner;
    mop_cfg[2] = OP;
    mop_cfg[3] = TT_OP_NOP;
    if (inner != 0)
    {
        mop_cfg[5] = OP;
        mop_cfg[6] = TT_OP_NOP;
        mop_cfg[7] = OP;
        mop_cfg[8] = operation;
    }
}

extern "C" __attribute__((noinline, used)) void update_count(const std::uint32_t count)
{
    mop::update(mop::inner_count, count);
}

extern "C" __attribute__((noinline, used)) void reference_update_count(const std::uint32_t count)
{
    LLK_ASSERT(count < 1024, "Template 1 MOP loop count must fit in 10 bits");
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[1] = count;
}

// Instruction words have no limit.
extern "C" __attribute__((noinline, used)) void update_operation(const std::uint32_t operation)
{
    mop::update(mop::last_inner, operation);
}

extern "C" __attribute__((noinline, used)) void reference_update_operation(const std::uint32_t operation)
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[8] = operation;
}

extern "C" __attribute__((noinline, used)) void update_count_and_operation(const std::uint32_t outer, const std::uint32_t operation)
{
    mop::update(mop::last_outer(operation), mop::outer_count(outer));
}

extern "C" __attribute__((noinline, used)) void reference_update_count_and_operation(const std::uint32_t outer, const std::uint32_t operation)
{
    LLK_ASSERT(outer < 1024, "Template 1 MOP loop count must fit in 10 bits");
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[7] = operation;
    mop_cfg[0] = outer;
}

extern "C" __attribute__((noinline, used)) void run_template0_runtime_count(const std::uint32_t count)
{
    Template0Runner::run<0x5>(count);
}

extern "C" __attribute__((noinline, used)) void reference_run_template0_runtime_count(const std::uint32_t count)
{
    const std::uint32_t loop_count = count - 1;
    LLK_ASSERT(loop_count < 128, "MOP encoded count must fit in 7 bits");
    TTI_MOP_CFG(0 /*zmask_hi16*/);
    TT_MOP(0 /*mop_type*/, loop_count, 0x5 /*zmask_lo16_or_loop_count*/);
}

// A compile-time count needs no runtime check.
extern "C" __attribute__((noinline, used)) void run_template0_runtime_mask(const std::uint32_t mask)
{
    Template0Runner::run_with_runtime_mask<4>(mask);
}

extern "C" __attribute__((noinline, used)) void reference_run_template0_runtime_mask(const std::uint32_t mask)
{
    TT_MOP_CFG(mask >> 16);
    TT_MOP(0 /*mop_type*/, 3 /*loop_count*/, mask & 0xffff);
}

extern "C" __attribute__((noinline, used)) std::uint32_t template0_runtime_preset_operation(const std::uint32_t count, const std::uint32_t mask)
{
    return Template0Runner {.count = count, .mask = mask, .mask_high = mop::MaskHigh::Preset}.operation();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_template0_runtime_preset_operation(const std::uint32_t count, const std::uint32_t mask)
{
    const std::uint32_t loop_count = count - 1;
    LLK_ASSERT(loop_count < 128, "MOP encoded count must fit in 7 bits");
    return TT_OP_MOP(0 /*mop_type*/, loop_count, mask & 0xffff);
}
