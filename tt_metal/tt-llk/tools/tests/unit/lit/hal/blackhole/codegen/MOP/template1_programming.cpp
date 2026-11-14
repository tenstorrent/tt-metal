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

// Template 1 MopCfg words: 0 = outer count, 1 = inner count, 2 = StartOp,
// 3 = EndOp0, 4 = EndOp1, 5 = LoopOp, 6 = LoopOp1, 7 = Loop0Last, 8 = Loop1Last.
// EndOp1 is written only with end ops, and words 5-8 only with a nonzero inner count.
inline volatile std::uint32_t *reference_mop_cfg()
{
    return reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);
}

inline constexpr std::uint32_t OP_START      = 0x10001000;
inline constexpr std::uint32_t OP_END        = 0x10002000;
inline constexpr std::uint32_t OP_SECOND_END = 0x10003000;
inline constexpr std::uint32_t OP_BODY       = 0x10004000;
inline constexpr std::uint32_t OP_ALTERNATE  = 0x10005000;
inline constexpr std::uint32_t OP_LAST_OUTER = 0x10006000;
inline constexpr std::uint32_t OP_LAST_INNER = 0x10007000;

inline constexpr auto complete = mop::loop(4, OP_START, mop::loop(8, OP_BODY, OP_ALTERNATE), OP_END, OP_SECOND_END)
                                     .patch(mop::last_outer, OP_LAST_OUTER)
                                     .patch(mop::last_inner, OP_LAST_INNER);

extern "C" __attribute__((noinline, used)) void program_complete()
{
    mop::program(complete);
}

extern "C" __attribute__((noinline, used)) void reference_program_complete()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = 4;
    mop_cfg[1] = 8;
    mop_cfg[2] = OP_START;
    mop_cfg[3] = OP_END;
    mop_cfg[4] = OP_SECOND_END;
    mop_cfg[5] = OP_BODY;
    mop_cfg[6] = OP_ALTERNATE;
    mop_cfg[7] = OP_LAST_OUTER;
    mop_cfg[8] = OP_LAST_INNER;
}

// Without a patch, both last-iteration ops repeat the body op.
extern "C" __attribute__((noinline, used)) void program_body_only()
{
    mop::program(mop::loop(2, mop::loop(3, OP_BODY)));
}

extern "C" __attribute__((noinline, used)) void reference_program_body_only()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = 2;
    mop_cfg[1] = 3;
    mop_cfg[2] = TT_OP_NOP;
    mop_cfg[3] = TT_OP_NOP;
    mop_cfg[5] = OP_BODY;
    mop_cfg[6] = TT_OP_NOP;
    mop_cfg[7] = OP_BODY;
    mop_cfg[8] = OP_BODY;
}

// With an alternate op, both last-iteration ops repeat it instead.
extern "C" __attribute__((noinline, used)) void program_alternate()
{
    mop::program(mop::loop(1, OP_START, mop::loop(2, OP_BODY, OP_ALTERNATE)));
}

extern "C" __attribute__((noinline, used)) void reference_program_alternate()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = 1;
    mop_cfg[1] = 2;
    mop_cfg[2] = OP_START;
    mop_cfg[3] = TT_OP_NOP;
    mop_cfg[5] = OP_BODY;
    mop_cfg[6] = OP_ALTERNATE;
    mop_cfg[7] = OP_ALTERNATE;
    mop_cfg[8] = OP_ALTERNATE;
}

extern "C" __attribute__((noinline, used)) void program_end()
{
    mop::program(mop::loop(2, mop::loop(3, OP_BODY), OP_END));
}

extern "C" __attribute__((noinline, used)) void reference_program_end()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = 2;
    mop_cfg[1] = 3;
    mop_cfg[2] = TT_OP_NOP;
    mop_cfg[3] = OP_END;
    mop_cfg[4] = TT_OP_NOP;
    mop_cfg[5] = OP_BODY;
    mop_cfg[6] = TT_OP_NOP;
    mop_cfg[7] = OP_BODY;
    mop_cfg[8] = OP_BODY;
}

extern "C" __attribute__((noinline, used)) void program_two_ends()
{
    mop::program(mop::loop(2, mop::loop(3, OP_BODY), OP_END, OP_SECOND_END));
}

extern "C" __attribute__((noinline, used)) void reference_program_two_ends()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = 2;
    mop_cfg[1] = 3;
    mop_cfg[2] = TT_OP_NOP;
    mop_cfg[3] = OP_END;
    mop_cfg[4] = OP_SECOND_END;
    mop_cfg[5] = OP_BODY;
    mop_cfg[6] = TT_OP_NOP;
    mop_cfg[7] = OP_BODY;
    mop_cfg[8] = OP_BODY;
}

extern "C" __attribute__((noinline, used)) void program_start_end()
{
    mop::program(mop::loop(2, OP_START, mop::loop(3, OP_BODY), OP_END));
}

extern "C" __attribute__((noinline, used)) void reference_program_start_end()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = 2;
    mop_cfg[1] = 3;
    mop_cfg[2] = OP_START;
    mop_cfg[3] = OP_END;
    mop_cfg[4] = TT_OP_NOP;
    mop_cfg[5] = OP_BODY;
    mop_cfg[6] = TT_OP_NOP;
    mop_cfg[7] = OP_BODY;
    mop_cfg[8] = OP_BODY;
}

// A last_outer patch leaves last_inner repeating the body op.
extern "C" __attribute__((noinline, used)) void program_patch_last_outer()
{
    mop::program(mop::loop(2, mop::loop(3, OP_BODY)).patch(mop::last_outer, OP_LAST_OUTER));
}

extern "C" __attribute__((noinline, used)) void reference_program_patch_last_outer()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = 2;
    mop_cfg[1] = 3;
    mop_cfg[2] = TT_OP_NOP;
    mop_cfg[3] = TT_OP_NOP;
    mop_cfg[5] = OP_BODY;
    mop_cfg[6] = TT_OP_NOP;
    mop_cfg[7] = OP_LAST_OUTER;
    mop_cfg[8] = OP_BODY;
}

extern "C" __attribute__((noinline, used)) void program_empty_inner()
{
    mop::program(mop::loop(3, OP_START, mop::loop(0, OP_BODY), OP_END));
}

extern "C" __attribute__((noinline, used)) void reference_program_empty_inner()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = 3;
    mop_cfg[1] = 0;
    mop_cfg[2] = OP_START;
    mop_cfg[3] = OP_END;
    mop_cfg[4] = TT_OP_NOP;
}

// With EndOp1 set, the count anomaly needs an outer count of 1, so this
// no-start, empty-inner configuration is accepted.
extern "C" __attribute__((noinline, used)) void program_empty_inner_second_end()
{
    mop::program(mop::loop(4, mop::loop(0, OP_BODY), OP_END, OP_SECOND_END));
}

extern "C" __attribute__((noinline, used)) void reference_program_empty_inner_second_end()
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = 4;
    mop_cfg[1] = 0;
    mop_cfg[2] = TT_OP_NOP;
    mop_cfg[3] = OP_END;
    mop_cfg[4] = OP_SECOND_END;
}

// A runtime inner count decides at runtime whether words 5-8 are written.
extern "C" __attribute__((noinline, used)) void program_runtime_counts(const std::uint32_t outer, const std::uint32_t inner)
{
    mop::program(mop::loop(outer, OP_START, mop::loop(inner, OP_BODY), OP_END));
}

extern "C" __attribute__((noinline, used)) void reference_program_runtime_counts(const std::uint32_t outer, const std::uint32_t inner)
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = outer;
    mop_cfg[1] = inner;
    mop_cfg[2] = OP_START;
    mop_cfg[3] = OP_END;
    mop_cfg[4] = TT_OP_NOP;
    if (inner != 0)
    {
        mop_cfg[5] = OP_BODY;
        mop_cfg[6] = TT_OP_NOP;
        mop_cfg[7] = OP_BODY;
        mop_cfg[8] = OP_BODY;
    }
}

extern "C" __attribute__((noinline, used)) void program_runtime_operations(const std::uint32_t body, const std::uint32_t last_inner)
{
    mop::program(mop::loop(2, mop::loop(4, body)).patch(mop::last_inner, last_inner));
}

extern "C" __attribute__((noinline, used)) void reference_program_runtime_operations(const std::uint32_t body, const std::uint32_t last_inner)
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[0] = 2;
    mop_cfg[1] = 4;
    mop_cfg[2] = TT_OP_NOP;
    mop_cfg[3] = TT_OP_NOP;
    mop_cfg[5] = body;
    mop_cfg[6] = TT_OP_NOP;
    mop_cfg[7] = body;
    mop_cfg[8] = last_inner;
}

extern "C" __attribute__((noinline, used)) void update_inner_count(const std::uint32_t count)
{
    mop::update(mop::inner_count, count);
}

extern "C" __attribute__((noinline, used)) void reference_update_inner_count(const std::uint32_t count)
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[1] = count;
}

extern "C" __attribute__((noinline, used)) void update_last_inner(const std::uint32_t operation)
{
    mop::update(mop::last_inner, operation);
}

extern "C" __attribute__((noinline, used)) void reference_update_last_inner(const std::uint32_t operation)
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[8] = operation;
}

// Several words share one sync and are written in argument order.
extern "C" __attribute__((noinline, used)) void update_several(const std::uint32_t outer, const std::uint32_t operation)
{
    mop::update(mop::last_outer(operation), mop::outer_count(outer));
}

extern "C" __attribute__((noinline, used)) void reference_update_several(const std::uint32_t outer, const std::uint32_t operation)
{
    volatile std::uint32_t *mop_cfg = reference_mop_cfg();
    ckernel::mop_sync();
    mop_cfg[7] = operation;
    mop_cfg[0] = outer;
}
