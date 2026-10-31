// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// Execution uses TTI_*/TT_* references; replay encoding uses TT_OP_* references.
// REPLAY operands are (start, count, execute while recording, record); a count
// of 64 is encoded as zero by the low six bits.

#include <cstdint>

#include "hal/mop.h"
#include "hal/replay.h"

namespace mop    = hal::mop;
namespace replay = hal::replay;

extern "C" __attribute__((noinline, used)) void run_full_buffer()
{
    replay::run<replay::BufferRange {0, 32}>();
}

extern "C" __attribute__((noinline, used)) void reference_run_full_buffer()
{
    TTI_REPLAY(0 /*start_idx*/, 32 /*len*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
}

// The range wraps past slot 31.
extern "C" __attribute__((noinline, used)) void run_wrapping()
{
    replay::run<replay::BufferRange {31, 2}>();
}

extern "C" __attribute__((noinline, used)) void reference_run_wrapping()
{
    TTI_REPLAY(31 /*start_idx*/, 2 /*len*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
}

// A constant range passed to the runtime overload still lowers to an immediate REPLAY.
extern "C" __attribute__((noinline, used)) void run_constant_runtime_range()
{
    replay::run(replay::BufferRange {5, 3});
}

extern "C" __attribute__((noinline, used)) void reference_run_constant_runtime_range()
{
    TTI_REPLAY(5 /*start_idx*/, 3 /*len*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
}

extern "C" __attribute__((noinline, used)) void run_runtime_count(const std::uint32_t count)
{
    replay::run(replay::BufferRange {5, count});
}

// lltt::replay issues the word through the instruction buffer like TT_REPLAY,
// in the same operation order as the HAL.
extern "C" __attribute__((noinline, used)) void reference_run_runtime_count(const std::uint32_t count)
{
    lltt::replay(5 /*start*/, count & 63);
}

extern "C" __attribute__((noinline, used)) std::uint32_t get_operation_max_count()
{
    return replay::get_operation<replay::BufferRange {4, 64}>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_get_operation_max_count()
{
    return TT_OP_REPLAY(4 /*start_idx*/, 0 /*len*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t get_operation_last_slot()
{
    return replay::get_operation<replay::BufferRange {31, 32}>();
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_get_operation_last_slot()
{
    return TT_OP_REPLAY(31 /*start_idx*/, 32 /*len*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);
}

extern "C" __attribute__((noinline, used)) std::uint32_t get_operation_runtime(const std::uint32_t start, const std::uint32_t count)
{
    return replay::get_operation({start, count});
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_get_operation_runtime(const std::uint32_t start, const std::uint32_t count)
{
    return TT_OP_REPLAY(start, count & 63, 0 /*execute_while_loading*/, 0 /*load_mode*/);
}

// An encoded replay is a MOP body op; both last-iteration ops repeat it.
inline constexpr replay::BufferRange row_epilogue {16, 2};

extern "C" __attribute__((noinline, used)) void program_mop_with_replay()
{
    mop::program(mop::loop(1, mop::loop(4, replay::get_operation<row_epilogue>())));
}

extern "C" __attribute__((noinline, used)) void reference_program_mop_with_replay()
{
    constexpr std::uint32_t replay_op = TT_OP_REPLAY(16 /*start_idx*/, 2 /*len*/, 0 /*execute_while_loading*/, 0 /*load_mode*/);

    volatile std::uint32_t *mop_cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_MOP_CFG_BASE);
    ckernel::mop_sync();
    mop_cfg[0] = 1;
    mop_cfg[1] = 4;
    mop_cfg[2] = TT_OP_NOP;
    mop_cfg[3] = TT_OP_NOP;
    mop_cfg[5] = replay_op;
    mop_cfg[6] = TT_OP_NOP;
    mop_cfg[7] = replay_op;
    mop_cfg[8] = replay_op;
}
