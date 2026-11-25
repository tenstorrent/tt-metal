// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -fno-ipa-icf -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -UENABLE_LLK_ASSERT -DENABLE_GATHERING -mno-tt-tensix-optimize-replay -fno-ipa-icf -c %s -o %t.gathering.o
// RUN: %{blackhole_objdump} -drz %t.gathering.o > %t.gathering.dump
// RUN: %{blackhole_compare_codegen} %t.gathering.dump
// clang-format on

// Compare each HAL function with its adjacent reference_ function. Disable
// identical-code folding so both bodies are emitted independently.
// Recorded bodies use HAL operations; references use TTI_*/TT_* instructions.
// REPLAY operands are (start, count, execute while recording, record).
// The gathering run checks that each record sits between disable_gathering() and
// enable_gathering(); -mno-tt-tensix-optimize-replay keeps the compiler from forming its own REPLAYs.

#include <cstdint>

#include "hal/replay.h"
#include "hal/scalar.h"

namespace replay = hal::replay;
namespace scalar = hal::scalar;

inline __attribute__((always_inline)) void reference_disable_gathering()
{
#if defined(ENABLE_GATHERING)
    ckernel::disable_gathering();
#endif
}

inline __attribute__((always_inline)) void reference_enable_gathering()
{
#if defined(ENABLE_GATHERING)
    ckernel::enable_gathering();
#endif
}

extern "C" __attribute__((noinline, used)) void record_only()
{
    replay::record<replay::BufferRange {16, 2}>(
        []
        {
            scalar::set_low<1>(hal::gpr<1>());
            scalar::set_low<2>(hal::gpr<2>());
        });
}

extern "C" __attribute__((noinline, used)) void reference_record_only()
{
    reference_disable_gathering();
    TTI_REPLAY(16 /*start_idx*/, 2 /*len*/, 0 /*execute_while_loading*/, 1 /*load_mode*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 1 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 2 /*RegIndex16b*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 2 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 4 /*RegIndex16b*/);
    reference_enable_gathering();
}

// The range ends at slot 31.
extern "C" __attribute__((noinline, used)) void record_and_execute_last_slots()
{
    replay::record<replay::BufferRange {28, 4}, replay::RecordBehavior::RecordAndExecute>(
        []
        {
            scalar::set_low<1>(hal::gpr<1>());
            scalar::set_low<2>(hal::gpr<2>());
            scalar::set_low<3>(hal::gpr<3>());
            scalar::set_low<4>(hal::gpr<4>());
        });
}

extern "C" __attribute__((noinline, used)) void reference_record_and_execute_last_slots()
{
    reference_disable_gathering();
    TTI_REPLAY(28 /*start_idx*/, 4 /*len*/, 1 /*execute_while_loading*/, 1 /*load_mode*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 1 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 2 /*RegIndex16b*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 2 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 4 /*RegIndex16b*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 3 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 6 /*RegIndex16b*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 4 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 8 /*RegIndex16b*/);
    reference_enable_gathering();
}

extern "C" __attribute__((noinline, used)) void record_forwarded_arguments()
{
    replay::record<replay::BufferRange {0, 3}>(
        [](const std::uint32_t count)
        {
            for (std::uint32_t i = 0; i < count; ++i)
            {
                scalar::set_low<7>(hal::gpr<1>());
            }
        },
        3);
}

extern "C" __attribute__((noinline, used)) void reference_record_forwarded_arguments()
{
    reference_disable_gathering();
    TTI_REPLAY(0 /*start_idx*/, 3 /*len*/, 0 /*execute_while_loading*/, 1 /*load_mode*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 7 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 2 /*RegIndex16b*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 7 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 2 /*RegIndex16b*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 7 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 2 /*RegIndex16b*/);
    reference_enable_gathering();
}

// A constant range passed to the runtime overload still lowers to an immediate REPLAY.
extern "C" __attribute__((noinline, used)) void record_constant_runtime_range()
{
    replay::record(replay::BufferRange {12, 1}, [] { scalar::set_low<1>(hal::gpr<1>()); });
}

extern "C" __attribute__((noinline, used)) void reference_record_constant_runtime_range()
{
    reference_disable_gathering();
    TTI_REPLAY(12 /*start_idx*/, 1 /*len*/, 0 /*execute_while_loading*/, 1 /*load_mode*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 1 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 2 /*RegIndex16b*/);
    reference_enable_gathering();
}

// A count chosen at runtime between constants is duplicated into each path, where
// it lowers to an immediate REPLAY like the raw recording builtin. With gathering the
// bracket keeps the paths joined, so both the HAL and the reference issue one runtime REPLAY.
extern "C" __attribute__((noinline, used)) void record_selected_count(const bool partial_face)
{
    const std::uint32_t count = partial_face ? 3 : 2;
    replay::record(
        {0, count},
        [partial_face]
        {
            scalar::set_low<1>(hal::gpr<1>());
            if (partial_face)
            {
                scalar::set_low<2>(hal::gpr<2>());
            }
            scalar::set_low<3>(hal::gpr<3>());
        });
}

extern "C" __attribute__((noinline, used)) void reference_record_selected_count(const bool partial_face)
{
    const std::uint32_t count = partial_face ? 3 : 2;
    reference_disable_gathering();
    lltt::record(0 /*start*/, count);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 1 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 2 /*RegIndex16b*/);
    if (partial_face)
    {
        TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 2 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 4 /*RegIndex16b*/);
    }
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 3 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 6 /*RegIndex16b*/);
    reference_enable_gathering();
}

extern "C" __attribute__((noinline, used)) void record_runtime_count(const std::uint32_t count)
{
    replay::record(
        {4, count},
        [count]
        {
            for (std::uint32_t i = 0; i < count; ++i)
            {
                scalar::set_low<1>(hal::gpr<1>());
            }
        });
}

// lltt::record issues the word through the instruction buffer like TT_REPLAY.
extern "C" __attribute__((noinline, used)) void reference_record_runtime_count(const std::uint32_t count)
{
    reference_disable_gathering();
    lltt::record(4 /*start*/, count);
    for (std::uint32_t i = 0; i < count; ++i)
    {
        TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 1 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 2 /*RegIndex16b*/);
    }
    reference_enable_gathering();
}

// A runtime start index is encoded in a register and issued through the instruction buffer.
extern "C" __attribute__((noinline, used)) void record_runtime_start(const std::uint32_t start)
{
    replay::record({start, 1}, [] { scalar::set_low<1>(hal::gpr<1>()); });
}

extern "C" __attribute__((noinline, used)) void reference_record_runtime_start(const std::uint32_t start)
{
    reference_disable_gathering();
    lltt::record(start, 1 /*count*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 1 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 2 /*RegIndex16b*/);
    reference_enable_gathering();
}
