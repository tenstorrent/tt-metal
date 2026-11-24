// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -fno-ipa-icf -fno-reorder-blocks -c %s -o %t.o
// RUN: %{blackhole_objdump} -drz %t.o > %t.dump
// RUN: %{blackhole_compare_codegen} %t.dump
// RUN: %{blackhole_tensix_compile} %{blackhole_math_thread} -DENABLE_LLK_ASSERT -DRECORD_BOUNDARIES -c %s -o %t.boundaries.o
// RUN: %{blackhole_objdump} -d %t.boundaries.o | FileCheck %s --check-prefix=BOUNDARY

// Compare each HAL function with its adjacent reference_ function with
// LLK_ASSERT enabled; references state each check as the HAL does, with
// hal::detail::require or LLK_ASSERT, before the operation. Disable identical-code folding so both bodies are emitted
// independently, and block reordering so the layout of the assert branches does
// not depend on GCC's placement heuristics.

#include <cstdint>

#include "hal/replay.h"
#include "hal/scalar.h"

namespace replay = hal::replay;
namespace scalar = hal::scalar;

// A valid constant range folds the check away.
extern "C" __attribute__((noinline, used)) void record_constant_count()
{
    replay::record(replay::BufferRange {12, 1}, [] { scalar::set_low<1>(hal::gpr<1>()); });
}

extern "C" __attribute__((noinline, used)) void reference_record_constant_count()
{
    TTI_REPLAY(12 /*start_idx*/, 1 /*len*/, 0 /*execute_while_loading*/, 1 /*load_mode*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 1 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 2 /*RegIndex16b*/);
}

extern "C" __attribute__((noinline, used)) void run_runtime_count(const std::uint32_t count)
{
    replay::run(replay::BufferRange {5, count});
}

extern "C" __attribute__((noinline, used)) void reference_run_runtime_count(const std::uint32_t count)
{
    LLK_ASSERT(count >= 1 && count <= 27, "Replay range requires count >= 1 and start + count <= 32");
    lltt::replay(5 /*start*/, count);
}

extern "C" __attribute__((noinline, used)) std::uint32_t get_operation_runtime(const std::uint32_t start, const std::uint32_t count)
{
    return replay::get_operation({start, count});
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_get_operation_runtime(const std::uint32_t start, const std::uint32_t count)
{
    const bool valid = start < 32 && count >= 1 && count <= 32 - start;
    hal::detail::require(valid, "Replay range requires count >= 1 and start + count <= 32");
    return TT_OP_REPLAY(start, count, 0 /*execute_while_loading*/, 0 /*load_mode*/);
}

// Both selectable counts are valid, so the check folds away and each path keeps
// its immediate REPLAY.
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
    lltt::record(0 /*start*/, partial_face ? 3 : 2 /*length*/);
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 1 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 2 /*RegIndex16b*/);
    if (partial_face)
    {
        TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 2 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 4 /*RegIndex16b*/);
    }
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 3 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, 6 /*RegIndex16b*/);
}

#ifdef RECORD_BOUNDARIES
// The runtime record() check folds differently from any independent reference, so it is
// checked here on its own. Each runtime check is called with the last valid and the first
// invalid value, which the compiler folds. The test checks only whether the LLK_ASSERT ebreak
// is emitted, and that it comes before the recording REPLAY. The REPLAY builtin rejects a
// constant count of 0 itself, so only the upper bound has an invalid case.
template <std::uint32_t Count>
inline __attribute__((always_inline)) void record_from_slot_4()
{
    replay::record(
        {4, Count},
        []
        {
            for (std::uint32_t i = 0; i < Count; ++i)
            {
                scalar::set_low<1>(hal::gpr<1>());
            }
        });
}

extern "C" __attribute__((noinline, used)) void record_count_max_valid()
{
    record_from_slot_4<28>();
}

// BOUNDARY-LABEL: <record_count_max_valid>:
// BOUNDARY-NOT: ebreak
// BOUNDARY: ttreplay 4,28,0,1
// BOUNDARY-NOT: ebreak
// BOUNDARY: ret

extern "C" __attribute__((noinline, used)) void record_count_too_large()
{
    record_from_slot_4<29>();
}

// BOUNDARY-LABEL: <record_count_too_large>:
// BOUNDARY: ebreak
// BOUNDARY: ttreplay 4,29,0,1

extern "C" __attribute__((noinline, used)) void record_count_min_valid()
{
    record_from_slot_4<1>();
}

// BOUNDARY-LABEL: <record_count_min_valid>:
// BOUNDARY-NOT: ebreak
// BOUNDARY: ttreplay 4,1,0,1
// BOUNDARY-NOT: ebreak
// BOUNDARY: ret
#endif
