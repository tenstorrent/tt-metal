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

#include "hal/replay.h"
#include "hal/scalar.h"

namespace replay = hal::replay;
namespace scalar = hal::scalar;

// A constant range is checked at compile time.
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
    LLK_ASSERT(count >= 1 && count <= 64, "Replay range requires start < 32 and count in [1, 64]");
    lltt::replay(5 /*start*/, count & 63);
}

extern "C" __attribute__((noinline, used)) std::uint32_t get_operation_runtime(const std::uint32_t start, const std::uint32_t count)
{
    return replay::get_operation({start, count});
}

extern "C" __attribute__((noinline, used)) std::uint32_t reference_get_operation_runtime(const std::uint32_t start, const std::uint32_t count)
{
    const bool valid = start < 32 && count >= 1 && count <= 64;
    LLK_ASSERT(valid, "Replay range requires start < 32 and count in [1, 64]");
    return TT_OP_REPLAY(start, count & 63, 0 /*execute_while_loading*/, 0 /*load_mode*/);
}
