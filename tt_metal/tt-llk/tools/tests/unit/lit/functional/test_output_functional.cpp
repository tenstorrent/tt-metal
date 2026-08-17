// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runtime behaviour of the reporting layer: impl.h's checks route into output.h's asserts, which
// print through the mocked DEVICE_PRINT and stop through the mocked LLK_ASSERT.
//
// Unlike test_hook_functional.cpp this TU supplies only llk::san::state -- output.h is the report
// sink, which is what this test exists to prove. The same source is built twice: the print build
// must emit the report text and keep running; the assert build must die on the first failed check.

// REQUIRES: fmt
// DEFINE: %{cflags} = -std=c++17 -Wall -Wextra -Werror -DLLK_SAN_ENABLE -DCOMPILE_FOR_TRISC=0 -DLLK_SAN_MOCK %{fmt_flags} -I %{sanitizer_include}
//
// RUN: %clangxx %{cflags} -DDEBUG_PRINT_ENABLED %s -o %t-print
// RUN: %t-print | grep "uninit() was given an Operand value that differs from the configured state"
// RUN: %t-print | grep "Current Kernel"
// RUN: %t-print | grep "Provided value ── 32"
//
// RUN: %clangxx %{cflags} -DENABLE_LLK_ASSERT %s -o %t-assert
// RUN: ! %t-assert

#include <cstdint>

#include "sanitizer/api.h"

using namespace llk::san;

using Tilize = OperationUnpackTilize;
using Unp    = Operand<Exu::Unpack>;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    configure<Thread::TRISC0>(StateVal<Unp::FaceHeightA>(16u));
    init<Tilize, Thread::TRISC0>(StateVal<Unp::FaceHeightA>(16u));

    // Mismatch: the print build reports and continues, the assert build aborts here.
    uninit<Tilize, Thread::TRISC0>(StateVal<Unp::FaceHeightA>(32u));

    return 0;
}
