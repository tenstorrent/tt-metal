// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: fmt
// DEFINE: %{cflags} = -std=c++17 -Wall -Wextra -Werror -DLLK_SAN_ENABLE -DCOMPILE_FOR_TRISC=0 -DLLK_SAN_SETTING_HOST_DEPS=1 -DDEBUG_PRINT_ENABLED %{fmt_flags}
// DEFINE: %{build} = %clangxx %{cflags} -I %{sanitizer_include} -o %t/run
// DEFINE: %{check} = %t/run | FileCheck --allow-empty
// DEFINE: %{case} = unset.cpp
// DEFINE: %{run} = %{build} %t/%{case} && %{check} %t/%{case}
// RUN: split-file %s %t
//
// REDEFINE: %{case} = zones_valid.cpp
// RUN: %{run}
// REDEFINE: %{case} = lifecycle_valid.cpp
// RUN: %{run}
// REDEFINE: %{case} = init_reseat_valid.cpp
// RUN: %{run}
// REDEFINE: %{case} = uninit_operation_unseated.cpp
// RUN: %{run}
// REDEFINE: %{case} = uninit_operation_foreign.cpp
// RUN: %{run}
// REDEFINE: %{case} = uninit_operation_superseded.cpp
// RUN: %{run}
// REDEFINE: %{case} = execute_operation_unseated.cpp
// RUN: %{run}

//--- zones_valid.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

static void zones()
{
    LLK_SAN_FUNCTION();
    LLK_SAN_SILENT_ZONE();
}

int main()
{
    zones();
    return 0;
}

// CHECK-NOT: llk::san

//--- lifecycle_valid.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::InputFormatA>(1u), StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    init<OperationUnpackTilize, Thread::TRISC0>(
        StateVal<OperationUnpackTilize::BlockCtDim>(2u), StateVal<OperationUnpackTilize::NarrowTile>(false), StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    execute<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u), StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    execute<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u));
    uninit<OperationUnpackTilize, Thread::TRISC0>(
        StateVal<OperationUnpackTilize::BlockCtDim>(2u), StateVal<OperationUnpackTilize::NarrowTile>(false), StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    return 0;
}

// CHECK-NOT: llk::san

//--- init_reseat_valid.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u));
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(3u));
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(3u));
    return 0;
}

// CHECK-NOT: llk::san

//--- uninit_operation_unseated.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    uninit<OperationUnpackTilize, Thread::TRISC0>();
    return 0;
}

// CHECK: UNINIT was called for an Operation the Exu is not initialized for
// CHECK: Status: UNINITIALIZED

//--- uninit_operation_foreign.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    init<OperationUnpackUnary, Thread::TRISC0>(StateVal<OperationUnpackUnary::UnpackToDest>(0u));
    uninit<OperationUnpackTilize, Thread::TRISC0>();
    return 0;
}

// CHECK: UNINIT was called for an Operation the Exu is not initialized for
// CHECK: Status: INITIALIZED

//--- uninit_operation_superseded.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u));
    init<OperationUnpackUnary, Thread::TRISC0>(StateVal<OperationUnpackUnary::UnpackToDest>(0u));
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u));
    return 0;
}

// CHECK: UNINIT was called for an Operation the Exu is not initialized for
// CHECK: Status: INITIALIZED

//--- execute_operation_unseated.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    execute<OperationUnpackTilize, Thread::TRISC0>();
    return 0;
}

// CHECK: EXECUTE was called for an Operation the Exu is not initialized for
// CHECK: Status: UNINITIALIZED
