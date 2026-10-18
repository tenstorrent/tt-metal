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
// REDEFINE: %{case} = uninit_nullary_valid.cpp
// RUN: %{run}
// REDEFINE: %{case} = discard_valid.cpp
// RUN: %{run}
// REDEFINE: %{case} = uninit_operation_first_field_differs.cpp
// RUN: %{run}
// REDEFINE: %{case} = uninit_operation_second_field_differs.cpp
// RUN: %{run}
// REDEFINE: %{case} = uninit_operation_field_differs_mixed.cpp
// RUN: %{run}
// REDEFINE: %{case} = uninit_operation_field_unseated.cpp
// RUN: %{run}
// REDEFINE: %{case} = execute_operation_field_differs.cpp
// RUN: %{run}

//--- uninit_nullary_valid.cpp
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
    uninit<OperationUnpackTilize, Thread::TRISC0>();
    return 0;
}

// CHECK-NOT: llk::san

//--- discard_valid.cpp
#include <cstdint>

#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u), StateDiscard<std::uint32_t>(7u));
    execute<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u), StateDiscard<std::uint32_t>(8u));
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateDiscard<std::uint32_t>(9u));
    return 0;
}

// CHECK-NOT: llk::san

//--- uninit_operation_first_field_differs.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u), StateVal<OperationUnpackTilize::NarrowTile>(false));
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(3u));
    return 0;
}

// CHECK: UNINIT was given an Operation value that differs from init()
// CHECK: llk::san::OperationUnpackTilize::BlockCtDim
// CHECK: Value: 2
// CHECK: Provided value: 3

//--- uninit_operation_second_field_differs.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u), StateVal<OperationUnpackTilize::NarrowTile>(false));
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::NarrowTile>(true));
    return 0;
}

// CHECK: UNINIT was given an Operation value that differs from init()
// CHECK: llk::san::OperationUnpackTilize::NarrowTile
// CHECK: Value: false
// CHECK: Provided value: true

//--- uninit_operation_field_differs_mixed.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u), StateVal<OperationUnpackTilize::NarrowTile>(false));
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u), StateVal<OperationUnpackTilize::NarrowTile>(true));
    return 0;
}

// CHECK: UNINIT was given an Operation value that differs from init()
// CHECK: llk::san::OperationUnpackTilize::NarrowTile
// CHECK: Value: false
// CHECK: Provided value: true

//--- uninit_operation_field_unseated.cpp
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
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::NarrowTile>(false));
    return 0;
}

// CHECK: UNINIT was given an Operation value that differs from init()
// CHECK: llk::san::OperationUnpackTilize::NarrowTile
// CHECK: UNKNOWN (value never recorded?)
// CHECK: Provided value: false

//--- execute_operation_field_differs.cpp
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
    execute<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(3u));
    return 0;
}

// CHECK: EXECUTE was given an Operation value that differs from init()
// CHECK: llk::san::OperationUnpackTilize::BlockCtDim
// CHECK: Value: 2
// CHECK: Provided value: 3
