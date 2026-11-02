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
// REDEFINE: %{case} = configure_discard_valid.cpp
// RUN: %{run}
// REDEFINE: %{case} = configure_unpack_valid.cpp
// RUN: %{run}
// REDEFINE: %{case} = configure_fpu_sfpu_valid.cpp
// RUN: %{run}
// REDEFINE: %{case} = configure_pack_valid.cpp
// RUN: %{run}
// REDEFINE: %{case} = init_operand_differs_valid.cpp
// RUN: %{run}
// REDEFINE: %{case} = reconfigure_unseated_field_valid.cpp
// RUN: %{run}
// REDEFINE: %{case} = uninit_operand_value_differs.cpp
// RUN: %{run}
// REDEFINE: %{case} = uninit_operand_field_unseated.cpp
// RUN: %{run}
// REDEFINE: %{case} = uninit_operand_reconfigured.cpp
// RUN: %{run}
// REDEFINE: %{case} = uninit_operand_reconfigured_nullary.cpp
// RUN: %{run}
// REDEFINE: %{case} = uninit_after_init_operand_differs.cpp
// RUN: %{run}

//--- configure_discard_valid.cpp
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
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u), StateDiscard<std::uint32_t>(7u));
    reconfigure<Thread::TRISC0>(StateDiscard<bool>(false));

    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    return 0;
}

// CHECK-NOT: llk::san

//--- configure_unpack_valid.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::InputFormatA>(7u), StateVal<Operand<Exu::Unpack>::NumFacesA>(4u));
    reconfigure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::InputFormatA>(9u));
    return 0;
}

// CHECK-NOT: llk::san

//--- configure_fpu_sfpu_valid.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    configure<Thread::TRISC1>(StateVal<Operand<Exu::Fpu>::Format>(1u), StateVal<Operand<Exu::Sfpu>::Format>(2u));
    reconfigure<Thread::TRISC1>(StateVal<Operand<Exu::Fpu>::Format>(3u));
    return 0;
}

// CHECK-NOT: llk::san

//--- configure_pack_valid.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    configure<Thread::TRISC2>(StateVal<Operand<Exu::Pack>::OutputFormat>(3u));
    reconfigure<Thread::TRISC2>(StateVal<Operand<Exu::Pack>::InputFormat>(1u));
    return 0;
}

// CHECK-NOT: llk::san

//--- init_operand_differs_valid.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(32u));
    return 0;
}

// CHECK-NOT: llk::san

//--- reconfigure_unseated_field_valid.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u), StateVal<Operand<Exu::Unpack>::InputFormatA>(1u));
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    reconfigure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::InputFormatA>(2u));
    uninit<OperationUnpackTilize, Thread::TRISC0>();
    return 0;
}

// CHECK-NOT: llk::san

//--- uninit_operand_value_differs.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(32u));
    return 0;
}

// CHECK: UNINIT was given an Operand value that differs from the configured state
// CHECK: Current Kernel
// CHECK: llk::san::Operand<(llk::san::Exu)0>::FaceHeightA
// CHECK: Value: 16
// CHECK: Provided value: 32

//--- uninit_operand_field_unseated.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    init<OperationUnpackTilize, Thread::TRISC0>();
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    return 0;
}

// CHECK: UNINIT was given an Operand value that differs from the configured state
// CHECK: llk::san::Operand<(llk::san::Exu)0>::FaceHeightA
// CHECK: UNKNOWN (value never recorded?)
// CHECK: Provided value: 16

//--- uninit_operand_reconfigured.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    reconfigure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(32u));
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(32u));
    return 0;
}

// CHECK: UNINIT found the Operand state changed since init()
// CHECK: Operation initialized here

//--- uninit_operand_reconfigured_nullary.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    reconfigure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(32u));
    uninit<OperationUnpackTilize, Thread::TRISC0>();
    return 0;
}

// CHECK: UNINIT found the Operand state changed since init()
// CHECK: Operation initialized here

//--- uninit_after_init_operand_differs.cpp
#include "sanitizer/api.h"

using namespace llk::san;

namespace llk::san
{

static State thread_state {};

State* const state = &thread_state;

} // namespace llk::san

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(16u));
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::FaceHeightA>(32u));
    uninit<OperationUnpackTilize, Thread::TRISC0>();
    return 0;
}

// CHECK: UNINIT found the Operand state changed since init()
// CHECK: Operation initialized here
