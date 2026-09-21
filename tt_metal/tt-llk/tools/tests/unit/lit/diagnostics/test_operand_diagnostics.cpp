// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: fmt
// DEFINE: %{cflags} = -std=c++17 -fsyntax-only -DLLK_SAN_ENABLE -DCOMPILE_FOR_TRISC=0 -DLLK_SAN_SETTING_HOST_DEPS=1 -DDEBUG_PRINT_ENABLED %{fmt_flags}
// DEFINE: %{verify} = -Xclang -verify -Xclang -verify-ignore-unexpected=note
// DEFINE: %{check} = %clangxx %{cflags} %{verify} -I %{sanitizer_include}
// RUN: split-file %s %t
//
// RUN: %{check} %t/thread_unsupported.cpp
// RUN: %{check} %t/argument_not_value.cpp
// RUN: %{check} %t/argument_value_malformed_converting.cpp
// RUN: %{check} %t/argument_value_malformed_defaulted.cpp
// RUN: %{check} %t/argument_two_defects.cpp
// RUN: %{check} %t/argument_value_operation.cpp
// RUN: %{check} %t/configure_value_foreign_operand.cpp
// RUN: %{check} %t/configure_value_foreign_operand_mixed.cpp

//--- thread_unsupported.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() is not supported on TRISC3.}}

int main()
{
    configure<Thread::TRISC3>(StateVal<Operand<Exu::Unpack>::InputFormatA>(1u));
    return 0;
}

//--- argument_not_value.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.}}

int main()
{
    configure<Thread::TRISC0>(8u);
    return 0;
}

//--- argument_value_malformed_converting.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.}}

int main()
{
    configure<Thread::TRISC0>(StateVal<int> {0});
    return 0;
}

//--- argument_value_malformed_defaulted.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.}}

int main()
{
    configure<Thread::TRISC0>(StateVal<int> {});
    return 0;
}

//--- argument_two_defects.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.}}

int main()
{
    configure<Thread::TRISC0>(5, StateVal<Operand<Exu::Pack>::OutputFormat>(3u));
    return 0;
}

//--- argument_value_operation.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() only accepts Operand fields; Operation fields belong to init(), execute() and uninit().}}

int main()
{
    configure<Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u));
    return 0;
}

//--- configure_value_foreign_operand.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() was given a field whose Exu this thread does not drive.}}

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Pack>::OutputFormat>(3u));
    return 0;
}

//--- configure_value_foreign_operand_mixed.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() was given a field whose Exu this thread does not drive.}}

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::InputFormatA>(7u), StateVal<Operand<Exu::Pack>::OutputFormat>(3u));
    return 0;
}
