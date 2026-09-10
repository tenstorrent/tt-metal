// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: fmt
// DEFINE: %{cflags} = -std=c++17 -fsyntax-only -DLLK_SAN_ENABLE -DCOMPILE_FOR_TRISC=0 -DLLK_SAN_SETTING_HOST_DEPS=1 -DDEBUG_PRINT_ENABLED %{fmt_flags}
// DEFINE: %{verify} = -Xclang -verify -Xclang -verify-ignore-unexpected=note
// DEFINE: %{check} = %clangxx %{cflags} %{verify} -I %{sanitizer_include}
// RUN: split-file %s %t
//
// RUN: %{check} %t/init_thread_unsupported.cpp
// RUN: %{check} %t/init_not_operation.cpp
// RUN: %{check} %t/init_operation_foreign_exu.cpp
// RUN: %{check} %t/init_operation_not_registered.cpp
// RUN: %{check} %t/init_argument_not_value.cpp
// RUN: %{check} %t/init_argument_operation_missmatch.cpp
// RUN: %{check} %t/init_argument_operand_foreign_exu.cpp
//
// RUN: %{check} %t/uninit_argument_operation_missmatch.cpp

//--- init_thread_unsupported.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{init() is not supported on TRISC3.}}

int main()
{
    init<OperationUnpackTilize, Thread::TRISC3>();
    return 0;
}

//--- init_not_operation.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{init() requires an Operation as its first template argument.}}

int main()
{
    init<int, Thread::TRISC0>();
    return 0;
}

//--- init_operation_foreign_exu.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{init() was given an Operation whose Exu this thread does not drive.}}

int main()
{
    init<OperationUnpackTilize, Thread::TRISC2>();
    return 0;
}

//--- init_operation_not_registered.cpp
#include <cstdint>

#include "sanitizer/api.h"

using namespace llk::san;

struct Unlisted : Operation<Exu::Unpack, Hoistable::Yes>
{
    template <typename T>
    using Field = StateField<Unlisted, T>;

    struct Value : Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<Unlisted, Value>;
};

// expected-error@*:* {{init() was given an Operation that its Exu's OperationList does not name.}}

int main()
{
    init<Unlisted, Thread::TRISC0>(StateVal<Unlisted::Value>(0u));
    return 0;
}

//--- init_argument_not_value.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{init() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.}}

int main()
{
    init<OperationUnpackUnary, Thread::TRISC0>(0u);
    return 0;
}

//--- init_argument_operation_missmatch.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{init() only accepts its own Operation's fields.}}

int main()
{
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackUnary::BroadcastType>(0u));
    return 0;
}

//--- init_argument_operand_foreign_exu.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{init() only accepts Operand fields of its own Operation's Exu.}}

int main()
{
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Pack>::OutputFormat>(3u));
    return 0;
}

//--- uninit_argument_operation_missmatch.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{uninit() only accepts its own Operation's fields.}}

int main()
{
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackUnary::BroadcastType>(0u));
    return 0;
}
