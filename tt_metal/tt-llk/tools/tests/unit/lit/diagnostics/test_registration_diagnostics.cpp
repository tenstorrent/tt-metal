// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DEFINE: %{cflags} = -std=c++17 -fsyntax-only
// DEFINE: %{verify} = -Xclang -verify -Xclang -verify-ignore-unexpected=note
// DEFINE: %{check} = %clangxx %{cflags} %{verify} -I %{sanitizer_include}
// RUN: split-file %s %t
//
// RUN: %{check} %t/exu_state_valid.cpp
// RUN: %{check} %t/exu_operations_not_list.cpp
// RUN: %{check} %t/exu_operations_member_not_operation.cpp
// RUN: %{check} %t/exu_operations_member_duplicate.cpp
// RUN: %{check} %t/exu_operations_member_foreign_exu.cpp

//--- exu_state_valid.cpp
#include "sanitizer/operation.h"

using namespace llk::san;

// expected-no-diagnostics

int main()
{
    [[maybe_unused]] ExuState<Exu::Unpack> unpack;
    [[maybe_unused]] State state;
    return 0;
}

//--- exu_operations_not_list.cpp
#include "sanitizer/types.h"

namespace llk::san
{

template <>
struct ExuOperations<Exu::Unpack>
{
    using type = int;
};

} // namespace llk::san

using namespace llk::san;

// expected-error@*:* {{ExuOperations<E>::type must be an OperationList<Ops...>.}}

using Registered = typename OperationUnion<Exu::Unpack>::List;

int main()
{
    return 0;
}

//--- exu_operations_member_not_operation.cpp
#include "sanitizer/types.h"

namespace llk::san
{

struct Registered : Operation<Exu::Unpack, Hoistable::Yes>
{
};

template <>
struct ExuOperations<Exu::Unpack>
{
    using type = OperationList<Registered, int>;
};

} // namespace llk::san

using namespace llk::san;

// expected-error@*:* {{ExuOperations may only list Operation<Exu, Hoistable> derivations.}}

using Registered = typename OperationUnion<Exu::Unpack>::List;

int main()
{
    return 0;
}

//--- exu_operations_member_duplicate.cpp
#include "sanitizer/types.h"

namespace llk::san
{

struct Registered : Operation<Exu::Unpack, Hoistable::Yes>
{
};

template <>
struct ExuOperations<Exu::Unpack>
{
    using type = OperationList<Registered, Registered>;
};

} // namespace llk::san

using namespace llk::san;

// expected-error@*:* {{ExuOperations must not list the same Operation twice.}}

using Registered = typename OperationUnion<Exu::Unpack>::List;

int main()
{
    return 0;
}

//--- exu_operations_member_foreign_exu.cpp
#include "sanitizer/types.h"

namespace llk::san
{

struct Foreign : Operation<Exu::Unpack, Hoistable::Yes>
{
};

template <>
struct ExuOperations<Exu::Pack>
{
    using type = OperationList<Foreign>;
};

} // namespace llk::san

using namespace llk::san;

// expected-error@*:* {{ExuOperations may only list Operations of its own Exu.}}

using Registered = typename OperationUnion<Exu::Pack>::List;

int main()
{
    return 0;
}
