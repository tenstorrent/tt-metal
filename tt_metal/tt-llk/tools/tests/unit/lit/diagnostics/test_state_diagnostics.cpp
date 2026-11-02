// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DEFINE: %{cflags} = -std=c++17 -fsyntax-only
// DEFINE: %{verify} = -Xclang -verify -Xclang -verify-ignore-unexpected=note
// DEFINE: %{check} = %clangxx %{cflags} %{verify} -I %{sanitizer_include}
// RUN: split-file %s %t
//
// RUN: %{check} %t/struct_member_not_field.cpp
// RUN: %{check} %t/struct_member_not_trivial.cpp
// RUN: %{check} %t/struct_member_foreign_field.cpp
// RUN: %{check} %t/struct_member_field_duplicate.cpp
// RUN: %{check} %t/struct_update_foreign_field.cpp
// RUN: %{check} %t/struct_equal_foreign_field.cpp
// RUN: %{check} %t/struct_contains_not_field.cpp
// RUN: %{check} %t/field_base_private.cpp
// RUN: %{check} %t/field_base_ambiguous.cpp
// RUN: %{check} %t/field_base_repeated.cpp

//--- struct_member_not_field.cpp

#include "sanitizer/types.h"

using namespace llk::san;

struct Group : Operation<Exu::Fpu, Hoistable::No>
{
    using Struct = StateStruct<Group, int, int>;
};

// expected-error@*:* {{StateStruct only accepts StateField<Group, Type> types.}}

int main()
{
    [[maybe_unused]] Group::Struct state;
    return 0;
}

//--- struct_member_not_trivial.cpp
#include "sanitizer/types.h"

using namespace llk::san;

#include <string>

struct Group : Operation<Exu::Sfpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Group, T>;

    struct Value : Field<std::string>
    {
    };

    using Struct = StateStruct<Group, Value>;
};

// expected-error@*:* {{StateField type must be trivially copyable}}

int main()
{
    [[maybe_unused]] Group::Struct state;
    return 0;
}

//--- struct_member_field_duplicate.cpp

#include "sanitizer/types.h"

using namespace llk::san;

struct Group : Operation<Exu::Fpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Group, T>;

    struct DuplicateFirst : Field<int>
    {
    };

    struct DuplicateSecond : Field<int>
    {
    };

    using Struct = StateStruct<Group, DuplicateFirst, DuplicateFirst, DuplicateSecond, DuplicateSecond>;
};

// expected-error@*:* {{StateStruct must have unique fields.}}

int main()
{
    [[maybe_unused]] Group::Struct state;
    return 0;
}

//--- struct_update_foreign_field.cpp

#include "sanitizer/types.h"

using namespace llk::san;

struct Group : Operation<Exu::Unpack, Hoistable::No>
{
    using Struct = StateStruct<Group>;
};

struct Foreign : Operation<Exu::Unpack, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Foreign, T>;

    struct Other : Field<int>
    {
    };

    using Struct = StateStruct<Foreign, Other>;
};

// expected-error@*:* {{Field is not a member of this StateStruct.}}

int main()
{
    Group::Struct state;
    state.update(StateVal<Foreign::Other>(1));
    return 0;
}

//--- struct_equal_foreign_field.cpp

#include "sanitizer/types.h"

using namespace llk::san;

struct Group : Operation<Exu::Unpack, Hoistable::No>
{
    using Struct = StateStruct<Group>;
};

struct Foreign : Operation<Exu::Pack, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Foreign, T>;

    struct Other : Field<int>
    {
    };

    using Struct = StateStruct<Foreign, Other>;
};

// expected-error@*:* {{Field is not a member of this StateStruct.}}

int main()
{
    Group::Struct state;
    state.equal(StateVal<Foreign::Other>(1));
    return 0;
}

//--- struct_contains_not_field.cpp

#include "sanitizer/types.h"

using namespace llk::san;

struct Group : Operation<Exu::Fpu, Hoistable::No>
{
    using Struct = StateStruct<Group>;
};

// expected-error@*:* {{StateStruct::contains() only accepts StateField<Group, Type>}}

int main()
{
    Group::Struct::contains<int>();
    return 0;
}

//--- struct_member_foreign_field.cpp

#include "sanitizer/types.h"

using namespace llk::san;

struct Foreign : Operation<Exu::Unpack, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Foreign, T>;

    struct Value : Field<int>
    {
    };

    using Struct = StateStruct<Foreign, Value>;
};

struct Group : Operation<Exu::Unpack, Hoistable::No>
{
    using Struct = StateStruct<Group, Foreign::Value>;
};

// expected-error@*:* {{StateFields must all belong to StateStruct's Group.}}

int main()
{
    [[maybe_unused]] Group::Struct state;
    return 0;
}

//--- field_base_private.cpp
#include <cstdint>

#include "sanitizer/types.h"

using namespace llk::san;

struct Group : Operation<Exu::Fpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Group, T>;

    struct Hidden : private Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<Group, Hidden>;
};

// expected-error@*:* {{StateStruct only accepts StateField<Group, Type> types.}}

int main()
{
    [[maybe_unused]] Group::Struct state;
    return 0;
}

//--- field_base_ambiguous.cpp
#include <cstdint>

#include "sanitizer/types.h"

using namespace llk::san;

struct Foreign : Operation<Exu::Pack, Hoistable::Yes>
{
};

struct Group : Operation<Exu::Fpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Group, T>;

    struct Both : Field<std::uint32_t>, StateField<Foreign, bool>
    {
    };

    using Struct = StateStruct<Group, Both>;
};

// expected-error@*:* {{StateStruct only accepts StateField<Group, Type> types.}}

int main()
{
    [[maybe_unused]] Group::Struct state;
    return 0;
}

//--- field_base_repeated.cpp
#include <cstdint>

#include "sanitizer/types.h"

using namespace llk::san;

struct Group : Operation<Exu::Fpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Group, T>;

    struct LeftField : Field<std::uint32_t>
    {
    };

    struct RightField : Field<std::uint32_t>
    {
    };

    struct Twice : LeftField, RightField
    {
    };

    using Struct = StateStruct<Group, Twice>;
};

// expected-error@*:* {{StateStruct only accepts StateField<Group, Type> types.}}

int main()
{
    [[maybe_unused]] Group::Struct state;
    return 0;
}
