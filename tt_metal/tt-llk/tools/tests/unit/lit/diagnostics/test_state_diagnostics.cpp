// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compile-time diagnostics tests for the llk::san state-tracking templates:
// StateField, StateVal and StateStruct, and for the registration they are held in --
// OperationList, ExuOperations and ExuState.
//
//     // expected-error@*:* {{StateStruct must have unique fields.}}
//
// @*:* matches a diagnostic at any line of any file, which is required because the
// static_assert fires inside the header, not in the case.
//
// clang fails a -verify run in BOTH directions, which is what makes these tests
// two-sided:
//
//   * a declared diagnostic that never appears -> "expected but not seen"
//     (a guard was weakened or deleted)
//   * any undeclared error that does appear    -> "seen but not expected"
//     (a guard fired but dragged a cascade behind it)
//
// Cascades are worth testing for because a failing static_assert does NOT stop
// instantiation. The compiler reports it and carries on through the rest of the class, so
// a carelessly placed guard yields the correct message buried under a dozen follow-on
// errors.
//
// split-file carves this file into one translation unit per case, each verified
// separately. That is what keeps the tests readable as a set while still isolating them:
// every part declares at most one expected error, so a diagnostic can only ever be
// attributed to the case that provoked it, no case can mask or satisfy another, and
// emission order never enters into it. Ordering would otherwise be a trap -- clang
// instantiates class templates eagerly at first use but defers function template bodies
// to the end of a translation unit, so in a single combined TU update() and equal() are
// reported after every later case.
//
// Each part declares only the fields it needs, rather than sharing a fixture. Most need
// none at all, and a reader should not have to look elsewhere to see what a case is built
// on. The templates come from sanitizer/types.h itself, reached through
// %{sanitizer_include} in the check line below; there is no test-local copy of them.
//
// Notes are exempted because clang emits a template instantiation backtrace for every
// static_assert, and its wording and depth vary by compiler version. Errors and warnings
// are still verified strictly.
//
// clang-only by construction: -verify and split-file are LLVM tools. That matches the
// kernel toolchain. Using clang here is not incidental -- at least one defect in this
// code was accepted by gcc and rejected by clang (a class-scope static_assert
// forward-referencing a member declared later in the same class), so verifying with gcc
// alone on the host would have let a hard kernel-build failure through.

// DEFINE: %{cflags} = -std=c++17 -fsyntax-only
// DEFINE: %{verify} = -Xclang -verify -Xclang -verify-ignore-unexpected=note
// DEFINE: %{check} = %clangxx %{cflags} %{verify} -I %{sanitizer_include}
// RUN: split-file %s %t
//
// RUN: %{check} %t/valid_usage.cpp
// RUN: %{check} %t/stateval_non_field_inert.cpp
// RUN: %{check} %t/statestruct_non_field.cpp
// RUN: %{check} %t/duplicate_fields.cpp
// RUN: %{check} %t/update_foreign_field.cpp
// RUN: %{check} %t/equal_foreign_field.cpp
// RUN: %{check} %t/non_trivially_copyable.cpp
// RUN: %{check} %t/contains_non_field.cpp
// RUN: %{check} %t/foreign_field_in_struct.cpp
// RUN: %{check} %t/distinct_same_type_fields.cpp
// RUN: %{check} %t/private_field_base.cpp
// RUN: %{check} %t/ambiguous_field_base.cpp
// RUN: %{check} %t/repeated_field_base.cpp
// RUN: %{check} %t/exu_state_valid.cpp
// RUN: %{check} %t/exu_operations_not_a_list.cpp
// RUN: %{check} %t/exu_operations_non_operation.cpp
// RUN: %{check} %t/exu_operations_duplicate_operation.cpp
// RUN: %{check} %t/exu_operations_wrong_exu.cpp

//--- valid_usage.cpp
// Correct use of the API must produce no diagnostics at all. This is the counterweight to
// the misuse cases: without it, a guard that rejected everything would still pass them.
#include <cstdint>

#include "sanitizer/types.h"

using namespace llk::san;

// A well-formed group declares its fields and its StateStruct in one place, and is an Operation
// derivation or an Operand specialization -- StateField admits no other Group, so a bare struct is
// not a group at all. Naming StateField<Alu, T> in the Field alias does not instantiate it, so Alu
// being incomplete at that point is fine.
struct Alu : Operation<Exu::Fpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Alu, T>;

    struct Fmt : Field<std::uint32_t>
    {
    };

    struct En : Field<bool>
    {
    };

    using Struct = StateStruct<Alu, Fmt, En>;
};

// A second group, so that its field is well formed but foreign to Alu::Struct.
struct Pck : Operation<Exu::Pack, Hoistable::Yes>
{
    template <typename T>
    using Field = StateField<Pck, T>;

    struct Fmt : Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<Pck, Fmt>;
};

// expected-no-diagnostics

int main()
{
    Alu::Struct state;

    state.update(StateVal<Alu::Fmt>(7u));
    state.update(StateVal<Alu::En>(true));

    const bool fmt_matches = state.equal(StateVal<Alu::Fmt>(7u));
    const bool en_matches  = state.equal(StateVal<Alu::En>(true));

    static_assert(Alu::Struct::contains<Alu::Fmt>());
    static_assert(Alu::Struct::contains<Alu::En>());
    static_assert(!Alu::Struct::contains<Pck::Fmt>());

    static_assert(Alu::Fmt::size() == 4 && Alu::Fmt::align() == 4);

    return (fmt_matches ? 1 : 0) + (en_matches ? 2 : 0);
}

//--- stateval_non_field_inert.cpp
// StateVal instantiated with something that is not a StateField is deliberately inert: it is
// diagnosed at the entry point it is handed to (see malformed_state_val in
// test_api_diagnostics.cpp), so on its own it must construct silently -- from a value AND by
// default, which pins the defaulted default constructor the swallow-anything template would
// otherwise suppress.
#include "sanitizer/types.h"

using namespace llk::san;

// expected-no-diagnostics

int main()
{
    StateVal<int> value {0};
    StateVal<int> empty {};
    (void)value;
    (void)empty;
    return 0;
}

//--- statestruct_non_field.cpp
// StateStruct declared over element types that are not StateFields.
#include "sanitizer/types.h"

using namespace llk::san;

struct Grp : Operation<Exu::Fpu, Hoistable::No>
{
};

// expected-error@*:* {{StateStruct only accepts StateField<Group, Type> types.}}

int main()
{
    StateStruct<Grp, int, float> state;
    (void)state;
    return 0;
}

//--- duplicate_fields.cpp
// The same StateField listed twice in one StateStruct.
#include "sanitizer/types.h"

using namespace llk::san;

struct Dup : Operation<Exu::Fpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Dup, T>;

    struct Value : Field<int>
    {
    };

    // The same field tag listed twice. A group cannot express this with two aliases of
    // StateField<Dup, int> -- those are one type, so the list would simply come out shorter.
    using Struct = StateStruct<Dup, Value, Value>;
};

using DupStruct = Dup::Struct;

// expected-error@*:* {{StateStruct must have unique fields.}}

int main()
{
    DupStruct state;
    (void)state;
    return 0;
}

//--- update_foreign_field.cpp
// update() called with a field belonging to a different group's StateStruct.
//
// This case and equal_foreign_field.cpp assert the same message. Keeping them in separate
// translation units is what makes them distinguishable: within one TU their directives
// would be interchangeable, and would still be satisfied if one guard fired twice and the
// other not at all.
#include "sanitizer/types.h"

using namespace llk::san;

struct Alu : Operation<Exu::Fpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Alu, T>;

    struct Value : Field<int>
    {
    };

    using Struct = StateStruct<Alu, Value>;
};

using AluStruct = Alu::Struct;

// Well formed in its own right, but not a member of AluStruct.
struct Pck : Operation<Exu::Pack, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Pck, T>;

    struct Value : Field<int>
    {
    };

    using Struct = StateStruct<Pck, Value>;
};

using PckStruct = Pck::Struct;

using PckField = Pck::Value;

// expected-error@*:* {{Field is not a member of this StateStruct.}}

int main()
{
    AluStruct state;
    state.update(StateVal<PckField>(1));
    return 0;
}

//--- equal_foreign_field.cpp
// equal() called with a field belonging to a different group's StateStruct.
#include "sanitizer/types.h"

using namespace llk::san;

struct Alu : Operation<Exu::Fpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Alu, T>;

    struct Value : Field<int>
    {
    };

    using Struct = StateStruct<Alu, Value>;
};

using AluStruct = Alu::Struct;

// Well formed in its own right, but not a member of AluStruct.
struct Pck : Operation<Exu::Pack, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Pck, T>;

    struct Value : Field<int>
    {
    };

    using Struct = StateStruct<Pck, Value>;
};

using PckStruct = Pck::Struct;

using PckField = Pck::Value;

// expected-error@*:* {{Field is not a member of this StateStruct.}}

int main()
{
    AluStruct state;
    return state.equal(StateVal<PckField>(1)) ? 1 : 0;
}

//--- non_trivially_copyable.cpp
// A StateField whose value type cannot be tracked by copy and compare.
#include "sanitizer/types.h"

using namespace llk::san;

#include <string>

struct Str : Operation<Exu::Sfpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Str, T>;

    struct Value : Field<std::string>
    {
    };

    using Struct = StateStruct<Str, Value>;
};

using StrStruct = Str::Struct;

// expected-error@*:* {{StateField type must be trivially copyable}}

int main()
{
    StrStruct state;
    (void)state;
    return 0;
}

//--- contains_non_field.cpp
// contains() queried with something that is not a StateField.
#include "sanitizer/types.h"

using namespace llk::san;

struct Alu : Operation<Exu::Fpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Alu, T>;

    struct Value : Field<int>
    {
    };

    using Struct = StateStruct<Alu, Value>;
};

using AluStruct = Alu::Struct;

// expected-error@*:* {{StateStruct::contains() only accepts StateField<Group, Type>}}

int main()
{
    return AluStruct::contains<int>() ? 1 : 0;
}

//--- foreign_field_in_struct.cpp
// A StateStruct listing a field that belongs to a different group.
//
// This is the declaration-side counterpart to update_foreign_field: that case catches a
// foreign field reaching update(), this one catches it being named as a member at all.
// Naming the group in StateStruct<Group, Fields...> is what makes the check possible --
// without it there is nothing to compare each field's group against.
#include "sanitizer/types.h"

using namespace llk::san;

struct Pck : Operation<Exu::Pack, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Pck, T>;

    struct Value : Field<int>
    {
    };

    using Struct = StateStruct<Pck, Value>;
};

using PckStruct = Pck::Struct;

// Well formed in its own right, but its group is Pck, not Alu.
using PckField = Pck::Value;

struct Alu : Operation<Exu::Fpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Alu, T>;

    struct Value : Field<int>
    {
    };

    using Struct = StateStruct<Alu, Value, PckField>;
};

using AluStruct = Alu::Struct;

// expected-error@*:* {{StateFields must all belong to StateStruct's Group.}}

int main()
{
    AluStruct state;
    (void)state;
    return 0;
}

//--- distinct_same_type_fields.cpp
// Several fields of the same Type in one group, declared as tag structs deriving from the
// group's Field alias. This is the form the sanitizer hooks are written in, and it must
// produce no diagnostics.
//
// It is also the case a bare alias cannot express: two fields declared as
// StateField<Alu, std::uint32_t> are one type, so a group could not hold both and the
// uniqueness guard would reject it. Deriving gives each field its own identity, which is
// why field detection matches a StateField base rather than the exact type.
#include <cstdint>

#include "sanitizer/types.h"

using namespace llk::san;

struct Alu : Operation<Exu::Fpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Alu, T>;

    struct InputFormat : Field<std::uint32_t>
    {
    };

    struct OutputFormat : Field<std::uint32_t>
    {
    };

    struct DestWidth32 : Field<bool>
    {
    };

    using Struct = StateStruct<Alu, InputFormat, OutputFormat, DestWidth32>;
};

// expected-no-diagnostics

int main()
{
    Alu::Struct state;

    state.update(StateVal<Alu::InputFormat>(3u));
    state.update(StateVal<Alu::OutputFormat>(5u));

    static_assert(Alu::Struct::contains<Alu::InputFormat>());
    static_assert(Alu::Struct::contains<Alu::OutputFormat>());
    static_assert(Alu::InputFormat::size() == 4 && Alu::DestWidth32::size() == sizeof(bool));

    // Same Type, tracked independently.
    return (state.equal(StateVal<Alu::InputFormat>(3u)) ? 1 : 0) + (state.equal(StateVal<Alu::OutputFormat>(3u)) ? 2 : 0);
}

//--- private_field_base.cpp
// A tag struct whose StateField base is private.
//
// This and the two cases below pin the failure mode of base detection itself. A tag with no
// StateField base anywhere is the easy half, already covered by statestruct_non_field; these
// are the half where the base exists but cannot be named unambiguously from outside, so
// deduction of G and T succeeds and only the pointer conversion fails. Detection has to
// report "not a field" for them, and it has to do so through the escape hatch rather than
// by letting the conversion error escape the header: an error phrased in terms of casts and
// base classes names neither the field nor the rule it broke, and its wording is the
// compiler's rather than ours. All three route through StateStruct, the declaration-side
// consumer of detection; StateVal is inert over a non-field and reports at the entry points.
#include <cstdint>

#include "sanitizer/types.h"

using namespace llk::san;

struct Alu : Operation<Exu::Fpu, Hoistable::No>
{
};

struct Hidden : private StateField<Alu, std::uint32_t>
{
};

// expected-error@*:* {{StateStruct only accepts StateField<Group, Type> types.}}

int main()
{
    StateStruct<Alu, Hidden> state;
    (void)state;
    return 0;
}

//--- ambiguous_field_base.cpp
// A tag struct with two different StateField bases, so neither Group nor Type is determined.
#include <cstdint>

#include "sanitizer/types.h"

using namespace llk::san;

struct Alu : Operation<Exu::Fpu, Hoistable::No>
{
};

struct Pck : Operation<Exu::Pack, Hoistable::Yes>
{
};

struct Both : StateField<Alu, std::uint32_t>, StateField<Pck, bool>
{
};

// expected-error@*:* {{StateStruct only accepts StateField<Group, Type> types.}}

int main()
{
    StateStruct<Alu, Both> state;
    (void)state;
    return 0;
}

//--- repeated_field_base.cpp
// One StateField base reached through two paths. Group and Type are unambiguous here, but
// the base itself is not, so the conversion detection relies on is still ill-formed.
//
// The intermediates are what keep this distinct from ambiguous_field_base: inheriting the
// same StateField twice directly would also trip -Winaccessible-base, and the point here is
// the ambiguity, not the warning.
#include <cstdint>

#include "sanitizer/types.h"

using namespace llk::san;

struct Alu : Operation<Exu::Fpu, Hoistable::No>
{
};

struct LeftField : StateField<Alu, std::uint32_t>
{
};

struct RightField : StateField<Alu, std::uint32_t>
{
};

struct Twice : LeftField, RightField
{
};

// expected-error@*:* {{StateStruct only accepts StateField<Group, Type> types.}}

int main()
{
    StateStruct<Alu, Twice> state;
    (void)state;
    return 0;
}

//--- exu_state_valid.cpp
// ExuState over a real Exu, naming no list: it reaches for its own through ExuOperations. This
// case includes sanitizer/operation.h rather than types.h alone, and needs to -- ExuState holds
// the Exu's operand StateStruct and the records of its registered operations, and the Operand and
// ExuOperations specializations that declare both live there.
//
// State itself is the point of this case. Three of the four Exus have an empty list today, so
// instantiating it exercises list_defect's folds over an empty pack, where anything but a binary
// fold would not be well-formed.
#include "sanitizer/operation.h"

using namespace llk::san;

// expected-no-diagnostics

int main()
{
    ExuState<Exu::Unpack> unpack;
    State state;
    (void)unpack;
    (void)state;
    return 0;
}

//--- exu_operations_not_a_list.cpp
// A registration naming something that is not an OperationList at all.
//
// This case and the three below include types.h alone and register their own operations, because
// operation.h already specializes ExuOperations for all four Exus and a translation unit may not
// specialize it twice. That is also why they are written against the registration rather than
// against ExuState: with the list no longer a template argument, a malformed registration is the
// only way left to express these mistakes, and OperationUnion is where they are caught.
//
// Naming OperationUnion's List member is what fires the check -- resolving it instantiates the
// class and its asserts. The alias is then left unused deliberately: using it would be a second
// error, because clang drops an alias whose type failed to instantiate.
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

//--- exu_operations_non_operation.cpp
// A registered list naming something that is not an Operation. Only the coarsest message may
// speak: the uniqueness and Exu rules, and the variant behind both, stay silent -- see
// detail::list_defect, which answers with the first defect rather than one bool per rule.
//
// The operation needs no Struct of its own. Every failing case here degrades to a variant over
// std::monostate alone, so nothing reaches OperationExtended to ask for one.
#include "sanitizer/types.h"

namespace llk::san
{

struct LocalUnpackOperation : Operation<Exu::Unpack, Hoistable::Yes>
{
};

template <>
struct ExuOperations<Exu::Unpack>
{
    using type = OperationList<LocalUnpackOperation, int>;
};

} // namespace llk::san

using namespace llk::san;

// expected-error@*:* {{ExuOperations may only list Operation<Exu, Hoistable> derivations.}}

using Registered = typename OperationUnion<Exu::Unpack>::List;

int main()
{
    return 0;
}

//--- exu_operations_duplicate_operation.cpp
// The same Operation registered twice. Worth its own case because std::variant accepts repeated
// alternatives without complaint and fails only later -- inside libstdc++, on any std::get<T> or
// holds_alternative<T> -- which is the failure mode this whole layer exists to keep out.
#include "sanitizer/types.h"

namespace llk::san
{

struct LocalUnpackOperation : Operation<Exu::Unpack, Hoistable::Yes>
{
};

template <>
struct ExuOperations<Exu::Unpack>
{
    using type = OperationList<LocalUnpackOperation, LocalUnpackOperation>;
};

} // namespace llk::san

using namespace llk::san;

// expected-error@*:* {{ExuOperations must not list the same Operation twice.}}

using Registered = typename OperationUnion<Exu::Unpack>::List;

int main()
{
    return 0;
}

//--- exu_operations_wrong_exu.cpp
// A well-formed Operation registered under an Exu that is not its own. The list is faultless in
// itself -- the mistake is the key it was filed under, which is why no rule about it could live in
// OperationList, and why the whole set of rules lives in OperationUnion instead.
//
// It also covers the mixed-Exu list, which has no rule of its own: a list whose entries disagree
// among themselves matches no single Exu, so it fails here too.
#include "sanitizer/types.h"

namespace llk::san
{

struct LocalUnpackOperation : Operation<Exu::Unpack, Hoistable::Yes>
{
};

template <>
struct ExuOperations<Exu::Pack>
{
    using type = OperationList<LocalUnpackOperation>;
};

} // namespace llk::san

using namespace llk::san;

// expected-error@*:* {{ExuOperations may only list Operations of its own Exu.}}

using Registered = typename OperationUnion<Exu::Pack>::List;

int main()
{
    return 0;
}
