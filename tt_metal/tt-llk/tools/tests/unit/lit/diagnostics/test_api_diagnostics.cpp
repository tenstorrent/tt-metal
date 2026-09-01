// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compile-time diagnostics tests for the five guarded entry points as a kernel writes them, using the
// real groups out of sanitizer/operation.h rather than fixtures. The rules under test:
//
//   every entry point            a StateVal for a tracked parameter, or a StateDiscard for one
//                                deliberately untracked, and nothing else
//   configure / reconfigure      Operand fields of an Exu this thread drives
//   init / execute / uninit      the named Operation's own fields, plus Operand fields of that
//                                Operation's Exu
//
// The Operation is a template argument, not inferred from the arguments. That is what lets a nullary
// uninit<Op>() still be checked, and it is why there is no "one call describes one operation" case
// here: with Op named, another operation's field is simply an argument that is not this one's.
//
// See test_state_diagnostics.cpp for how -verify, split-file and @*:* work, and why every part
// declares at most one expected error.
//
// Two things this suite needs that the state suites do not:
//
//   * LLK_SAN_ENABLE, because api.h compiles the entry points only under it. Without it they are
//     no-ops that accept anything, so a case would fail as "expected but not seen" and say nothing
//     about the guards.
//   * COMPILE_FOR_TRISC, which is what Thread T defaults to. Cases name their thread explicitly, so
//     the value is irrelevant -- it only has to exist, because the default argument is still parsed.
//
// The diagnostics themselves live in impl.h, not api.h. That is deliberate: api.h is arch specific,
// and what a thread may be handed is not, so the rules are stated once in the arch-independent layer
// and are identical for every target. api.h only supplies the thread and the state object.

// api.h reaches output.h, so the host build needs deps/host.h and libfmt.
// REQUIRES: fmt
// DEFINE: %{cflags} = -std=c++17 -fsyntax-only -DLLK_SAN_ENABLE -DCOMPILE_FOR_TRISC=0 -DLLK_SAN_SETTING_HOST_DEPS=1 -DDEBUG_PRINT_ENABLED %{fmt_flags}
// DEFINE: %{verify} = -Xclang -verify -Xclang -verify-ignore-unexpected=note
// DEFINE: %{check} = %clangxx %{cflags} %{verify} -I %{sanitizer_include}
// RUN: split-file %s %t
//
// RUN: %{check} %t/valid_operand.cpp
// RUN: %{check} %t/valid_operation.cpp
// RUN: %{check} %t/not_state_value.cpp
// RUN: %{check} %t/not_state_value_to_init.cpp
// RUN: %{check} %t/malformed_state_val.cpp
// RUN: %{check} %t/malformed_state_val_mixed.cpp
// RUN: %{check} %t/operation_field_to_configure.cpp
// RUN: %{check} %t/foreign_exu_to_configure.cpp
// RUN: %{check} %t/trisc3_configure.cpp
// RUN: %{check} %t/trisc3_nullary_uninit.cpp
// RUN: %{check} %t/non_operation_as_op.cpp
// RUN: %{check} %t/operation_on_wrong_thread.cpp
// RUN: %{check} %t/unregistered_operation.cpp
// RUN: %{check} %t/foreign_operation_field.cpp
// RUN: %{check} %t/foreign_operation_field_to_uninit.cpp
// RUN: %{check} %t/foreign_exu_operand_to_init.cpp

//--- valid_operand.cpp
// The operand family used correctly, including a discarded parameter. This and valid_operation are
// the counterweight to the rest: without them a guard that rejected everything would pass every
// negative case.
#include <cstdint>

#include "sanitizer/api.h"

using namespace llk::san;
using Unp = Operand<Exu::Unpack>;

// expected-no-diagnostics

int main()
{
    configure<Thread::TRISC0>(StateVal<Unp::InputFormatA>(1u), StateVal<Unp::FaceHeightA>(16u), StateDiscard<std::uint32_t>(7u));
    reconfigure<Thread::TRISC0>(StateVal<Unp::OutputFormatA>(2u));

    // A call carrying nothing but discards is still a call: an LLK function whose every parameter is
    // deliberately untracked has to be expressible.
    configure<Thread::TRISC0>(StateDiscard<bool>(false));
    return 0;
}

//--- valid_operation.cpp
// The operation family on its owning thread: the operation's own fields, an Operand field of that
// operation's Exu, a discard, and a nullary uninit.
//
// All three hooks take the same arguments, so the three calls below differ only in which one is being
// exercised. uninit() restating the operation's own fields is the case worth watching: it is what lets
// an operation torn down under different parameters than it was set up with be caught, and it is
// checked at runtime in tests/unit/functional/test_hook_functional.cpp.
//
// Nullary is worth covering on its own. Every per-argument layer is a fold, and a fold over an empty
// pack is vacuously true, so nothing but the thread and operation layers stand between uninit<Op>()
// and acceptance -- which is exactly why the operation is named rather than inferred.
#include <cstdint>

#include "sanitizer/api.h"

using namespace llk::san;
using Tilize = OperationUnpackTilize;
using Unp    = Operand<Exu::Unpack>;

// expected-no-diagnostics

int main()
{
    init<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u), StateVal<Tilize::NarrowTile>(false), StateVal<Unp::FaceHeightA>(16u));
    execute<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u), StateDiscard<std::uint32_t>(0u));
    uninit<Tilize, Thread::TRISC0>(StateVal<Tilize::BlockCtDim>(2u), StateVal<Unp::FaceHeightA>(16u), StateDiscard<std::uint32_t>(0u));
    uninit<Tilize, Thread::TRISC0>();
    return 0;
}

//--- not_state_value.cpp
// Something that is neither a StateVal nor a StateDiscard. Nothing else diagnoses this, so it can
// only be caught at the entry point.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.}}

int main()
{
    configure<Thread::TRISC0>(8u);
    return 0;
}

//--- malformed_state_val.cpp
// A StateVal over something that is not a StateField. The type itself is deliberately inert --
// see stateval_non_field_inert in test_state_diagnostics.cpp -- so the entry point is the single
// reporting site, and it rejects the argument at the same layer, with the same message, as a bare
// int: a StateVal over a non-field is simply not a StateVal<StateField<...>>.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.}}

int main()
{
    configure<Thread::TRISC0>(StateVal<int> {});
    return 0;
}

//--- malformed_state_val_mixed.cpp
// A malformed StateVal next to a well-formed field of a foreign Exu: two rules broken in one call,
// and still exactly one diagnostic -- the first broken argument's. A malformed StateVal is held to
// one defect the same way the bare int in two_rules_broken (test_thread_diagnostics.cpp) is.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.}}

int main()
{
    configure<Thread::TRISC0>(StateVal<int> {}, StateVal<Operand<Exu::Pack>::OutputFormat>(3u));
    return 0;
}

//--- operation_field_to_configure.cpp
// An Operation field handed to the operand family. The Exu is right and the thread drives it; only
// the family is wrong, so that is what the message has to say.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() only accepts Operand fields; Operation fields belong to init(), execute() and uninit().}}

int main()
{
    configure<Thread::TRISC0>(StateVal<OperationUnpackTilize::BlockCtDim>(2u));
    return 0;
}

//--- foreign_exu_to_configure.cpp
// A well-formed Operand field carried to a thread that does not drive its Exu.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() was given a field whose Exu this thread does not drive.}}

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Pack>::OutputFormat>(3u));
    return 0;
}

//--- trisc3_configure.cpp
// TRISC3 drives no Exu, so the thread is rejected before the arguments are judged. The argument here
// is well formed, which is the point: the message must be about the thread.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() is not supported on TRISC3.}}

int main()
{
    configure<Thread::TRISC3>(StateVal<Operand<Exu::Unpack>::InputFormatA>(1u));
    return 0;
}

//--- trisc3_nullary_uninit.cpp
// The same rejection with nothing to fold over, which is the case a per-argument ownership check
// cannot reach: a fold over an empty pack is vacuously true. Deriving thread support from the
// ownership table, rather than asking it per argument, is what closes this.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{uninit() is not supported on TRISC3.}}

int main()
{
    uninit<OperationUnpackTilize, Thread::TRISC3>();
    return 0;
}

//--- non_operation_as_op.cpp
// The template argument is not an Operation at all. Every later layer asks something about the
// operation's Exu, so this has to be caught first, and the questions behind those layers have to
// answer false here rather than fail to compile.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{init() requires an Operation as its first template argument.}}

int main()
{
    init<int, Thread::TRISC0>();
    return 0;
}

//--- operation_on_wrong_thread.cpp
// An Unpack operation initialized from the pack thread. Ownership is the same table the operand
// family consults; only the group-kind detection differs.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{init() was given an Operation whose Exu this thread does not drive.}}

int main()
{
    init<OperationUnpackTilize, Thread::TRISC2>();
    return 0;
}

//--- unregistered_operation.cpp
// An Operation on the right thread and right Exu, but absent from that Exu's OperationList.
//
// The registration layer exists entirely for this case, and it is the one place where being a layer
// short does visible damage. Without it the call reaches std::variant, which rejects an alternative
// it does not have -- several errors from inside libstdc++, phrased in terms of __exactly_once and
// alternatives, naming neither the operation nor the list it is missing from.
#include <cstdint>

#include "sanitizer/api.h"

using namespace llk::san;

// A well-formed Unpack operation that UnpackOperations does not name.
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

//--- foreign_operation_field.cpp
// Another operation's field passed to init<Op>. Both are registered Unpack operations on the owning
// thread, so every other layer passes.
//
// This is what replaced the old "one call describes one operation" layer. Naming the operation turns
// an ambiguity question into a membership question, and a membership question needs no extra layer.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{init() only accepts its own Operation's fields.}}

int main()
{
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackUnary::BroadcastType>(0u));
    return 0;
}

//--- not_state_value_to_init.cpp
// A bare value handed to the operation family. not_state_value.cpp covers the same rule for
// configure(), but the two families reach it through different chains and at different depths --
// the operand family asks about parameters second, the operation family fifth, after everything
// about the operation itself has held. Only a case per family proves the rule is wired to the right
// message in both, rather than to the kind rule that sits next to it.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{init() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.}}

int main()
{
    init<OperationUnpackUnary, Thread::TRISC0>(0u);
    return 0;
}

//--- foreign_operation_field_to_uninit.cpp
// The membership rule again, on the hook that used to admit no Operation field at all. uninit() takes
// the same arguments as init() and execute() -- see the uninit() calls in valid_operation.cpp -- so the
// only Operation field it rejects is another operation's, and it has to be rejected with its own
// hook's wording rather than init()'s.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{uninit() only accepts its own Operation's fields.}}

int main()
{
    uninit<OperationUnpackTilize, Thread::TRISC0>(StateVal<OperationUnpackUnary::BroadcastType>(0u));
    return 0;
}

//--- foreign_exu_operand_to_init.cpp
// An Operand field of an Exu other than the operation's own. On TRISC0 this is also a foreign Exu,
// but the rule is narrower than the thread's ownership on purpose: the operation's snapshot has room
// for one Exu's operand state, its own, so a Pack field could not be stored even on a thread that
// drove both.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{init() only accepts Operand fields of its own Operation's Exu.}}

int main()
{
    init<OperationUnpackTilize, Thread::TRISC0>(StateVal<Operand<Exu::Pack>::OutputFormat>(3u));
    return 0;
}
