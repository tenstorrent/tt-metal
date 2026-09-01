// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compile-time diagnostics tests for two properties of the guards that are not about any single
// entry point, and so do not belong in test_api_diagnostics.cpp:
//
//   * the thread/Exu ownership table, asked per argument
//   * the layering, which holds one mistake to one diagnostic
//
// The table under test:
//
//     TRISC0  Unpack          TRISC2  Pack
//     TRISC1  Fpu, Sfpu       TRISC3  nothing -- unsupported
//
// TRISC1 owning two Exus is the reason ownership is asked per argument rather than per call: one
// configure() there legitimately carries Fpu and Sfpu fields together. Nothing else in the suite
// forces that shape, which is why the mixed case below is the important one.
//
// These cases all use the operand family. The operation family cannot exercise the table the same
// way, because it is the operation's own Exu that is checked and the only Exu with registered
// operations today is Unpack -- see UnpackOperations in sanitizer/operation.h. Registering an Fpu
// operation here is not an option either: ExuOperations is already specialized for every Exu in
// that header, and a test may not specialize it a second time. test_api_diagnostics.cpp covers the
// operation family's own layers instead.
//
// Groups come from sanitizer/operation.h rather than from fixtures. That is not incidental: an
// earlier version of this suite declared its own, and a fixture group is free to be shaped in a way
// no real group is -- one draft had them deriving from Operand<E>, which passed every case here
// while rejecting all four real Operand groups, because a full specialization inherits no exu()
// from the primary it replaces.
//
// See test_state_diagnostics.cpp for how -verify, split-file and @*:* work, and
// test_api_diagnostics.cpp for why LLK_SAN_ENABLE and COMPILE_FOR_TRISC are needed here.

// api.h reaches output.h, so the host build needs deps/host.h and libfmt.
// REQUIRES: fmt
// DEFINE: %{cflags} = -std=c++17 -fsyntax-only -DLLK_SAN_ENABLE -DCOMPILE_FOR_TRISC=0 -DLLK_SAN_SETTING_HOST_DEPS=1 -DDEBUG_PRINT_ENABLED %{fmt_flags}
// DEFINE: %{verify} = -Xclang -verify -Xclang -verify-ignore-unexpected=note
// DEFINE: %{check} = %clangxx %{cflags} %{verify} -I %{sanitizer_include}
// RUN: split-file %s %t
//
// RUN: %{check} %t/valid_trisc0.cpp
// RUN: %{check} %t/valid_trisc1_mixed.cpp
// RUN: %{check} %t/valid_trisc2.cpp
// RUN: %{check} %t/wrong_thread_mixed_args.cpp
// RUN: %{check} %t/two_rules_broken.cpp

//--- valid_trisc0.cpp
// The unpacker thread configuring its own Exu. This and the two cases below are the counterweight to
// the negative ones: without them, a table that owned nothing would pass every rejection case.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-no-diagnostics

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::InputFormatA>(7u), StateVal<Operand<Exu::Unpack>::NumFacesA>(4u));
    reconfigure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::InputFormatA>(9u));
    return 0;
}

//--- valid_trisc1_mixed.cpp
// The math thread drives two Exus, and one call may carry both. This is the case that forces
// ownership to be a per-argument question: a per-call check keyed on a single Exu could not express
// it, and it is the one place where "native to the thread" is genuinely wider than "one Exu".
#include "sanitizer/api.h"

using namespace llk::san;

// expected-no-diagnostics

int main()
{
    configure<Thread::TRISC1>(StateVal<Operand<Exu::Fpu>::Format>(1u), StateVal<Operand<Exu::Sfpu>::Format>(2u));
    return 0;
}

//--- valid_trisc2.cpp
#include "sanitizer/api.h"

using namespace llk::san;

// expected-no-diagnostics

int main()
{
    configure<Thread::TRISC2>(StateVal<Operand<Exu::Pack>::OutputFormat>(3u));
    return 0;
}

//--- wrong_thread_mixed_args.cpp
// One admissible argument and one not, in a single call, and still exactly one diagnostic.
//
// The layers are folds over the whole pack, so this pins down that the fold is a conjunction: an
// admissible argument must not satisfy the layer on behalf of an inadmissible one. It says nothing
// about how many times a message is emitted -- clang collapses repeated failures of the same
// static_assert, so two arguments wrong the same way would report once regardless. two_rules_broken
// is where the layering itself is tested.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() was given a field whose Exu this thread does not drive.}}

int main()
{
    configure<Thread::TRISC0>(StateVal<Operand<Exu::Unpack>::InputFormatA>(7u), StateVal<Operand<Exu::Pack>::OutputFormat>(3u));
    return 0;
}

//--- two_rules_broken.cpp
// A call that breaks two rules at once: one argument is not a StateVal at all, and the other is a
// field this thread does not drive. Exactly one message is expected -- the first broken argument's.
//
// This is the case that makes the per-argument chain load-bearing, and it fails loudly without it.
// Each argument yields at most one defect, and the entry point reports one defect per call, so the
// caller is never told their Operation fields belong elsewhere when they passed no Operation field
// at all -- a diagnostic must not describe a mistake that was not made.
#include "sanitizer/api.h"

using namespace llk::san;

// expected-error@*:* {{configure() only accepts StateVal<StateField<Group, Type>> and StateDiscard<Type> arguments.}}

int main()
{
    configure<Thread::TRISC0>(5, StateVal<Operand<Exu::Pack>::OutputFormat>(3u));
    return 0;
}
