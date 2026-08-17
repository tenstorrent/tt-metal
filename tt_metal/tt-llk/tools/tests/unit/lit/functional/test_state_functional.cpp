// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runtime behaviour of the llk::san state-tracking templates: StateField, StateVal and
// StateStruct.
//
// This is the counterweight to tests/unit/diagnostics/test_state_diagnostics.cpp. Those
// cases only prove that misuse is rejected; without a case that exercises correct use, a
// guard that rejected everything would still pass every one of them. Keeping it in a
// separate translation unit is forced rather than chosen: -verify rejects
// expected-no-diagnostics alongside any other directive, so the no-diagnostics case
// cannot share a file with the misuse cases.
//
// Compiling and running it, rather than only checking that it compiles, costs nothing
// here and covers what a syntax-only check cannot: that update() and equal() actually
// address the right tuple slot, and that a field reads back as unequal until it has been
// set. The templates come from sanitizer/types.h itself, reached through
// %{sanitizer_include}; there is no test-local copy of them.

// RUN: %clangxx -std=c++17 -Wall -Wextra -Werror -I %{sanitizer_include} %s -o %t
// RUN: %t

#include <cassert>
#include <cstdint>

#include "sanitizer/types.h"

using namespace llk::san;

// A well-formed group declares its fields and its StateStruct in one place, and is either an
// Operation derivation or an Operand specialization -- StateField admits nothing else, so a bare
// struct is not a group. Fields are tag structs deriving from the group's Field alias rather than
// aliases of StateField directly, because two aliases of the same StateField<G, T> would be one
// type and a group could not then hold two fields of the same Type.
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

static_assert(Alu::Struct::contains<Alu::Fmt>());
static_assert(Alu::Struct::contains<Alu::En>());
static_assert(!Alu::Struct::contains<Pck::Fmt>());

static_assert(Alu::Fmt::size() == sizeof(std::uint32_t));
static_assert(Alu::Fmt::align() == alignof(std::uint32_t));

int main()
{
    Alu::Struct state;

    // Nothing has been recorded yet, so no value compares equal -- not even the one a
    // default-constructed tuple happens to hold.
    assert(!state.equal(StateVal<Alu::Fmt>(0u)));
    assert(!state.equal(StateVal<Alu::En>(false)));

    state.update(StateVal<Alu::Fmt>(7u));

    assert(state.equal(StateVal<Alu::Fmt>(7u)));
    assert(!state.equal(StateVal<Alu::Fmt>(8u)));

    // Updating one field must not mark any other as known.
    assert(!state.equal(StateVal<Alu::En>(true)));
    assert(!state.equal(StateVal<Alu::En>(false)));

    state.update(StateVal<Alu::En>(true));

    assert(state.equal(StateVal<Alu::En>(true)));
    assert(!state.equal(StateVal<Alu::En>(false)));

    // ...and must not have disturbed the field set before it.
    assert(state.equal(StateVal<Alu::Fmt>(7u)));

    // Overwriting a known field tracks the new value, not the old.
    state.update(StateVal<Alu::Fmt>(9u));

    assert(state.equal(StateVal<Alu::Fmt>(9u)));
    assert(!state.equal(StateVal<Alu::Fmt>(7u)));

    Alu::Struct snap;
    Alu::Struct operand;

    assert(snap.subset_of(operand));

    operand.update(StateVal<Alu::Fmt>(7u));
    assert(snap.subset_of(operand));

    snap.update(StateVal<Alu::Fmt>(7u));
    assert(snap.subset_of(operand));

    snap.update(StateVal<Alu::En>(true));
    assert(!snap.subset_of(operand));

    operand.update(StateVal<Alu::En>(false));
    assert(!snap.subset_of(operand));

    operand.update(StateVal<Alu::En>(true));
    assert(snap.subset_of(operand));

    return 0;
}
