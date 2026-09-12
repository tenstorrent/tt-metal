// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expected-no-diagnostics alongside any other directive, so the no-diagnostics case

// RUN: %clangxx -std=c++17 -Wall -Wextra -Werror -I %{sanitizer_include} %s -o %t
// RUN: %t

#include <cassert>
#include <cstdint>

#include "sanitizer/types.h"

using namespace llk::san;

struct Group : Operation<Exu::Fpu, Hoistable::No>
{
    template <typename T>
    using Field = StateField<Group, T>;

    struct Fmt : Field<std::uint32_t>
    {
    };

    struct En : Field<bool>
    {
    };

    struct Fmt2 : Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<Group, Fmt, En, Fmt2>;
};

struct Foreign : Operation<Exu::Pack, Hoistable::Yes>
{
    template <typename T>
    using Field = StateField<Foreign, T>;

    struct Fmt : Field<std::uint32_t>
    {
    };

    using Struct = StateStruct<Foreign, Fmt>;
};

static_assert(Group::Struct::contains<Group::Fmt>());
static_assert(Group::Struct::contains<Group::En>());
static_assert(Group::Struct::contains<Group::Fmt2>());
static_assert(!Group::Struct::contains<Foreign::Fmt>());

static_assert(Group::Fmt::size() == sizeof(std::uint32_t));
static_assert(Group::Fmt::align() == alignof(std::uint32_t));

int main()
{
    Group::Struct state;

    assert(!state.equal(StateVal<Group::Fmt>(0u)));
    assert(!state.equal(StateVal<Group::En>(false)));

    state.update(StateVal<Group::Fmt>(7u));

    assert(state.equal(StateVal<Group::Fmt>(7u)));
    assert(!state.equal(StateVal<Group::Fmt>(8u)));

    assert(!state.equal(StateVal<Group::En>(true)));
    assert(!state.equal(StateVal<Group::En>(false)));

    state.update(StateVal<Group::En>(true));

    assert(state.equal(StateVal<Group::En>(true)));
    assert(!state.equal(StateVal<Group::En>(false)));

    assert(state.equal(StateVal<Group::Fmt>(7u)));

    state.update(StateVal<Group::Fmt>(9u));

    assert(state.equal(StateVal<Group::Fmt>(9u)));
    assert(!state.equal(StateVal<Group::Fmt>(7u)));

    assert(!state.equal(StateVal<Group::Fmt2>(9u)));

    state.update(StateVal<Group::Fmt2>(11u));

    assert(state.equal(StateVal<Group::Fmt2>(11u)));
    assert(state.equal(StateVal<Group::Fmt>(9u)));
    assert(!state.equal(StateVal<Group::Fmt>(11u)));
    assert(!state.equal(StateVal<Group::Fmt2>(9u)));

    Group::Struct snap;
    Group::Struct operand;

    assert(snap.is_subset(operand));

    operand.update(StateVal<Group::Fmt>(7u));
    assert(snap.is_subset(operand));

    snap.update(StateVal<Group::Fmt>(7u));
    assert(snap.is_subset(operand));

    snap.update(StateVal<Group::En>(true));
    assert(!snap.is_subset(operand));

    operand.update(StateVal<Group::En>(false));
    assert(!snap.is_subset(operand));

    operand.update(StateVal<Group::En>(true));
    assert(snap.is_subset(operand));

    return 0;
}
