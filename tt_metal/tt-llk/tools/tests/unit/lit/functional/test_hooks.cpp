// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The disabled build -- LLK_SAN_ENABLE off. Nothing else in the suite compiles this branch, and it
// is the one every kernel that does not opt in is built with.
//
// Two things are checked. The five entry points, thread_init() and both zone macros still exist and
// still accept what the enabled build accepts, so firmware and the LLK can call them unguarded. And
// SAN_HOOK still passes its arguments, so a value a hook site names is read here too -- which is what
// keeps -Wall -Wextra -Werror, the flags below, from failing a build over a local nothing else uses.

// DEFINE: %{cflags} = -std=c++17 -Wall -Wextra -Werror -DCOMPILE_FOR_TRISC=0
// RUN: %clangxx %{cflags} -I %{sanitizer_include} %s -o %t
// RUN: %t

#include <cstdint>

#include "sanitizer/api.h"

using namespace llk::san;

// The entry points constrain neither the Operation nor the field, so an Operation that was never
// declared is still something a disabled build accepts.
struct AbsentOperation;

// A hook site as the LLK writes it. api.h reaches operation.h here as well, so the Operand
// specializations and Operation types these name are the real ones.
static void hook_sites()
{
    SAN_HOOK(init<OperationUnpackTilize>(StateVal<OperationUnpackTilize::BlockCtDim>(2u)));
    SAN_HOOK(execute<OperationPack>(StateVal<Operand<Exu::Pack>::FaceHeight>(16u)));
}

// Why SAN_HOOK passes its arguments: nothing but the hook site reads this local, so dropping the
// tokens would leave it unused and -Werror would fail the build.
static void hook_only_local(const std::uint32_t face_height)
{
    const std::uint32_t doubled = face_height * 2u;

    SAN_HOOK(configure(StateVal<Operand<Exu::Unpack>::FaceHeightA>(doubled)));
}

int main()
{
    LLK_SAN_FUNCTION();
    LLK_SAN_SILENT_ZONE();

    thread_init();

    configure(StateDiscard<std::uint32_t>(1u));
    reconfigure(StateDiscard<std::uint32_t>(2u));
    init<AbsentOperation>(StateDiscard<std::uint32_t>(3u));
    execute<AbsentOperation>(StateDiscard<std::uint32_t>(4u));
    uninit<AbsentOperation>();

    hook_sites();
    hook_only_local(16u);

    return 0;
}
