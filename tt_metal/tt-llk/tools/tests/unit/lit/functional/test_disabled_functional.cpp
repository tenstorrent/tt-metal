// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DEFINE: %{cflags} = -std=c++17 -Wall -Wextra -Werror -DCOMPILE_FOR_TRISC=0
// RUN: %clangxx %{cflags} -I %{sanitizer_include} %s -o %t
// RUN: %t

#include <cstdint>

#include "sanitizer/api.h"

struct AbsentOperation;

static void unused_no_error(const std::uint32_t face_height)
{
    // When sanitizer is disabled, this is unused. The compilation shouldn't stop because of the unused variable
    const std::uint32_t doubled = face_height * 2u;

    SAN_HOOK(configure(StateVal<Operand<Exu::Unpack>::FaceHeightA>(doubled)));
}

int main()
{
    LLK_SAN_FUNCTION();
    LLK_SAN_SILENT_ZONE();

    SAN_HOOK(thread_init());

    SAN_HOOK(configure(StateDiscard<std::uint32_t>(1u)));
    SAN_HOOK(reconfigure(StateDiscard<std::uint32_t>(2u)));
    SAN_HOOK(init<AbsentOperation>(StateDiscard<std::uint32_t>(3u)));
    SAN_HOOK(execute<AbsentOperation>(StateDiscard<std::uint32_t>(4u)));
    SAN_HOOK(uninit<AbsentOperation>());
    SAN_HOOK(unsupported());

    unused_no_error(16u);

    return 0;
}
