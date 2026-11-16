// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DEFINE: %{cflags} = -std=c++17 -Wall -Wextra -Werror -DCOMPILE_FOR_TRISC=0
// RUN: %clangxx %{cflags} -I %{sanitizer_include} %s -o %t
// RUN: %t

#include <cstdint>

#include "sanitizer/api.h"

struct AbsentOperation;

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

    return 0;
}
