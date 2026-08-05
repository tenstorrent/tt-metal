// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 host-API version of watcher_stack.cpp.
// Compiled only for TENSIX cores (BRISC / NCRISC / TRISC / DM). Idle-ethernet callers
// continue to use watcher_stack.cpp via the legacy host API.

#include "internal/debug/stack_usage.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t usage = get_arg(args::usage);
    uint32_t* stack_base = get_stack_base();
    auto point = &stack_base[usage / sizeof(uint32_t)];
    uint32_t* sp;
    asm("mv %0,sp" : "=r"(sp));

    // When usage == 0 we're testing full-overflow detection; write unconditionally
    // since SP may already be below the stack base on cores with small stacks.
    if (sp > point || usage == 0) {
        *point = 0;
    }
}
