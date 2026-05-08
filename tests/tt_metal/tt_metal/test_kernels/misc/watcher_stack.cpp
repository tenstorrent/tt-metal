// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scribble on the stack to check stack usage detection.

#include "api/compile_time_args.h"
#include "internal/debug/stack_usage.h"

void kernel_main() {
    uint32_t usage = get_compile_time_arg_val (0);
    uint32_t* stack_base = get_stack_base();
    auto point = &stack_base[usage / sizeof(uint32_t)];
    uint32_t *sp;
    asm ("mv %0,sp" : "=r"(sp));

    // When usage == 0 we're testing full-overflow detection; write
    // unconditionally since SP may already be below the stack base
    // on cores with small stacks
    if (sp > point || usage == 0) {
        *point = 0;
    }
}
