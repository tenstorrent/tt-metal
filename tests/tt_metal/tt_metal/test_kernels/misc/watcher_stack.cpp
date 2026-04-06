// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

    // Do not scribble above stack pointer.
    if (sp > point)
        *point = 0;
}
