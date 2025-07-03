// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scribble on the stack to check stack usage detection.

#include "compile_time_args.h"
#include <dev_mem_map.h>

extern uint32_t __stack_base[];

#if defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#else
void kernel_main() {
#endif
    uint32_t usage = get_compile_time_arg_val (0);
    auto point = &__stack_base[usage / sizeof(uint32_t)];
    uint32_t *sp;
    asm ("mv %0,sp" : "=r"(sp));

    // Do not scribble above stack pointer.
    if (sp > point)
        *point = 0;
}
#if defined(COMPILE_FOR_TRISC)
} // namespace NAMESPACE
#endif
