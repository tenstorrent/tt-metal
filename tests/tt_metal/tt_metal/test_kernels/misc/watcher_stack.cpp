// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scribble on the stack to check stack usage detection.

#include "compile_time_args.h"
#include <dev_mem_map.h>

static uint32_t get_stack_base() {
#if defined(COMPILE_FOR_BRISC)
    return MEM_BRISC_STACK_TOP - MEM_BRISC_STACK_SIZE;
#elif defined(COMPILE_FOR_NCRISC)
    return MEM_NCRISC_STACK_TOP - MEM_NCRISC_STACK_SIZE;
#elif defined(COMPILE_FOR_IDLE_ERISC)
#if COMPILE_FOR_IDLE_ERISC == 0
    return MEM_IERISC_STACK_TOP - MEM_IERISC_STACK_SIZE;
#elif COMPILE_FOR_IDLE_ERISC == 1
    return MEM_SLAVE_IERISC_STACK_TOP - MEM_SLAVE_IERISC_STACK_SIZE;
#else
#error "idle erisc get_stack_base unknown"
#endif
#elif defined(COMPILE_FOR_TRISC)
#if COMPILE_FOR_TRISC == 0
    return MEM_TRISC0_STACK_TOP - MEM_TRISC0_STACK_SIZE;
#elif COMPILE_FOR_TRISC == 1
    return MEM_TRISC1_STACK_TOP - MEM_TRISC1_STACK_SIZE;
#elif COMPILE_FOR_TRISC == 2
    return MEM_TRISC2_STACK_TOP - MEM_TRISC2_STACK_SIZE;
#else
#error "trisc get_stack_base unknown"
#endif
#else
#error "get_stack_base unknown"
#endif
}

#if defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#else
void kernel_main() {
#endif
    uint32_t usage = get_compile_time_arg_val (0);
    auto base = (uint32_t tt_l1_ptr *)get_stack_base();
    auto point = &base[usage/sizeof(uint32_t)];
    uint32_t *sp;
    asm ("mv %0,sp" : "=r"(sp));

    // Do not scribble above stack pointer.
    if (sp > point)
        *point = 0;
}
#if defined(COMPILE_FOR_TRISC)
} // namespace NAMESPACE
#endif
