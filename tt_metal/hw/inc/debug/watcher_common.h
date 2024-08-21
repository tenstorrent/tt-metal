// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_msgs.h"

#if defined(WATCHER_ENABLED)

#if defined(COMPILE_FOR_ERISC)
// Forward declare these functions to avoid including erisc.h -> circular dependency via noc_nonblocking_api.h, sanitize_noc.h.
namespace internal_ {
void __attribute__((section("code_l1"))) risc_context_switch();
void disable_erisc_app();
}
extern "C" void erisc_early_exit(std::int32_t stack_save_addr);
#endif

inline uint32_t debug_get_which_riscv()
{
#if defined(COMPILE_FOR_BRISC)
    return DebugBrisc;
#elif defined(COMPILE_FOR_NCRISC)
    return DebugNCrisc;
#elif defined(COMPILE_FOR_ERISC)
    return DebugErisc;
#elif defined(COMPILE_FOR_IDLE_ERISC)
    return DebugIErisc;
#else
    return DebugTrisc0 + COMPILE_FOR_TRISC;
#endif
}


#endif // WATCHER_ENABLED
