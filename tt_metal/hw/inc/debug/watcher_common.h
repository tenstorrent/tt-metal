// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_msgs.h"

#if defined(WATCHER_ENABLED)

inline uint32_t debug_get_which_riscv()
{
#if defined(COMPILE_FOR_BRISC)
    return DebugBrisc;
#elif defined(COMPILE_FOR_NCRISC)
    return DebugNCrisc;
#elif defined(COMPILE_FOR_ERISC)
    return DebugErisc;
#else
    return DebugTrisc0 + COMPILE_FOR_TRISC;
#endif
}


#endif // WATCHER_ENABLED
