// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Experiment only: math hangs on purpose (spin), unpack and pack finish.
#include <cstdint>

#include "ckernel.h"

struct RuntimeParams
{
};

#if defined(LLK_TRISC_MATH)
void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
{
#if 0
    // An FPU op with no SrcA/SrcB valid from unpack: the math Tensix thread stalls, and trisc.cpp's tensix_sync never returns
    TTI_ELWADD(0, 0, 0, 0, 0);
#else
    for (;;)
    {
        asm volatile("nop");
    }
#endif
}
#else
void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
{
}
#endif
