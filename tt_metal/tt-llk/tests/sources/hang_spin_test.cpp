// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Math hangs in a RISC-V spin (what an LLK assert's ebreak looks like to the launch); the other threads finish.
#include <cstdint>

#include "ckernel.h"

struct RuntimeParams
{
};

void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
{
#if defined(LLK_TRISC_MATH)
    for (;;)
    {
        asm volatile("nop");
    }
#endif
}
