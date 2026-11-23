// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Math hangs in a RISC-V spin; the other threads finish. The launch times out (not an assert/ebreak).
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
