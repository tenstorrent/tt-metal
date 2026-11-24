// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Math hangs in a Tensix stall (an FPU op with no SrcA/SrcB valid, so tensix_sync never returns); the other threads finish.
#include <cstdint>

#include "ckernel.h"

struct RuntimeParams
{
};

void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
{
#if defined(LLK_TRISC_MATH)
    TTI_ELWADD(0, 0, 0, 0, 0);
#endif
}
