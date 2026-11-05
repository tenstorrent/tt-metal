// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Every thread finishes: shows that a core is usable again after a hang.
#include <cstdint>

#include "ckernel.h"

struct RuntimeParams
{
};

void run_kernel([[maybe_unused]] const struct RuntimeParams& params)
{
#if defined(LLK_TRISC_MATH)

#endif
}
