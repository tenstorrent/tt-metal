// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

#ifdef LLK_TRISC_UNPACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
}

#endif

#ifdef LLK_TRISC_MATH

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
}

#endif

#ifdef LLK_TRISC_PACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
}
#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
}

#endif
