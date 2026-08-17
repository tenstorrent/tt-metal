// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Stub for SFPU TRISC when test does not use the 4-TRISC SFPU pipeline.
// Include in Quasar test sources.

#pragma once

#ifdef LLK_TRISC_ISOLATE_SFPU

// build.h uses p_unpacr::UNP_A etc; bring ckernel namespace into scope
using namespace ckernel;

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
    // Stub: SFPU TRISC not used in this test; trisc.cpp will signal completion
}

#endif
