// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// mac_init with ENABLE_GATHERING defined for the whole translation unit, as tt-metal builds
// kernels under TT_METAL_ENABLE_GATHERING=1. The harness has no mac test (ttnn drives the
// kernel directly), so this driver only instantiates the init that records the replay
// buffer. Build-only: test_gathering_config.py inspects the math ELF and never runs it.

#define ENABLE_GATHERING

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_sfpu/ckernel_sfpu_mac.h"

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
    // bf16 Dest: the six-instruction body (three SFPLOADs, SFPMAD, SFPSTOCHRND, SFPSTORE).
    ckernel::sfpu::mac_init<false, false, DataFormat::Float16_b>();
    // fp32 Dest: the five-instruction body, recorded into the same slot.
    ckernel::sfpu::mac_init<false, true, DataFormat::Float32>();
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
}

#endif
