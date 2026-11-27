// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Build-only driver: instantiates mac_init under ENABLE_GATHERING so test_gathering_config.py can
// inspect its replay record in the math ELF (the harness has no mac test; ttnn drives the kernel).

#ifndef ENABLE_GATHERING
#define ENABLE_GATHERING
#endif

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
    ckernel::sfpu::mac_init<false /*APPROXIMATE*/, false /*is_fp32_dest_acc_en*/, DataFormat::Float16_b>();
    // fp32 Dest: the five-instruction body, recorded into the same slot.
    ckernel::sfpu::mac_init<false /*APPROXIMATE*/, true /*is_fp32_dest_acc_en*/, DataFormat::Float32>();
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
}

#endif
