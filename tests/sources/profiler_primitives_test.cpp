// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "profiler.h"

// Globals
uint32_t unp_cfg_context        = 0;
uint32_t pack_sync_tile_dst_ptr = 0;

#ifdef LLK_TRISC_UNPACK

void run_kernel()
{
    ZONE_SCOPED("TEST_ZONE")
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel()
{
    TIMESTAMP("TEST_TIMESTAMP")
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel()
{
    TIMESTAMP_DATA("TEST_TIMESTAMP_DATA", 0xBADC0FFE0DDF00D);
}

#endif
