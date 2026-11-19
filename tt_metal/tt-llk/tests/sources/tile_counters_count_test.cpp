// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

using namespace ckernel;

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// About 20 ms of spinning with no NoC traffic of its own, so the host's polls are the only reads the tile serves.
constexpr std::uint32_t ITERATIONS = 2000000;

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
    START_PERF_MEASURE("TILE_LOOP")
    asm volatile(
        "li t0, %0\n"
        "1:\n"
        "addi t0, t0, -1\n"
        ".rept 8\n"
        "nop\n"
        ".endr\n"
        "bnez t0, 1b\n"
        :
        : "i"(ITERATIONS)
        : "t0", "memory");
}
