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

// Kept equal to SHORT_ITERATIONS and LONG_ITERATIONS in test_riscv_pmc_count.py.
constexpr std::uint32_t SHORT_ITERATIONS = 100;
constexpr std::uint32_t LONG_ITERATIONS  = 1100;

// One li, then ten instructions per iteration: the decrement, eight nops and the branch. Both counts fit li's single
// instruction form, so the two zones differ by exactly ten instructions per extra iteration.
template <std::uint32_t ITERATIONS>
inline __attribute__((always_inline)) void spin()
{
    static_assert(ITERATIONS > 0 && ITERATIONS < 2048);
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

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
    {
        START_PERF_MEASURE("INIT")
        spin<SHORT_ITERATIONS>();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        spin<LONG_ITERATIONS>();
    }
}
