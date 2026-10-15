// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Does tensix_sync() on its own wait for a MOP that is still expanding, or does it need a
// mop_sync() in front of it?
//
// Each TRISC programs a MOP whose only instruction adds one to a GPR, runs it, synchronises in
// one of four ways (no sync, tensix_sync, mop_sync, mop_sync then tensix_sync), and then reads
// the GPR back from the RISC. If the sync returned while the MOP was still expanding, the read
// comes back short of the programmed number of increments.
//
// Two shapes of MOP are used: one long enough that the no-sync read is far short of the total
// (which shows the probe can see an unfinished MOP at all), and a single-increment MOP repeated
// many times with the counter checked after every run (which catches a sync that returns a few
// cycles early). Every thread runs the same probe on its own GPR file, MOP decoder and PC buffer.
//
// Result words per thread, at word offset 256 * thread (0 unpack, 1 math, 2 pack):
//   0       magic
//   1       increments programmed into the long MOP
//   2..5    long MOP: counter read after arm 0..3
//   6..9    long MOP: wall-clock cycles from the MOP run to the end of the sync, arm 0..3
//   10..13  short MOP: trials whose read did not see the increment, arm 0..3
//   14      short MOP trials per arm
// Arms: 0 = no sync, 1 = tensix_sync, 2 = mop_sync, 3 = mop_sync then tensix_sync.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_template.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{

using namespace ckernel;

constexpr std::uint32_t RESULT_MAGIC = 0x4D4F5053;

// No LLK runs on any thread in this test, so every GPR is free; 62 is used as the counter.
constexpr std::uint32_t COUNTER_GPR = 62;

// Loop lengths are 10-bit fields in the MOP configuration.
constexpr std::uint32_t LONG_MOP_OUTER      = 1000;
constexpr std::uint32_t LONG_MOP_INNER      = 1000;
constexpr std::uint32_t LONG_MOP_INCREMENTS = LONG_MOP_OUTER * LONG_MOP_INNER;
constexpr std::uint32_t SHORT_MOP_TRIALS    = 4096;

constexpr std::uint32_t WORDS_PER_THREAD = 256;
constexpr std::uint32_t NUM_ARMS         = 4;

// counter = counter + 1 on the scalar unit (operand B is the constant 1).
constexpr std::uint32_t OP_INCREMENT_COUNTER = TT_OP_ADDDMAREG(1, COUNTER_GPR, 1, COUNTER_GPR);

inline void sync_for_arm(const std::uint32_t arm)
{
    if (arm & 2)
    {
        mop_sync();
    }
    if (arm & 1)
    {
        tensix_sync();
    }
}

inline void reset_counter()
{
    // Drain whatever the previous arm left in flight, zero the counter, and make sure the zero has
    // landed before the next MOP is pushed.
    tensix_sync();
    TTI_SETDMAREG(0, 0, 0, LO_16(COUNTER_GPR));
    TTI_SETDMAREG(0, 0, 0, HI_16(COUNTER_GPR));
    tensix_sync();
}

inline std::uint32_t read_counter()
{
    return regfile[COUNTER_GPR];
}

inline void run_probe(volatile std::uint32_t* out)
{
    for (std::uint32_t i = 0; i < WORDS_PER_THREAD; i++)
    {
        out[i] = 0;
    }
    out[0]  = RESULT_MAGIC;
    out[1]  = LONG_MOP_INCREMENTS;
    out[14] = SHORT_MOP_TRIALS;

    ckernel_template long_mop(LONG_MOP_OUTER, LONG_MOP_INNER, OP_INCREMENT_COUNTER);
    long_mop.program();
    for (std::uint32_t arm = 0; arm < NUM_ARMS; arm++)
    {
        reset_counter();
        const std::uint64_t start = read_wall_clock();
        long_mop.run();
        sync_for_arm(arm);
        const std::uint64_t end = read_wall_clock();
        out[2 + arm]            = read_counter();
        out[6 + arm]            = static_cast<std::uint32_t>(end - start);
    }

    tensix_sync();
    ckernel_template short_mop(1, 1, OP_INCREMENT_COUNTER);
    short_mop.program();
    for (std::uint32_t arm = 0; arm < NUM_ARMS; arm++)
    {
        reset_counter();
        std::uint32_t misses = 0;
        for (std::uint32_t trial = 1; trial <= SHORT_MOP_TRIALS; trial++)
        {
            short_mop.run();
            sync_for_arm(arm);
            if (read_counter() != trial)
            {
                misses++;
            }
        }
        out[10 + arm] = misses;
    }
    tensix_sync();
}

} // namespace

#ifdef LLK_TRISC_UNPACK
constexpr std::uint32_t THREAD_RESULT_OFFSET = 0 * WORDS_PER_THREAD;
#endif
#ifdef LLK_TRISC_MATH
constexpr std::uint32_t THREAD_RESULT_OFFSET = 1 * WORDS_PER_THREAD;
#endif
#ifdef LLK_TRISC_PACK
constexpr std::uint32_t THREAD_RESULT_OFFSET = 2 * WORDS_PER_THREAD;
#endif

void run_kernel(RUNTIME_PARAMETERS params)
{
    volatile std::uint32_t* out = reinterpret_cast<volatile std::uint32_t*>(params.buffer_Res[0]) + THREAD_RESULT_OFFSET;
    run_probe(out);
}
