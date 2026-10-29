// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// tensix_sync() on its own must wait for a MOP that is still expanding. Each TRISC runs a MOP whose
// only instruction increments a GPR, with or without a tensix_sync() behind it, and reads the GPR
// back from the RISC: a read short of the programmed increments means the sync returned early.
//
// Result words per thread, at word offset 256 * thread (0 unpack, 1 math, 2 pack):
//   0     magic
//   1     increments programmed into the long MOP
//   2, 3  long MOP: counter read with no sync, with tensix_sync
//   4, 5  short MOP: runs whose read missed an increment, with no sync, with tensix_sync
//   6     short MOP runs per arm
//   7     increments per short MOP run

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

constexpr std::uint32_t RESULT_MAGIC          = 0x4D4F5053;
constexpr std::uint32_t WORDS_PER_THREAD      = 256;
constexpr std::uint32_t LONG_INCREMENTS_WORD  = 1;
constexpr std::uint32_t LONG_COUNT_WORD       = 2;
constexpr std::uint32_t SHORT_MISS_WORD       = 4;
constexpr std::uint32_t SHORT_TRIALS_WORD     = 6;
constexpr std::uint32_t SHORT_INCREMENTS_WORD = 7;

// No LLK runs on any thread in this test, so every GPR is free.
constexpr std::uint32_t COUNTER_GPR = 62;

// Loop lengths are 10-bit fields in the MOP configuration.
constexpr std::uint32_t LONG_MOP_OUTER       = 1000;
constexpr std::uint32_t LONG_MOP_INNER       = 1000;
constexpr std::uint32_t LONG_MOP_INCREMENTS  = LONG_MOP_OUTER * LONG_MOP_INNER;
constexpr std::uint32_t SHORT_MOP_OUTER      = 1;
constexpr std::uint32_t SHORT_MOP_INNER      = 64;
constexpr std::uint32_t SHORT_MOP_INCREMENTS = SHORT_MOP_OUTER * SHORT_MOP_INNER;
constexpr std::uint32_t SHORT_MOP_TRIALS     = 4096;

constexpr std::uint32_t NO_SYNC     = 0;
constexpr std::uint32_t TENSIX_SYNC = 1;
constexpr std::uint32_t NUM_ARMS    = 2;

// counter = counter + 1 on the scalar unit: operand B is the immediate 1, not a register.
constexpr std::uint32_t OP_INCREMENT_COUNTER = TT_OP_ADDDMAREG(1 /*OpBisConst*/, COUNTER_GPR, 1 /*OpBRegIndex, the immediate*/, COUNTER_GPR);

inline void sync_for_arm(const std::uint32_t arm)
{
    if (arm == TENSIX_SYNC)
    {
        tensix_sync();
    }
}

inline void reset_counter()
{
    // Drain the previous arm, zero the counter, and let the zero land before the next MOP is pushed.
    tensix_sync();
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 0 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, LO_16(COUNTER_GPR));
    TTI_SETDMAREG(0 /*Payload_SigSelSize*/, 0 /*Payload_SigSel*/, 0 /*SetSignalsMode*/, HI_16(COUNTER_GPR));
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
    out[0]                     = RESULT_MAGIC;
    out[LONG_INCREMENTS_WORD]  = LONG_MOP_INCREMENTS;
    out[SHORT_TRIALS_WORD]     = SHORT_MOP_TRIALS;
    out[SHORT_INCREMENTS_WORD] = SHORT_MOP_INCREMENTS;

    ckernel_template long_mop(LONG_MOP_OUTER, LONG_MOP_INNER, OP_INCREMENT_COUNTER);
    long_mop.program();
    for (std::uint32_t arm = NO_SYNC; arm < NUM_ARMS; arm++)
    {
        reset_counter();
        long_mop.run();
        sync_for_arm(arm);
        out[LONG_COUNT_WORD + arm] = read_counter();
    }

    tensix_sync();
    ckernel_template short_mop(SHORT_MOP_OUTER, SHORT_MOP_INNER, OP_INCREMENT_COUNTER);
    short_mop.program();
    for (std::uint32_t arm = NO_SYNC; arm < NUM_ARMS; arm++)
    {
        reset_counter();
        std::uint32_t misses = 0;
        for (std::uint32_t trial = 1; trial <= SHORT_MOP_TRIALS; trial++)
        {
            short_mop.run();
            sync_for_arm(arm);
            if (read_counter() != trial * SHORT_MOP_INCREMENTS)
            {
                misses++;
            }
        }
        out[SHORT_MISS_WORD + arm] = misses;
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
