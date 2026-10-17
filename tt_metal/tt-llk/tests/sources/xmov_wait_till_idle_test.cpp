// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for `xmov_wait_till_idle`: once it returns, the move that was issued just before
// it has been carried out, so the destination can be read straight away.
//
// Each iteration PACK writes a fresh pattern to the source buffer, issues an L1-to-L1 move with
// `xmov_l1_to_l1_non_compact`, calls `xmov_wait_till_idle`, and then reads the destination back
// (last word first, since the mover writes it last) and compares it with the pattern. An iteration
// in which any destination word does not yet hold the pattern of this iteration is counted as
// stale.
//
// The source is read back after it is written, so the pattern is in L1 before the move is issued
// and a stale destination cannot be blamed on the source. The RISC data cache is left as the
// harness booted it; the wait itself drops it on Blackhole.

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_xmov.h"

using namespace ckernel;

// Globals the harness links against.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
constexpr std::uint32_t WORDS_PER_16B = 4;
constexpr std::uint32_t WORDS         = XMOV_XFER_16B * WORDS_PER_16B;
constexpr std::uint32_t NO_STALE      = 0xFFFFFFFFu;

// Pattern constants: odd multipliers, so consecutive iterations differ in every word.
constexpr std::uint32_t ITERATION_STRIDE = 0x9E3779B1u;
constexpr std::uint32_t WORD_STRIDE      = 0x01000193u;
constexpr std::uint32_t PATTERN_SEED     = 0xA5A5A5A5u;

constexpr std::uint32_t pattern(const std::uint32_t iteration, const std::uint32_t word)
{
    return (iteration * ITERATION_STRIDE) ^ (word * WORD_STRIDE) ^ PATTERN_SEED;
}

// cfg0 as booted, reported for the record: bit 3 clear means the L0 data cache was enabled.
inline std::uint32_t read_cfg0()
{
#if defined(ARCH_BLACKHOLE)
    std::uint32_t cfg0;
    asm volatile("csrr %0, 0x7c0" : "=r"(cfg0));
    return cfg0;
#else
    return 0;
#endif
}
} // namespace

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS)
{
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS)
{
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t src_addr = params.buffer_A[0];
    const std::uint32_t dst_addr = params.buffer_B[0];
    volatile std::uint32_t* src  = reinterpret_cast<volatile std::uint32_t*>(src_addr);
    volatile std::uint32_t* dst  = reinterpret_cast<volatile std::uint32_t*>(dst_addr);

    const std::uint32_t cfg0 = read_cfg0();

    std::uint32_t stale_iterations  = 0;
    std::uint32_t first_stale_iter  = NO_STALE;
    std::uint32_t first_stale_word  = NO_STALE;
    std::uint32_t first_stale_words = 0;
    std::uint32_t src_mismatches    = 0;

    // Start from a destination that matches no iteration's pattern.
    for (std::uint32_t w = 0; w < WORDS; w++)
    {
        dst[w] = pattern(ITERATIONS, w);
    }

    for (std::uint32_t i = 0; i < ITERATIONS; i++)
    {
        for (std::uint32_t w = 0; w < WORDS; w++)
        {
            src[w] = pattern(i, w);
        }
        // The read-back is ordered behind the store to the same address, so the pattern is in L1
        // before the move is issued.
        for (std::uint32_t w = 0; w < WORDS; w++)
        {
            src_mismatches += src[w] != pattern(i, w) ? 1 : 0;
        }

        xmov_l1_to_l1_non_compact(src_addr >> 4 /* src_addr_16B */, dst_addr >> 4 /* dst_addr_16B */, XMOV_XFER_16B /* xfer_size_16B */);
        xmov_wait_till_idle();

        for (std::uint32_t g = 0; g < XMOV_READ_GAP; g++)
        {
            asm volatile("nop");
        }

        std::uint32_t stale_words = 0;
        std::uint32_t stale_word  = NO_STALE;
        for (std::uint32_t w = WORDS; w-- > 0;)
        {
            if (dst[w] != pattern(i, w))
            {
                stale_words++;
                if (stale_word == NO_STALE)
                {
                    stale_word = w;
                }
            }
        }
        if (stale_words != 0)
        {
            stale_iterations++;
            if (first_stale_iter == NO_STALE)
            {
                first_stale_iter  = i;
                first_stale_word  = stale_word;
                first_stale_words = stale_words;
            }
        }
    }

    auto* res = reinterpret_cast<std::uint32_t*>(params.buffer_Res[0]);
    res[0]    = ITERATIONS;
    res[1]    = stale_iterations;
    res[2]    = first_stale_iter;
    res[3]    = first_stale_word;
    res[4]    = first_stale_words;
    res[5]    = src_mismatches;
    res[6]    = XMOV_XFER_16B;
    res[7]    = XMOV_READ_GAP;
    res[8]    = cfg0;
}

#endif
