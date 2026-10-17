// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for `xmov_wait_till_idle`: once it returns, the move that was issued just before
// it has been carried out, so the destination can be read straight away.
//
// Each iteration PACK writes a fresh pattern to the source buffer, issues an L1-to-L1 move with
// `xmov_l1_to_l1_non_compact`, calls `xmov_wait_till_idle`, and then reads the destination back
// (last word first, since the mover writes it last) and compares it with the pattern. An iteration
// in which any destination word still holds the previous pattern is counted as stale.
//
// The source is read back after it is written, so the pattern is in L1 before the move is issued
// and a stale destination cannot be blamed on the source. On Blackhole the RISC L0 data cache is
// switched off for the duration of the test (as tt-metal firmware does on every RISC), since it
// does not observe writes made by the mover.

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

// Differs between consecutive iterations in every word.
constexpr std::uint32_t pattern(const std::uint32_t iteration, const std::uint32_t word)
{
    return (iteration * 0x9E3779B1u) ^ (word * 0x01000193u) ^ 0xA5A5A5A5u;
}

#if defined(ARCH_BLACKHOLE)
// cfg0.DisLowCash: the L0 data cache is not coherent with other L1 writers such as the mover.
constexpr std::uint32_t CFG0_DIS_LOW_CACHE = 1u << 3;

inline std::uint32_t disable_l0_data_cache()
{
    std::uint32_t before;
    asm volatile("csrrs %0, 0x7c0, %1" : "=r"(before) : "r"(CFG0_DIS_LOW_CACHE));
    asm volatile("fence" ::: "memory");
    return before;
}

inline void restore_cfg0(const std::uint32_t value)
{
    asm volatile("csrw 0x7c0, %0" : : "r"(value));
    asm volatile("fence" ::: "memory");
}
#endif
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

    std::uint32_t cfg0_before = 0;
    std::uint32_t cfg0_during = 0;
#if defined(ARCH_BLACKHOLE)
    cfg0_before = disable_l0_data_cache();
    asm volatile("csrr %0, 0x7c0" : "=r"(cfg0_during));
#endif

    std::uint32_t stale_iterations  = 0;
    std::uint32_t first_stale_iter  = 0xFFFFFFFFu;
    std::uint32_t first_stale_word  = 0xFFFFFFFFu;
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

        xmov_l1_to_l1_non_compact(src_addr >> 4, dst_addr >> 4, XMOV_XFER_16B);
        xmov_wait_till_idle();

        for (std::uint32_t g = 0; g < XMOV_READ_GAP; g++)
        {
            asm volatile("nop");
        }

        std::uint32_t stale_words = 0;
        std::uint32_t stale_word  = 0xFFFFFFFFu;
        for (std::uint32_t w = WORDS; w-- > 0;)
        {
            if (dst[w] != pattern(i, w))
            {
                stale_words++;
                if (stale_word == 0xFFFFFFFFu)
                {
                    stale_word = w;
                }
            }
        }
        if (stale_words != 0)
        {
            stale_iterations++;
            if (first_stale_iter == 0xFFFFFFFFu)
            {
                first_stale_iter  = i;
                first_stale_word  = stale_word;
                first_stale_words = stale_words;
            }
        }
    }

#if defined(ARCH_BLACKHOLE)
    restore_cfg0(cfg0_before);
#endif

    auto* res = reinterpret_cast<std::uint32_t*>(params.buffer_Res[0]);
    res[0]    = ITERATIONS;
    res[1]    = stale_iterations;
    res[2]    = first_stale_iter;
    res[3]    = first_stale_word;
    res[4]    = first_stale_words;
    res[5]    = src_mismatches;
    res[6]    = XMOV_XFER_16B;
    res[7]    = XMOV_READ_GAP;
    res[8]    = cfg0_before;
    res[9]    = cfg0_during;
}

#endif
