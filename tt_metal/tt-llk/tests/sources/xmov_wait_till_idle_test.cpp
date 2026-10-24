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
// harness booted it; the wait itself drops it on Blackhole. Right before each move the last
// destination line is loaded, so it sits in that cache holding the previous iteration's contents
// whatever the move size, and a wait that did not drop the cache returns the stale line.

#include <cstdint>

#include "build.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_xmov.h"

using namespace ckernel;

// Globals the harness links against.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
constexpr std::uint32_t WORDS_PER_16B = 4;
constexpr std::uint32_t BYTES_PER_16B = WORDS_PER_16B * sizeof(std::uint32_t);
constexpr std::uint32_t WORDS         = XMOV_XFER_16B * WORDS_PER_16B;
constexpr std::uint32_t NO_STALE      = 0xFFFFFFFFu;

// buffer_A and buffer_B are one Int32 tile each, laid out back to back with buffer_Res.
constexpr std::uint32_t BUFFER_BYTES = TILE_R_DIM * TILE_C_DIM * sizeof(std::uint32_t);
static_assert(XMOV_XFER_16B > 0 && XMOV_XFER_16B * BYTES_PER_16B <= BUFFER_BYTES, "the move must stay inside one buffer and not be empty");

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
    return csr_read<0x7c0 /*csr_num*/, false /*fence*/>();
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
        // before the move is issued: the blocking load consumes the last read-back, and the TDMA
        // register stores below are emitted only after it.
        for (std::uint32_t w = 0; w < WORDS; w++)
        {
            src_mismatches += src[w] != pattern(i, w) ? 1 : 0;
        }
        (void)load_blocking(&src[WORDS - 1]);
        // Leave the last destination line, the one read first below, in the RISC data cache with
        // the previous iteration's contents (Blackhole; Wormhole has no data cache). The source
        // read-back above touches more lines than the cache holds for all but the smallest move,
        // so without this load the destination reads would miss the cache and pass whether or not
        // the wait dropped it. As above, the blocking load is issued before the TDMA stores below.
        (void)load_blocking(&dst[WORDS - 1]);

        xmov_l1_to_l1_non_compact(src_addr >> 4 /*src_addr_16B*/, dst_addr >> 4 /*dst_addr_16B*/, XMOV_XFER_16B /*xfer_size_16B*/);
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
