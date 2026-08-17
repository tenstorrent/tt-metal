// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "params.h"
#include "stream.h"

// Globals required by LLK infrastructure
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uintptr_t STREAM_ADDRESS = 0x70000;
constexpr std::uint32_t STREAM_DEPTH    = 128;

using StreamType = llk::Stream<STREAM_DEPTH>;

constexpr std::size_t PACKET_COUNT = 10;
constexpr std::uint32_t DATA_SEED  = 0xDEADBEEF;

constexpr std::uint32_t PRODUCER_DELAY_SEED = 0xCAFEBABE;
constexpr std::uint32_t CONSUMER_DELAY_SEED = 0xFEEDFACE;

struct prng_t
{
    std::uint32_t state;

    constexpr prng_t(std::uint32_t seed) : state(seed ? seed : 1)
    {
    }

    constexpr std::uint32_t next()
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    constexpr char byte()
    {
        return static_cast<char>(next() & 0xFF);
    }
};

inline void delay(prng_t& rng)
{
    std::uint32_t nops = rng.next() % 11;
    for (std::uint32_t i = 0; i < nops; ++i)
    {
        asm volatile("nop");
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_assert.h"

// Unpack is used as the producer when the host is not.
void run_kernel(RUNTIME_PARAMETERS params)
{
    if (params.HOST_IS_STREAM_PRODUCER)
    {
        return;
    }

    auto write_idx = *reinterpret_cast<const volatile std::uint32_t*>(STREAM_ADDRESS);
    LLK_ASSERT(write_idx == 0, "write_idx not initialized to zero");

    asm volatile("" ::: "memory");

    StreamType& stream = *reinterpret_cast<StreamType*>(STREAM_ADDRESS);
    char send[512];

    prng_t rand(DATA_SEED);
    prng_t delay_rand(PRODUCER_DELAY_SEED);

    for (std::size_t i = 0; i < PACKET_COUNT; ++i)
    {
        const std::uint32_t packet_size = rand.next() % 100 + 1;

        for (std::uint32_t j = 0; j < packet_size; ++j)
        {
            send[j] = rand.byte();
        }

        stream.push(send, packet_size);
        delay(delay_rand);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_assert.h"

// Math is used as the consumer when the host is not.
void run_kernel(RUNTIME_PARAMETERS params)
{
    if (params.HOST_IS_STREAM_CONSUMER)
    {
        return;
    }

    auto read_idx = *reinterpret_cast<const volatile std::uint32_t*>(STREAM_ADDRESS + 4);
    LLK_ASSERT(read_idx == 0, "read_idx not initialized to zero");

    asm volatile("" ::: "memory");

    StreamType& stream = *reinterpret_cast<StreamType*>(STREAM_ADDRESS);
    char recv[512];

    prng_t rand(DATA_SEED);
    prng_t delay_rand(CONSUMER_DELAY_SEED);
    std::uint32_t errors = 0;

    for (std::size_t i = 0; i < PACKET_COUNT; ++i)
    {
        delay(delay_rand);

        const std::uint32_t packet_size = rand.next() % 100 + 1;

        stream.pop(recv, packet_size);

        for (std::uint32_t j = 0; j < packet_size; ++j)
        {
            const char expected = rand.byte();
            if (recv[j] != expected)
            {
                ++errors;
            }
        }
    }

    LLK_ASSERT(errors == 0, "Stream data verification failed");
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
}

#endif
