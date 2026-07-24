// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>

#include "build.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t total_tiles = params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK;
    for (std::uint32_t tile = 0; tile < total_tiles; tile += 3)
    {
        const auto* A = reinterpret_cast<const std::int32_t*>(params.buffer_A[tile]);
        const auto* B = reinterpret_cast<const std::int32_t*>(params.buffer_B[tile]);
        auto* C       = reinterpret_cast<std::int32_t*>(params.buffer_Res[tile]);
        std::transform(A, A + 1024, B, C, std::plus<std::int32_t>());
    }
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t total_tiles = params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK;
    for (std::uint32_t tile = 1; tile < total_tiles; tile += 3)
    {
        const auto* A = reinterpret_cast<const std::int32_t*>(params.buffer_A[tile]);
        const auto* B = reinterpret_cast<const std::int32_t*>(params.buffer_B[tile]);
        auto* C       = reinterpret_cast<std::int32_t*>(params.buffer_Res[tile]);
        std::transform(A, A + 1024, B, C, std::plus<std::int32_t>());
    }
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t total_tiles = params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK;
    for (std::uint32_t tile = 2; tile < total_tiles; tile += 3)
    {
        const auto* A = reinterpret_cast<const std::int32_t*>(params.buffer_A[tile]);
        const auto* B = reinterpret_cast<const std::int32_t*>(params.buffer_B[tile]);
        auto* C       = reinterpret_cast<std::int32_t*>(params.buffer_Res[tile]);
        std::transform(A, A + 1024, B, C, std::plus<std::int32_t>());
    }
}

#endif
