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

void run_kernel(const volatile struct RuntimeParams* params)
{
    std::int32_t* A = reinterpret_cast<std::int32_t*>(buffer_A[0]);
    std::int32_t* B = reinterpret_cast<std::int32_t*>(buffer_B[0]);
    std::int32_t* C = reinterpret_cast<std::int32_t*>(buffer_Res[0]);

    std::transform(A, A + 1024, B, C, std::plus<std::int32_t>());
}

#endif

#ifdef LLK_TRISC_MATH

void run_kernel(const volatile struct RuntimeParams* params)
{
    std::int32_t* A = reinterpret_cast<std::int32_t*>(buffer_A[1]);
    std::int32_t* B = reinterpret_cast<std::int32_t*>(buffer_B[1]);
    std::int32_t* C = reinterpret_cast<std::int32_t*>(buffer_Res[1]);

    std::transform(A, A + 1024, B, C, std::plus<std::int32_t>());
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(const volatile struct RuntimeParams* params)
{
    std::int32_t* A = reinterpret_cast<std::int32_t*>(buffer_A[2]);
    std::int32_t* B = reinterpret_cast<std::int32_t*>(buffer_B[2]);
    std::int32_t* C = reinterpret_cast<std::int32_t*>(buffer_Res[2]);

    std::transform(A, A + 1024, B, C, std::plus<std::int32_t>());
}

#endif
