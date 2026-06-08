// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "ckernel.h"

// FIXME: this shouldn't be statically allocated
constexpr std::uint32_t PERF_INPUT_A = 0x21000;
constexpr std::uint32_t PERF_INPUT_B = PERF_INPUT_A + 16 * 4096;
constexpr std::uint32_t PERF_INPUT_C = PERF_INPUT_B + 16 * 4096;
constexpr std::uint32_t PERF_OUTPUT  = PERF_INPUT_C + 16 * 4096;

constexpr std::uint32_t PERF_ADDRESS(std::uint32_t buffer, std::uint32_t tile)
{
    std::uint32_t address = buffer + (tile % 16) * 4096; // Loop every 16 tiles, to prevent escaping memory
    return address / 16 - 1;                             // Correct the L1 Address for Tensix
}

enum class PerfRunType
{
    L1_TO_L1,
    UNPACK_ISOLATE,
    MATH_ISOLATE,
    PACK_ISOLATE,
    L1_CONGESTION
};

inline void _perf_unpack_set_valid(std::uint32_t source)
{
    std::uint32_t set_a = source == ckernel::SrcA ? 1 : 0;
    std::uint32_t set_b = source == ckernel::SrcB ? 1 : 0;

    std::uint32_t wait_clear_a = set_a ? ckernel::p_stall::SRCA_CLR : 0;
    std::uint32_t wait_clear_b = set_b ? ckernel::p_stall::SRCB_CLR : 0;

#ifdef ARCH_QUASAR
    TT_STALLWAIT(ckernel::p_stall::STALL_TDMA, wait_clear_a, wait_clear_b, 0);
    TT_SETDVALID((set_b << 1) | set_a);

#else
    TT_STALLWAIT(ckernel::p_stall::STALL_TDMA, wait_clear_a | wait_clear_b);
    TT_SETDVALID((set_b << 1) | set_a);

#endif
}

inline void _perf_math_clear_valid(std::uint32_t source)
{
    std::uint32_t clear_a = source == ckernel::SrcA ? 1 : 0;
    std::uint32_t clear_b = source == ckernel::SrcB ? 1 : 0;

    std::uint32_t wait_valid_a = clear_a ? ckernel::p_stall::SRCA_VLD : 0;
    std::uint32_t wait_valid_b = clear_b ? ckernel::p_stall::SRCB_VLD : 0;

#ifdef ARCH_QUASAR
    TT_STALLWAIT(ckernel::p_stall::STALL_MATH, wait_valid_a, wait_valid_b, 0);
    TT_CLEARDVALID((clear_b << 1) | clear_a, 0, 0, 0, 0, 0);

#else
    TT_STALLWAIT(ckernel::p_stall::STALL_MATH, wait_valid_a | wait_valid_b);
    TT_CLEARDVALID((clear_b << 1) | clear_a, 0);

#endif
}

template <bool set_a, bool set_b>
inline void _perf_unpack_loop_set_valid(std::uint32_t iterations)
{
    while (iterations-- > 0)
    {
        constexpr std::uint32_t cond_clear_a = set_a ? ckernel::p_stall::SRCA_CLR : 0;
        constexpr std::uint32_t cond_clear_b = set_b ? ckernel::p_stall::SRCB_CLR : 0;

#ifdef ARCH_QUASAR
        TTI_STALLWAIT(ckernel::p_stall::STALL_TDMA, cond_clear_a, cond_clear_b, 0);
#else
        TTI_STALLWAIT(ckernel::p_stall::STALL_TDMA, cond_clear_a | cond_clear_b);
#endif
        TTI_SETDVALID((set_b << 1) | set_a);
    }
}

template <bool clear_a, bool clear_b>
inline void _perf_math_loop_clear_valid(std::uint32_t iterations)
{
    while (iterations-- > 0)
    {
        constexpr std::uint32_t cond_valid_a = clear_a ? ckernel::p_stall::SRCA_VLD : 0;
        constexpr std::uint32_t cond_valid_b = clear_b ? ckernel::p_stall::SRCB_VLD : 0;
#ifdef ARCH_QUASAR
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a, cond_valid_b, 0);
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0, 0, 0, 0, 0);
#else
        TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, cond_valid_a | cond_valid_b);
        TTI_CLEARDVALID((clear_b << 1) | clear_a, 0);
#endif
    }
}

inline void _perf_unpack_matmul_mock(std::uint32_t loop_factor, std::uint32_t rt_dim, std::uint32_t kt_dim, std::uint32_t ct_dim)
{
    // fixme: add quasar support

    for (std::uint32_t loop = 0; loop < loop_factor; loop++)
    {
        for (std::uint32_t j = 0; j < kt_dim; j++)
        {
            const std::uint32_t reuse_reg  = ct_dim >= rt_dim ? ckernel::SrcB : ckernel::SrcA;
            const std::uint32_t reuse_loop = std::min(ct_dim, rt_dim);

            const std::uint32_t reload_reg  = ct_dim >= rt_dim ? ckernel::SrcA : ckernel::SrcB;
            const std::uint32_t reload_loop = std::max(ct_dim, rt_dim);

#ifdef ARCH_WORMHOLE

            /*
             * WORMHOLE SCHEME:
             * Utilizes both source register banks to maximize throughput.
             * IF CT_DIM >= RT_DIM ->
             *   SRCB, SRCB, SRCA * CT_DIM, SRCB, SRCB, ...
             * ELSE ->
             *   SRCA, SRCA, SRCB * RT_DIM, SRCA, SRCA, ...
             */

            std::uint32_t reuse_iter = 0;
            while (reuse_iter < reuse_loop)
            {
                const std::uint32_t reuse_burst = std::min(static_cast<std::uint32_t>(2), reuse_loop - reuse_iter);

                for (std::uint32_t i = 0; i < reuse_burst; i++)
                {
                    _perf_unpack_set_valid(reuse_reg);
                }

                for (std::uint32_t i = 0; i < reload_loop; i++)
                {
                    _perf_unpack_set_valid(reload_reg);
                }

                reuse_iter += reuse_burst;
            }
#endif

#ifdef ARCH_BLACKHOLE

            /*
             * BLACKHOLE SCHEME:
             * Utilizes only one source register bank because bandwidth is better on BH
             * IF CT_DIM >= RT_DIM ->
             *   SRCB, SRCA * CT_DIM, SRCB, SRCA * CT_DIM, ...
             * ELSE ->
             *   SRCA, SRCB * RT_DIM, SRCA, SRCB * RT_DIM, ...
             */

            for (std::uint32_t reuse_iter = 0; reuse_iter < reuse_loop; reuse_iter++)
            {
                _perf_unpack_set_valid(reuse_reg);

                for (std::uint32_t i = 0; i < reload_loop; i++)
                {
                    _perf_unpack_set_valid(reload_reg);
                }
            }
#endif
        }
    }
}

inline void _perf_math_matmul_mock(std::uint32_t loop_factor, std::uint32_t rt_dim, std::uint32_t kt_dim, std::uint32_t ct_dim)
{
    // fixme: add quasar support

    for (std::uint32_t loop = 0; loop < loop_factor; loop++)
    {
        for (std::uint32_t j = 0; j < kt_dim; j++)
        {
            const std::uint32_t reuse_reg  = ct_dim >= rt_dim ? ckernel::SrcB : ckernel::SrcA;
            const std::uint32_t reuse_loop = std::min(ct_dim, rt_dim);

            const std::uint32_t reload_reg  = ct_dim >= rt_dim ? ckernel::SrcA : ckernel::SrcB;
            const std::uint32_t reload_loop = std::max(ct_dim, rt_dim);

#ifdef ARCH_WORMHOLE

            /*
             * WORMHOLE SCHEME:
             * Utilizes both source register banks to maximize throughput.
             * IF CT_DIM >= RT_DIM ->
             *  SRCA * CT_DIM, SRCB, SRCB, SRCA * CT_DIM, ...
             * ELSE ->
             *  SRCB * RT_DIM, SRCA, SRCA, SRCB * RT_DIM, ...
             */

            std::uint32_t reuse_iter = 0;
            while (reuse_iter < reuse_loop)
            {
                const std::uint32_t reuse_burst = std::min(static_cast<std::uint32_t>(2), reuse_loop - reuse_iter);

                for (std::uint32_t i = 0; i < reload_loop; i++)
                {
                    _perf_math_clear_valid(reload_reg);
                }

                for (std::uint32_t i = 0; i < reuse_burst; i++)
                {
                    _perf_math_clear_valid(reuse_reg);
                }

                reuse_iter += reuse_burst;
            }
#endif

#ifdef ARCH_BLACKHOLE

            /*
             * BLACKHOLE SCHEME:
             * Utilizes only one source register bank because bandwidth is better on BH
             * IF CT_DIM >= RT_DIM ->
             *  SRCA * CT_DIM, SRCB, SRCA * CT_DIM, SRCB, ...
             * ELSE ->
             *  SRCB * RT_DIM, SRCA, SRCB * RT_DIM, SRCA, ...
             */

            for (std::uint32_t reuse_iter = 0; reuse_iter < reuse_loop; reuse_iter++)
            {
                for (std::uint32_t i = 0; i < reload_loop; i++)
                {
                    _perf_math_clear_valid(reload_reg);
                }

                _perf_math_clear_valid(reuse_reg);
            }
#endif
        }
    }
}
