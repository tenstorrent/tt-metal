// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"

static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1, "dest_sync_chain only supports the L1_TO_L1 run type");

constexpr std::uint32_t DEST_SECTION_REPEATS = 256;

constexpr std::uint32_t chain_count(const ChainOp op)
{
    std::uint32_t count = 0;
    for (std::uint32_t i = 0; i < CHAIN_LENGTH; i++)
    {
        if (CHAIN_OPS[i] == op)
        {
            count++;
        }
    }
    return count;
}

constexpr std::uint32_t NUM_L1_ITERATIONS = chain_count(ChainOp::PACK);

constexpr bool CHAIN_HAS_UNPACK_TO_DEST = chain_count(ChainOp::UNPACK) > 0;
constexpr bool CHAIN_HAS_FPU            = chain_count(ChainOp::FPU) > 0;
constexpr bool CHAIN_HAS_SFPU           = chain_count(ChainOp::SFPU) > 0;

constexpr std::uint32_t iteration_begin(const std::uint32_t iteration)
{
    std::uint32_t begin = 0;
    for (std::uint32_t seen = 0; seen < iteration; begin++)
    {
        if (CHAIN_OPS[begin] == ChainOp::PACK)
        {
            seen++;
        }
    }
    return begin;
}

constexpr std::uint32_t iteration_num_ops(const std::uint32_t iteration)
{
    std::uint32_t count = 0;
    for (std::uint32_t i = iteration_begin(iteration); CHAIN_OPS[i] != ChainOp::PACK; i++)
    {
        count++;
    }
    return count;
}

constexpr ChainOp iteration_op(const std::uint32_t iteration, const std::uint32_t position)
{
    return CHAIN_OPS[iteration_begin(iteration) + position];
}

constexpr std::uint32_t iteration_pos(const std::uint32_t iteration, const ChainOp op)
{
    const std::uint32_t num_ops = iteration_num_ops(iteration);
    for (std::uint32_t position = 0; position < num_ops; position++)
    {
        if (iteration_op(iteration, position) == op)
        {
            return position;
        }
    }
    return num_ops;
}

constexpr bool iteration_has(const std::uint32_t iteration, const ChainOp op)
{
    return iteration_pos(iteration, op) < iteration_num_ops(iteration);
}

constexpr bool iteration_first(const std::uint32_t iteration, const ChainOp op)
{
    return iteration_op(iteration, 0) == op;
}

constexpr bool iteration_last(const std::uint32_t iteration, const ChainOp op)
{
    return iteration_op(iteration, iteration_num_ops(iteration) - 1) == op;
}

constexpr ChainOp iteration_pred(const std::uint32_t iteration, const ChainOp op)
{
    const std::uint32_t position = iteration_pos(iteration, op);
    return position == 0 ? ChainOp::PACK : iteration_op(iteration, position - 1);
}

constexpr bool uniform_iterations()
{
    for (std::uint32_t i = 1; i < NUM_L1_ITERATIONS; i++)
    {
        if (iteration_num_ops(i) != iteration_num_ops(0))
        {
            return false;
        }
        for (std::uint32_t position = 0; position < iteration_num_ops(i); position++)
        {
            if (iteration_op(i, position) != iteration_op(0, position))
            {
                return false;
            }
        }
    }
    return true;
}

constexpr bool UNIFORM_CHAIN = uniform_iterations();

constexpr std::uint32_t dvalid_ctrl(const std::uint32_t wait_mask, const std::uint32_t wait_polarity, const std::uint32_t toggle_mask)
{
    return (wait_mask << 0) | (wait_polarity << 4) | (toggle_mask << 8);
}

template <ckernel::dest_dvalid_client CLIENT>
inline void configure_dvalid_role(const std::uint32_t iteration, const ChainOp op)
{
    static constexpr std::uint32_t ctrl_regs[] = {
        UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32,
        MATH_DEST_DVALID_CTRL_wait_mask_ADDR32,
        SFPU_DEST_DVALID_CTRL_wait_mask_ADDR32,
        PACK_DEST_DVALID_CTRL_wait_mask_ADDR32};

    const std::uint32_t num_ops  = iteration_num_ops(iteration);
    const std::uint32_t position = op == ChainOp::PACK ? num_ops : iteration_pos(iteration, op);
    const std::uint32_t base     = UNIFORM_CHAIN ? 0u : (iteration & 1u) * 2u;

    std::uint32_t value = 0;
    if (position == 0)
    {
        const std::uint32_t slots = ((1u << num_ops) - 1u) << base;
        value                     = dvalid_ctrl(slots, 0, 1u << base);
    }
    else if (position == num_ops)
    {
        const std::uint32_t last = 1u << (base + num_ops - 1);
        value                    = dvalid_ctrl(last, last, last);
    }
    else
    {
        const std::uint32_t in = 1u << (base + position - 1);
        value                  = dvalid_ctrl(in, in, in | (in << 1));
    }

    if constexpr (dest_sync == ckernel::DstSync::SyncFull)
    {
        value |= 1u << UNPACK_TO_DEST_DVALID_CTRL_disable_auto_bank_id_toggle_SHAMT;
    }

    auto cfg                                 = (volatile std::uint32_t*)TENSIX_CFG_BASE;
    cfg[ctrl_regs[static_cast<int>(CLIENT)]] = value;
}

template <std::uint32_t CLIENT>
inline void toggle_dest_dvalid()
{
    TTI_CLEARDVALID(0, 0, 0, 0, CLIENT, 0);
}

#ifdef LLK_TRISC_UNPACK

#include "llk_sync.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        ZONE_SCOPED("INIT")
        if constexpr (USE_DVALID_SCHEME)
        {
            if constexpr (UNIFORM_CHAIN && CHAIN_HAS_UNPACK_TO_DEST)
            {
                configure_dvalid_role<ckernel::dest_dvalid_client::UNPACK>(0, ChainOp::UNPACK);
            }
        }
        else
        {
            set_up_zero_dest_dvalid_handshake_for_unpack();
            if constexpr (CHAIN_HAS_UNPACK_TO_DEST)
            {
                _llk_sync_init_(semaphore::UNPACK_MATH, dest_sync == ckernel::DstSync::SyncHalf ? 2 : 1, 0);
            }
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t iteration = 0; iteration < NUM_L1_ITERATIONS; iteration++)
        {
            if (!iteration_has(iteration, ChainOp::UNPACK))
            {
                continue;
            }

            if constexpr (USE_DVALID_SCHEME && !UNIFORM_CHAIN)
            {
                ckernel::tensix_sync();
                configure_dvalid_role<ckernel::dest_dvalid_client::UNPACK>(iteration, ChainOp::UNPACK);
            }

            for (std::uint32_t rep = 0; rep < DEST_SECTION_REPEATS; rep++)
            {
                if constexpr (USE_DVALID_SCHEME)
                {
                    toggle_dest_dvalid<p_cleardvalid::UNPACK_TO_DEST>();
                }
                else
                {
                    _llk_sync_wait_<p_stall::STALL_UNPACK, p_stall::STALL_ON_MAX>(semaphore::MATH_PACK, semaphore::UNPACK_MATH);
                    _llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_MATH);
                    if constexpr (dest_sync == ckernel::DstSync::SyncHalf)
                    {
                        _llk_sync_advance_dest_section_<TRISC_ID, true, p_stall::UNPACK0>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_sync.h"

using namespace ckernel;

inline void take_semaphore(const std::uint32_t sem)
{
    _llk_sync_wait_<p_stall::STALL_MATH | p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO>(sem);
    _llk_sync_get_<p_stall::MATH, p_stall::WAIT_SFPU>(sem);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        ZONE_SCOPED("INIT")
        if constexpr (USE_DVALID_SCHEME)
        {
            if constexpr (UNIFORM_CHAIN && CHAIN_HAS_FPU)
            {
                configure_dvalid_role<dest_dvalid_client::FPU>(0, ChainOp::FPU);
            }
        }
        else
        {
            set_up_zero_dest_dvalid_handshake_for_math();
            _llk_math_pack_sync_init_<dest_sync>();
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t iteration = 0; iteration < NUM_L1_ITERATIONS; iteration++)
        {
            const bool has_fpu   = iteration_has(iteration, ChainOp::FPU);
            const bool fpu_first = iteration_first(iteration, ChainOp::FPU);
            const bool fpu_last  = iteration_last(iteration, ChainOp::FPU);
            const ChainOp pred   = iteration_pred(iteration, ChainOp::FPU);
            const bool middleman = !USE_DVALID_SCHEME && !has_fpu && iteration_num_ops(iteration) == 1 && iteration_first(iteration, ChainOp::UNPACK);

            if (!has_fpu && !middleman)
            {
                continue;
            }

            if constexpr (USE_DVALID_SCHEME && !UNIFORM_CHAIN)
            {
                tensix_sync();
                configure_dvalid_role<dest_dvalid_client::FPU>(iteration, ChainOp::FPU);
            }

            for (std::uint32_t rep = 0; rep < DEST_SECTION_REPEATS; rep++)
            {
                if constexpr (USE_DVALID_SCHEME)
                {
                    toggle_dest_dvalid<p_cleardvalid::FPU>();
                }
                else
                {
                    if (fpu_first || middleman)
                    {
                        _llk_math_wait_for_dest_available_();
                    }
                    if (pred == ChainOp::UNPACK || middleman)
                    {
                        take_semaphore(semaphore::UNPACK_MATH);
                    }
                    else if (pred == ChainOp::SFPU)
                    {
                        take_semaphore(semaphore::SFPU_FPU);
                    }

                    if (fpu_last || middleman)
                    {
                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                    else
                    {
                        _llk_sync_wait_<p_stall::STALL_MATH | p_stall::STALL_SYNC, p_stall::STALL_ON_MAX>(semaphore::FPU_SFPU);
                        _llk_sync_post_<p_stall::MATH>(semaphore::FPU_SFPU);
                        if constexpr (dest_sync == ckernel::DstSync::SyncHalf)
                        {
                            _llk_sync_advance_dest_section_<TRISC_ID, is_fp32_dest_acc_en, p_stall::MATH>();
                        }
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_sync.h"

using namespace ckernel;
using namespace ckernel::math;

inline void take_semaphore(const std::uint32_t sem)
{
    _llk_sync_wait_<p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO>(sem);
    _llk_sync_get_<p_stall::WAIT_SFPU>(sem);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        ZONE_SCOPED("INIT")
        if constexpr (USE_DVALID_SCHEME)
        {
            if constexpr (UNIFORM_CHAIN && CHAIN_HAS_SFPU)
            {
                configure_dvalid_role<dest_dvalid_client::SFPU>(0, ChainOp::SFPU);
            }
        }
        else
        {
            set_up_zero_dest_dvalid_handshake_for_sfpu();
            _reset_dest_register_offset_();
            _set_dest_section_base_<TRISC_ID>(_get_dest_buffer_base_());
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t iteration = 0; iteration < NUM_L1_ITERATIONS; iteration++)
        {
            const bool sfpu_first = iteration_first(iteration, ChainOp::SFPU);
            const bool sfpu_last  = iteration_last(iteration, ChainOp::SFPU);
            const ChainOp pred    = iteration_pred(iteration, ChainOp::SFPU);

            if (!iteration_has(iteration, ChainOp::SFPU))
            {
                continue;
            }

            if constexpr (USE_DVALID_SCHEME && !UNIFORM_CHAIN)
            {
                tensix_sync();
                configure_dvalid_role<dest_dvalid_client::SFPU>(iteration, ChainOp::SFPU);
            }

            for (std::uint32_t rep = 0; rep < DEST_SECTION_REPEATS; rep++)
            {
                if constexpr (USE_DVALID_SCHEME)
                {
                    toggle_dest_dvalid<p_cleardvalid::SFPU>();
                }
                else
                {
                    if (sfpu_first)
                    {
                        _llk_sync_wait_<p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::STALL_ON_MAX>(semaphore::MATH_PACK);
                    }
                    else if (pred == ChainOp::UNPACK)
                    {
                        take_semaphore(semaphore::UNPACK_MATH);
                    }
                    else
                    {
                        take_semaphore(semaphore::FPU_SFPU);
                    }

                    if (sfpu_last)
                    {
                        _llk_sync_post_<p_stall::WAIT_SFPU>(semaphore::MATH_PACK);
                    }
                    else
                    {
                        _llk_sync_wait_<p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::STALL_ON_MAX>(semaphore::SFPU_FPU);
                        _llk_sync_post_<p_stall::WAIT_SFPU>(semaphore::SFPU_FPU);
                    }
                    if constexpr (dest_sync == ckernel::DstSync::SyncHalf)
                    {
                        _llk_sync_advance_dest_section_<TRISC_ID, is_fp32_dest_acc_en, p_stall::WAIT_SFPU>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack_common.h"
#include "llk_sync.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        ZONE_SCOPED("INIT")
        if constexpr (USE_DVALID_SCHEME)
        {
            if constexpr (UNIFORM_CHAIN)
            {
                configure_dvalid_role<ckernel::dest_dvalid_client::PACK>(0, ChainOp::PACK);
            }
        }
        else
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
            _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        for (std::uint32_t iteration = 0; iteration < NUM_L1_ITERATIONS; iteration++)
        {
            if constexpr (USE_DVALID_SCHEME && !UNIFORM_CHAIN)
            {
                ckernel::tensix_sync();
                configure_dvalid_role<ckernel::dest_dvalid_client::PACK>(iteration, ChainOp::PACK);
            }

            for (std::uint32_t rep = 0; rep < DEST_SECTION_REPEATS; rep++)
            {
                if constexpr (USE_DVALID_SCHEME)
                {
                    toggle_dest_dvalid<p_cleardvalid::PACK>();
                }
                else
                {
                    _llk_packer_wait_for_math_done_();
                    _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        ckernel::wait_pack_idle();
        PROFILER_SYNC();
    }
}

#endif
