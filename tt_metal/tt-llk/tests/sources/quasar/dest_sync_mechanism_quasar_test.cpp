// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

namespace
{
constexpr std::uint32_t TC_FREE = 30;

constexpr std::uint32_t UNPACK_WAIT_RES = p_stall::STALL_UNPACK;
constexpr std::uint32_t PACK_WAIT_RES   = p_stall::STALL_PACK;

constexpr std::uint32_t PUSH_WAIT_PACK0      = 0x1;
constexpr std::uint32_t POP_WAIT_ALL_UNPACKS = 0x7;

template <ckernel::DstSync DST>
constexpr std::uint32_t sections_in_flight()
{
    return DST == ckernel::DstSync::SyncHalf ? 2 : 1;
}

inline std::uint32_t dest_bank_tile_offset()
{
    return ckernel::trisc::_get_dest_buffer_base_() >> ckernel::trisc::get_dest_tile_size_log2(ckernel::trisc::DstTileShape::Tile32x32);
}
} // namespace

#ifdef LLK_TRISC_UNPACK

#include "ckernel_buf_desc.h"
#include "llk_sync.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand_to_dest.h"
#include "params.h"

inline void unpack_section(const std::uint32_t l1_tile_idx, const std::uint32_t tiles_in_block)
{
    _llk_unpack_unary_operand_to_dest_block_(l1_tile_idx, 0 /*dst_tile_idx*/, tiles_in_block);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const Operand& buffer_A                = params.buffer_A;
#endif
    const std::uint32_t num_blocks     = static_cast<std::uint32_t>(NUM_BLOCKS);
    const std::uint32_t tiles_in_block = static_cast<std::uint32_t>(NUM_TILES_IN_BLOCK);
    constexpr std::uint32_t N          = sections_in_flight<dest_sync>();

    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 && dest_sync_uses_dest_dvalid)
        {
            set_up_unpack_to_pack_dest_dvalid_chain<ckernel::dest_dvalid_client::UNPACK>();
        }
        else
        {
            set_up_zero_dest_dvalid_handshake_for_unpack();
        }

        const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);
        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(tensor_shape_A, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        constexpr std::uint8_t bfd_unpack = ckernel::trisc::BFD_ID_UNP0;

        _llk_unpack_configure_unary_<p_unpacr::UNP_DEST>(static_cast<DataFormat>(formats.unpack_A_dst));
        _llk_unpack_dest_init_();
        _llk_unpack_unary_operand_to_dest_init_(bfd_unpack);

        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 && dest_sync_uses_semaphore)
        {
            _llk_sync_init_(semaphore::UNPACK_PACK, N /*max*/, 0 /*init*/);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    unpack_section(block * tiles_in_block, tiles_in_block);
                    if constexpr (dest_sync == ckernel::DstSync::SyncHalf)
                    {
                        _llk_unpack_dest_section_advance_<is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    if constexpr (dest_sync_uses_tile_counters)
                    {
                        TTI_WAIT_TILES(UNPACK_WAIT_RES, 1 /*num_tiles*/, TC_FREE);
                        unpack_section(block * tiles_in_block, tiles_in_block);
                        TTI_POP_TILES(POP_WAIT_ALL_UNPACKS, 1 /*num_tiles*/, TC_FREE);
                    }
                    else if constexpr (dest_sync_uses_dest_dvalid)
                    {
                        unpack_section(block * tiles_in_block, tiles_in_block);
                        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                    }
                    else
                    {
                        _llk_sync_wait_<UNPACK_WAIT_RES, p_stall::STALL_ON_MAX>(semaphore::UNPACK_PACK);
                        unpack_section(block * tiles_in_block, tiles_in_block);
                        _llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_PACK);
                    }
                    if constexpr (dest_sync == ckernel::DstSync::SyncHalf && !dest_sync_uses_dest_dvalid)
                    {
                        _llk_unpack_dest_section_advance_<is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    {
        START_PERF_MEASURE("INIT")
        set_up_zero_dest_dvalid_handshake_for_math();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_sync.h"
#include "params.h"

inline void pack_section(const std::uint32_t l1_tile_idx, const ckernel::TensorShape& tensor_shape)
{
    _llk_pack_(dest_bank_tile_offset() /*start_math_dest_tile_idx*/, l1_tile_idx, tensor_shape);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
    const std::uint32_t num_blocks     = static_cast<std::uint32_t>(NUM_BLOCKS);
    const std::uint32_t tiles_in_block = static_cast<std::uint32_t>(NUM_TILES_IN_BLOCK);

    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 && dest_sync_uses_dest_dvalid)
        {
            set_up_unpack_to_pack_dest_dvalid_chain<ckernel::dest_dvalid_client::PACK>();
        }
        else
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }

        const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);
        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(tensor_shape_A, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        constexpr std::uint8_t bfd_pack = ckernel::trisc::BFD_ID_PACK0;
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(bfd_pack, tensor_shape_A, tiles_in_block);
        _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();

        if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 && dest_sync_uses_tile_counters)
        {
            constexpr std::uint32_t N                             = sections_in_flight<dest_sync>();
            ckernel::trisc::tile_counters[TC_FREE].f.reset        = 1;
            ckernel::trisc::tile_counters[TC_FREE].f.buf_capacity = N;
            ckernel::trisc::tile_counters[TC_FREE].f.posted       = N;
            [[maybe_unused]] const std::uint32_t free_sections    = ckernel::trisc::tile_counters[TC_FREE].f.posted;
            LLK_ASSERT((free_sections & 0xFFFF) == N, "FREE tile counter seed did not land");
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);

        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    pack_section(block * tiles_in_block, tensor_shape_A);
                    if constexpr (dest_sync == ckernel::DstSync::SyncHalf)
                    {
                        ckernel::trisc::_update_dest_register_offset_<is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    if constexpr (dest_sync_uses_tile_counters)
                    {
                        TTI_WAIT_FREE(PACK_WAIT_RES, 1 /*num_tiles*/, TC_FREE);
                        pack_section(block * tiles_in_block, tensor_shape_A);
                        TTI_PUSH_TILES(PUSH_WAIT_PACK0, 1 /*num_tiles*/, TC_FREE);
                    }
                    else if constexpr (dest_sync_uses_dest_dvalid)
                    {
                        pack_section(block * tiles_in_block, tensor_shape_A);
                        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::WAIT_SFPU, p_stall::PACK);
                        TTI_CLEARDVALID(0, 0, 0, 0, p_cleardvalid::PACK, 0);
                        if constexpr (dest_sync == ckernel::DstSync::SyncFull)
                        {
                            TTI_CLEARDVALID(0, 0, 0, p_cleardvalid::PACK, p_cleardvalid::PACK, 0);
                        }
                    }
                    else
                    {
                        if (loop == 0 && block == 0)
                        {
                            _llk_sync_spin_until_posted_(semaphore::UNPACK_PACK);
                        }
                        _llk_sync_wait_<PACK_WAIT_RES, p_stall::STALL_ON_ZERO>(semaphore::UNPACK_PACK);
                        pack_section(block * tiles_in_block, tensor_shape_A);
                        _llk_sync_get_<p_stall::PACK0>(semaphore::UNPACK_PACK);
                    }
                    if constexpr (dest_sync == ckernel::DstSync::SyncHalf && !dest_sync_uses_dest_dvalid)
                    {
                        ckernel::trisc::_update_dest_register_offset_<is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }

    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 && dest_sync_uses_tile_counters)
    {
        TTI_WAIT_TILES(PACK_WAIT_RES, sections_in_flight<dest_sync>() /*num_tiles*/, TC_FREE);
        ckernel::wait_sync_idle();
        ckernel::trisc::tile_counters[TC_FREE].f.reset = 1;
    }
}

#endif
