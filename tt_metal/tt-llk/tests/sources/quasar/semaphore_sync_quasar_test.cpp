// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"
#include "tensor_shape.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_math_common.h" // _llk_math_upk_to_dest_hw_configure_: DEST format for UNP_DEST writes, programmed from this thread
#include "llk_sync.h"
#include "llk_unpack_common.h"
#include "llk_unpack_reduce.h"
#include "llk_unpack_unary_operand_to_dest.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // allocate srcA (order matters: A before B)
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src);
    // allocate srcB
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp1>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_B[0]), formats.unpack_B_src);

    if constexpr (unpack_to_dest)
    {
        // Unpack-to-dest under the semaphore scheme. The placers carry no synchronization, so this thread runs the
        // section handshake itself, as tt-metal's tile_regs_acquire / tile_regs_commit hooks do: stall until a DEST
        // bank is free (UNPACK_PACK below its max, the count pack releases), place the tile, post UNPACK_PACK (bank
        // occupied) and then UNPACK_MATH (data ready), and in SyncHalf move this thread to the other bank.
        _llk_unpack_configure_unary_<p_unpacr::UNP_DEST>(static_cast<DataFormat>(formats.unpack_A_dst));
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
        _llk_unpack_dest_init_();
        _llk_unpack_unary_operand_to_dest_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>());
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_sync_wait_<p_stall::STALL_UNPACK, p_stall::STALL_ON_MAX>(semaphore::UNPACK_PACK);
            _llk_unpack_unary_operand_to_dest_tile_(i /*l1_tile_idx*/, 0 /*dst_tile_idx*/);
            _llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_PACK);
            _llk_sync_post_<p_stall::UNPACK0>(semaphore::UNPACK_MATH);
            if constexpr (dest_sync == ckernel::DstSync::SyncHalf)
            {
                _llk_unpack_dest_section_advance_<is_fp32_dest_acc_en>();
            }
        }
    }
    else
    {
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_B_dst));
        _llk_unpack_reduce_init_<POOL_TYPE, REDUCE_DIM>(
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(),
            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp1>(),
            ckernel::DEFAULT_TENSOR_SHAPE,
            1 /*num_tiles_per_unpack*/); // tiny-tiles not yet supported with reduce
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_unpack_reduce_(i, 0, ckernel::DEFAULT_TENSOR_SHAPE);
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_reduce.h"
#include "llk_sync.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    if constexpr (unpack_to_dest)
    {
        // Middleman only, no FPU work. Seed UNPACK_MATH and UNPACK_PACK next to MATH_PACK (max N = sections in flight:
        // 1 in SyncFull, 2 in SyncHalf). Per section: wait for the unpacker's data, hand the section to pack (MATH_PACK
        // post and, in SyncHalf, the math-side bank flip), then release the unpacker's data-ready credit.
        _llk_math_pack_sync_init_<dest_sync>();
        constexpr std::uint32_t N = (dest_sync == DstSync::SyncFull) ? 1 : 2;
        _llk_sync_init_(semaphore::UNPACK_MATH, N, 0);
        _llk_sync_init_(semaphore::UNPACK_PACK, N, 0);
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_math_wait_for_dest_available_();
            _llk_sync_wait_<p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO>(semaphore::UNPACK_MATH);
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            _llk_sync_get_<p_stall::MATH, p_stall::WAIT_SFPU>(semaphore::UNPACK_MATH);
        }
    }
    else
    {
        DataFormat src_format = static_cast<DataFormat>(formats.math);

        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /* int32 dest */>(src_format, src_format);
        _llk_math_pack_sync_init_<dest_sync>();
        _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, MATH_FIDELITY>(ckernel::DEFAULT_TENSOR_SHAPE); // tiny-tiles not yet supported with reduce
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_math_wait_for_dest_available_();
            _llk_math_reduce_<POOL_TYPE, REDUCE_DIM>(0 /*dest_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_sync.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst);

    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles_per_pack*/);
    if constexpr (unpack_to_dest)
    {
        // Pack side of the unpack-to-dest handshake, as tt-metal's llk_pack_dest_section_done: behind the PACK0 drain,
        // release the bank to the unpacker (UNPACK_PACK, the count it stalls on) and the section to math (MATH_PACK);
        // in SyncHalf move the packer's DEST read base to the other bank. No ZEROACC: the unpacker overwrites the bank.
        _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_packer_wait_for_math_done_();
            _llk_pack_(0 /*dest_idx*/, i, ckernel::DEFAULT_TENSOR_SHAPE);
            _llk_sync_get_<p_stall::PACK0>(semaphore::UNPACK_PACK);
            _llk_sync_get_<p_stall::PACK0>(semaphore::MATH_PACK);
            if constexpr (dest_sync == ckernel::DstSync::SyncHalf)
            {
                ckernel::trisc::_update_dest_register_offset_<is_fp32_dest_acc_en>();
                _llk_stall_cfg_on_<p_stall::PACK0>();
                ckernel::trisc::_set_packer_dest_registers_<p_pacr::PACK0, dest_sync>();
            }
        }
    }
    else
    {
        _llk_pack_reduce_mask_config_<REDUCE_DIM>(ckernel::DEFAULT_TENSOR_SHAPE);
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_packer_wait_for_math_done_();
            _llk_pack_(0 /*dest_idx*/, i, ckernel::DEFAULT_TENSOR_SHAPE);
            _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
        }
        _llk_pack_reduce_mask_clear_();
    }
}
#endif
