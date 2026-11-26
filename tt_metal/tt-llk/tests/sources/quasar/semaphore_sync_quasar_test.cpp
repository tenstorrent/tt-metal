// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_dest_borrow.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "tensor_shape.h"

#ifdef LLK_TRISC_UNPACK

#include "ckernel_buf_desc.h"
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TEST_FACE_R_DIM    = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces          = params.num_faces;
    const Operand& buffer_A                = params.buffer_A;
    const Operand& buffer_B                = params.buffer_B;
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
#endif
    // One DEST section per block of NUM_TILES_IN_BLOCK tiles.
    const std::uint32_t num_blocks = dest_section_count(TILE_CNT, NUM_TILES_IN_BLOCK);
    // allocate srcA (order matters: A before B)
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(
        ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
    constexpr std::uint8_t bfd_a = ckernel::trisc::BFD_ID_UNP0;
    // allocate srcB
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP1>(
        ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_B[0]), formats.unpack_B_src);
    constexpr std::uint8_t bfd_b = ckernel::trisc::BFD_ID_UNP1;

    if constexpr (unpack_to_dest)
    {
        // Write Dest in the section's accumulation format so PACK reads the same format.
        _llk_unpack_configure_unary_<p_unpacr::UNP_DEST>(static_cast<DataFormat>(formats.pack_src));
        _llk_unpack_dest_init_();
        _llk_unpack_unary_operand_to_dest_init_(bfd_a);
        for (std::uint32_t block = 0; block < num_blocks; ++block)
        {
            if (block % 3 == 1)
            {
                _llk_tile_regs_borrow_<DstClient::PACK, DstClient::UNPACK>();
            }
            else
            {
                _llk_tile_regs_borrow_<DstClient::MATH, DstClient::UNPACK>();
            }
            for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
            {
                _llk_unpack_unary_operand_to_dest_tile_(block * NUM_TILES_IN_BLOCK + tile /*l1_tile_idx*/, tile /*dst_tile_idx*/);
            }
            if (block % 3 == 1)
            {
                _llk_tile_regs_return_<DstClient::PACK, DstClient::UNPACK>();
            }
            else
            {
                _llk_tile_regs_return_<DstClient::MATH, DstClient::UNPACK>();
            }
        }
    }
    else
    {
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_B_dst));
        _llk_unpack_reduce_init_<POOL_TYPE, REDUCE_DIM>(
            bfd_a, bfd_b, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles_per_unpack*/); // tiny-tiles not yet supported with reduce
        for (std::uint32_t i = 0; i < TILE_CNT; ++i)
        {
            _llk_unpack_reduce_<POOL_TYPE, REDUCE_DIM>(i, 0, ckernel::DEFAULT_TENSOR_SHAPE);
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
#endif
    const std::uint32_t num_blocks = dest_section_count(TILE_CNT, NUM_TILES_IN_BLOCK);

    if constexpr (unpack_to_dest)
    {
        const auto src_format = static_cast<DataFormat>(formats.math);
        _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(src_format, src_format);
        _llk_math_pack_sync_init_<dest_sync>();
        for (std::uint32_t block = 0; block < num_blocks; ++block)
        {
            _llk_math_wait_for_dest_available_();
            if (block % 3 != 1)
            {
                _llk_tile_regs_borrow_<DstClient::MATH, DstClient::UNPACK>();
                _llk_tile_regs_return_<DstClient::MATH, DstClient::UNPACK>();
                _llk_tile_regs_borrow_<DstClient::MATH, DstClient::SFPU>();
                _llk_tile_regs_return_<DstClient::MATH, DstClient::SFPU>();
            }
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
    else
    {
        DataFormat src_format = static_cast<DataFormat>(formats.math);

        _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(src_format, src_format);
        _llk_math_pack_sync_init_<dest_sync>();
        _llk_math_reduce_init_<POOL_TYPE, REDUCE_DIM, is_fp32_dest_acc_en, MATH_FIDELITY>(
            static_cast<DataFormat>(formats.math),
            static_cast<DataFormat>(formats.math),
            ckernel::DEFAULT_TENSOR_SHAPE); // tiny-tiles not yet supported with reduce
        for (std::uint32_t block = 0; block < num_blocks; ++block)
        {
            _llk_math_wait_for_dest_available_();
            for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
            {
                _llk_math_reduce_<POOL_TYPE, REDUCE_DIM>(tile /*dest_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
            }
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "llk_sync.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TEST_FACE_R_DIM    = params.TEST_FACE_R_DIM;
    const std::uint32_t num_faces          = params.num_faces;
    const Operand& buffer_Res              = params.buffer_Res;
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
#endif
    const std::uint32_t num_blocks = dest_section_count(TILE_CNT, NUM_TILES_IN_BLOCK);
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(
        ckernel::tensor_shape_from_num_faces(TEST_FACE_R_DIM, num_faces), L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
    constexpr std::uint8_t bfd_pack = ckernel::trisc::BFD_ID_PACK0;

    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(bfd_pack, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles_per_pack*/);
    // Start on bank 0 like math: a previous program that ran an odd number of SyncHalf sections leaves the packer on bank 1.
    _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
    if constexpr (unpack_to_dest)
    {
        for (std::uint32_t block = 0; block < num_blocks; ++block)
        {
            _llk_packer_wait_for_math_done_();
            if (block % 3 == 1)
            {
                _llk_tile_regs_borrow_<DstClient::PACK, DstClient::UNPACK>();
                _llk_tile_regs_return_<DstClient::PACK, DstClient::UNPACK>();
                _llk_tile_regs_borrow_<DstClient::PACK, DstClient::SFPU>();
                _llk_tile_regs_return_<DstClient::PACK, DstClient::SFPU>();
            }
            for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
            {
                _llk_pack_(tile /*dest_idx*/, block * NUM_TILES_IN_BLOCK + tile, ckernel::DEFAULT_TENSOR_SHAPE);
            }
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
        for (std::uint32_t block = 0; block < num_blocks; ++block)
        {
            _llk_packer_wait_for_math_done_();
            for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
            {
                _llk_pack_(tile /*dest_idx*/, block * NUM_TILES_IN_BLOCK + tile, ckernel::DEFAULT_TENSOR_SHAPE);
            }
            _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
        }
        _llk_pack_reduce_mask_clear_();
    }
}
#endif

#ifdef LLK_TRISC_ISOLATE_SFPU
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu/ckernel_sfpu_negative.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
#endif
    if constexpr (unpack_to_dest)
    {
        _llk_math_eltwise_sfpu_init_();
        const auto num_blocks = dest_section_count(TILE_CNT, NUM_TILES_IN_BLOCK);
        for (std::uint32_t block = 0; block < num_blocks; ++block)
        {
            if (block % 3 == 1)
            {
                _llk_tile_regs_borrow_<DstClient::PACK, DstClient::SFPU>();
            }
            else
            {
                _llk_tile_regs_borrow_<DstClient::MATH, DstClient::SFPU>();
            }
            const auto expected_base = dest_sync == DstSync::SyncHalf ? (block % 2) * (is_fp32_dest_acc_en ? 256 : 512) : 0;
            LLK_ASSERT(trisc::_get_dest_buffer_base_() == expected_base, "SFPU borrowed the wrong DST bank");
            for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
            {
                if (static_cast<DataFormat>(formats.math) == DataFormat::Int32)
                {
                    _llk_math_eltwise_unary_sfpu_params_(sfpu::_calculate_negative_int_<false, 8>, tile);
                }
                else
                {
                    _llk_math_eltwise_unary_sfpu_params_(sfpu::_calculate_negative_<false, 8>, tile);
                }
            }
            if (block % 3 == 1)
            {
                _llk_tile_regs_return_<DstClient::PACK, DstClient::SFPU>();
            }
            else
            {
                _llk_tile_regs_return_<DstClient::MATH, DstClient::SFPU>();
            }
        }
    }
}
#endif
