// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Standalone tt-llk test for Quasar pack rows to an L1 address. Every input tile is copied to DEST, and the first
// NUM_ROWS_TO_PACK rows of it are packed into its own two-tile result slot, tile % 4 16B units from the slot start, so
// every run mixes aligned and misaligned targets. The host fills the result buffer with guard bytes before the run and
// checks that nothing outside the packed rows is written.
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"
#include "tensor_shape.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const Operand& buffer_A                = params.buffer_A;
#endif
    constexpr std::uint32_t SELECTED_UNPACKER = unpack_to_dest ? p_unpacr::UNP_DEST : p_unpacr::UNP_A;

    if constexpr (unpack_to_dest)
    {
        set_up_unpack_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
    }

    const auto bfd_unpack =
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(ckernel::DEFAULT_TENSOR_SHAPE, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);

    if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
    }
    else
    {
        _llk_unpack_configure_unary_<SELECTED_UNPACKER>(static_cast<DataFormat>(formats.unpack_A_dst));
    }
    _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(bfd_unpack, ckernel::DEFAULT_TENSOR_SHAPE, NUM_TILES_IN_BLOCK);

    for (std::uint32_t block = 0; block < NUM_BLOCKS; block++)
    {
        _llk_unpack_unary_operand_<SELECTED_UNPACKER>(block * NUM_TILES_IN_BLOCK /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
        if constexpr (unpack_to_dest)
        {
            _llk_unpack_dest_dvalid_section_done_<dest_sync>();
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
#endif
    // Unpack to dest fills DEST directly, so math has nothing to do.
    if constexpr (!unpack_to_dest)
    {
        set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();

        _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.math), static_cast<DataFormat>(formats.math));
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(
            ckernel::trisc::NUM_FACES * FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);

        for (std::uint32_t block = 0; block < NUM_BLOCKS; block++)
        {
            for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; tile++)
            {
                _llk_math_eltwise_unary_datacopy_(tile);
            }
            _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack_common.h"
#include "llk_pack_rows.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t NUM_BLOCKS         = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t NUM_ROWS_TO_PACK   = params.NUM_ROWS_TO_PACK;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
    if constexpr (unpack_to_dest)
    {
        set_up_unpack_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
    }
    else
    {
        set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
    }

    // The L1 base is set on every _llk_pack_rows_ call.
    const auto bfd_pack =
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(ckernel::DEFAULT_TENSOR_SHAPE, 0 /*l1_base_addr*/, formats.pack_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_rows_init_(bfd_pack, NUM_ROWS_TO_PACK);

    for (std::uint32_t block = 0; block < NUM_BLOCKS; block++)
    {
        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; tile++)
        {
            const std::uint32_t input_tile = block * NUM_TILES_IN_BLOCK + tile;
            const std::uint32_t l1_address = L1_ADDRESS(buffer_Res[2 * input_tile]) + input_tile % 4;
            _llk_pack_rows_(bfd_pack, tile /*dest_tile_idx*/, l1_address);
        }
        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
    _llk_pack_rows_uninit_();
}
#endif
