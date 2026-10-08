// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Pack untilize under the semaphore math <-> pack sync scheme, the only scheme that moves
// dest_register_offset, so the packer reads dest bank 1 in DestSync::SyncHalf.
// Each dest section holds BLOCK_RT_DIM tile rows; tile row r of a section sits at dest tile r * BLOCK_CT_DIM.

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
    const std::uint32_t TILE_CNT = params.TILE_CNT;
    const Operand& buffer_A      = params.buffer_A;
#endif
    const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);

    // CFG persists across variants in a session; the dest-dvalid scheme must stay disarmed.
    set_up_zero_dest_dvalid_handshake_for_unpack();

    const auto bfd_unpack =
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(tensor_shape_A, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);

    if constexpr (is_fp32_dest_acc_en)
    {
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
            static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
    }
    else
    {
        _llk_unpack_configure_unary_<p_unpacr::UNP_A>(static_cast<DataFormat>(formats.unpack_A_dst));
    }
    _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, is_fp32_dest_acc_en>(bfd_unpack, tensor_shape_A, TILE_CNT);

    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(0 /*l1_tile_idx*/, tensor_shape_A);
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
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
#endif
    set_up_zero_dest_dvalid_handshake_for_math();

    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.math), static_cast<DataFormat>(formats.math));
    _llk_math_pack_sync_init_<dest_sync>();
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);

    static_assert(FULL_RT_DIM % BLOCK_RT_DIM == 0, "FULL_RT_DIM must be divisible by BLOCK_RT_DIM");

    for (std::uint32_t section = 0; section < FULL_RT_DIM / BLOCK_RT_DIM; section++)
    {
        _llk_math_wait_for_dest_available_();
        for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
        {
            for (std::uint32_t block_ct = 0; block_ct < BLOCK_CT_DIM; block_ct++)
            {
                _llk_math_eltwise_unary_datacopy_(block_rt * BLOCK_CT_DIM + block_ct);
            }
        }
        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "cpack_common.h"
#include "llk_bfd_alloc.h"
#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_Res = params.buffer_Res;
#endif
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    set_up_zero_dest_dvalid_handshake_for_pack();

    std::uint8_t bfd_pack;
    if (tensor_shape.face_r_dim < ckernel::pack::PACR_STRIDE_OFFSET_ROWS)
    {
        // PACR_STRIDE quirk: tiny-tiles index L1 rows as tiles, so the BD is built with y_dim = 1.
        bfd_pack = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0, ckernel::trisc::L1AccessMode::Strided>(
            tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
    }
    else
    {
        bfd_pack = ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0, ckernel::trisc::L1AccessMode::Continuous>(
            tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
    }

    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();

    if (tensor_shape.total_num_faces() == NUM_FACES)
    {
        _llk_pack_untilize_init_<FULL_CT_DIM, BLOCK_CT_DIM>(bfd_pack, tensor_shape);
    }
    else
    {
        _llk_pack_untilize_strided_init_<FULL_CT_DIM, BLOCK_CT_DIM>(bfd_pack, tensor_shape);
    }

    const std::uint32_t y_stride_external = FULL_CT_DIM * tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;

    for (std::uint32_t section = 0; section < FULL_RT_DIM / BLOCK_RT_DIM; section++)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
        {
            const std::uint32_t y        = section * BLOCK_RT_DIM + block_rt;
            const std::uint32_t dest_idx = block_rt * BLOCK_CT_DIM;
            if (tensor_shape.total_num_faces() == NUM_FACES)
            {
                _llk_pack_untilize_set_dst_offset_(tensor_shape, y * y_stride_external);
                _llk_pack_untilize_(dest_idx, 0 /*l1_tile_idx*/);
            }
            else
            {
                _llk_pack_untilize_strided_<FULL_CT_DIM>(bfd_pack, tensor_shape, y * y_stride_external, dest_idx /*src_tile_idx*/);
            }
        }
        _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
