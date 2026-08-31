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
#include "llk_unpack_common.h"
#include "llk_unpack_reduce.h"
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

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_reduce.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

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

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
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
    _llk_pack_reduce_mask_config_<REDUCE_DIM>(ckernel::DEFAULT_TENSOR_SHAPE);
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_(0 /*dest_idx*/, i, ckernel::DEFAULT_TENSOR_SHAPE);
        _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
    }
    _llk_pack_reduce_mask_clear_();
}
#endif
