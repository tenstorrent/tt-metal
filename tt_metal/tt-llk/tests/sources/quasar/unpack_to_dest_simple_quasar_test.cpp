// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_math_common.h"
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
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    // Dest dvalid handshake is UNPACK -> PACK; the FPU is not involved.
    set_up_unpack_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();

    // Dest width follows the variant;
    _llk_math_upk_to_dest_hw_configure_<true /*EN_IMPLIED_MATH_FORMAT*/, is_fp32_dest_acc_en, false /*EN_INT32_MATH_FORMAT*/>();

    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(tensor_shape, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);

    _llk_unpack_configure_unary_<p_unpacr::UNP_DEST>(static_cast<DataFormat>(formats.unpack_A_dst));
    _llk_unpack_unary_operand_init_<p_unpacr::UNP_DEST, false /*TRANSPOSE_EN*/, is_fp32_dest_acc_en>(
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), tensor_shape, TILE_CNT);

    _llk_unpack_unary_operand_<p_unpacr::UNP_DEST>(0 /*l1_tile_idx*/, tensor_shape);
    _llk_unpack_dest_dvalid_section_done_<DstSync::SyncFull>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    // Unpack writes straight into Dest; math takes no part in this path.
    (void)params;
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t TILE_CNT = params.TILE_CNT;
    const Operand& buffer_Res    = params.buffer_Res;
#endif
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    set_up_unpack_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();

    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), tensor_shape, TILE_CNT);

    _llk_pack_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/, tensor_shape);
    _llk_pack_dest_dvalid_section_done_<DstSync::SyncFull, is_fp32_dest_acc_en>();
}
#endif
