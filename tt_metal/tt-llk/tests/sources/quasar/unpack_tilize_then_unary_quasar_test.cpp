// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unpack tilize followed by a plain unary unpack on the same unpacker, with no teardown between them.
// tilize_uninit() is a no-op on Quasar, so the second op relies only on its own init and execute: any unpacker
// state tilize left behind (L1 source offset, tile/face/row counters, stride config) would corrupt the second
// half of the result. Section 0 of the result holds the tilized input, section 1 the input copied tile by tile.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

// Op 0 is the tilize, op 1 the unary unpack; each fills one DEST section.
constexpr std::uint32_t NUM_OPS = 2;

#ifdef LLK_TRISC_UNPACK

#include "ckernel_buf_desc.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    static_assert(!is_fp32_dest_acc_en, "This test covers the SrcA/SrcB tilize without FP32 dest");
    const FormatConfig& formats             = params.formats;
    const std::uint32_t TILE_CNT            = params.TILE_CNT;
    const Operand& buffer_A                 = params.buffer_A;
    const Operand& buffer_B                 = params.buffer_B;
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();

    const std::uint32_t l1_addr_16B   = (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B) ? L1_ADDRESS(buffer_B[0]) : L1_ADDRESS(buffer_A[0]);
    constexpr auto unp_res            = (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B) ? ckernel::trisc::BFD_ID_UNP1 : ckernel::trisc::BFD_ID_UNP0;
    constexpr std::uint8_t bfd_unpack = unp_res;

    _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));

    // Op 0: unpack tilize of the whole input.
    if (_llk_unpack_tilize_is_strided_(tensor_shape))
    {
        ckernel::trisc::program_bfd<unp_res, ckernel::trisc::L1AccessMode::Strided>(tensor_shape, l1_addr_16B, formats.unpack_A_src);
    }
    else
    {
        ckernel::trisc::program_bfd<unp_res, ckernel::trisc::L1AccessMode::Continuous>(tensor_shape, l1_addr_16B, formats.unpack_A_src);
    }
    _llk_unpack_tilize_init_<UNPACKER_ENGINE_SEL, false /*EN_32BIT_DEST*/>(bfd_unpack, FULL_CT_DIM, BLOCK_CT_DIM, tensor_shape);
    const std::uint32_t y_stride_external = FULL_CT_DIM * tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;
    for (std::uint32_t y = 0; y < BLOCK_RT_DIM; y++)
    {
        for (std::uint32_t x = 0; x < FULL_CT_DIM; x += BLOCK_CT_DIM)
        {
            _llk_unpack_tilize_<UNPACKER_ENGINE_SEL>(y * y_stride_external + x /*l1_tile_idx*/, BLOCK_CT_DIM, tensor_shape);
        }
    }

    // No teardown: op 1 starts from whatever state op 0 left.

    // Op 1: plain unary unpack of the same L1 data, one tile per call.
    ckernel::trisc::program_bfd<unp_res, ckernel::trisc::L1AccessMode::Continuous>(tensor_shape, l1_addr_16B, formats.unpack_A_src);
    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, false /*EN_32BIT_DEST*/>(bfd_unpack, tensor_shape, 1 /*num_tiles*/);
    for (std::uint32_t i = 0; i < TILE_CNT; i++)
    {
        _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(i /*l1_tile_idx*/, tensor_shape);
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
    const FormatConfig& formats  = params.formats;
    const std::uint32_t TILE_CNT = params.TILE_CNT;

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
    const DataFormat math_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
    _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(
        params.num_faces * params.TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);

    for (std::uint32_t op = 0; op < NUM_OPS; op++)
    {
        for (std::uint32_t i = 0; i < TILE_CNT; i++)
        {
            _llk_math_eltwise_unary_datacopy_(i);
        }
        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats             = params.formats;
    const std::uint32_t TILE_CNT            = params.TILE_CNT;
    const Operand& buffer_Res               = params.buffer_Res;
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
    constexpr std::uint8_t bfd_pack = ckernel::trisc::BFD_ID_PACK0;
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(bfd_pack, tensor_shape, TILE_CNT);

    for (std::uint32_t op = 0; op < NUM_OPS; op++)
    {
        _llk_pack_(0 /*start_math_dest_tile_idx*/, op * TILE_CNT /*start_l1_tile_idx*/, tensor_shape);
        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif
