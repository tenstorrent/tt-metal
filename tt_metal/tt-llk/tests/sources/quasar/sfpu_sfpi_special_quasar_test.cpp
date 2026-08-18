// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Functional harness for the full-tile-only SFPI kernels that do not fit the
// regular elementwise-unary operation table.  The reduction kernels use fixed
// face/register offsets, and add_int uses dst_reg[32] for the second full tile.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

#include "cfg_defines.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    constexpr std::uint32_t buf_desc_id     = 0;
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);

    LLK_ASSERT(params.TEST_FACE_R_DIM == 16 && params.TEST_FACE_C_DIM == 16, "Special SFPI kernels require 16x16 faces");
    LLK_ASSERT(params.num_faces == 4, "Special SFPI kernels require one full 32x32 tile");

    constexpr auto unpack_dest = unpack_to_dest ? dest_dvalid_client::UNPACK : dest_dvalid_client::FPU;
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({unpack_dest, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    if constexpr (unpack_to_dest)
    {
        if constexpr (SPECIAL_DATA_FORMAT == DataFormat::Int32)
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>();
        }
        else
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>();
        }
    }

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_A[0]);
    bd_val.f.format            = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    tdma_descriptor_t td_val;
    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = buf_desc_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);
    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);

    if constexpr (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
    }

    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, tensor_shape, params.TILE_CNT);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/, tensor_shape);

    if constexpr (unpack_to_dest)
    {
        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"
// The compatibility layer must precede the exact ported headers.
#include "llk_sfpu/ckernel_sfpu_alt_complex_rotate90.h"
#include "llk_sfpu/ckernel_sfpu_bitwise.h"
#include "llk_sfpu/ckernel_sfpu_int_sum.h"
#include "llk_sfpu/ckernel_sfpu_tiled_prod.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpu/ckernel_sfpu_compat.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

namespace
{

constexpr std::uint32_t kRotate90  = 0;
constexpr std::uint32_t kBitAnd    = 1;
constexpr std::uint32_t kBitOr     = 2;
constexpr std::uint32_t kBitXor    = 3;
constexpr std::uint32_t kSumCol    = 4;
constexpr std::uint32_t kSumRow    = 5;
constexpr std::uint32_t kAddOffset = 6;
constexpr std::uint32_t kTiledProd = 7;

inline void init_special_sfpi()
{
    if constexpr (SPECIAL_SFPU_OP == kRotate90)
    {
        alt_complex_rotate90_init();
    }
    else if constexpr (SPECIAL_SFPU_OP == kBitAnd)
    {
        bitwise_and_init();
    }
    else if constexpr (SPECIAL_SFPU_OP == kBitOr)
    {
        bitwise_or_init();
    }
    else if constexpr (SPECIAL_SFPU_OP == kBitXor)
    {
        bitwise_xor_init();
    }
    else if constexpr (SPECIAL_SFPU_OP == kSumCol || SPECIAL_SFPU_OP == kSumRow || SPECIAL_SFPU_OP == kAddOffset)
    {
        sum_int_init<false>();
    }
    else
    {
        tiled_prod_init();
    }
}

inline void call_special_sfpi(std::uint32_t dst_index, std::uint32_t scalar)
{
    if constexpr (SPECIAL_SFPU_OP == kRotate90)
    {
        SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_alt_complex_rotate90, (false, 4), dst_index, VectorMode::RC);
    }
    else if constexpr (SPECIAL_SFPU_OP == kBitAnd)
    {
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            calculate_sfpu_unary_bitwise,
            (false, UnaryBitwiseOp::AND, SPECIAL_DATA_FORMAT, 8),
            dst_index,
            VectorMode::RC,
            scalar);
    }
    else if constexpr (SPECIAL_SFPU_OP == kBitOr)
    {
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            calculate_sfpu_unary_bitwise,
            (false, UnaryBitwiseOp::OR, SPECIAL_DATA_FORMAT, 8),
            dst_index,
            VectorMode::RC,
            scalar);
    }
    else if constexpr (SPECIAL_SFPU_OP == kBitXor)
    {
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            calculate_sfpu_unary_bitwise,
            (false, UnaryBitwiseOp::XOR, SPECIAL_DATA_FORMAT, 8),
            dst_index,
            VectorMode::RC,
            scalar);
    }
    else if constexpr (SPECIAL_SFPU_OP == kSumCol)
    {
        SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_sum_int_col, (false), dst_index, VectorMode::R);
    }
    else if constexpr (SPECIAL_SFPU_OP == kSumRow)
    {
        SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_sum_int_row, (false), dst_index, VectorMode::C);
    }
    else if constexpr (SPECIAL_SFPU_OP == kAddOffset)
    {
        // The ported implementation ignores its argument and reads dst_reg[32].
        // Therefore the only supported offset is exactly one full 32x32 tile.
        SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, add_int, (false, 8), dst_index, VectorMode::RC, 1u);
    }
    else
    {
        SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_tiled_prod, (false, 8), dst_index, VectorMode::RC);
    }
}

} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    LLK_ASSERT(params.TEST_FACE_R_DIM == 16 && params.TEST_FACE_C_DIM == 16, "Special SFPI kernels require 16x16 faces");
    LLK_ASSERT(params.num_faces == 4, "Special SFPI kernels require one full 32x32 tile");
    if constexpr (SPECIAL_SFPU_OP == kAddOffset)
    {
        LLK_ASSERT(params.TILE_CNT == 2, "add_int requires two consecutive full tiles");
    }
    else
    {
        LLK_ASSERT(params.TILE_CNT == 1, "Special unary SFPI harness expects one input tile");
    }

    if constexpr (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    const DataFormat math_format = static_cast<DataFormat>(formats.math);
    if constexpr (SPECIAL_DATA_FORMAT == DataFormat::Int32)
    {
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>(math_format, math_format);
    }
    else
    {
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>(math_format, math_format);
    }

    if constexpr (!unpack_to_dest)
    {
        const std::uint32_t num_rows = params.num_faces * params.TEST_FACE_R_DIM;
        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_rows, 1);
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_math_eltwise_unary_datacopy_(params.DST_INDEX + i);
        }
        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }

    _llk_math_eltwise_sfpu_init_();
    init_special_sfpi();
    call_special_sfpi(params.DST_INDEX, params.SPECIAL_SFPU_SCALAR);

    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    constexpr std::uint32_t buf_desc_id        = 8;
    constexpr std::uint32_t num_tiles_per_pack = 1;
    const ckernel::TensorShape tensor_shape    = TENSOR_SHAPE_FROM_PARAMS(params);

    constexpr auto unpack_dest = unpack_to_dest ? dest_dvalid_client::UNPACK : dest_dvalid_client::FPU;
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({unpack_dest, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_Res[0]);
    bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(tdma_desc, ckernel::ReluConfig::none());
    _llk_pack_init_(buf_desc_id, tensor_shape, num_tiles_per_pack);
    _llk_pack_(params.DST_INDEX, 0 /*l1_tile_idx*/, tensor_shape);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_PACK
