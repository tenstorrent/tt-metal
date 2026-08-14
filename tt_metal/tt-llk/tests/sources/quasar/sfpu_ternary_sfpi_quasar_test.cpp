// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Functional harness for the SFPI implementations of addcdiv, lerp, and
// snake_beta.  Three consecutive tiles in buffer_A are staged into Dest[0:3),
// the selected ternary kernel writes Dest[0], and PACK emits that one result.

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

    LLK_ASSERT(params.TILE_CNT == 3, "Ternary SFPI harness expects exactly three input tiles");
    LLK_ASSERT(params.TEST_FACE_R_DIM == 16 && params.TEST_FACE_C_DIM == 16, "Ported ternary SFPI kernels require 16x16 faces");
    LLK_ASSERT(params.num_faces == 4, "Ported ternary SFPI kernels require full 32x32 tiles");

    constexpr auto unpack_dest = unpack_to_dest ? dest_dvalid_client::UNPACK : dest_dvalid_client::FPU;
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({unpack_dest, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    if constexpr (unpack_to_dest)
    {
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
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
        // Quasar's 32-bit A2D fallback consumes both SrcA and SrcB dvalid.
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
// Load the Quasar compatibility layer before any ported kernel so its
// comparison declarations precede the reciprocal/log include cycle.
#include "sfpu/ckernel_sfpu_compat.h"
#if defined(QUASAR_SFPI_TERNARY_ADDCDIV)
#include "llk_sfpu/ckernel_sfpu_addcdiv.h"
#elif defined(QUASAR_SFPI_TERNARY_LERP)
#include "llk_sfpu/ckernel_sfpu_lerp.h"
#elif defined(QUASAR_SFPI_TERNARY_SNAKE_BETA)
#include "llk_sfpu/ckernel_sfpu_snake_beta.h"
#else
#error "A Quasar SFPI ternary operation must be selected"
#endif
#include "llk_sfpu/llk_math_eltwise_ternary_sfpu_macros.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

template <DataFormat sfpu_data_format>
inline void call_ternary_sfpi(
    std::uint32_t dst_in0,
    std::uint32_t dst_in1,
    std::uint32_t dst_in2,
    std::uint32_t dst_out,
    VectorMode vector_mode,
    [[maybe_unused]] std::uint32_t scalar_bits)
{
    constexpr bool approximation_mode = false;
    constexpr int iterations          = 8;

#if defined(QUASAR_SFPI_TERNARY_ADDCDIV)
    SFPU_TERNARY_CALL(
        dest_sync,
        is_fp32_dest_acc_en,
        calculate_addcdiv,
        (approximation_mode, is_fp32_dest_acc_en, sfpu_data_format, iterations),
        dst_in0,
        dst_in1,
        dst_in2,
        dst_out,
        vector_mode,
        scalar_bits);
#elif defined(QUASAR_SFPI_TERNARY_LERP)
    SFPU_TERNARY_CALL(
        dest_sync,
        is_fp32_dest_acc_en,
        calculate_lerp,
        (approximation_mode, is_fp32_dest_acc_en, sfpu_data_format, iterations),
        dst_in0,
        dst_in1,
        dst_in2,
        dst_out,
        vector_mode);
#elif defined(QUASAR_SFPI_TERNARY_SNAKE_BETA)
    SFPU_TERNARY_CALL(
        dest_sync,
        is_fp32_dest_acc_en,
        calculate_snake_beta,
        (approximation_mode, is_fp32_dest_acc_en, sfpu_data_format, iterations),
        dst_in0,
        dst_in1,
        dst_in2,
        dst_out,
        vector_mode);
#endif
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    LLK_ASSERT(params.TILE_CNT == 3, "Ternary SFPI harness expects exactly three input tiles");
    LLK_ASSERT(params.TEST_FACE_R_DIM == 16 && params.TEST_FACE_C_DIM == 16, "Ported ternary SFPI kernels require 16x16 faces");
    LLK_ASSERT(params.num_faces == 4, "Ported ternary SFPI kernels require full 32x32 tiles");

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
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>(math_format, math_format);

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

#if defined(QUASAR_SFPI_TERNARY_ADDCDIV)
    SFPU_TERNARY_INIT_FN(addcdiv, sfpu::init_addcdiv, (false));
#elif defined(QUASAR_SFPI_TERNARY_LERP)
    SFPU_TERNARY_INIT(lerp);
#elif defined(QUASAR_SFPI_TERNARY_SNAKE_BETA)
    SFPU_TERNARY_INIT_FN(snake_beta, sfpu::snake_beta_init, (false));
#endif

    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);
    const VectorMode vector_mode            = tensor_shape.total_num_faces() == 4   ? VectorMode::RC
                                              : tensor_shape.total_num_faces() == 1 ? VectorMode::None
                                              : tensor_shape.num_faces_r_dim == 1   ? VectorMode::R
                                                                                    : VectorMode::C;

    const std::uint32_t dst_base = params.DST_INDEX;
    const DataFormat sfpu_format = static_cast<DataFormat>(formats.sfpu_math);
    if (sfpu_format == DataFormat::Float32 || sfpu_format == DataFormat::Tf32)
    {
        call_ternary_sfpi<DataFormat::Float32>(dst_base, dst_base + 1, dst_base + 2, dst_base, vector_mode, params.TERNARY_SCALAR_BITS);
    }
    else
    {
        // The L1 Float16 and MX cases retain their runtime SFPU configuration;
        // the header's format template is only a Float32-vs-16-bit contract.
        call_ternary_sfpi<DataFormat::Float16_b>(dst_base, dst_base + 1, dst_base + 2, dst_base, vector_mode, params.TERNARY_SCALAR_BITS);
    }

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
