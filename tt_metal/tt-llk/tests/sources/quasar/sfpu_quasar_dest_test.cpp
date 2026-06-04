// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Unified Quasar SFPU test — unpack-to-dest data path.
//
// Data path: UNPACK → Dest directly → SFPU → Pack
// dvalid chain on all three threads: {UNPACK, SFPU, PACK}
//
// unpack_to_dest is always true on this path.  Use sfpu_quasar_srca_test.cpp
// for the SrcA / FPU path.
//
// Same op-selection preprocessor defines as sfpu_quasar_srca_test.cpp.
// SFPU dispatch block is identical between the two files; only the dvalid
// setup and the absence of the FPU datacopy step differ.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

inline bool is_int_fill_format(DataFormat fmt)
{
    return fmt == DataFormat::Int32 || fmt == DataFormat::Int16 || fmt == DataFormat::Int8 || fmt == DataFormat::UInt8;
}

// ---------------------------------------------------------------------------
// LLK_TRISC_UNPACK
// ---------------------------------------------------------------------------
#ifdef LLK_TRISC_UNPACK

#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t buf_desc_id = 0;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    // hw_configure for unpack-to-dest: fill needs int vs float dest mode
#ifdef SFPU_IS_FILL_OP
    {
        const bool is_int_fill = is_int_fill_format(static_cast<DataFormat>(formats.unpack_A_src));
        if (is_int_fill)
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>();
        }
        else
        {
            _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>();
        }
    }
#else
    _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>();
#endif

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

    _llk_unpack_configure_unary_<p_unpacr::UNP_DEST>(td_val);

    _llk_unpack_unary_operand_init_<p_unpacr::UNP_DEST, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, params.TILE_CNT);
    _llk_unpack_unary_operand_<p_unpacr::UNP_DEST>(0);
    _llk_unpack_dest_dvalid_section_done_<dest_sync>();
}

#endif // LLK_TRISC_UNPACK

// ---------------------------------------------------------------------------
// LLK_TRISC_MATH
// ---------------------------------------------------------------------------
#ifdef LLK_TRISC_MATH

constexpr bool is_int_fpu_en = false;

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_ternary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "params.h"
#include "sfpu_quasar_ops.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    DataFormat src_format = static_cast<DataFormat>(formats.math);

    // hw_configure: needed even on the dest path to set SFPU lane mode
#ifdef SFPU_IS_FILL_OP
    {
        const bool is_int_fill = is_int_fill_format(static_cast<DataFormat>(formats.unpack_A_src));
        if (is_int_fill)
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>(src_format, src_format);
        }
        else
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>(src_format, src_format);
        }
    }
#elif defined(SFPU_IS_BINARY_INT_OP)
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, false /*int32_dest*/>(src_format, src_format);
#elif defined(SFPU_IS_BINARY_MAX_MIN_OP)
    {
        DataFormat pack_src_format = static_cast<DataFormat>(formats.pack_src);
        if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Float32)
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, true /*fp32_dest*/, false /*int32_dest*/>(src_format, src_format);
        }
        else if (is_fp32_dest_acc_en && pack_src_format == DataFormat::Int32)
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>(src_format, src_format);
        }
        else
        {
            _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, false /*int32_dest*/>(src_format, src_format);
        }
    }
#else
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);
#endif

    // No FPU datacopy — unpack already placed data in Dest

    // SFPU dispatch (identical block to sfpu_quasar_srca_test.cpp)
    const std::uint32_t n = params.TEST_FACE_R_DIM / ckernel::math::SFP_ROWS;
    _llk_math_eltwise_sfpu_init_();

#ifdef SFPU_IS_FILL_OP
    {
        constexpr int num_sfpu_iter_fill = static_cast<int>(FACE_R_DIM / SFP_ROWS);
        const bool is_int_fill           = is_int_fill_format(static_cast<DataFormat>(formats.unpack_A_src));
        if (is_int_fill)
        {
            constexpr std::uint32_t FILL_INT_VALUE = 5;
            for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
            {
                _llk_math_eltwise_unary_sfpu_params_(
                    [](std::uint32_t v) { ckernel::sfpu::_calculate_fill_int_<FILL_INT_FORMAT, num_sfpu_iter_fill>(v); }, params.DST_INDEX + i, FILL_INT_VALUE);
            }
        }
        else
        {
            constexpr float FILL_CONST = 5.0f;
            for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
            {
                _llk_math_eltwise_unary_sfpu_params_([](float v) { ckernel::sfpu::_calculate_fill_<num_sfpu_iter_fill>(v); }, params.DST_INDEX + i, FILL_CONST);
            }
        }
    }
#elif defined(SFPU_IS_BINARY_INT_OP)
    {
        constexpr int tile_stride = NUM_FACES * FACE_R_DIM;
        const int in0_offset      = params.SRC0_TILE_IDX * tile_stride;
        const int in1_offset      = params.SRC1_TILE_IDX * tile_stride;
        const int out_offset      = params.DST_TILE_IDX * tile_stride;
#if defined(SFPU_INT_OP_MUL)
        _llk_math_eltwise_binary_sfpu_params_<false>(_mul_int32_<false, 8>, 0, static_cast<std::uint32_t>(n), in0_offset, in1_offset, out_offset);
#elif defined(SFPU_INT_OP_GT)
        _llk_math_eltwise_binary_sfpu_params_<false>(
            calculate_binary_comp_int32<false, 8, SfpuType::gt>, 0, static_cast<std::uint32_t>(n), in0_offset, in1_offset, out_offset);
#elif defined(SFPU_INT_OP_LT)
        _llk_math_eltwise_binary_sfpu_params_<false>(
            calculate_binary_comp_int32<false, 8, SfpuType::lt>, 0, static_cast<std::uint32_t>(n), in0_offset, in1_offset, out_offset);
#elif defined(SFPU_INT_OP_LE)
        _llk_math_eltwise_binary_sfpu_params_<false>(
            calculate_binary_comp_int32<false, 8, SfpuType::le>, 0, static_cast<std::uint32_t>(n), in0_offset, in1_offset, out_offset);
#elif defined(SFPU_INT_OP_GE)
        _llk_math_eltwise_binary_sfpu_params_<false>(
            calculate_binary_comp_int32<false, 8, SfpuType::ge>, 0, static_cast<std::uint32_t>(n), in0_offset, in1_offset, out_offset);
#else
        // default: add_int
        _llk_math_eltwise_binary_sfpu_params_<false>(
            _add_int_<false, 8, 0, false>, 0, src_format, static_cast<std::uint32_t>(n), in0_offset, in1_offset, out_offset);
#endif
    }
#elif defined(SFPU_BINARY_OPERATION)
    {
        sfpu_binary_init<false /*APPROXIMATION_MODE*/, SFPU_BINARY_OPERATION>();
        _llk_math_eltwise_sfpu_params_(
            calculate_sfpu_binary<false, SFPU_BINARY_OPERATION, is_fp32_dest_acc_en>,
            0,
            static_cast<std::uint32_t>(n),
            params.SRC0_TILE_IDX,
            params.SRC1_TILE_IDX,
            params.DST_TILE_IDX);
    }
#elif defined(SFPU_IS_BINARY_MAX_MIN_OP)
    {
        DataFormat math_format = static_cast<DataFormat>(formats.math);
        _init_binary_max_min_();
        if (math_format == DataFormat::Int32)
        {
            _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_binary_max_min<DataFormat::Int32, IS_MAX_OP, 8>, params.DST_INDEX, 0U, 1U, 2U);
        }
        else
        {
            _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_binary_max_min<DataFormat::Float32, IS_MAX_OP, 8>, params.DST_INDEX, 0U, 1U, 2U);
        }
    }
#elif defined(SFPU_IS_WHERE_OP)
    {
        _llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>();
        init_where();
        _llk_math_eltwise_ternary_sfpu_params_(sfpu::calculate_where<false>, 0u, 1u, 2u, 0u, VECTOR_MODE);
    }
#else
    // Standard unary ops + swiglu
    if constexpr (SFPU_UNARY_OPERATION == SfpuType::swiglu)
    {
        _llk_math_eltwise_sfpu_start_(params.DST_INDEX);
        ckernel::sfpu::_init_swiglu_();
        const std::uint32_t DEST_ROWS_PER_TILE = params.num_faces * params.TEST_FACE_R_DIM;
        for (std::uint32_t face = 0; face < params.num_faces; ++face)
        {
            ckernel::sfpu::_calculate_swiglu_(n, 0, DEST_ROWS_PER_TILE, 2 * DEST_ROWS_PER_TILE);
            _llk_math_eltwise_sfpu_inc_dst_face_addr_();
        }
        _llk_math_eltwise_sfpu_done_();
    }
    else
    {
        quasar_sfpu_init<SFPU_UNARY_OPERATION>();
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            quasar_sfpu_call<SFPU_UNARY_OPERATION>(static_cast<int>(params.DST_INDEX + i), static_cast<int>(n));
        }
    }
#endif

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
}

#endif // LLK_TRISC_MATH

// ---------------------------------------------------------------------------
// LLK_TRISC_PACK
// ---------------------------------------------------------------------------
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
    const std::uint32_t buf_desc_id = 8;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = params.buffer_Res[0] / 16;
    bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);

#if defined(SFPU_IS_BINARY_INT_OP) || defined(SFPU_BINARY_OPERATION)
    _llk_pack_init_(buf_desc_id, 1);
    _llk_pack_(params.DST_TILE_IDX, 0);
#elif defined(SFPU_IS_BINARY_MAX_MIN_OP)
    _llk_pack_init_(buf_desc_id, 1);
    _llk_pack_(params.DST_INDEX + 2, 0);
#elif defined(SFPU_IS_WHERE_OP)
    _llk_pack_init_(buf_desc_id, 1);
    _llk_pack_(params.DST_INDEX, 0);
#else
    // fill, unary ops, swiglu
    if constexpr (SFPU_UNARY_OPERATION == SfpuType::swiglu)
    {
        _llk_pack_init_(buf_desc_id, 1);
        _llk_pack_(params.DST_INDEX + 2, 0);
    }
    else
    {
        _llk_pack_init_(buf_desc_id, params.TILE_CNT);
        _llk_pack_(params.DST_INDEX, 0);
    }
#endif

    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_PACK
