// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-04-23_fill_quasar_e9608a59

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

using namespace ckernel;
#include "params.h" // FILL_INT_FORMAT, IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en

// The kernel path (int_fill vs fill) is selected at runtime from formats.unpack_A_src.
// FILL_INT_FORMAT (forwarded by the harness) drives the SFPMEM store mode used by
// _calculate_fill_int_. Because the kernel compiles both branches, the harness must
// always pass a FILL_INT_FORMAT that is safe for _calculate_fill_int_'s static_assert
// (one of Int32/Int16/Int8/UInt8); on float-fill variants it is a placeholder that
// is never executed at runtime.

// Returns true when the unpack source format is one of the integer formats supported
// by _calculate_fill_int_ (Int32/Int16/Int8/UInt8).
inline bool is_int_fill_format(DataFormat fmt)
{
    return fmt == DataFormat::Int32 || fmt == DataFormat::Int16 || fmt == DataFormat::Int8 || fmt == DataFormat::UInt8;
}

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
    const std::uint32_t num_tiles   = params.TILE_CNT;

    // fill always uses unpack_to_dest (SFPU test — no FPU datacopy path)
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    const bool is_int_fill = is_int_fill_format(static_cast<DataFormat>(formats.unpack_A_src));
    if (is_int_fill)
    {
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>();
    }
    else
    {
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>();
    }

    buffer_descriptor_u bd_val = {0};

    bd_val.f.l1_addr_16B = L1_ADDRESS(params.buffer_A[0]);
    bd_val.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_val.f.x_dim       = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim       = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim       = params.num_faces;

    tdma_descriptor_t td_val;
    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = buf_desc_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);
    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);

    _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, num_tiles);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0);

    _llk_unpack_dest_dvalid_section_done_<dest_sync>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "experimental/ckernel_sfpu_fill.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // fill always uses unpack_to_dest path
    set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    DataFormat math_format = static_cast<DataFormat>(formats.math);
    const bool is_int_fill = is_int_fill_format(static_cast<DataFormat>(formats.unpack_A_src));

    if (is_int_fill)
    {
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>(math_format, math_format);
    }
    else
    {
        _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>(math_format, math_format);
    }

    constexpr int num_sfpu_iterations = static_cast<int>(FACE_R_DIM / SFP_ROWS);

    _llk_math_eltwise_unary_sfpu_init_();

    if (is_int_fill)
    {
        // Int path: _calculate_fill_int_ writes FILL_INT_VALUE to every element of Dest
        // via SFPLOADI + SFPSTORE; the SFPMEM store mode is selected by FILL_INT_FORMAT
        // at compile time (no runtime dispatch).
        constexpr std::uint32_t FILL_INT_VALUE = 5;

        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_math_eltwise_unary_sfpu_params_(
                [](std::uint32_t v) { ckernel::sfpu::_calculate_fill_int_<FILL_INT_FORMAT, num_sfpu_iterations>(v); }, params.DST_INDEX + i, FILL_INT_VALUE);
        }
    }
    else
    {
        // Float path: _calculate_fill_ uses SFPU DEFAULT store mode, which supports
        // all float formats (Float16, Float16_b, Float32).
        constexpr float FILL_CONST = 5.0f;

        for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
        {
            _llk_math_eltwise_unary_sfpu_params_([](float v) { _calculate_fill_<num_sfpu_iterations>(v); }, params.DST_INDEX + i, FILL_CONST);
        }
    }

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
}

#endif

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
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = params.TILE_CNT;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

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

    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_(buf_desc_id, num_tiles_per_pack);
    _llk_pack_(params.DST_INDEX, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
