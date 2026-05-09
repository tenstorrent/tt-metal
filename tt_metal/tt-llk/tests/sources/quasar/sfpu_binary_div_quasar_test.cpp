// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

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
    const std::uint32_t buf_desc_id          = 0;
    const std::uint32_t num_tiles_per_unpack = params.TILE_CNT;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();

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

    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, num_tiles_per_unpack);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0);

    _llk_unpack_dest_dvalid_section_done_<dest_sync>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "params.h"
#include "sfpu/ckernel_sfpu_binary.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false>(src_format, src_format);

    // The BH-style div helper iterates 1 face's worth of SFP rows per call and
    // is invoked once per face from the harness loop below. SFP_ROWS = 2 on
    // Quasar, so a face (TEST_FACE_R_DIM = 16 rows) corresponds to 8 SFP iters.
    const std::uint32_t num_sfpu_iterations = params.TEST_FACE_R_DIM / ckernel::math::SFP_ROWS;

    _llk_math_eltwise_binary_sfpu_init_();

    // Programmable-constant init for the sfpi reciprocal helper. Sets
    // `sfpi::vConstFloatPrgm0 = 2.0f` (the constant used by the in-helper
    // Newton-Raphson refinement). No-op when APPROXIMATION_MODE = true.
    _sfpu_binary_init_<false /*APPROXIMATION_MODE*/, SFPU_BINARY_OPERATION>();

    _llk_math_eltwise_binary_sfpu_start_(0);

    for (std::uint32_t face = 0; face < NUM_FACES; face++)
    {
        // BH-style sfpi vFloat divide: reads operand tiles via
        // `dst_reg[idx * 32]` (sfpi tile stride) from the current dest base
        // and writes the result at `dst_reg[dst_idx * 32]`. The dest base is
        // advanced one face between calls by `_llk_math_eltwise_binary_sfpu_
        // inc_dst_face_addr_()` below.
        _calculate_sfpu_binary_div_<false /*APPROXIMATION_MODE*/, SFPU_BINARY_OPERATION, is_fp32_dest_acc_en>(
            num_sfpu_iterations, params.SRC0_TILE_IDX, params.SRC1_TILE_IDX, params.DST_TILE_IDX);
        _llk_math_eltwise_binary_sfpu_inc_dst_face_addr_();
    }

    _llk_math_eltwise_binary_sfpu_done_();

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
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
    std::uint32_t const buf_desc_id = 8;

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
    _llk_pack_init_(buf_desc_id, 1);

    _llk_pack_(params.DST_TILE_IDX, 0);

    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
