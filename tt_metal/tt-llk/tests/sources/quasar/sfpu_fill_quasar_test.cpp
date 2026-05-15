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
//
// Test flow (per tile):
//   T0 unpack: stage buffer_A from L1 into DEST via the unpack-to-dest path.
//              The data is a placeholder — the SFPU overwrites every DEST lane.
//   T1 math:   run _calculate_fill_int_ / _calculate_fill_ to write the constant
//              (FILL_INT_VALUE or FILL_CONST) into every DEST lane.
//   T2 pack:   pack the filled DEST tile out to buffer_Res in L1.

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
    const std::uint32_t buf_desc_id = 0; // T0 source descriptor slot for buffer_A
    const std::uint32_t num_tiles   = params.TILE_CNT;

    // DEST DVALID handshake: T0 is the producer, T1 (SFPU) and T2 (PACK) are the consumers.
    // fill always uses unpack_to_dest (SFPU test — no FPU datacopy path).
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    // hw_configure: int-fill needs DEST in int32 mode; float-fill follows is_fp32_dest_acc_en.
    const bool is_int_fill = is_int_fill_format(static_cast<DataFormat>(formats.unpack_A_src));
    if (is_int_fill)
    {
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, false /*fp32_dest*/, true /*int32_dest*/>();
    }
    else
    {
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*int32_dest*/>();
    }

    // Source descriptor: buffer_A in L1, L1-side format = formats.unpack_A_src,
    // face geometry from the harness.
    buffer_descriptor_u bd_val = {0};

    bd_val.f.l1_addr_16B = L1_ADDRESS(params.buffer_A[0]);
    bd_val.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_val.f.x_dim       = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim       = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim       = params.num_faces;

    // TDMA descriptor: bind the buffer descriptor to slot 0; reg_data_format =
    // unpack_A_dst is the DEST-side (post-conversion) format.
    tdma_descriptor_t td_val;
    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = buf_desc_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);
    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);

    // Configure unpacker → init unary operand path → unpack tile 0 from L1 into DEST.
    _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, num_tiles);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0);

    // Release DEST section to the SFPU consumer.
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
    // Math acts as the SFPU client of the DEST DVALID chain.
    // fill always uses unpack_to_dest path.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    // srcAB hw_configure: srcA/srcB both use formats.math; DEST mode tracks the
    // int-fill / float-fill split (int32 for int fills, otherwise is_fp32_dest_acc_en).
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

    // SFPU iterates SFP_ROWS rows at a time; one face is FACE_R_DIM rows tall.
    constexpr int num_sfpu_iterations = static_cast<int>(FACE_R_DIM / SFP_ROWS);

    _llk_math_eltwise_unary_sfpu_init_();

    if (is_int_fill)
    {
        // Int path: _calculate_fill_int_ writes FILL_INT_VALUE to every element of Dest
        // via SFPLOADI + SFPSTORE; the SFPMEM store mode is selected by FILL_INT_FORMAT
        // at compile time (no runtime dispatch).
        constexpr std::uint32_t FILL_INT_VALUE = 5;

        // Walk every tile in DEST starting at DST_INDEX, filling all lanes with FILL_INT_VALUE.
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

        // Walk every tile in DEST starting at DST_INDEX, filling all lanes with FILL_CONST.
        for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
        {
            _llk_math_eltwise_unary_sfpu_params_([](float v) { _calculate_fill_<num_sfpu_iterations>(v); }, params.DST_INDEX + i, FILL_CONST);
        }
    }

    // Hand DEST off to PACK.
    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    // Drain SFPU/FPU/MOP queues before this thread returns.
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
    std::uint32_t const buf_desc_id        = 8; // T2 destination descriptor slot for buffer_Res
    const std::uint32_t num_tiles_per_pack = params.TILE_CNT;

    // PACK is the final consumer of the DEST DVALID chain.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    // Destination descriptor: buffer_Res in L1, L1-side format = formats.pack_dst,
    // face geometry from the harness.
    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_Res[0]);
    bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    // TDMA descriptor: bind buffer_Res to slot 8; reg_data_format = pack_src is
    // the DEST-side format the packer reads.
    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

    // Configure pack engine 0 → init → pack tile from DST_INDEX into buffer_Res → release section.
    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_(buf_desc_id, num_tiles_per_pack);
    _llk_pack_(params.DST_INDEX, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
