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
//              5 into every DEST lane.
//   T2 pack:   pack the filled DEST tile out to buffer_Res in L1.

// Returns true when the unpack source format is one of the integer formats supported
// by _calculate_fill_int_ (Int32/Int16/Int8/UInt8).
inline bool is_int_fill_format(DataFormat fmt)
{
    return fmt == DataFormat::Int32 || fmt == DataFormat::Int16 || fmt == DataFormat::Int8 || fmt == DataFormat::UInt8;
}

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
    // face geometry from the harness. reg_data_format = unpack_A_dst is the
    // DEST-side (post-conversion) format.
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src);

    // Configure unpacker → init unary operand path → unpack tile 0 from L1 into DEST.
    _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), ckernel::DEFAULT_TENSOR_SHAPE, params.TILE_CNT);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0 /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);

    // Release DEST section to the SFPU consumer.
    _llk_unpack_dest_dvalid_section_done_<dest_sync>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "experimental/ckernel_sfpu_fill.h"
#include "llk_math_common.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
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

    _llk_math_eltwise_sfpu_init_();

    if (is_int_fill)
    {
        // Int path: _calculate_fill_int_ writes 5 to every element of Dest
        // via SFPLOADI + SFPSTORE; the SFPMEM store mode is selected by FILL_INT_FORMAT
        // at compile time (no runtime dispatch).
        // Fill every DEST lane in each tile with integer value 5.
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            SFPU_UNARY_CALL(
                dest_sync, is_fp32_dest_acc_en, _calculate_fill_int_, (FILL_INT_FORMAT, SFPU_ITERATIONS), params.DST_INDEX + i, VectorMode::RC, 5 /*value*/);
        }
    }
    else
    {
        // Float path: _calculate_fill_ uses SFPU DEFAULT store mode, which supports
        // all float formats (Float16, Float16_b, Float32).
        // Walk every tile in DEST starting at DST_INDEX, filling all lanes with 5.0f.
        for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
        {
            SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, _calculate_fill_, (SFPU_ITERATIONS), params.DST_INDEX + i, VectorMode::RC, 5.0f /*value*/);
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
#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    // PACK is the final consumer of the DEST DVALID chain.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    // Destination descriptor: buffer_Res in L1, L1-side format = formats.pack_dst,
    // face geometry from the harness. reg_data_format = pack_src is the DEST-side
    // format the packer reads.
    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst);

    // Configure pack engine 0 → init → pack tile from DST_INDEX into buffer_Res → release section.
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), ckernel::DEFAULT_TENSOR_SHAPE, params.TILE_CNT);
    _llk_pack_(params.DST_INDEX, 0 /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
