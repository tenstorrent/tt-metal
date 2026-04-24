// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — ternary SFPU where kernel test for Quasar.

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

// UNPACK: unpack 3 tiles packed in buffer_A (cond, true_val, false_val).
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t buf_desc_id          = 0;
    const std::uint32_t num_tiles_per_unpack = params.TILE_CNT;

    if (unpack_to_dest)
    {
        // Direct UNPACK-to-DEST path: UNPACK writes DEST; SFPU reads/writes DEST; PACK reads DEST.
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
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

    if (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        // If Dst is 32b and MATH uses FPU datacopy (MOVA2D → ELWADD fallback), we need both SrcA and SrcB formats configured.
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
    }

    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, num_tiles_per_unpack);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0);

    if (unpack_to_dest)
    {
        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
    }
}

#endif

#ifdef LLK_TRISC_MATH

const bool is_int_fpu_en = false;

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "params.h"
#include "sfpu/ckernel_sfpu_where.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

// MATH: datacopy the three input tiles (condition, true_val, false_val) into DEST
// at tile indices 0, 1, 2, then run the SFPU where kernel face-by-face with the
// output overwriting DEST tile 0 (matches ttnn_where_test convention).
// Per-face offsets (in SFPU dest_reg_addr units, i.e. rows × 2):
//   - condition tile (DEST tile 0) → 0
//   - true_val  tile (DEST tile 1) → 64  (32 rows × 2)
//   - false_val tile (DEST tile 2) → 128 (64 rows × 2)
//   - output   tile (DEST tile 0) → 0
// Face stride is 16 (`_inc_dst_face_addr_<16>()`), advancing RWC_D between faces.
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    if (unpack_to_dest)
    {
        // Chain: UNPACK (writes DEST) → SFPU (reads/writes DEST) → PACK (reads DEST).
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        // Chain: UNPACK → SrcA → FPU datacopy (MOVA2D) → DEST → SFPU → PACK.
        set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

    if (!unpack_to_dest)
    {
        // FPU path: datacopy all 3 tiles from SrcA into DEST at tile indices 0, 1, 2.
        const std::uint32_t num_rows = params.num_faces * params.TEST_FACE_R_DIM;
        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_rows, 1);

        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_math_eltwise_unary_datacopy_(num_rows, params.DST_INDEX + i);
        }

        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }

    _llk_math_eltwise_unary_sfpu_init_();
    _init_where_();

    // Per-tile offsets in SFPU dest_reg_addr units (rows × 2; tile stride = 64 for 32x32 tiles).
    // We set section_base to tile DST_INDEX below so offsets are relative to DST_INDEX's tile.
    constexpr int TILE_STRIDE       = 64;
    constexpr int cond_tile_offset  = 0 * TILE_STRIDE;  // tile DST_INDEX + 0 = condition
    constexpr int true_tile_offset  = 1 * TILE_STRIDE;  // tile DST_INDEX + 1 = true_val
    constexpr int false_tile_offset = 2 * TILE_STRIDE;  // tile DST_INDEX + 2 = false_val
    constexpr int out_tile_offset   = cond_tile_offset; // overwrite condition tile

    // Bracket the per-face SFPU section. `_set_dst_write_addr_<Tile32x32>(DST_INDEX)` sets
    // the section base to DST_INDEX's tile start so the offsets above resolve to the
    // correct tile (DST_INDEX + 0, +1, +2).
    _llk_math_eltwise_unary_sfpu_start_(params.DST_INDEX);

    const std::uint32_t num_sfpu_iterations = params.TEST_FACE_R_DIM / ckernel::math::SFP_ROWS;
    for (std::uint32_t face = 0; face < params.num_faces; face++)
    {
        _calculate_where_(static_cast<int>(num_sfpu_iterations), cond_tile_offset, true_tile_offset, false_tile_offset, out_tile_offset);
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
    }

    _llk_math_eltwise_unary_sfpu_done_();

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    // Wait for all math execution units this thread has driven to drain before PACK handover.
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

// PACK: write a single output tile (DEST tile 0) to buffer_Res.
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    std::uint32_t const buf_desc_id        = 8;
    const std::uint32_t num_tiles_per_pack = 1;

    if (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

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
    _llk_pack_init_(buf_desc_id, num_tiles_per_pack);
    _llk_pack_(params.DST_INDEX, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
