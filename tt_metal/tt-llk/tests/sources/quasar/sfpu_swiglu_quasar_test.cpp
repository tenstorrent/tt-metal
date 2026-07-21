// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-04-24_swiglu_quasar_9a4f086d

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

// Swiglu is a binary SFPU kernel: it requires TWO input tiles (gate, up) in
// Dest at distinct tile indices, and produces one output tile at a third
// index. Layout used by this test:
//   buffer_A: 2 concatenated tiles (tile 0 = gate, tile 1 = up)
//   Dest:     tile 0 = gate, tile 1 = up, tile 2 = output
//   buffer_Res: 1 output tile (the value of dest tile 2 packed out)
// We unpack 2 tiles to Dest via the unary operand path (unpacker advances
// through tiles 0,1 automatically with num_tiles=2).
void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t buf_desc_id     = 0;
    const std::uint32_t num_input_tiles = 2; // gate + up

    if (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
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

    if (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        // Same workaround as unary tests: MOVA2D for 32-bit dest requires both
        // SrcA/B format configs because the datacopy uses ELWADD internally.
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
    }

    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_input_tiles);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0, ckernel::DEFAULT_TENSOR_SHAPE);

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
#include "experimental/ckernel_sfpu_swiglu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // Binary SFPU: there are 2 input tiles (gate, up) and 1 output tile.
    // gate lives at Dest tile index 0, up at Dest tile index 1,
    // output written to Dest tile index 2. One SFPU "section" covers all
    // three tiles so we just need the SFPU dvalid chain.
    if (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

    constexpr std::uint32_t NUM_INPUT_TILES = 2; // gate + up

    if (!unpack_to_dest)
    {
        // FPU path: datacopy BOTH input tiles from SrcA to Dest before the
        // SFPU section reads them. Tile 0 = gate at Dest[0], tile 1 = up at
        // Dest[1].
        const std::uint32_t num_rows = params.num_faces * params.TEST_FACE_R_DIM;
        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_rows, 1);

        for (std::uint32_t i = 0; i < NUM_INPUT_TILES; ++i)
        {
            _llk_math_eltwise_unary_datacopy_(params.DST_INDEX + i);
        }

        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }

    _llk_math_eltwise_sfpu_init_();

    // SFPU section for swiglu. Base the Dest write address at the gate tile
    // (DST_INDEX + 0). Tile offsets are in Dest rows: one tile spans
    // num_faces * TEST_FACE_R_DIM rows (= 64 for Tile32x32 with 4 faces × 16
    // rows), so:
    //   gate at Dest tile 0  → offset 0
    //   up   at Dest tile 1  → offset 1 * DEST_ROWS_PER_TILE
    //   out  at Dest tile 2  → offset 2 * DEST_ROWS_PER_TILE
    // _calculate_swiglu_ reads/writes 2 rows per iteration (SFP_ROWS=2), and
    // _llk_math_eltwise_sfpu_inc_dst_face_addr_() advances the base by
    // TEST_FACE_R_DIM rows (one face) between face iterations, so the same
    // relative offsets work for every face.
    _llk_math_eltwise_sfpu_start_(params.DST_INDEX);

    // Load the 3 hoisted constants (+L, +2L, alpha) into LREG4/5/6 once for
    // the whole SFPU section. They persist across every per-face call below.
    ckernel::sfpu::_init_swiglu_();

    const std::uint32_t DEST_ROWS_PER_TILE = params.num_faces * params.TEST_FACE_R_DIM;
    for (std::uint32_t face = 0; face < params.num_faces; ++face)
    {
        ckernel::sfpu::_calculate_swiglu_<SFPU_ITERATIONS>(
            /*gate_offset_idx=*/0,
            /*up_offset_idx=*/DEST_ROWS_PER_TILE,
            /*out_offset_idx=*/2 * DEST_ROWS_PER_TILE);
        _llk_math_eltwise_sfpu_inc_dst_face_addr_();
    }

    _llk_math_eltwise_sfpu_done_();

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    // Drain any in-flight SFPU / FPU / MOP work before PACK reads Dest.
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
    std::uint32_t const buf_desc_id          = 8;
    constexpr std::uint32_t num_output_tiles = 1;

    // Declare the same dvalid client chain that UNPACK/MATH used, seen from
    // PACK's side. The chain must match on all three threads.
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
    _llk_pack_init_(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, num_output_tiles);

    // Output lives at Dest tile index 2 — this is the layout *this driver*
    // uses (see "Layout used by this test" at the top of the file): gate=0,
    // up=1, out=2 relative to DST_INDEX. The kernel itself is layout-agnostic
    // and accepts arbitrary (gate, up, out) Dest offsets via
    // `_calculate_swiglu_`'s parameters; +2 is not a property of swiglu.
    _llk_pack_(params.DST_INDEX + 2, 0, ckernel::DEFAULT_TENSOR_SHAPE);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif
