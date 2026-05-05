// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// AI-generated — run_id: 2026-04-08_abs_quasar_2f52d870

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
    const std::uint32_t buf_desc_id = 0;
    const std::uint32_t num_tiles   = params.TILE_CNT;

    if (unpack_to_dest)
    {
        // Direct path: UNPACK writes data straight into Dest — no FPU datacopy needed.
        // Requires format bit-width to match Dest mode (e.g., 16-bit input with 16-bit Dest).
        // dvalid clients: UNPACK (writes Dest), SFPU (reads/writes Dest), PACK (reads Dest).
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        // When unpacking directly to Dest (bypassing FPU), MATH HW still needs format configuration
        // so the SFPU reads Dest data in the correct format.
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
    }
    else
    {
        // FPU path: UNPACK -> SrcA -> FPU datacopy (MOVA2D) -> Dest.
        // Needed when input format bit-width differs from Dest mode (format conversion required).
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    buffer_descriptor_u bd_val = {0};

    bd_val.f.l1_addr_16B = L1_ADDRESS(params.buffer_A[0]);
    bd_val.f.format      = static_cast<std::uint8_t>(formats.unpack_A_src);
    // Buffer descriptor: x_dim = columns per face, y_dim = rows per face, z_dim = number of faces
    bd_val.f.x_dim = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim = params.num_faces;

    tdma_descriptor_t td_val;
    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = buf_desc_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);
    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);

    if (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        // When Dest is 32-bit (fp32_dest_acc) and data goes through the FPU path,
        // MOVA2D/MOVB2D requires both SrcA and SrcB format registers configured,
        // so use binary unpack configuration.
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
    }

    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, num_tiles);
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
#include "experimental/ckernel_sfpu_abs.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
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
    // dvalid clients for Dest are the producer/consumer set this MATH thread
    // participates in. The chain MUST match what UNPACK declared, otherwise
    // the producer will not hand off Dest sections correctly.
    if (unpack_to_dest)
    {
        // Direct path: UNPACK writes Dest, SFPU reads/writes Dest, PACK reads Dest.
        // No FPU datacopy, so FPU is not in the chain.
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        // FPU path: UNPACK -> SrcA -> FPU datacopy (MOVA2D) -> Dest -> SFPU -> PACK.
        // Declare the chain both as FPU-producer (for the datacopy handoff)
        // and SFPU-producer (for the SFPU handoff to PACK).
        set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

    // SFPU iterations per face. `_llk_math_eltwise_unary_sfpu_params_`
    // processes one whole tile per call: for each face in the tile it invokes
    // `_calculate_abs_(num_sfpu_iterations)`, which advances the Dest row
    // cursor `num_sfpu_iterations` times (each step covers SFP_ROWS rows,
    // so num_sfpu_iterations*SFP_ROWS == TEST_FACE_R_DIM rows per face).
    //
    // Note: Quasar SFPU only supports VectorMode::RC (the default and only
    // mode), so iterating over a single face in the per-face loop inside
    // `_llk_math_eltwise_unary_sfpu_params_` is equivalent to covering the
    // whole tile. The outer `for (i < TILE_CNT)` below therefore iterates
    // tile-by-tile, not face-by-face.
    const std::uint32_t num_sfpu_iterations = params.TEST_FACE_R_DIM / ckernel::math::SFP_ROWS;

    if (!unpack_to_dest)
    {
        // FPU path: datacopy tiles from SrcA to Dest via MOVA2D before SFPU can operate on them.
        // (This kernel does not use the ELWADD-as-datacopy workaround from WH/BH —
        //  that worked around an 8-row MOVA2D bug which is not present on Quasar.)
        const std::uint32_t num_rows = params.num_faces * params.TEST_FACE_R_DIM;
        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_rows, 1);

        // Datacopy each tile into Dest, honouring the runtime DST_INDEX offset
        // so MATH writes the same Dest region PACK will later read.
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_math_eltwise_unary_datacopy_(num_rows, params.DST_INDEX + i);
        }

        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }

    _llk_math_eltwise_unary_sfpu_init_();

    // Apply SFPU abs (SFPABS) in-place on Dest for each tile.
    // Tile index must match the one used by the producer (datacopy above, or
    // UNPACK-to-Dest), so it is offset by params.DST_INDEX.
    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_calculate_abs_, params.DST_INDEX + i, num_sfpu_iterations);
    }

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    // Idle all execution units this MATH thread has driven before PACK takes
    // over: SFPU (the abs loop), FPU (datacopy on the !unpack_to_dest path),
    // and MOP (any macro-op sequences issued from the SFPU helpers).
    // No wait_replay_idle() because this kernel emits straight-line SFPU
    // code and does not install a replay buffer.
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
    // Buffer descriptor: x_dim = columns per face, y_dim = rows per face, z_dim = number of faces
    bd_val.f.x_dim = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim = params.num_faces;

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
