// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Isolated SFPU add (binary): operand B via UNPACK0 (UNP_DEST) -> Dest (UNPACK thread),
// operand A via UNPACK2 (UNP_S) -> SrcS (ISOLATE_SFPU thread). SFPU reads A from SrcS
// and B from Dest, adds them, stores result to SrcS output slice, then PACK1 -> L1.
// MATH and PACK threads are stubbed.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

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
    const std::uint32_t num_tiles = params.TILE_CNT;

    // Dest dvalid chain: UNPACK -> SFPU (no PACK since result goes to SrcS, not Dest)
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU});
    _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();

    constexpr std::uint32_t buf_desc_id_dest = 1;

    buffer_descriptor_u bd_dest = {0};
    tdma_descriptor_t td_dest;

    // Unpack BD for operand B: L1 input B -> Dest (UNP_DEST, standard face dimensions)
    bd_dest.f.l1_addr_16B   = L1_ADDRESS(params.buffer_B[0]);
    bd_dest.f.format        = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_dest.f.x_dim         = params.TEST_FACE_C_DIM;
    bd_dest.f.y_dim         = params.TEST_FACE_R_DIM;
    bd_dest.f.z_dim         = params.num_faces;
    td_dest.buf_desc        = bd_dest;
    td_dest.buf_desc_id     = buf_desc_id_dest;
    td_dest.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);
    _configure_buf_desc_table_(td_dest.buf_desc_id, td_dest.buf_desc);
    _llk_unpack_configure_unary_<p_unpacr::UNP_DEST>(td_dest);

    // Set dest base for UNPACK thread (SEC0). Mimics bank switching: in a real
    // scenario this would be called per tile-batch with _get_dest_buffer_base_()
    // toggling between banks. Here all tiles fit in bank 0.
    ckernel::unpack::_set_dst_write_addr_<ckernel::trisc::DstTileShape::Tile32x32>(0);

    // Program MOP and unpack all tiles of operand B to Dest
    _llk_unpack_unary_operand_init_<p_unpacr::UNP_DEST, false, is_fp32_dest_acc_en>(buf_desc_id_dest, num_tiles);
    _llk_unpack_unary_operand_<p_unpacr::UNP_DEST>(0);

    _llk_unpack_dest_dvalid_section_done_<dest_sync>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "llk_srcs_tdma.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const volatile FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_tiles = params.TILE_CNT;

    // Dest dvalid chain: UNPACK -> SFPU (SFPU consumes Dest data written by UNPACK)
    set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU});

    // -------------------------------------------------------------------------
    // Data format inference and dimensions
    // -------------------------------------------------------------------------

    const bool PARAM_SRCS_32BIT_MODE =
        static_cast<DataFormat>(formats.unpack_S_dst) == DataFormat::Float32 || static_cast<DataFormat>(formats.unpack_S_dst) == DataFormat::Int32;
    constexpr std::uint32_t PARAM_SRCS_XDIM         = 16; // datums per row of SrcS slice
    constexpr std::uint32_t PARAM_SRCS_ZDIM         = 1;
    constexpr std::uint32_t PARAM_SRCS_YDIM_BASE    = 8; // rows per slice if SrcS were 16-bit columns
    const std::uint32_t PARAM_SRCS_YDIM             = PARAM_SRCS_32BIT_MODE ? (PARAM_SRCS_YDIM_BASE / 2) : PARAM_SRCS_YDIM_BASE;
    const std::uint32_t PARAM_SRCS_SLICE_COUNT      = (32 * 32) / (PARAM_SRCS_XDIM * PARAM_SRCS_YDIM * PARAM_SRCS_ZDIM);
    constexpr std::uint32_t PARAM_SRCS_INSTRN_COUNT = 1;

    // -------------------------------------------------------------------------
    // Buffer descriptors and HW setup (SrcS unpack + SrcS pack)
    // -------------------------------------------------------------------------

    constexpr std::uint32_t buf_desc_id_srcs = 0; // Operand A -> SrcS (UNP_S)
    constexpr std::uint32_t buf_desc_id_pack = 8;

    buffer_descriptor_u bd_srcs = {0};
    tdma_descriptor_t td_srcs;
    buffer_descriptor_u bd_pack = {0};
    tdma_descriptor_t td_pack;

    // Unpack BD for operand A: L1 input A -> SrcS (UNP_S, SrcS slice dimensions)
    bd_srcs.f.l1_addr_16B   = L1_ADDRESS(params.buffer_A[0]);
    bd_srcs.f.format        = static_cast<std::uint8_t>(formats.unpack_S_src);
    bd_srcs.f.x_dim         = PARAM_SRCS_XDIM;
    bd_srcs.f.y_dim         = PARAM_SRCS_YDIM;
    bd_srcs.f.z_dim         = PARAM_SRCS_ZDIM;
    td_srcs.buf_desc        = bd_srcs;
    td_srcs.buf_desc_id     = buf_desc_id_srcs;
    td_srcs.reg_data_format = static_cast<std::uint8_t>(formats.unpack_S_dst);
    _configure_buf_desc_table_(td_srcs.buf_desc_id, td_srcs.buf_desc);
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(td_srcs);

    // Pack BD: SrcS output slice -> L1 output
    bd_pack.f.l1_addr_16B   = L1_ADDRESS(params.buffer_Res[0]);
    bd_pack.f.format        = static_cast<std::uint8_t>(formats.pack_S_dst);
    bd_pack.f.x_dim         = PARAM_SRCS_XDIM;
    bd_pack.f.y_dim         = PARAM_SRCS_YDIM;
    bd_pack.f.z_dim         = PARAM_SRCS_ZDIM;
    td_pack.buf_desc        = bd_pack;
    td_pack.buf_desc_id     = buf_desc_id_pack;
    td_pack.reg_data_format = static_cast<std::uint8_t>(formats.pack_S_src);
    _configure_buf_desc_table_(td_pack.buf_desc_id, td_pack.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK1>(td_pack);

    // Implied math format disable for SrcS and sfpmem mod selection
    cfg[DISABLE_IMPLIED_SRCS_FORMAT_ADDR32 + TRISC_ID] = !IMPLIED_MATH_FORMAT;
    const std::uint32_t sfpmem_mod                     = IMPLIED_MATH_FORMAT ? p_sfpu::sfpmem::DEFAULT : _sfpu_sfpmem_type_(formats.unpack_S_dst);
    const std::uint32_t sfpmem_mod_dest                = IMPLIED_MATH_FORMAT ? p_sfpu::sfpmem::DEFAULT : _sfpu_sfpmem_type_(formats.unpack_A_dst);

    // -------------------------------------------------------------------------
    // SFPU configuration and execution
    // -------------------------------------------------------------------------

    // Unary SrcS: 1 instruction per auto-loop iteration (operand A only)
    if (PARAM_SRCS_32BIT_MODE)
    {
        _llk_unpack_srcs_config_<PARAM_SRCS_INSTRN_COUNT, 16>();
        _llk_pack_srcs_config_<PARAM_SRCS_INSTRN_COUNT, 16>();
    }
    else
    {
        _llk_unpack_srcs_config_<PARAM_SRCS_INSTRN_COUNT, 8>();
        _llk_pack_srcs_config_<PARAM_SRCS_INSTRN_COUNT, 8>();
    }

    _llk_math_eltwise_unary_sfpu_init_();

    // SFPU add: operand A from SrcS slice 0, operand B from Dest, result to SrcS slice 2.
    // Dest addressing is purely SW — no counter increments needed.
    // D counter stays at 0 (set by _llk_math_eltwise_unary_sfpu_init_) and
    // ADDR_MOD_7 has dest.incr=0, so the full offset is in dest_reg_addr.
    // Tile positioning uses dest_section_base (via _set_dst_write_addr_),
    // intra-tile slice offsets use SfpuDestSlice/SfpuSrcsSlice.
    const SfpuSrcsSlice srcs {static_cast<int>(PARAM_SRCS_YDIM)};
    const SfpuDestSlice dest {static_cast<int>(PARAM_SRCS_YDIM)};
    const int num_sfpu_iterations = PARAM_SRCS_YDIM >> 1; // SFP_ROWS == 2
    for (std::uint32_t i = 0; i < num_tiles; ++i)
    {
        // Set dest_section_base for this tile (SEC3 for ISOLATE_SFPU).
        // Matches the UNPACK thread's dest base so both agree on tile positioning.
        _set_dst_write_addr_<trisc::DstTileShape::Tile32x32>(i);

        // Unpack operand A to SrcS (unary: single operand with dvalid)
        _llk_unpack_srcs_<PARAM_SRCS_INSTRN_COUNT>(buf_desc_id_srcs, i * PARAM_SRCS_SLICE_COUNT);

        _llk_pack_srcs_<PARAM_SRCS_INSTRN_COUNT>(buf_desc_id_pack, i * PARAM_SRCS_SLICE_COUNT);

        for (std::uint32_t slice = 0; slice < PARAM_SRCS_SLICE_COUNT; slice++)
        {
#pragma GCC unroll 8
            for (int d = 0; d < num_sfpu_iterations; d++)
            {
                TT_SFPLOAD(p_sfpu::LREG0, sfpmem_mod, ADDR_MOD_7, 0, srcs[0] + (d << 1));
                TT_SFPLOAD(p_sfpu::LREG1, sfpmem_mod_dest, ADDR_MOD_7, 0, dest[slice] + (d << 1));
                // Add LREG0 (A from SrcS) + LREG1 (B from Dest), store result in LREG2
                TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0x0);
                TTI_NOP;
                // Store result back to SrcS output slice
                TT_SFPSTORE(p_sfpu::LREG2, sfpmem_mod, ADDR_MOD_7, 0, srcs[2] + (d << 1));
            }

            _llk_math_eltwise_unary_sfpu_srcs_clear_vlds_<0x1, 0x1>(); // Clears dvalid for SFPU read and write
        }
    }

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    // Wait for all operations to complete
    wait_sfpu_idle();
    wait_unpack_idle();
    wait_pack_idle();
}

#endif

#ifdef LLK_TRISC_PACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
}

#endif
