// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Direct helper-correctness check for the PR #45127 canonical-baseline helpers (C4):
//   `canonical_unpA_y_stride`, `canonical_unpA_z_stride`, `canonical_unpA_tile_x_dim_cntx`.
//
// These helpers are the single source of truth that `_llk_unpack_tilize_uninit_`,
// `_llk_unpack_bcastA_B_uninit_`, and `_llk_unpack_reconfig_data_format_srca_impl_`
// restore to. The whole restore design is only correct if the helpers reproduce the
// EXACT register state that `configure_unpack_AB` (`_llk_unpack_hw_configure_`)
// programs for a given (dst_format, face_r_dim, num_faces). Every restore test
// exercises the helpers transitively; this test pins them down DIRECTLY.
//
// Flow (unpack thread only):
//   1. `_llk_unpack_hw_configure_(dst, face_r_dim, num_faces)` programs the SrcA
//      tile descriptor + stride registers from its OWN inline formulas (which are
//      written independently of the helpers — see cunpack_common.h:807/928).
//   2. Read back the four canonical SrcA registers:
//        - UNP0 ch1 Y-stride   (UNP0_ADDR_CTRL_XY_REG_1_Ystride)
//        - UNP0 ch1 Z-stride   (UNP0_ADDR_CTRL_ZW_REG_1_Zstride)
//        - Tile_x_dim_cntx0     (THCON_SEC0_REG5_Tile_x_dim_cntx0)
//        - tile-descriptor Z-dim (THCON_SEC0_REG0_TileDescriptor word 1, bits 31:16)
//   3. LLK_ASSERT each equals the helper value (Z-dim equals num_faces) on-device.
//
// If a helper ever drifts from `configure_unpack_AB`'s inline programming, the
// readback diverges from the helper and the on-device assert fails. There is no
// actual unpack/datacopy here — the registers are the deliverable, so no stimuli /
// golden are needed.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals referenced by the LLK config helpers (configure_unpack_AB / sync_regfile_write).
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_faces  = params.num_faces;
    const std::uint32_t face_r_dim = params.TEST_FACE_R_DIM;
    const std::uint32_t dst_format = formats.unpack_A_dst;

    // Program the canonical SrcA baseline for this (dst_format, face_r_dim, num_faces).
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r_dim, face_r_dim, num_faces, num_faces);

    // Drain config writes before reading them back (mirrors are_unpackers_AB_configured_correctly).
    tensix_sync();
    for (std::uint32_t i = 0; i < 10; i++)
    {
        asm volatile("nop");
    }

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    const std::uint32_t act_y_stride =
        (cfg[UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32] & UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK) >> UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT;
    const std::uint32_t act_z_stride =
        (cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] & UNP0_ADDR_CTRL_ZW_REG_1_Zstride_MASK) >> UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT;
    const std::uint32_t act_tile_x_dim = cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32];
    // tile_descriptor word 1: z_dim at bits [31:16].
    const std::uint32_t act_z_dim = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1] >> 16;

    const std::uint32_t exp_y_stride   = canonical_unpA_y_stride(dst_format);
    const std::uint32_t exp_z_stride   = canonical_unpA_z_stride(dst_format);
    const std::uint32_t exp_tile_x_dim = canonical_unpA_tile_x_dim_cntx(face_r_dim);
    const std::uint32_t exp_z_dim      = num_faces;

    LLK_ASSERT(act_y_stride == exp_y_stride, "canonical_unpA_y_stride mismatch vs configure_unpack_AB");
    LLK_ASSERT(act_z_stride == exp_z_stride, "canonical_unpA_z_stride mismatch vs configure_unpack_AB");
    LLK_ASSERT(act_tile_x_dim == exp_tile_x_dim, "canonical_unpA_tile_x_dim_cntx mismatch vs configure_unpack_AB");
    LLK_ASSERT(act_z_dim == exp_z_dim, "tile-descriptor Z-dim (num_faces) mismatch vs configure_unpack_AB");
}

#endif

#ifdef LLK_TRISC_MATH

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS)
{
}

#endif

#ifdef LLK_TRISC_PACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS)
{
}

#endif
