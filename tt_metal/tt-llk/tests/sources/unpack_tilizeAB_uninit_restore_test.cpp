// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Register-level restore test for `_llk_unpack_tilizeA_B_uninit_`.
//
// Companion to `unpack_canonical_baseline_check_test.cpp` and the single-operand
// `unpack_tilize_uninit_restore*` tests, but for the FUSED tilize-A / unpack-B
// teardown path. It closes the same class of teardown gap as tt-llk#1161: the
// uninit used to restore `Tile_x_dim_cntx0` from the hardcoded FACE_DIM_16x16 GPR
// (= 256 | (256 << 16)), which is correct only for face_r_dim == 16 and leaked a
// 16-row tile geometry into the next operand for tiny tiles (face_r_dim < 16).
//
// Flow (unpack thread only — like the canonical-baseline check, no math/pack, so
// no inter-thread handshake and no data golden are needed):
//   1. `_llk_unpack_hw_configure_` programs operand A's canonical SrcA baseline for
//      (dst_format, face_r_dim, num_faces), including Tile_x_dim_cntx0 =
//      canonical_unpA_tile_x_dim_cntx(face_r_dim).
//   2. `_llk_unpack_tilizeA_B_init_` mutates the tilize-A state (on WH it drives
//      Tile_x_dim_cntx0 to the 1x16 tilize value; on BH it mutates the SrcA
//      Y-stride).
//   3. `_llk_unpack_tilizeA_B_uninit_(dst, tensor_shape)` — the RESTORE under test.
//   4. Read back Tile_x_dim_cntx0 (and, on BH, the SrcA Y-stride) and LLK_ASSERT
//      each equals the canonical baseline for the operand's face_r_dim.
//
// A broken uninit (e.g. the old FACE_DIM_16x16 restore) diverges from
// canonical_unpA_tile_x_dim_cntx(face_r_dim) for every tiny face_r_dim, failing
// the on-device assert. `_llk_unpack_tilizeA_B_` (the executor) is intentionally
// NOT run: init already establishes the state that uninit must revert, and the
// registers are the deliverable.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

// Globals referenced by the LLK config helpers (configure_unpack_AB / mop programming).
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t num_faces  = params.num_faces;
    const std::uint32_t face_r_dim = params.TEST_FACE_R_DIM;
    const std::uint32_t dst_format = formats.unpack_A_dst;

    // Establish operand A's canonical SrcA baseline for this (dst_format, face_r_dim, num_faces).
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, face_r_dim, face_r_dim, num_faces, num_faces);

    // Program the fused tilize-A / unpack-B state (mutates the SrcA baseline this op owns).
#ifdef ARCH_WORMHOLE
    _llk_unpack_tilizeA_B_init_<>(formats.unpack_A_src, formats.unpack_A_dst, false /* narrow_tile */, 1 /* ct_dim */, num_faces, face_r_dim, face_r_dim);
#else
    _llk_unpack_tilizeA_B_init_<>(formats.unpack_A_src, formats.unpack_A_dst, 1 /* ct_dim */, num_faces, face_r_dim);
#endif

    // Restore under test: revert the tilize-A state back to operand A's canonical baseline.
    _llk_unpack_tilizeA_B_uninit_(dst_format, ckernel::tensor_shape_from_num_faces(num_faces, face_r_dim));

    // Drain config writes before reading them back (mirrors unpack_canonical_baseline_check_test).
    tensix_sync();
    for (std::uint32_t i = 0; i < 10; i++)
    {
        asm volatile("nop");
    }

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    const std::uint32_t act_tile_x_dim = cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32];
    const std::uint32_t exp_tile_x_dim = canonical_unpA_tile_x_dim_cntx(face_r_dim);
    LLK_ASSERT(act_tile_x_dim == exp_tile_x_dim, "_llk_unpack_tilizeA_B_uninit_ did not restore Tile_x_dim_cntx0 to the canonical face_r_dim baseline");

#ifdef ARCH_BLACKHOLE
    // Blackhole's tilizeA_B mutates the SrcA Y-stride and the uninit restores it; verify that too.
    const std::uint32_t act_y_stride =
        (cfg[UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32] & UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK) >> UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT;
    const std::uint32_t exp_y_stride = canonical_unpA_y_stride(dst_format);
    LLK_ASSERT(act_y_stride == exp_y_stride, "_llk_unpack_tilizeA_B_uninit_ did not restore the canonical SrcA Y-stride");
#endif
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
