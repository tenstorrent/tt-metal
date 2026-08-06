// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Cross-op MIXED-TILE transition test (register-state design): a real `unpack_tilize`
// at one tile geometry G0 (the "polluter") followed by the transition under test —
// `_llk_unpack_tilize_uninit_` (PR C1) + `_llk_unpack_reconfig_data_format_srca_impl_`
// `<.. FACE_ROW_MAJOR>` (PR C2) — that retargets the SrcA baseline to a DIFFERENT
// geometry G1. The test then reads the SrcA config registers back and verifies they
// are the canonical G1 baseline.
//
// This closes the last open `tilize_uninit` cell: "normal->tiny / tiny->normal with
// reconfig" (axis F = tile-size transition, axis E = with-reconfig). Phase 3b covers
// tilize->matmul geometry changes via a full `_llk_unpack_hw_configure_`; here the
// re-establishment uses the lightweight C2 reconfig path, the ONLY mechanism that both
// (a) re-commits the canonical SrcA Y-stride (the PR's new write) and (b) RETARGETS the
// geometry — it rewrites `Tile_x_dim_cntx0 = canonical_unpA_tile_x_dim_cntx(G1.face_r_dim)`
// and the tile-descriptor Z-dim = G1.num_faces (llk_unpack_common.h:170-174).
//
// Why a register-state check (not a 2nd data op): the run-1 victim of such a transition
// is a SrcA datacopy whose correctness is *entirely* a function of these four SrcA
// registers, and Phases 1/2 already prove "correct registers => correct data". Reading
// the registers directly verifies the transition for BOTH directions with no second
// operand / golden (which a single-geometry stimuli harness can't lay out cleanly for
// two different tile shapes).
//
//   Run 0 : real `unpack_tilize` of a G0 tile (full unpack->math->pack pipeline,
//           output discarded to scratch) — leaves SrcA in the G0 tilize state.
//   Transition (only when DO_RESTORE): `_llk_unpack_tilize_uninit_(G0)` +
//           `_llk_unpack_reconfig_data_format_srca_impl_<.. FACE_ROW_MAJOR>(G1)`.
//   Readback: UNP0 Y-stride / Z-stride, Tile_x_dim_cntx0, tile-descriptor Z-dim,
//           compared against the canonical G1 baseline on-device via LLK_ASSERT.
//
// DO_RESTORE=false is the negative control: with no transition the registers stay in
// the G0 tilize state, so actual != G1-expected (the Tile_x_dim / Z-dim differ between
// G0 and G1) — the kernel LLK_ASSERTs that the G1 baseline is NOT reproduced, proving
// the transition is load-bearing.

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Scratch L1 address for the discarded run-0 (polluter) tilize output.
constexpr std::uint32_t buffer_polluter_scratch = 0xA0000;

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    // Polluter (run-0) tilize geometry.
    const std::uint32_t g0_face_r_dim = params.TEST_FACE_R_DIM;
    const std::uint32_t g0_num_faces  = params.num_faces;
    // Victim (run-1) target geometry (compile-time).
    constexpr std::uint32_t g1_face_r_dim = VICTIM_FACE_R_DIM;
    constexpr std::uint32_t g1_num_faces  = VICTIM_NUM_FACES;
    constexpr std::uint32_t g1_tile_size  = g1_face_r_dim * FACE_C_DIM * g1_num_faces;

    // ---- Run 0: real tilize "polluter" (output discarded) ----
    const std::uint32_t g0_block_ct  = _llk_unpack_tilize_block_ct_dim_wrapper_(1);
    const std::uint32_t g0_tilize_nf = _llk_unpack_tilize_num_faces_wrapper_(g0_num_faces);

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats_array[0].unpack_A_src,
        formats_array[0].unpack_B_src,
        formats_array[0].unpack_A_dst,
        formats_array[0].unpack_B_dst,
        g0_face_r_dim,
        g0_face_r_dim,
        g0_num_faces,
        g0_num_faces);
    _llk_unpack_tilize_init_wrapper_(
        formats_array[0].unpack_A_src, formats_array[0].unpack_A_dst, 1 /* ct_dim */, g0_face_r_dim, false /* narrow_tile */, g0_tilize_nf);
    _llk_unpack_tilize_wrapper_(
        L1_ADDRESS(params.buffer_A[0]),
        0 /* tile_index */,
        formats_array[0].unpack_A_src,
        formats_array[0].unpack_A_dst,
        g0_block_ct,
        g0_face_r_dim,
        g0_tilize_nf,
        false /* narrow_tile */);

    // Wait until the run-0 packer has drained before touching unpacker state.
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    // ---- Transition under test: uninit (C1) + reconfig (C2) retarget G0 -> G1 ----
    if constexpr (DO_RESTORE)
    {
        // C1: tear down the tilize-mutated SrcA baseline at the G0 geometry (WH 3-arg
        // uninit threads face_r_dim so it restores the G0 operand baseline exactly).
        _llk_unpack_tilize_uninit_wrapper_(formats_array[0].unpack_A_dst, g0_num_faces, g0_face_r_dim);
        // C2: the FACE_ROW_MAJOR reconfig re-commits the canonical SrcA Y/Z-stride AND
        // retargets the geometry to G1 (Tile_x_dim_cntx0 from g1_face_r_dim, descriptor
        // Z-dim from g1_num_faces). Same format both runs, so this is purely the
        // geometry/stride retarget (no data-format change).
        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::FACE_ROW_MAJOR, false>(
            formats_array[1].unpack_A_src, formats_array[1].unpack_A_dst, g1_tile_size, g1_face_r_dim, g1_num_faces);
    }

    // ---- Read back the SrcA baseline and compare against the canonical G1 state ----
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
    const std::uint32_t act_z_dim      = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1] >> 16;

    const std::uint32_t dst_format     = formats_array[1].unpack_A_dst;
    const std::uint32_t exp_y_stride   = canonical_unpA_y_stride(dst_format);
    const std::uint32_t exp_z_stride   = canonical_unpA_z_stride(dst_format);
    const std::uint32_t exp_tile_x_dim = canonical_unpA_tile_x_dim_cntx(g1_face_r_dim);
    const std::uint32_t exp_z_dim      = g1_num_faces;

    if constexpr (DO_RESTORE)
    {
        // The transition must reproduce the canonical G1 SrcA baseline exactly.
        LLK_ASSERT(act_y_stride == exp_y_stride, "transition: SrcA Y-stride not the canonical G1 baseline");
        LLK_ASSERT(act_z_stride == exp_z_stride, "transition: SrcA Z-stride not the canonical G1 baseline");
        LLK_ASSERT(act_tile_x_dim == exp_tile_x_dim, "transition: Tile_x_dim_cntx0 not retargeted to G1 face_r_dim");
        LLK_ASSERT(act_z_dim == exp_z_dim, "transition: tile-descriptor Z-dim not retargeted to G1 num_faces");
    }
    else
    {
        // Negative control: without the transition the SrcA baseline stays in the G0
        // tilize state, so the full G1 baseline must NOT be reproduced (the
        // geometry-bearing Tile_x_dim / Z-dim differ between G0 and G1).
        const bool all_match =
            (act_y_stride == exp_y_stride) && (act_z_stride == exp_z_stride) && (act_tile_x_dim == exp_tile_x_dim) && (act_z_dim == exp_z_dim);
        LLK_ASSERT(!all_match, "negative control: SrcA already matches the G1 baseline without the transition");
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    const std::uint32_t g0_num_faces = params.num_faces;
    const bool is_int_fpu_en         = false;
    const std::uint32_t res_dst_idx  = 0;

    // ---- Run 0: datacopy consuming the polluter tilize ----
    static constexpr bool TILIZE = true;
    _llk_math_eltwise_unary_datacopy_init_wrapper_<
        DataCopyType::A2D,
        is_fp32_dest_acc_en,
        BroadcastType::NONE,
        is_int_fpu_en,
        llk_test_pack_mode_v<false, TILIZE>>(g0_num_faces, formats_array[0].math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats_array[0].math, formats_array[0].math);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        res_dst_idx, formats_array[0].math, formats_array[0].math, g0_num_faces);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig(&formats_array)[2] = params.formats;
#endif
    const std::uint32_t g0_face_r_dim = params.TEST_FACE_R_DIM;
    const std::uint32_t g0_num_faces  = params.num_faces;
    const std::uint32_t res_dst_idx   = 0;
    static constexpr bool UNTILIZE    = false;
    static constexpr bool TILIZE      = true;

    // ---- Run 0: pack the (discarded) polluter tilize result to scratch ----
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>>(
        formats_array[0].pack_src,
        formats_array[0].pack_dst,
        g0_face_r_dim * params.TEST_FACE_C_DIM * g0_num_faces /* tile_size */,
        g0_face_r_dim,
        TILE_C_DIM,
        g0_num_faces);
    _llk_pack_init_wrapper_<llk_unpack_tilize_sweep_pack_cfg_mode_v<UNTILIZE, TILIZE>, false /* zero_output */>(
        formats_array[0].pack_dst, g0_face_r_dim, TILE_C_DIM, g0_num_faces);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, pack_exec_mode_v<UNTILIZE>>(res_dst_idx, L1_ADDRESS(buffer_polluter_scratch));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Signal the unpacker that run-0 has fully drained.
    t6_semaphore_post<>(semaphore::PACK_DONE);
}

#endif
