// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// custom_mm_reuse_dest_srcb  (Blackhole-only experimental LLK, PR #53297)
//
// This LLK is the SECOND matmul of a fused chain.  Its in0 operand is NOT
// unpacked from L1 — it is moved straight out of DEST (where a preceding
// custom_mm left its output) into SrcB with MOVD2B.  Only in1 (the weights) is
// unpacked, into SrcA.  See llk_math_custom_mm_reuse_dest_srcb.h.
//
// Because the reuse LLK has no meaning without a DEST-resident in0 in the exact
// layout custom_mm's Dense layout produces, this test drives the WHOLE fused
// chain, exactly as the header documents it:
//
//   producer:  custom_mm<DenseSplitAcc>               out0 = A0 @ B0      -> DEST
//   consumer:  custom_mm_reuse_dest_srcb<r=8>         out  = out0 @ B1    -> DEST
//   pack:      the dense 16-rows-per-tile accumulator layout               -> L1
//
// -------------------------- Golden derivation --------------------------------
// Let  A0  be the producer in0 tile of shape [8, 32*KP]   (KP producer K-tiles)
//      B0  be the producer weights           [32*KP, 32*KT]  (KT output tiles)
//      B1  be the consumer weights           [32*KT, 32*NT]
//
// PRODUCER (custom_mm<DenseSplitAcc>, in0_r_dim=8, ct_dim=KT, kt_dim=KP):
//   out0[8, 32*KT] = A0[8, 32*KP] @ B0[32*KP, 32*KT]
//   In DEST, dense_packing lays the KT output tiles 32 rows apart; within each
//   tile face0 (cols 0..15) is at rows base+0..7 and face1 (cols 16..31) at
//   rows base+16..23  (custom_mm split-acc dense output; header lines 21-31).
//
// CONSUMER (custom_mm_reuse_dest_srcb<in0_tile_r_dim=8>, kt_dim=KT, nt_dim=NT):
//   Reads those KT DEST tiles as in0 (SrcB, one every src_tile_stride=32 rows),
//   multiplies by the unpacked weights B1 (SrcA) and reduces over K:
//     out[8, 32*NT] = out0[8, 32*KT] @ B1[32*KT, 32*NT]
//   The 4 MVMULs per output tile are (2 SrcA K-faces) x (2 N-faces), all
//   accumulating into the 16 DEST rows of that tile.  MOVD2B places source
//   face0 (DEST row+0) at SrcB row 0 and source face1 (DEST row+16) at SrcB
//   row 8, so the K reduction over the two 16-wide K-faces is correct.
//
// Therefore the end-to-end golden is a plain chained matmul:
//     golden = (A0 @ B0) @ B1
// with LoFi products and the producer result narrowed to the DEST format.  The
// packer writes each output tile's `r`(=8) defined rows as two 8-row faces
// (cols 0..15, then cols 16..31); the rest of the tile is undefined and the
// python side validates only the defined rows.
//
// Fidelity is LoFi only (both custom_mm and the reuse LLK are LoFi-only).
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"

using namespace ckernel;

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

// Chain geometry (compile-time; see param dataclass CUSTOM_MM_REUSE_CFG).
//   IN0_TILE_R_DIM  height of in0 tile (1/2/4/8); we test the full-height 8 case
//   PRODUCER_KT     producer inner dim in tiles (its kt_dim)
//   REUSE_KT        consumer inner dim in tiles (== producer ct_dim, # of DEST in0 tiles)
//   REUSE_NT        consumer output width in tiles
// src_tile_stride for a Dense-layout custom_mm producer output is 32 DEST rows.
static constexpr std::uint32_t SRC_TILE_STRIDE = 32;
static constexpr std::uint32_t ISRC            = 0;   // DEST row of first producer output (in0) tile
static constexpr std::uint32_t IDST            = 256; // DEST row of first consumer output tile (upper half)

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_custom_mm.h"
#include "experimental/llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb.h"
#include "experimental/llk_unpack_A_sdpa.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    {
        START_PERF_MEASURE("INIT")
        // ---- producer custom_mm unpack: in0 (SrcB) = buffer_A, weights (SrcA) = buffer_B ----
        // hw_configure swaps operands: SrcA carries the weights, SrcB carries in0.
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_B_src,
            formats.unpack_A_src,
            formats.unpack_B_dst,
            formats.unpack_A_dst,
            FACE_R_DIM,
            IN0_TILE_R_DIM,
            params.num_faces_B,
            params.num_faces_A,
            params.TILE_SIZE_UNPACK_B,
            params.TILE_SIZE_UNPACK_A);

        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t k = 0; k < PRODUCER_KT; ++k)
                {
                    _perf_unpack_loop_set_valid<true, true>(1);
                    if constexpr (REUSE_KT > 1)
                    {
                        _perf_unpack_loop_set_valid<true, false>(REUSE_KT - 1);
                    }
                }
                _perf_unpack_loop_set_valid<false, true>(1);
                _perf_unpack_loop_set_valid<true, false>(REUSE_KT * REUSE_NT);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                // custom_mm and reuse both program unpack MOP bank 0 and replay
                // buffer 0. Initialize each operation immediately before its
                // execute so the consumer cannot overwrite the producer.
                _llk_unpack_AB_custom_mm_init_<false /* transpose */, true /* clear_src */>(
                    IN0_TILE_R_DIM, formats.unpack_B_dst /* weights land in SrcA */, REUSE_KT /* producer ct_dim */);

                // Signature: (base_address_a, base_address_b, tile_index_a, tile_index_b,
                //             tile_size_a, tile_size_b, kt_dim, ct_dim) where _a == SrcA
                //             (weights = buffer_B) and _b == SrcB (in0 = buffer_A).
                _llk_unpack_AB_custom_mm_<false /* read_transposed */>(
                    L1_ADDRESS(params.buffer_B[0]) /* base_address_a: producer weights -> SrcA */,
                    L1_ADDRESS(params.buffer_A[0]) /* base_address_b: in0 -> SrcB */,
                    0 /* tile_index_a (weights) */,
                    0 /* tile_index_b (in0) */,
                    params.TILE_SIZE_UNPACK_B /* tile_size_a (weights) */,
                    params.TILE_SIZE_UNPACK_A /* tile_size_b (in0) */,
                    PRODUCER_KT /* producer kt_dim */,
                    REUSE_KT /* producer ct_dim */);

                // The production API publishes one dummy SrcB token before
                // reuse; MOVD2B replaces its data and the final SETRWC clears it.
                _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_init_(REUSE_NT, FACE_R_DIM, 4 /* unpA_num_faces */);
                _llk_unpack_A_sdpa_set_srcb_dummy_valid_();
                _llk_unpack_AB_sdpa_custom_mm_reuse_dest_srcb_(
                    L1_ADDRESS(params.buffer_C[0]) /* consumer weights base -> SrcA */,
                    0 /* tile_index_a */,
                    params.TILE_SIZE_UNPACK_B /* tile_size_a (same shape/format as producer weights) */,
                    REUSE_KT /* kt_dim */,
                    REUSE_NT /* nt_dim */,
                    REUSE_NT /* in1_k_stride == nt_dim for contiguous weights */);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_custom_mm.h"
#include "experimental/llk_math_custom_mm_reuse_dest_srcb.h"
#include "llk_math_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t k = 0; k < PRODUCER_KT; ++k)
                {
                    if constexpr (REUSE_KT > 1)
                    {
                        _perf_math_loop_clear_valid<true, false>(REUSE_KT - 1);
                    }
                    _perf_math_loop_clear_valid<true, true>(1);
                }
                _perf_math_loop_clear_valid<true, false>(REUSE_KT * REUSE_NT);
                _perf_math_loop_clear_valid<false, true>(1);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_wait_for_dest_available_<DST_SYNC>();
                }

                // Reuse rewrites all address modifiers, so restore custom_mm
                // before every producer execute.
                _llk_math_custom_mm_init_<false /* transpose */, CustomMmDestLayout::DenseSplitAcc>(IN0_TILE_R_DIM, REUSE_KT /* producer ct_dim */);

                // Producer output starts at DEST row ISRC (dst_index is a tile slot; ISRC=0).
                _llk_math_custom_mm_<CustomMmDestLayout::DenseSplitAcc, true /* finalize */>(
                    IN0_TILE_R_DIM, 0 /* dst_index (tile slot) */, PRODUCER_KT, REUSE_KT /* ct_dim */);

                // Must follow the producer's last instruction because custom_mm
                // rewrites all eight address modifiers.
                _llk_math_custom_mm_reuse_dest_srcb_init_<true /* load_replay */>();

                // src_index / dst_index are bank-relative DEST *row* offsets (not tile slots).
                _llk_math_custom_mm_reuse_dest_srcb_<IN0_TILE_R_DIM>(
                    ISRC /* first in0 tile DEST row */,
                    IDST /* first output tile DEST row */,
                    REUSE_KT /* kt_dim */,
                    REUSE_NT /* nt_dim */,
                    SRC_TILE_STRIDE /* 32 for a Dense-layout custom_mm */);

                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "experimental/llk_pack_block.h"
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
#endif

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
        _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

        // The reuse accumulator layout: faces 8 DEST rows apart, tiles 16 rows apart
        // (custom_mm_reuse_dest_srcb_pack_init).  Program those DEST-read strides, then
        // pack the NT output tiles contiguously.  Only the DEST-side read strides
        // change; the bytes packed to L1 are unchanged.
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_C_DIM * 8 * 2);
        cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * 8 * 2);

        // Block-contiguous MOP so the NT tiny (16-row) tiles pack into a dense L1 block.
        _llk_pack_block_contiguous_mop_config_<false /* zero_output */>(FACE_R_DIM, TILE_NUM_FACES);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_packer_wait_for_math_done_();
                }

                // IDST is a DEST row; the reuse layout uses 16-row tile slots, so tile_index =
                // IDST / 16.  pack NT tiles to buffer_Res.
                _llk_pack_block_contiguous_<DST_SYNC, is_fp32_dest_acc_en>(IDST / 16 /* tile_index */, L1_ADDRESS(params.buffer_Res[0]), REUSE_NT);

                if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                {
                    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }

    // Restore the default packer strides (persistent state).
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Zstride_RMW>(FACE_C_DIM * FACE_R_DIM * 2);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
}

#endif
