// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#ifdef ARCH_BLACKHOLE
#include "experimental/llk_math_fast_tilize_api.h"
#endif
#include "llk_math_reduce_api.h"
#ifndef ARCH_QUASAR
#include "llk_math_matmul_api.h"
#endif
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_tilize_api.h"
#ifdef ARCH_BLACKHOLE
#include "experimental/llk_unpack_fast_tilize_api.h"
#endif
#include "llk_unpack_common_api.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack_tile_api.h"
#if defined(ARCH_BLACKHOLE)
#include "experimental/llk_pack_fast_tilize_api.h"
#elif defined(ARCH_WORMHOLE)
#include "llk_pack_fast_tilize_api.h"
#endif
#endif

#ifdef ARCH_QUASAR
#include "api/debug/dprint.h"  // DEBUG (conv tilize-pack PACR0_TILE_INC localizer, remove after)
#endif

namespace ckernel {

#if defined(ARCH_QUASAR) && defined(QSR_TILIZE_UNPACK_TO_DEST)
// Batched unpack-to-dest tilize (tt-metal #49445): the LLK-intended tilize-to-DEST unpacks a whole tile-row
// into DISTINCT DEST slots with ONE section_done per DEST section — unlike the single-tile UNP_DEST path,
// which lands every tile in DEST slot 0 with a per-tile section_done and mis-orders the tilized data (PCC~0).
// A conv tilize block is one 32-row tile-row of `block_width` column-tiles; when block_width exceeds the DEST
// tile capacity the row is split into equal compile-time column chunks that each fit DEST.

// DEST tile capacity in the CURRENT sync/accum mode: full DEST holds DEST_REGISTER_FULL_SIZE/(NUM_FACES*
// FACE_R_DIM) full tiles in fp16 (== DEST_NUM_TILES_FP16); half-sync holds half, and 32-bit dest (accum)
// halves it again. Derived from ckernel_trisc_common.h constants (pulled in by the unpack/math/pack LLK
// headers this file includes) rather than ckernel_defs.h's DEST_NUM_TILES_FP16, which the compute-API build
// does not transitively include.
constexpr uint32_t qsr_tilize_dest_tile_cap() {
    constexpr uint32_t full_tiles =
        ckernel::trisc::DEST_REGISTER_FULL_SIZE / (ckernel::trisc::NUM_FACES * ckernel::trisc::FACE_R_DIM);
    uint32_t cap = (DST_SYNC_MODE == DstSync::SyncFull) ? full_tiles : (full_tiles >> 1);
    return DST_ACCUM_MODE ? (cap >> 1) : cap;
}

// Largest column-chunk width that (a) fits DEST and (b) divides block_width evenly, so every chunk reuses the
// same compile-time MOP (no tail re-program). Degenerates to 1 (per-tile) only for block widths sharing no
// divisor <= cap other than 1.
constexpr uint32_t qsr_tilize_chunk_width(uint32_t block_width) {
    uint32_t cap = qsr_tilize_dest_tile_cap();
    uint32_t c = (block_width < cap) ? block_width : cap;
    while (c > 1 && (block_width % c) != 0) {
        --c;
    }
    return c;
}
#endif

// clang-format off
/**
 * Initializes the tilize operation. Should be called once at the beginning of a kernel.
 *
 * Return value: None
 *
 * | Param Type | Name   | Description                              | Type     | Valid Range | Required |
 * |----------- |--------|------------------------------------------|----------|-------------|----------|
 * | Function   | icb    | Input circular buffer identifier         | uint32_t | 0 to 31     | True     |
 * | Function   | block  | Size of tile block to work on            | uint32_t | > 0         | True     |
 * | Function   | ocb    | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 */
// clang-format on
template <uint32_t block_ct_dim_ct = 0>
ALWI void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);
    UNPACK((llk_unpack_tilize_init(icb, block)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false /*is_int_en*/,
          PackMode::Tilize>(icb)));
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_init<PackMode::Tilize, false /* zero_output */>(ocb, 1 /* num_tiles */, icb)));
#endif
#else
    // TODO(SK) #42757: Quasar unpack tilize could issue block_ct_dim tiles per MOP invocation, but scheduling
    // block_ct_dim against full_ct_dim would need a compute-API-level workaround since BH/WH operate
    // tile-by-tile and have no equivalent concept. Deferred: not on the Quasar critical path.
#ifdef QSR_TILIZE_UNPACK_TO_DEST
    // UnpackToDestEn bypass (tt-metal #49445): route the tilize unpacker into DEST (UNP_DEST) so the
    // per-tile MATH A2D datacopy issues NO MOP (sync-only forwarder) — sidestepping the Quasar tilize
    // datacopy DEST-section-release fault (ERROR_TRISC1 0x19). The math init is skipped for unpack_to_dest
    // (llk_math_unary_datacopy_api.h:47). Enabled only for the conv Program-A tilize (factory-injected define).
    if constexpr (block_ct_dim_ct > 0) {
        // BATCHED path: program the block MOP with the compile-time column-chunk width (must match the chunk
        // width the batched tilize_block below computes from the same block_ct_dim_ct). FULL_CT_DIM = the full
        // source-row width in tiles (== block_ct_dim_ct).
        constexpr uint32_t chunk = qsr_tilize_chunk_width(block_ct_dim_ct);
        UNPACK((llk_unpack_tilize_block_to_dest_init<block_ct_dim_ct /*FULL_CT_DIM*/, chunk /*BLOCK_CT_DIM*/>(icb)));
    } else {
        // Fallback single-tile MOP (block_ct_dim_ct not threaded by the caller).
        UNPACK((llk_unpack_tilize_init<true /*unpack_to_dest*/>(icb, block /*full_ct_dim*/)));
    }
    // Direct UNPACK<->PACK DEST double-buffer handshake (replaces the racy dvalid section scheme). UNPACK is the
    // sole DEST producer (UNPACR_TILIZE, SET_DVALID=0), PACK the sole consumer (PACR). A single semaphore
    // (UNPACK_MATH, reused purely as the unpack->pack token) with max=N (2 SyncHalf / 1 SyncFull) bounds the
    // unpacker to <=1 DEST bank ahead of the packer; each thread flips its OWN DEST section base in lockstep (in
    // tilize_block). Deterministic: the unpacker provably cannot lap the packer. MATH issues NO DEST ops on this
    // path (no MOVA2D/datacopy MOP -> the FPU dest-dvalid ring is never advanced -> ERROR_TRISC1 0x19 cannot
    // recur). The SEMINIT is done by UNPACK inside llk_unpack_tilize_dest_sync_init (the PRODUCER thread), NOT
    // MATH: UNPACK's acquire (SEMWAIT STALL_ON_MAX) latches the sem max at issue, so the SEMINIT (max=N) must be
    // ordered before it on the SAME thread — SEMINIT-on-MATH raced -> acquire latched the reset max=0 -> waited
    // for val<0 -> first-section deadlock (dprint_tr10). PACK only reads the value (reset 0), so it needs no
    // SEMINIT ordering. MATH is idle (no init call). The dvalid section scheme was insufficient: its masks came
    // from the single-section reference test and the unpack side never flipped its DEST bank id -> desync/lap.
    UNPACK((llk_unpack_tilize_dest_sync_init<DST_SYNC_MODE>()));
    PACK((llk_pack_tilize_dest_sync_init<DST_SYNC_MODE>()));
#else
    UNPACK((llk_unpack_tilize_init(icb, block /*full_ct_dim*/)));  // block_ct_dim defaults to 1
    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE>(icb)));
#endif
#endif
}

#if (defined(REDUCE_OP) and defined(REDUCE_DIM)) or defined(__DOXYGEN__)

// clang-format off
/**
 * Initializes the tilize operation with reduction. Should be called once at the beginning of a kernel.
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                              | Type     | Valid Range | Required |
 * |------------|----------------|------------------------------------------|----------|-------------|----------|
 * | Template   | neginf_srcA    | NegInf source A flag                     | bool     | true/false  | False    |
 * | Template   | zero_srcA_reduce| Zero source A for reduce flag           | bool     | true/false  | False    |
 * | Function   | icb0           | Input circular buffer A identifier       | uint32_t | 0 to 31     | True     |
 * | Function   | icb1_scaler    | Input circular buffer for scaler         | uint32_t | 0 to 31     | True     |
 * | Function   | block          | Size of tile block to work on            | uint32_t | > 0         | True     |
 * | Function   | ocb            | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 *
 * Unpack face geometry for operand A comes from circular-buffer metadata (JIT unpack_tile_* arrays), e.g.
 * set_unpack_face_geometry / set_tile_dims on the host.
 */
// clang-format on
template <bool neginf_srcA = true, bool zero_srcA_reduce = false>
ALWI void tilizeA_B_reduce_init(
    uint32_t icb0, uint32_t icb1_scaler, uint32_t block, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1_scaler, ocb, call_line);
#ifndef ARCH_QUASAR
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1_scaler)));
    UNPACK((llk_unpack_tilizeA_B_init<neginf_srcA, true /*reload_srcB*/, false /*zero_srcA*/, zero_srcA_reduce>(
        icb0, icb1_scaler, block)));

    MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, DST_ACCUM_MODE, MATH_FIDELITY>(icb0, icb1_scaler)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1_scaler)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>(ocb)));
#else
    UNPACK((llk_unpack_hw_configure(icb0, icb1_scaler)));
    UNPACK((llk_unpack_tilizeA_B_init<neginf_srcA, true /*reload_srcB*/, false /*zero_srcA*/, zero_srcA_reduce>(
        icb0, icb1_scaler, block)));

    MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, DST_ACCUM_MODE, MATH_FIDELITY>(icb0, icb1_scaler)));
    MATH((llk_math_pack_sync_init()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1_scaler)));

    PACK((llk_pack_hw_configure(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init()));
#endif
}

// clang-format off
/**
 * Re-initializes the fused tilize+reduce for a new input CB and/or a changed block size WITHOUT re-running the
 * one-time hw_configure that tilizeA_B_reduce_init performs. Call tilizeA_B_reduce_init once at kernel start,
 * then use this lighter variant to re-bind the unpack-tilize / math-reduce to a different input CB or a
 * changed `block` (tiles-to-reduce) mid-kernel -- e.g. per split-reader stream, or per channel-block when the
 * tile count changes. Re-running hw_configure every iteration corrupts unpacker state (UNPACKER fault), so
 * this issues only the per-use unpack/math inits (the pack side is re-init'd separately, e.g. via
 * pack_untilize_dest_init). These two llk inits are identical across WH/BH/Quasar, so no arch split is needed.
 *
 * | Param Type | Name             | Description                          | Type     | Valid Range | Required |
 * |------------|------------------|--------------------------------------|----------|-------------|----------|
 * | Template   | neginf_srcA      | NegInf source A flag                 | bool     | true/false  | False    |
 * | Template   | zero_srcA_reduce | Zero source A for reduce flag        | bool     | true/false  | False    |
 * | Function   | icb0             | Input circular buffer A identifier   | uint32_t | 0 to 31     | True     |
 * | Function   | icb1_scaler      | Input circular buffer for scaler     | uint32_t | 0 to 31     | True     |
 * | Function   | block            | Size of tile block to work on        | uint32_t | > 0         | True     |
 */
// clang-format on
template <bool neginf_srcA = true, bool zero_srcA_reduce = false>
ALWI void tilizeA_B_reduce_init_short(uint32_t icb0, uint32_t icb1_scaler, uint32_t block, uint32_t ocb) {
    // Identical to tilizeA_B_reduce_init but WITHOUT the three llk_*_hw_configure calls (unpack/math/pack),
    // which are one-time hardware setup and corrupt engine state if re-run per iteration. The per-use inits
    // below -- including the MATH pack-sync and the PACK init/dest-init -- must all be re-issued together so
    // UNPACK, MATH and PACK stay coherent when `block` (tiles-to-reduce) or the input CB changes mid-kernel.
    // On Quasar this matters especially because pack_untilize_dest_init only issues llk_pack_untilize_init and
    // relies on llk_pack_init/llk_pack_dest_init here for the packer's dest-offset / MATH-PACK sync state.
    UNPACK((llk_unpack_tilizeA_B_init<neginf_srcA, true /*reload_srcB*/, false /*zero_srcA*/, zero_srcA_reduce>(
        icb0, icb1_scaler, block)));
    MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, DST_ACCUM_MODE, MATH_FIDELITY>(icb0, icb1_scaler)));
#ifndef ARCH_QUASAR
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>(ocb)));
#else
    MATH((llk_math_pack_sync_init()));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init()));
#endif
}
#endif  // (REDUCE_OP && REDUCE_DIM) || __DOXYGEN__

#ifndef ARCH_QUASAR
// clang-format off
/**
 * Re-initializes the tilize operation and reconfigures the unpacker with CB data type.
 *
 * Return value: None
 *
 * | Param Type | Name     | Description                              | Type     | Valid Range | Required |
 * |----------- |----------|------------------------------------------|----------|-------------|----------|
 * | Function   | old_icb  | Previous input circular buffer identifier| uint32_t | 0 to 31     | True     |
 * | Function   | new_icb  | New input circular buffer identifier     | uint32_t | 0 to 31     | True     |
 * | Function   | block    | Size of tile block to work on            | uint32_t | > 0         | True     |
 * | Function   | ocb      | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void tilize_init_short_with_dt(uint32_t old_icb, uint32_t new_icb, uint32_t block, uint32_t ocb) {
    MATH((llk_math_eltwise_unary_datacopy_init<
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false /*is_int_en*/,
          PackMode::Tilize>(new_icb)));
    // This reconfig call checks if old operand has different data format to
    // new operand idx, otherwise no reconfig call occurs
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(old_icb, new_icb)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_icb, new_icb)));
    UNPACK((llk_unpack_tilize_init(new_icb, block)));

#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_init<PackMode::Tilize, false /* zero_output */>(ocb, 1 /* num_tiles */, new_icb)));
#endif
}
#endif  // !ARCH_QUASAR

// clang-format off
/**
 * Performs the tilize operation on a block.
 *
 * Return value: None
 *
 * | Param Type | Name             | Description                              | Type     | Valid Range | Required |
 * |----------- |------------------|------------------------------------------|----------|-------------|----------|
 * | Function   | icb              | Input circular buffer identifier         | uint32_t | 0 to 31     | True     |
 * | Function   | block            | Size of tile block to work on            | uint32_t | > 0         | True     |
 * | Function   | ocb              | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 * | Function   | input_tile_index | Index of the input tile in the icb       | uint32_t | >= 0        | False    |
 * | Function   | output_tile_index| Index of the output tile in the ocb      | uint32_t | >= 0        | False    |
 */
// clang-format on
template <uint32_t block_ct_dim_ct = 0>
ALWI void tilize_block(
    uint32_t icb, uint32_t block, uint32_t ocb, uint32_t input_tile_index = 0, uint32_t output_tile_index = 0) {
#ifdef ARCH_QUASAR
    // DEBUG (conv tilize 0x19 localizer): the LAST TZBLK/TZPK that flushes before the fault tells us how far
    // the tilize got. Same tilize_block PASSES in test_tilize.py but FAULTS in the conv2d stem -> the code is
    // fine, the DFB SETUP differs. Dump the tile geometry / formats of both CBs so the two runs can be diffed.
    // The per-tile PACK stride is set by the OUTPUT tile geometry (face_r_dim * num_faces * entry): if the
    // conv's ocb (act_tilized, borrowed/aliased) reports frdim != 16 or nf != 4 (a partial/2x2-face tile) the
    // pack address drifts off the whole-tile grid -> OOB -> ERROR_TRISC1 0x19. Remove after diagnosis.
    PACK(DPRINT(
        "TZBLK icb={} ocb={} block={} oti={}\n",
        (uint32_t)icb,
        (uint32_t)ocb,
        (uint32_t)block,
        (uint32_t)output_tile_index));
    PACK(DPRINT(
        "TZCFG-OUT ocb={} out_nf={} out_frdim={} out_narrow={} dstacc={} syncfull={}\n",
        (uint32_t)ocb,
        (uint32_t)get_output_num_faces(ocb),
        (uint32_t)get_output_face_r_dim(ocb),
        (uint32_t)get_output_narrow_tile(ocb),
        (uint32_t)DST_ACCUM_MODE,
        (uint32_t)(DST_SYNC_MODE == DstSync::SyncFull)));
    UNPACK(DPRINT(
        "TZCFG-IN icb={} in_srcfmt={} in_dstfmt={} in_nf={} in_frdim={}\n",
        (uint32_t)icb,
        (uint32_t)get_operand_src_format(icb),
        (uint32_t)get_operand_dst_format(icb),
        (uint32_t)get_operand_num_faces(icb),
        (uint32_t)get_operand_face_r_dim(icb)));
#endif
#if defined(ARCH_QUASAR) && defined(QSR_TILIZE_UNPACK_TO_DEST)
    // UnpackToDestEn bypass (tt-metal #49445): the unpacker tilizes DIRECTLY into DEST (UNP_DEST), so the
    // per-tile MATH A2D datacopy MOP (ERROR_TRISC1 0x19, Quasar DEST-section leak) is never issued. Sync is the
    // UNPACK<->PACK DEST-dvalid section-done handshake, NOT the MATH semaphore; MATH is bypassed entirely.
    if constexpr (block_ct_dim_ct > 0) {
        // BATCHED unpack-to-dest (the LLK-intended tilize-to-DEST — see the reference in tt-llk
        // tests/sources/quasar/unpack_tilize_quasar_test.cpp). One tilize block is a single 32-row tile-row of
        // block_ct_dim_ct column-tiles; the batched MOP tilizes a run of column-tiles into DISTINCT DEST slots
        // (advancing the L1 source by SRC_Z_STRIDE and DEST by one full tile per tile) with ONE section_done
        // per DEST section — fixing the single-tile path's slot-0 overwrite + per-tile section_done that
        // mis-ordered the tilized data (PCC~0). When the row is wider than DEST, split into equal compile-time
        // column chunks that each fit DEST; each chunk fills DEST slots 0..chunk-1 then packs them out.
        constexpr uint32_t chunk = qsr_tilize_chunk_width(block_ct_dim_ct);
        constexpr uint32_t num_chunks = block_ct_dim_ct / chunk;  // exact by construction of chunk
        static_assert(chunk * num_chunks == block_ct_dim_ct, "column chunks must tile the row exactly");
        for (uint32_t c = 0; c < num_chunks; c++) {
            // UNPACK: block until a DEST bank is free (<=1 bank ahead of PACK), tilize `chunk` column-tiles into
            // slots 0..chunk-1 of that bank, then publish it to PACK and flip UNPACK to the other bank.
            UNPACK((llk_unpack_tilize_dest_acquire()));
            UNPACK((llk_unpack_tilize_block_to_dest(icb, input_tile_index, c * chunk, 0 /*dest slot*/)));
            UNPACK((llk_unpack_tilize_dest_release<DST_SYNC_MODE, DST_ACCUM_MODE>()));
            // PACK: wait for UNPACK's published bank, pack all `chunk` slots (dslot j -> out tile j, confirmed
            // via dprint_utd2), then free the bank back to UNPACK and flip PACK to the other bank.
            PACK((llk_pack_tilize_dest_wait()));
            for (uint32_t j = 0; j < chunk; j++) {
                PACK((llk_pack<true /*out_of_order*/>(j /*DEST slot*/, ocb, output_tile_index + c * chunk + j)));
            }
            PACK((llk_pack_tilize_dest_release<DST_SYNC_MODE, DST_ACCUM_MODE>()));
            // MATH: intentionally absent — no DEST MOP is issued, so no MOVA2D and no 0x19.
        }
        return;
    }
    // Fallback single-tile UNP_DEST (block_ct_dim_ct not threaded by the caller): each tile -> DEST slot 0 with
    // a per-tile section_done. Mis-orders wide blocks; kept only for API completeness. The conv
    // (compute_kernel_lib::tilize) always threads block_ct_dim_ct, so the batched path above is what runs.
    for (uint32_t t = 0; t < block; t++) {
        UNPACK((llk_unpack_tilize_to_dest(icb, input_tile_index, t)));   // tile t -> DEST slot 0
        UNPACK((llk_unpack_dest_dvalid_section_done<DST_SYNC_MODE>()));  // mark DEST section valid for PACK
        PACK((llk_pack<true /*out_of_order*/>(
            0 /*tile index*/, ocb, t + output_tile_index)));                         // PACR waits on DEST dvalid
        PACK((llk_pack_dest_dvalid_section_done<DST_SYNC_MODE, DST_ACCUM_MODE>()));  // clear dvalid, free DEST bank
    }
    return;
#endif

    UNPACK((llk_unpack_tilize_block(icb, block, input_tile_index)));

    for (uint32_t t = 0; t < block; t++) {
        // Acquire dst
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));

#ifndef ARCH_QUASAR
        // Datacopy
        MATH((llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            0 /*dst index*/, icb)));
        PACK((llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(0 /*tile index*/, ocb, t + output_tile_index)));
#else
        MATH((llk_math_eltwise_unary_datacopy(0 /*dst index*/, icb)));
        // DEBUG: print immediately BEFORE the tilize pack; last TZPK before the fault = the faulting tile.
        PACK(DPRINT("TZPK t={} l1idx={}\n", (uint32_t)t, (uint32_t)(t + output_tile_index)));
        PACK((llk_pack<true /*out_of_order*/>(0 /*tile index*/, ocb, t + output_tile_index)));
#endif
        // Release dest
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// clang-format off
/**
 * Unpacks and tilizes a block from two input CBs.
 *
 * Return value: None
 *
 * | Param Type | Name             | Description                              | Type         | Valid Range | Required |
 * |------------|------------------|------------------------------------------|--------------|-------------|----------|
 * | Template   | neginf_srcA      | NegInf source A flag                     | bool         | true/false  | False    |
 * | Template   | reload_srcB      | Reload source B flag                     | std::uint32_t| true/false  | False    |
 * | Template   | zero_srcA        | Zero source A flag                       | bool         | true/false  | False    |
 * | Template   | zero_srcA_reduce | Zero source A for reduce flag            | bool         | true/false  | False    |
 * | Function   | icb0             | Input circular buffer A identifier       | uint32_t     | 0 to 31     | True     |
 * | Function   | icb1             | Input circular buffer B identifier       | uint32_t     | 0 to 31     | True     |
 * | Function   | block            | Size of tile block to work on            | uint32_t     | > 0         | True     |
 * | Function   | tile_idx_b       | Tile index for source B                  | uint32_t     | >= 0        | True     |
 *
 * Operand A face geometry is read from circular-buffer unpack metadata.
 */
// clang-format on
template <
    bool neginf_srcA = true,
    std::uint32_t reload_srcB = true,
    bool zero_srcA = false,
    bool zero_srcA_reduce = false>
ALWI void unpack_tilizeA_B_block(uint32_t icb0, uint32_t icb1, uint32_t block, uint32_t tile_idx_b) {
    UNPACK((llk_unpack_tilizeA_B_block<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
        icb0, icb1, block, tile_idx_b)));
}

// clang-format off
/**
 * Uninitializes the tilize operation before re-initializing for another operation.
 *
 * NOTE: This function is not in line with our programming model, and will be removed by the end of 2025
 * as a part of tt-metal#22904.
 * NOTE: Does nothing on Quasar because there is no persistent tilize unpack/pack state to undo.
 *
 * Return value: None
 *
 * | Param Type | Name   | Description                              | Type     | Valid Range | Required |
 * |----------- |--------|------------------------------------------|----------|-------------|----------|
 * | Function   | icb    | Input circular buffer identifier         | uint32_t | 0 to 31     | True     |
 * | Function   | ocb    | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 */
// clang-format on

ALWI void tilize_uninit(uint32_t icb, uint32_t ocb) {
    UNPACK((llk_unpack_tilize_uninit(icb)));
#if defined(ARCH_QUASAR) && defined(QSR_TILIZE_UNPACK_TO_DEST)
    // Clean the DEST dvalid ring + reset both banks to 0 for a FOLLOWING matmul in the same kernel. The semaphore
    // tilize above orders UNPACK<->PACK via the UNPACK_MATH count but does NOT touch the HW dvalid bits, whereas
    // the OLD dvalid section_done cleared them as a side effect — and the fused conv's matmul (utd1) relied on
    // inheriting that clean dvalid ring. The fused kernel's llk_math_pack_sync_init/llk_pack_dest_init reset the
    // semaphore + bank id but not the raw dvalid bits, so without this the matmul hangs at its first partials op
    // (dprint_utd10/11). The SyncFull section_done variant issues CLEARDVALID for BOTH banks and resets SEC->0.
    UNPACK((llk_unpack_dest_dvalid_section_done<DstSync::SyncFull>()));
    PACK((llk_pack_dest_dvalid_section_done<DstSync::SyncFull, DST_ACCUM_MODE>()));
#endif
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_init<PackMode::Default>(ocb)));
#endif
}

#ifndef ARCH_QUASAR
// clang-format off
/**
 * Uninitializes the tilize operation and reconfigures the unpacker with CB data types.
 *
 * NOTE: This function is not in line with our programming model, and will be removed by the end of 2025
 * as a part of tt-metal#22904.
 *
 * Return value: None
 *
 * | Param Type | Name     | Description                              | Type     | Valid Range | Required |
 * |----------- |----------|------------------------------------------|----------|-------------|----------|
 * | Function   | old_icb  | Previous input circular buffer identifier| uint32_t | 0 to 31     | True     |
 * | Function   | new_icb  | New input circular buffer identifier     | uint32_t | 0 to 31     | True     |
 * | Function   | ocb      | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void tilize_uninit_with_dt(uint32_t old_icb, uint32_t new_icb, uint32_t ocb) {
    UNPACK((llk_unpack_tilize_uninit(old_icb)));
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(old_icb, new_icb)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_icb, new_icb)));
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_init(ocb)));
#endif
}

namespace fast_tilize_detail {

template <bool configure_remap>
ALWI void fast_tilize_init_impl(uint32_t icb, uint32_t full_dim, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
#ifdef ARCH_BLACKHOLE
    if (full_dim == 1) {
        tilize_init(icb, full_dim, ocb, call_line);
        return;
    }
#endif

    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);

#ifdef ARCH_BLACKHOLE
    // first_chunk = decompose_row(full_dim)[0]: avoids first reinit_xdim in block loop.
    uint32_t first_chunk = (full_dim > 5) ? 4 : (full_dim == 5) ? 2 : full_dim;
    UNPACK((llk_unpack_fast_tilize_init(icb, full_dim, first_chunk)));
    if constexpr (configure_remap) {
        MATH((llk_math_fast_tilize_init(icb)));
    } else {
        MATH((llk_math_fast_tilize_init_skip_remap(icb)));
    }
    PACK((llk_pack_fast_tilize_init(icb, ocb, first_chunk)));
#else
    UNPACK((llk_unpack_fast_tilize_init(icb, full_dim)));
    MATH((llk_math_fast_tilize_init(icb, full_dim == 1 ? 1 : 2)));
    PACK((llk_pack_fast_tilize_init(icb, ocb, full_dim == 1 ? 1 : 2)));
#endif
}

}  // namespace fast_tilize_detail

ALWI void fast_tilize_init(uint32_t icb, uint32_t full_dim, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    fast_tilize_detail::fast_tilize_init_impl<true>(icb, full_dim, ocb, call_line);
}

ALWI void fast_tilize_init_skip_remap(
    uint32_t icb, uint32_t full_dim, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    fast_tilize_detail::fast_tilize_init_impl<false>(icb, full_dim, ocb, call_line);
}

namespace fast_tilize_detail {

template <bool configure_remap>
ALWI void fast_tilize_init_with_dt_impl(uint32_t icb, uint32_t full_dim, uint32_t ocb) {
    // Reconfig both SrcA and SrcB to match WH: some activation-reuse call sites
    // leave SrcB in a prior matmul-weights config that's incompatible with the
    // fast-tilize path, producing garbage output.
    UNPACK((llk_unpack_reconfig_data_format<DST_ACCUM_MODE, p_dim_stride_target::IGNORE>(icb, icb)));
    MATH((llk_math_reconfig_data_format<true, true>(icb, icb)));

    fast_tilize_init_impl<configure_remap>(icb, full_dim, ocb);
}

}  // namespace fast_tilize_detail

ALWI void fast_tilize_init_with_dt(uint32_t icb, uint32_t full_dim, uint32_t ocb) {
    fast_tilize_detail::fast_tilize_init_with_dt_impl<true>(icb, full_dim, ocb);
}

ALWI void fast_tilize_init_with_dt_skip_remap(uint32_t icb, uint32_t full_dim, uint32_t ocb) {
    fast_tilize_detail::fast_tilize_init_with_dt_impl<false>(icb, full_dim, ocb);
}

ALWI void fast_tilize_uninit(uint32_t icb, uint32_t ocb, uint32_t full_dim) {
#ifdef ARCH_BLACKHOLE
    if (full_dim == 1) {
        tilize_uninit(icb, ocb);
        return;
    }
#endif

    UNPACK((llk_unpack_fast_tilize_uninit<DST_ACCUM_MODE>()));
    MATH((llk_math_fast_tilize_uninit<DST_ACCUM_MODE>(icb)));
    PACK((llk_pack_fast_tilize_uninit<DST_ACCUM_MODE>(ocb)));
}

ALWI void fast_tilize_block(
    uint32_t icb, uint32_t block, uint32_t ocb, uint32_t input_tile_index = 0, uint32_t output_tile_index = 0) {
#ifdef ARCH_BLACKHOLE
    if (block == 1) {
        tilize_block(icb, block, ocb, input_tile_index, output_tile_index);
        return;
    }
    ASSERT(block > 1);

    // BH fast-tilize: each row chunk calls llk_unpack_fast_tilize_block directly.
    // Pack programs output L1 destination once per call; replay advances per tile.
    {
        input_tile_index = input_tile_index % block + (input_tile_index / block) * block * TILE_R_DIM;

        uint32_t tiles_done = 0;
        // Always program the current unit dim at block entry.
        uint32_t prev_chunk = 0;

        PACK((llk_pack_fast_tilize_row_begin(ocb, output_tile_index)));

        while (tiles_done < block) {
            // BH fast-tilize MOP supports unit_dim 2, 3, 4 (not 1).
            // Avoid chunk=1 by splitting: remaining=5 → 2+3 instead of 4+1.
            // Matches LLK decompose_row order.
            uint32_t remaining = block - tiles_done;
            uint32_t chunk = (remaining > 5) ? 4 : (remaining == 5) ? 2 : remaining;

            MATH((llk_math_wait_for_dest_available()));
            PACK((llk_packer_wait_for_math_done()));

            if (chunk != prev_chunk) {
                UNPACK((llk_unpack_fast_tilize_reinit_xdim(chunk)));
                PACK((llk_pack_fast_tilize_reinit_unit_dim(ocb, chunk)));
                prev_chunk = chunk;
            }
            UNPACK((llk_unpack_fast_tilize_block(icb, input_tile_index, chunk, tiles_done)));
            MATH((llk_math_fast_tilize_block_(0, icb, 4)));
            PACK((llk_pack_fast_tilize_row_chunk(0, ocb, chunk)));

            MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
            PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));

            tiles_done += chunk;
        }

        PACK((llk_pack_fast_tilize_row_end()));
    }
#else
    uint32_t full_dim = block;

    // Not sure if input_tile_index can be arbitrary but it works for moving across rows of files,
    // i.e. input_tile_index % full_dim == 0
    input_tile_index = input_tile_index % full_dim + (input_tile_index / full_dim) * full_dim * TILE_R_DIM;

    uint32_t packed_tiles = 0;
    uint32_t remaining_tiles = block;
    uint32_t dest_size = DST_ACCUM_MODE ? 4 : 8;
    uint32_t unit_dim = full_dim == 1 ? 1 : 2;
    uint32_t num_units = dest_size / unit_dim;

    while (packed_tiles < block) {
        UNPACK(uint32_t read_tile_index = input_tile_index + packed_tiles);
        PACK(uint32_t write_tile_index = output_tile_index + packed_tiles);

        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));

        if (remaining_tiles > 2 * dest_size) {
            // Three or more dests
            UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
            MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
            PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
            packed_tiles += dest_size;
            remaining_tiles -= dest_size;
        } else if (remaining_tiles > dest_size) {
            // Two dests
            uint32_t even_remainder = remaining_tiles / 2 + ((remaining_tiles / 2) % 2);
            num_units = even_remainder / unit_dim;
            UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
            MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
            PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
            packed_tiles += even_remainder;
            remaining_tiles -= even_remainder;
        } else {
            // Last dest
            if (remaining_tiles % 2 == 0 || unit_dim == 1) {
                // Single sequence
                num_units = remaining_tiles / unit_dim;
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
                MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
                PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
            } else if (remaining_tiles == 3) {
                // only odd pack
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, 3, 1, full_dim)));
                MATH((llk_math_fast_tilize_block_(0, icb, 3, 1)));
                PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, 3, 1)));
            } else {
                // even packs plus odd pack
                num_units = (remaining_tiles - 3) / unit_dim;
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
                MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
                PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));

                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index + remaining_tiles - 3, 3, 1, full_dim)));
                MATH((llk_math_fast_tilize_block_(remaining_tiles - 3, icb, 3, 1)));
                PACK((llk_pack_fast_tilize_block(
                    remaining_tiles - 3, ocb, write_tile_index + remaining_tiles - 3, 3, 1)));
            }
            packed_tiles += remaining_tiles;
            remaining_tiles = 0;
        }

        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
#endif
}

#endif  // !ARCH_QUASAR

// clang-format off
/**
 * Uninitializes the unpack tilizeA_B configuration and restores unpacker state
 * modified by _llk_unpack_tilizeA_B_init_.
 *
 * Return value: None
 *
 * Parameters:
 *
 * | Param Type | Name | Description           | Type     | Valid Range | Required |
 * |------------|------|-----------------------|----------|-------------|----------|
 * | Function   | icb  | Input circular buffer | uint32_t | 0 - 31.     | True     |
 *
 * Restored hardware state:
 *
 * | Field / Setting           | Scope      | Description                                           | Restored value / behavior                                                                  |
 * |---------------------------|------------|-------------------------------------------------------|--------------------------------------------------------------------------------------------|
 * | X-dim & base (ADCXX)      | UNP_A/B    | Face X-extent for address counters                    | face_r_dim * FACE_C_DIM elements, start at 0                                               |
 * | XY address counters       | UNP_A/B    | X/Y counters used by tilizeA_B y-stride pattern       | Counters reset to 0 (mask selects CH0/CH1 X/Y)                                             |
 * | ZW address counters       | UNP_A/B    | Z/W counters used for face/row stepping               | Counters reset to 0 for both unpackers                                                     |
 * | Out_data_format/config[0] | THCON_SEC0 | Unpack config[0]: out format, throttle, tilize, shift | out_data_format = unpack_dst_format; throttle_mode = 2; tileize_mode = 0; shift_amount = 0 |
 * | Tile_x_dim (cntx0)        | THCON_SEC0 | Tile X dimension per context for unpacker             | Restored to FACE_DIM_16x16 (16 | (16 << 16))                                               |
 */
// clang-format on
ALWI void unpack_tilizeA_B_uninit(uint32_t icb) { UNPACK((llk_unpack_tilizeA_B_uninit(icb))); }
}  // namespace ckernel
