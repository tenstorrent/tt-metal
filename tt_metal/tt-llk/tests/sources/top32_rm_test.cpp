// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPU top32_rm test: DeepSeek row-major top-32 with paired indices. Blackhole-only.
//
// This test is the sole coverage for the seven promoted experimental wrappers that
// the DeepSeek top32_rm compute kernels use:
//   UNPACK: ckernel::_llk_unpack_A_top32_rm_init_ / _llk_unpack_A_top32_rm_
//           (experimental/llk_unpack_A_top32_rm.h)
//   MATH:   ckernel::_llk_math_top32_rm_init_ / _llk_math_top32_rm_
//           (experimental/llk_math_top32_rm.h)
//   SFPU:   ckernel::sfpu::_top32_rm_init_,
//           _bitonic_top32_phases_steps_, _bitonic_top32_merge_, _bitonic_top32_rebuild_,
//           _bitonic_top32_of_1024_rm_pre_sorted_{prep,combine,final}_
//           (sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h)
//
// It mirrors the two on-silicon demo compute kernels exactly:
//   * TOP32_MODE == 0  -> tests/.../compute/top32_rm_dev_compute.cpp
//                         (the "< 1024 elements", 64-elements-at-a-time path)
//   * TOP32_MODE == 1  -> tests/.../compute/top32_rm_dev_compute_v2.cpp
//                         (the ">= 1024 elements", whole-1024-chunk pre-sort path)
// and the gtest tests/.../llk/test_top32_rm_dev.cpp selects the same kernel per
// row_elements (< 1024 -> mode 0, >= 1024 -> mode 1).
//
// -----------------------------------------------------------------------------
// GOLDEN DERIVATION (mirrors the gtest reference verify_top32_outputs()):
//
//   Inputs are two flat row-major streams for one row of `TOP32_RM_ROW_ELEMENTS`
//   elements:
//     buffer_A[i]  = bf16 score of element i         (value input, bf16)
//     buffer_B[i]  = uint32 index of element i = i   (index input, uint32)
//
//   The kernel unpacks scores into DEST value tiles and indices into DEST index
//   tiles (offset by 2 tiles), then runs an index-tracking bitonic sort. Index
//   tracking means each score carries its paired index word through every
//   compare-exchange, so the surviving top-32 scores come out paired with the
//   index words they entered with. Because buffer_B[i] == i, the reported index
//   of a surviving score is that score's original row-major position.
//
//   The reference is therefore a plain descending value-sort with the paired
//   index:
//     rank the (score, orig_idx) pairs by score DESCENDING, ties broken by
//     smaller orig_idx (mirrors the gtest std::sort comparator), then take the
//     first 32. Value compares are done on the bf16 value (the SFPSWAP compares
//     the INT32-reinterpreted fused words, and every stimulus here is exactly
//     representable in bf16, so score order == fp32 order of the bf16 values).
//
//   VALIDATED LANES: the packer is set to a single output row (step 6/10 does
//   TTI_SETADCXX(PAC, 1-1) -> pack a single row), so only the top-32 result row is
//   written and, with a single row, it lands at the first 32 contiguous words of each
//   packed tile (words 0..31), exactly as the on-silicon gtest reads out0/out1. Every
//   other lane in the packed tiles is undefined and must NOT be validated. The golden
//   is checked as: the 32 defined values are the top-32 score multiset, and each
//   defined (index, value) pair is consistent with the input (index -> input[index] ==
//   value). Index SET equality is only asserted for strictly-distinct-score inputs,
//   since the sort is not stable across ties (same rule the gtest applies: it
//   validates scores, not indices).
//
//   Output layout in buffer_Res (per the demo kernels' step 6 / step 10; both
//   tiles are UInt32-sized, 4 bytes/element):
//     buffer_Res[0] = value tile  (top-32 scores in row 0)   packed bf16 -> Float32
//                     so each score is a full fp32 word (the golden reads it as fp32)
//     buffer_Res[1] = index tile  (top-32 indices in row 0)  raw UInt32 words
// -----------------------------------------------------------------------------
//
// Compile-time parameters (TOP32_RM param class):
//   TOP32_RM_ROW_ELEMENTS : elements in the row (row_elements axis)
//   TOP32_RM_MODE         : 0 = 64-at-a-time path (< 1024), 1 = 1024-chunk path (>= 1024)
//   TOP32_RM_NUM_IN_TILES : ceil(row_elements / 1024) input tiles per stream

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

using namespace ckernel;

constexpr std::uint32_t VALUE_OFFSET_TILES = 0;
constexpr std::uint32_t INDEX_OFFSET_TILES = 2;
constexpr std::uint32_t CHUNK_SIZE         = 1024;

// The bitonic merge across the running top-32 always keeps the max half, and the
// per-step sort directions / skip_second flags are fixed by the demo-kernel
// structure this test mirrors (they reorder how the survivors are rebuilt, not
// which survive), so they are hardcoded per step below rather than parameterised.
constexpr bool DECREASING = false;
constexpr bool INCREASING = true;

#ifdef LLK_TRISC_UNPACK

// _llk_unpack_A_top32_rm_init_ takes an unpack_src_format it never references (the
// init only needs the dst format for the x-stride), which trips the build's
// -Werror=unused-parameter. For a template the diagnostic is evaluated in the
// diagnostic state at the point of DEFINITION, so silence it around the header
// include; the header is a promoted experimental wrapper we must not edit here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_unpack_A_top32_rm.h"
#pragma GCC diagnostic pop
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

// Replicates llk_unpack_A_top32_rm() from the metal wrapper for one 64-element
// group (bf16 value path: same_src_format=false, transpose within face enabled).
// `stream_base` is the L1 address of element 0 of the stream; the wrapper adds a
// per-group offset of (64>>4)*datum_size*group_index.
inline void unpack_top32_group(
    std::uint32_t stream_base, std::uint32_t group_index, std::uint32_t num_faces, std::uint32_t src_format, std::uint32_t dst_format)
{
    const DataFormat dst_masked  = static_cast<DataFormat>(dst_format & 0x3);
    const std::uint32_t datum_sz = dst_masked == DataFormat::Float32 ? 4 : dst_masked == DataFormat::Float16 ? 2 : 1;
    const std::uint32_t address  = stream_base + (64 >> 4) * datum_sz * group_index;

    // unpack_to_dest=true mirrors the demo kernels: the templates decide per-format
    // whether the stream lands directly in DEST (the 32-bit uint32 index stream) or is
    // fed to SrcA for the MATH datacopy (the non-32-bit bf16 value stream).
    _llk_unpack_A_top32_rm_init_<true>(true /* within_face_16x16_transpose */, src_format, dst_format);
    _llk_unpack_A_top32_rm_<true>(num_faces, address, src_format, dst_format);
}

// The whole-1024-chunk path (mode 1) transpose-unpacks a full tile per chunk,
// exactly as the demo v2 kernel's transpose_tile does. transpose_tile lowers to
// an A2D datacopy on SrcA plus a SrcB dummy-valid feeding the MATH transpose_dest,
// so UNPACK feeds one SrcA tile then one dummy SrcB valid per chunk-tile.
inline void unpack_transpose_tile(std::uint32_t l1_address, std::uint32_t src_format, std::uint32_t dst_format)
{
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 1 /* within_face_16x16_transpose */, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4), src_format, dst_format);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(l1_address, src_format, dst_format);
    _llk_unpack_set_srcb_dummy_valid_();
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    // buffer_A: bf16 scores (value stream), buffer_B: uint32 indices (index stream).
    const std::uint32_t val_src_format = formats.unpack_A_src;
    const std::uint32_t val_dst_format = formats.unpack_A_dst;
    const std::uint32_t idx_src_format = formats.unpack_B_src;
    const std::uint32_t idx_dst_format = formats.unpack_B_dst;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        val_src_format,
        idx_src_format,
        val_dst_format,
        idx_dst_format,
        FACE_R_DIM,
        FACE_R_DIM,
        TILE_NUM_FACES /* unpA_num_faces */,
        TILE_NUM_FACES /* unpB_num_faces */);

    const std::uint32_t val_base = L1_ADDRESS(params.buffer_A[0]);
    const std::uint32_t idx_base = L1_ADDRESS(params.buffer_B[0]);

    if constexpr (TOP32_RM_MODE == 1)
    {
        // Whole-1024-chunk path (top32_rm_dev_compute_v2): each 1024-element chunk
        // is transpose-unpacked as one full tile (values then indices), then the
        // presorted-1024 SFPU prep/combine/final run on MATH. The < 1024 tail is
        // then processed 64-at-a-time (same as mode 0).
        for (std::uint32_t chunk = 0; chunk < TOP32_RM_NUM_IN_TILES; chunk++)
        {
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                val_src_format, val_dst_format, params.TILE_SIZE_UNPACK_A);
            unpack_transpose_tile(L1_ADDRESS(params.buffer_A[chunk]), val_src_format, val_dst_format);
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                idx_src_format, idx_dst_format, params.TILE_SIZE_UNPACK_B);
            unpack_transpose_tile(L1_ADDRESS(params.buffer_B[chunk]), idx_src_format, idx_dst_format);
        }

        // Tail: any remaining < 1024 elements are unpacked 64-at-a-time. The tail
        // begins at input element TOP32_RM_NUM_IN_TILES * 1024, i.e. group index
        // (NUM_IN_TILES * 1024 / 64) within the stream.
        for (std::uint32_t i = TOP32_RM_NUM_IN_TILES * CHUNK_SIZE; i < TOP32_RM_ROW_ELEMENTS; i += 64)
        {
            const std::uint32_t num_faces = (i + 64 > TOP32_RM_ROW_ELEMENTS) ? 2 : 4;
            const std::uint32_t group     = i / 64;

            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                val_src_format, val_dst_format, params.TILE_SIZE_UNPACK_A);
            unpack_top32_group(val_base, group, num_faces, val_src_format, val_dst_format);
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                idx_src_format, idx_dst_format, params.TILE_SIZE_UNPACK_B);
            unpack_top32_group(idx_base, group, num_faces, idx_src_format, idx_dst_format);
        }
    }
    else
    {
        // 64-at-a-time path (top32_rm_dev_compute). Group 0 seeds the running top-32,
        // then each subsequent 64-element group is unpacked and merged.
        for (std::uint32_t i = 0; i < TOP32_RM_ROW_ELEMENTS; i += 64)
        {
            const std::uint32_t num_faces = (i + 64 > TOP32_RM_ROW_ELEMENTS) ? 2 : 4;
            const std::uint32_t group     = i / 64;

            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                val_src_format, val_dst_format, params.TILE_SIZE_UNPACK_A);
            unpack_top32_group(val_base, group, num_faces, val_src_format, val_dst_format);
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                idx_src_format, idx_dst_format, params.TILE_SIZE_UNPACK_B);
            unpack_top32_group(idx_base, group, num_faces, idx_src_format, idx_dst_format);
        }
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

// The SFPU bitonic body is large; keep TRISC1 under the code budget.
#pragma GCC optimize("O2")

// llk_math_top32_rm_configure_mop takes a total_rows it never references (the MOP
// loop bounds are hardcoded), tripping -Werror=unused-parameter. Same rule as the
// UNPACK header: silence around the include (definition-point diagnostic state);
// this is a promoted experimental wrapper we must not edit here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "experimental/llk_math_top32_rm.h"
#pragma GCC diagnostic pop
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpu/experimental/ckernel_sfpu_deepseek_top32_rm.h"

using namespace ckernel;

constexpr bool APPROX = false; // The bitonic wrappers take APPROXIMATION_MODE but never use it.

// Datacopy a 64-element (or 1024-element chunk) group of one stream into DEST.
// Mirrors llk_math_top32_rm(): the bf16 value / uint32 index streams have
// distinct formats, so MATH reconfigs the srca format per stream.
inline void math_top32_group(std::uint32_t dst_tile, std::uint32_t num_faces, std::uint32_t src_format, std::uint32_t dst_format)
{
    _llk_math_top32_rm_init_<is_fp32_dest_acc_en>(num_faces, dst_format);
    // The unpack->dest handshake in _llk_math_top32_rm_<...,unpack_to_dest=true>
    // (math_unpack_to_dest_math_ready / mailbox_write / math_unpack_to_dest_tile_ready) is
    // matched, on the unpack side (_llk_unpack_A_top32_rm_), by a half that runs ONLY when
    // is_32bit_input() is true. Hardcoding unpack_to_dest=true here deadlocks the bf16
    // value stream: MATH does the full handshake but the 16-bit UNPACR never posts the
    // MATH_DONE / UNPACK_TO_DEST tokens MATH waits on. Gate on the SAME is_32bit_input()
    // predicate the unpacker uses, exactly as the metal wrapper (llk_math_top32_rm_api.h)
    // splits int32 vs. non-int32 -- so the 32-bit index stream takes the unpack->dest path
    // and the bf16 value stream takes the SrcA->dest MOP path, symmetrically on both TRISCs.
    if (math::is_32bit_input(src_format, dst_format))
    {
        _llk_math_top32_rm_<dest_sync, is_fp32_dest_acc_en, true>(dst_tile, src_format, dst_format, num_faces);
    }
    else
    {
        _llk_math_top32_rm_<dest_sync, is_fp32_dest_acc_en, false>(dst_tile, src_format, dst_format, num_faces);
    }
}

// Mode-1 whole-1024-chunk load: A2D datacopy the tile into DEST then transpose it
// in place, mirroring the demo v2 kernel's transpose_tile.
inline void math_transpose_tile(std::uint32_t dst_tile, std::uint32_t math_format)
{
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(4, math_format);
    _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        dst_tile, math_format, math_format);

    _llk_math_transpose_dest_init_<false /* transpose_of_faces */, is_fp32_dest_acc_en>();
    _llk_math_transpose_dest_wrapper_<is_fp32_dest_acc_en, false /* transpose_of_faces */, is_fp32_dest_acc_en>(dst_tile);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t val_src_format = formats.unpack_A_src;
    const std::uint32_t val_dst_format = formats.unpack_A_dst;
    const std::uint32_t idx_src_format = formats.unpack_B_src;
    const std::uint32_t idx_dst_format = formats.unpack_B_dst;
    const std::uint32_t math_format    = formats.math;

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
    _llk_math_wait_for_dest_available_<dest_sync>();

    if constexpr (TOP32_RM_MODE == 1)
    {
        // --- Whole-1024-chunk path (top32_rm_dev_compute_v2) ---
        // step 1: transpose-load chunk 0 values + indices into value/index tiles.
        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(val_src_format);
        math_transpose_tile(VALUE_OFFSET_TILES, math_format);
        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(idx_src_format);
        math_transpose_tile(INDEX_OFFSET_TILES, math_format);

        // step 2: presort the first 1024 elements.
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
        ckernel::sfpu::_top32_rm_init_();
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            _bitonic_top32_of_1024_rm_pre_sorted_prep_,
            (APPROX, is_fp32_dest_acc_en, DECREASING),
            VALUE_OFFSET_TILES,
            VectorMode::RC_custom,
            VALUE_OFFSET_TILES);

        // loop over remaining 1024-chunks.
        for (std::uint32_t chunk = 1; chunk < TOP32_RM_NUM_IN_TILES; chunk++)
        {
            // step 3: transpose-load next chunk into scratch value/index tiles.
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(val_src_format);
            math_transpose_tile(VALUE_OFFSET_TILES + 1, math_format);
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(idx_src_format);
            math_transpose_tile(INDEX_OFFSET_TILES + 1, math_format);

            // step 4: presort the new chunk (opposite direction).
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _bitonic_top32_of_1024_rm_pre_sorted_prep_,
                (APPROX, is_fp32_dest_acc_en, INCREASING),
                VALUE_OFFSET_TILES + 1,
                VectorMode::RC_custom,
                VALUE_OFFSET_TILES + 1);

            // step 5: merge+rebuild across the two chunks into the running top-32.
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _bitonic_top32_of_1024_rm_pre_sorted_combine_,
                (APPROX, is_fp32_dest_acc_en),
                VALUE_OFFSET_TILES,
                VectorMode::RC_custom,
                VALUE_OFFSET_TILES);
        }

        // step 6: reduce the 16 columns of F0/F1 into the final top-32.
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            _bitonic_top32_of_1024_rm_pre_sorted_final_,
            (APPROX, is_fp32_dest_acc_en),
            VALUE_OFFSET_TILES,
            VectorMode::RC_custom,
            VALUE_OFFSET_TILES);

        // tail: any remaining < 1024 elements are processed 64-at-a-time and merged.
        for (std::uint32_t i = TOP32_RM_NUM_IN_TILES * CHUNK_SIZE; i < TOP32_RM_ROW_ELEMENTS; i += 64)
        {
            const std::uint32_t num_faces = (i + 64 > TOP32_RM_ROW_ELEMENTS) ? 2 : 4;

            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(val_src_format);
            math_top32_group(VALUE_OFFSET_TILES + 1, num_faces, val_src_format, val_dst_format);
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(idx_src_format);
            math_top32_group(INDEX_OFFSET_TILES + 1, num_faces, idx_src_format, idx_dst_format);

            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _bitonic_top32_rebuild_,
                (APPROX, is_fp32_dest_acc_en),
                VALUE_OFFSET_TILES + 1,
                VectorMode::RC_custom,
                DECREASING,
                false);
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _bitonic_top32_merge_,
                (APPROX, is_fp32_dest_acc_en, false),
                VALUE_OFFSET_TILES + 1,
                VectorMode::RC_custom,
                false);
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _bitonic_top32_rebuild_,
                (APPROX, is_fp32_dest_acc_en),
                VALUE_OFFSET_TILES + 1,
                VectorMode::RC_custom,
                INCREASING,
                true);
            SFPU_UNARY_CALL(
                dest_sync, is_fp32_dest_acc_en, _bitonic_top32_merge_, (APPROX, is_fp32_dest_acc_en, false), VALUE_OFFSET_TILES, VectorMode::RC_custom, true);
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _bitonic_top32_rebuild_,
                (APPROX, is_fp32_dest_acc_en),
                VALUE_OFFSET_TILES,
                VectorMode::RC_custom,
                DECREASING,
                true);
        }
    }
    else
    {
        // --- 64-at-a-time path (top32_rm_dev_compute) ---
        // step 1: unpack first 64 elements (values + indices) into the running tiles.
        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(val_src_format);
        math_top32_group(VALUE_OFFSET_TILES, 4, val_src_format, val_dst_format);
        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(idx_src_format);
        math_top32_group(INDEX_OFFSET_TILES, 4, idx_src_format, idx_dst_format);

        // step 2: sort the seed group into a decreasing top-32.
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
        ckernel::sfpu::_top32_rm_init_();
        SFPU_UNARY_CALL(
            dest_sync, is_fp32_dest_acc_en, _bitonic_top32_phases_steps_, (APPROX, is_fp32_dest_acc_en), VALUE_OFFSET_TILES, VectorMode::RC_custom, DECREASING);
        SFPU_UNARY_CALL(
            dest_sync, is_fp32_dest_acc_en, _bitonic_top32_merge_, (APPROX, is_fp32_dest_acc_en, false), VALUE_OFFSET_TILES, VectorMode::RC_custom, false);
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            _bitonic_top32_rebuild_,
            (APPROX, is_fp32_dest_acc_en),
            VALUE_OFFSET_TILES,
            VectorMode::RC_custom,
            DECREASING,
            true);

        for (std::uint32_t i = 64; i < TOP32_RM_ROW_ELEMENTS; i += 64)
        {
            const std::uint32_t num_faces = (i + 64 > TOP32_RM_ROW_ELEMENTS) ? 2 : 4;

            // step 3: unpack next 64 elements into the scratch value/index tiles.
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(val_src_format);
            math_top32_group(VALUE_OFFSET_TILES + 1, num_faces, val_src_format, val_dst_format);
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(idx_src_format);
            math_top32_group(INDEX_OFFSET_TILES + 1, num_faces, idx_src_format, idx_dst_format);

            // step 4: sort the new group increasing, then merge/rebuild it.
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _bitonic_top32_phases_steps_,
                (APPROX, is_fp32_dest_acc_en),
                VALUE_OFFSET_TILES + 1,
                VectorMode::RC_custom,
                DECREASING);
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _bitonic_top32_merge_,
                (APPROX, is_fp32_dest_acc_en, false),
                VALUE_OFFSET_TILES + 1,
                VectorMode::RC_custom,
                false);
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _bitonic_top32_rebuild_,
                (APPROX, is_fp32_dest_acc_en),
                VALUE_OFFSET_TILES + 1,
                VectorMode::RC_custom,
                INCREASING,
                true);

            // step 5: merge the new group's top-32 across tiles into the running top-32.
            SFPU_UNARY_CALL(
                dest_sync, is_fp32_dest_acc_en, _bitonic_top32_merge_, (APPROX, is_fp32_dest_acc_en, false), VALUE_OFFSET_TILES, VectorMode::RC_custom, true);
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                _bitonic_top32_rebuild_,
                (APPROX, is_fp32_dest_acc_en),
                VALUE_OFFSET_TILES,
                VectorMode::RC_custom,
                DECREASING,
                true);
        }
    }

    _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t val_pack_src = formats.pack_src;

    // buffer_Res is UInt32 (4 bytes/element) for both tiles. The value DEST tile is
    // bf16, but it is packed out as Float32 so the score comes back as a full 32-bit
    // fp32 word (the exact bf16 value widened to fp32); the golden reads it as fp32.
    // The index DEST tile is packed as raw UInt32 words.
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        val_pack_src, (std::uint32_t)DataFormat::Float32, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>((std::uint32_t)DataFormat::Float32, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();

    // step 6/10 of the demo kernels: the final top-32 lives in ROW 0 of the value
    // tile and ROW 0 of the index tile. The pack row-count is set to a single row
    // (TTI_SETADCXX(PAC, 1-1)) so only row 0 (the 32 top elements) is written.
    TTI_SETADCXX(p_setadc::PAC, 1 - 1, 0x0);
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(VALUE_OFFSET_TILES, L1_ADDRESS(params.buffer_Res[0]));

    // The index tile is packed as raw uint32 words (index stream format).
    _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en>(
        val_pack_src, (std::uint32_t)DataFormat::UInt32, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */);
    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(INDEX_OFFSET_TILES, L1_ADDRESS(params.buffer_Res[1]));

    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif // LLK_TRISC_PACK
