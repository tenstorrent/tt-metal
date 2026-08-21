// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#ifdef TRISC_UNPACK
#ifdef ARCH_BLACKHOLE
#include "experimental/llk_unpack_A_topk_xl_copy_api.h"
#endif
#endif
#ifdef TRISC_MATH
#ifdef ARCH_BLACKHOLE
#include "experimental/llk_math_topk_xl_copy_api.h"
#include "experimental/llk_sfpu/llk_math_eltwise_unary_sfpu_topk_xl.h"
#endif
#endif
#ifdef TRISC_PACK
#ifdef ARCH_BLACKHOLE
#include "experimental/llk_sfpu/llk_math_eltwise_unary_sfpu_topk_xl.h"
#endif
#endif

namespace ckernel {

/**
 * Performs local sort on 512, 1024, or 2048 elements in DST registers.
 * 512 requires half of a tile in DST.
 * 1024 requires 1 tile in DST.
 * 2048 requires 2 tiles in DST.
 * Sorts elements in bitonic order for later merge stages.
 *
 * This implements a full bitonic sorting network for K elements,
 * with values and indices fused as (bf16 value | u16 index) in FP32 format.
 *
 * Return value: None
 *
 * | Argument   | Description                                                                | Type     | Valid Range |
 * Required |
 * |------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | K          | Number of elements to sort                                                 | uint32_t | 512, 1024, or
 * 2048                                    | True     | | idst       | The index of the tile in DST register buffer to
 * perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | |
 * ascending  | Sort direction: true for ascending, false for descending                   | bool     | true, false |
 * True     |
 */
template <std::uint32_t K>
ALWI void topk_xl_local_sort(std::uint32_t idst, bool ascending) {
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_local_sort<K>(idst, ascending)));
}

/**
 * Per-column-isolated local sort. Runs the same bitonic network as
 * topk_xl_local_sort up to the length-64 build, then (with early_exit_K64)
 * returns before the cross-column merge phases, leaving each 64-row column
 * sorted in isolation — used by the sparse-K reader to sink all-zero packed
 * mask words to the bottom of each column.
 *
 * With early_exit_K64 = true this does NOT issue llk_unpack_set_srcb_dummy_valid:
 * the early-exit column sort never consumes a real SrcB operand, so the dummy
 * valid is dead config, and dropping it saves one UNPACK-thread issue per call.
 * The default (full-sort) instantiation runs the same network as
 * topk_xl_local_sort and does issue it.
 *
 * early_exit_K64 requires K >= 1024, and the full sort K = 512 or K = 1024: the
 * generic network does not converge at K = 2048, so that size has to go through
 * topk_xl_local_sort and its fast path. Both are enforced by static_assert.
 */
template <std::uint32_t K, bool early_exit_K64 = false>
ALWI void topk_xl_local_sort_generic(std::uint32_t idst, bool ascending) {
    if constexpr (!early_exit_K64) {
        UNPACK((llk_unpack_set_srcb_dummy_valid()));
    }
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_local_sort_generic<K, early_exit_K64>(idst, ascending)));
}

/**
 * Merges two sorted sequences of K elements each, such that the top K elements are moved to the first sequence.
 *
 * If fused is true, the data is fused as (bf16 value | u16 index) in FP32 format.
 * If fused is false, the data is not fused and is stored as FP32 values and UInt32 indices.
 *
 * Expects data to start at DST idst and the 2 sequences to merge to be densely packed.
 * If unfused, order of operands in DST is [values0, indices0, values1, indices1].
 * For K=512 the operands are in the top half of a tile, with the bottom half padded with -inf.
 * For K=1024 the operands each require a full tile.
 * For K=2048 the operands each require two full tiles.
 *
 * This is part of a bitonic merge-sort algorithm for finding top-k.
 *
 * Return value: None
 *
 * | Argument   | Description                                                                | Type     | Valid Range |
 * Required |
 * |------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | K          | Number of elements per sorted sequence                                     | uint32_t | 512, 1024, or
 * 2048                                    | True     | | idst       | The index of the tile in DST register buffer to
 * perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | | fused |
 * Whether values + indices are fused as single FP32 datum in DST             | bool     | true, false | False    |
 */
template <std::uint32_t K, bool fused = true>
ALWI void topk_xl_merge(std::uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_merge<K, fused>(idst)));
}

/**
 * Rebuilds bitonic sequences after merge for continued merging.
 *
 * Re-establishes the bitonic property for the merged K elements,
 * allowing subsequent merge operations to combine this sequence
 * with other sorted sequences.
 *
 * If fused is true, the data is fused as (bf16 value | u16 index) in FP32 format.
 * If fused is false, the data is not fused and is stored as FP32 values and UInt32 indices.
 *
 * This is part of a bitonic merge-sort algorithm for finding top-k.
 *
 * Return value: None
 *
 * | Argument   | Description                                                                | Type     | Valid Range |
 * Required |
 * |------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | K          | Number of elements in the bitonic sequence                                 | uint32_t | 512, 1024, or
 * 2048                                    | True     | | idst       | The index of the tile in DST register buffer to
 * perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     | |
 * ascending  | Sort direction for rebuild: true for ascending, false for descending       | bool     | true, false |
 * True     | | fused      | Whether values + indices are fused as single FP32 datum in DST             | bool     |
 * true, false                                           | False    |
 */
template <std::uint32_t K, bool fused = true>
ALWI void topk_xl_rebuild(std::uint32_t idst, bool ascending) {
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_rebuild<K, fused>(idst, ascending)));
}

/**
 * Initialize TopK-XL SFPU state.
 *
 * Must be called once before topk_xl_local_sort/topk_xl_merge/topk_xl_rebuild.
 * Programs all of:
 *   * ADDR_MOD_1..7 for the bitonic load/store strides (incl. the +24 / +40 /
 *     +48 stride-folding slots that the inner loops depend on),
 *   * the math-thread MOP Expander with the merge body template (16-issue
 *     body for fused AND for the default macro-scheduled unfused path;
 *     18 for unfused only when built with -DDISABLE_TOPK_XL_SFPLOADMACRO),
 *   * the SFPU index-tracking config in unfused mode.
 *
 * Because every merge/rebuild/local_sort relies on the ADDR_MOD programming
 * above, callers should call this exactly once at the top of a query — and
 * once again at the fused → unfused mode switch in the extended 256K path.
 * The hot loop must not re-call this per stage.
 */
template <std::uint32_t K, bool fused = true>
ALWI void topk_xl_init() {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_init<K, fused>()));
}

/**
 * Initialize unpack/math state for topk_xl_copy_tile.
 */
ALWI void topk_xl_copy_tile_init(std::uint32_t cbid, std::uint32_t call_line = __builtin_LINE()) {
    // TOPK_LARGE_INDICES ADDITION: the low-level copy wrapper only initializes
    // the TopK XL copy LLKs. This TTNN op enters through the standard compute
    // API, so it must also configure SRCA unpack/math state for the input CB.
    state_configure<Operand::SRCA>(cbid, call_line);
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(cbid)));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(cbid, cbid)));
    UNPACK((llk_unpack_topk_xl_copy_init(cbid)));
    MATH((llk_math_topk_xl_copy_init(cbid)));
}

/**
 * Copies up to K elements from up to two consecutive input tiles into one or two DST tiles.
 *
 * Unpacks input CB tiles starting at in_tile_index_base into the DST register buffer starting at
 * dst_start_tile_index. The number of input/DST tiles touched depends on K:
 *   - K = 512  : 1 input tile -> 1 DST tile (only the top half-tile is populated; remainder is -inf padded)
 *   - K = 1024 : 1 input tile -> 1 DST tile
 *   - K = 2048 : 2 input tiles -> 2 DST tiles
 *
 * num_elements selects a partial unpack in the range 1..K. Lanes beyond the active element count
 * are cleared to negative infinity before unpack so inactive entries sort last.
 *
 * Return value: None
 *
 * | Argument               | Description                                                        | Type     | Valid
 * Range                                           | Required |
 * |------------------------|--------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | K                      | Maximum number of elements                                         | uint32_t | 512, 1024,
 * or 2048                                    | True     | | in_cb_id               | Input circular buffer ID for
 * unpack                                | uint32_t | Must match the CB passed to topk_xl_copy_tile_init    | True     |
 * | dst_start_tile_index   | First DST tile index                                               | uint32_t |
 * dst_start_tile_index+(K>1024?1:0) must fit in DST     | True     | | in_tile_index_base     | First input tile index
 * in the CB                                   | uint32_t | Must be within the CB tile capacity                   | True
 * | | num_elements           | Number of elements to copy (partial-tile unpack)                   | uint32_t | 1 .. K
 * | True     |
 */
template <std::uint32_t K>
ALWI void topk_xl_copy_tile(
    std::uint32_t in_cb_id,
    std::uint32_t dst_start_tile_index,
    std::uint32_t in_tile_index_base,
    std::uint32_t num_elements) {
    constexpr std::uint32_t elements_per_tile = TILE_R_DIM * TILE_C_DIM;
    if constexpr (K <= elements_per_tile) {
        UNPACK((llk_unpack_topk_xl_copy_one_tile_unpack(in_cb_id, in_tile_index_base, num_elements)));
        MATH((llk_math_topk_xl_copy_one_tile_math(in_cb_id, dst_start_tile_index, num_elements)));
    } else {
        const std::uint32_t n1 = num_elements < elements_per_tile ? num_elements : elements_per_tile;
        const std::uint32_t n2 = num_elements > elements_per_tile ? (num_elements - elements_per_tile) : 0;

        UNPACK((llk_unpack_topk_xl_copy_one_tile_unpack(in_cb_id, in_tile_index_base, n1)));
        MATH((llk_math_topk_xl_copy_one_tile_math(in_cb_id, dst_start_tile_index, n1)));
        UNPACK((llk_unpack_topk_xl_copy_one_tile_unpack(in_cb_id, in_tile_index_base + 1, n2)));
        MATH((llk_math_topk_xl_copy_one_tile_math(in_cb_id, dst_start_tile_index + 1, n2)));
    }
    UNPACK(TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0));
}

/**
 * Initializes the state for adding LSB indices to the topk_xl_copy_tile output.
 */
ALWI void topk_xl_add_lsb_indices_init() { MATH((llk_math_eltwise_unary_sfpu_topk_xl_add_lsb_indices_init())); }

/**
 * Adds LSB indices to the topk_xl_copy_tile output.

 * | Argument   | Description                                                                | Type     | Valid Range |
 Required |
 * |------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | K          | Number of elements in the bitonic sequence                                 | uint32_t | 512, 1024, or
 2048                                    | True     |
 * | core_id    | The ID of the core that forms the upper 5 bits of the index                | uint32_t | 0 .. 31 | True
 |
 * | idst       | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less
 than the size of the DST register buffer | True     |
 */
template <std::uint32_t K, std::uint32_t core_id, bool row_major = false>
ALWI void topk_xl_add_lsb_indices(std::uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_add_lsb_indices<K, core_id, row_major>(idst)));
}

/**
 * Reprogram only the MOP Expander after topk_xl_copy_tile_init, instead of a
 * full topk_xl_init. copy init clobbers ADDR_MOD_0/3 and the MOP; the fused
 * merge/rebuild kernels use ADDR_MOD_1/5/6/7, so for fused = true only the MOP
 * needs reinstalling — saving the CFG/ADDR_MOD writes a full init would issue
 * per merge stage on the recv path.
 */
template <bool fused = true>
ALWI void topk_xl_reinit_mop_after_copy() {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_reinit_mop_after_copy<fused>()));
}

/**
 * Restore the subset of unfused TopK state clobbered by copy_tile_init.
 *
 * In addition to the shared MOP Expander, unfused rebuild consumes ADDR_MOD_3,
 * which copy init rewrites, and ADDR_MOD_2, which is (re)established for the
 * unfused stride in case the preceding phase was fused. The remaining TopK
 * ADDR_MODs and SFPU index-tracking state stay live.
 *
 * NOT restored: ADDR_MOD_4. copy init does not touch it, but
 * topk_xl_add_lsb_indices_init reprograms it to +16 while unfused rebuild
 * needs +8 — so if add_lsb_indices_init has run since the last full
 * topk_xl_init<fused=false>, run that full init instead of this helper.
 * (tt-blaze's callers always follow add_lsb with a full init, so this
 * sequence does not arise there.)
 */
ALWI void topk_xl_reinit_unfused_rebuild_after_copy() {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_reinit_unfused_rebuild_after_copy()));
}

/**
 * Runtime-chunk-id variant of topk_xl_add_lsb_indices: the 5-bit id in index
 * bits [15:11] is a runtime argument, so one instantiation stamps every chunk
 * of a fused end-to-end row (chunk_id in 0..31). Same init.
 */
template <std::uint32_t K>
ALWI void topk_xl_add_lsb_indices_rt(std::uint32_t idst, std::uint32_t chunk_id) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_add_lsb_indices_rt<K>(idst, chunk_id)));
}

/**
 * Initializes the state for removing 16 MSB values from the topk_xl data.
 *
 * Programs ADDR_MOD_0 with the +2 stride of one 32-lane group. Runs on PACK
 * (TRISC2) because in the extended 256K path the value half is packed out
 * first, the hi16 half is then overwritten with zero, and the indices are
 * packed out second.
 *
 * The kernel static_asserts DstSync::SyncFull as MATH and PACK would contend
 * on LRegs otherwise.
 */
ALWI void topk_xl_remove_msb_values_init() { PACK((llk_math_eltwise_unary_sfpu_topk_xl_remove_msb_values_init())); }

/**
 * Removes MSB values from the topk_xl data, leaving only the indices.
 *
 * This function strips the upper 16 bits (values) from the fused
 * (bf16 value | u16 index) format in FP32, leaving only the
 * indices in the lower 16 bits.
 *
 * Requires the kernel to be built with DST_SYNC_MODE == DstSync::SyncFull.
 *
 * | Argument   | Description                                                                | Type     | Valid Range |
 * Required |
 * |------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | K          | Number of elements in the bitonic sequence                                 | uint32_t | 512, 1024, or
 * 2048                                    | True     | | idst       | The index of the tile in DST register buffer to
 * perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
template <std::uint32_t K>
ALWI void topk_xl_remove_msb_values(std::uint32_t idst) {
    PACK((llk_math_eltwise_unary_sfpu_topk_xl_remove_msb_values<K, DST_SYNC_MODE>(idst)));
}

/**
 * Initializes the state for separating indices from fused topk_xl data.
 *
 * This prepares the SFPU for extracting the 16-bit indices from the fused
 * (bf16 value | u16 index) format into a separate location for non-fused mode.
 *
 * The group_id_bit_shift parameter controls at which bit position the group_id
 * gets placed in the resulting indices during topk_xl_separate_indices.
 *
 * | Argument             | Description                                                                | Type     |
 * Valid Range                                           | Required |
 * |----------------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | group_id_bit_shift   | Bit position at which group_id is placed in the indices                    | uint32_t | 0 ..
 * 31                                               | True     |
 */
ALWI void topk_xl_separate_indices_init(std::uint32_t group_id_bit_shift) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_init(group_id_bit_shift)));
}

// TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split compute API.
// These entry points are used by this TTNN op to convert the fused low-16 tile
// coordinate into public row-major UINT32 indices while preserving the existing
// unfused TopK XL merge/rebuild data layout.
/**
 * Initializes state for separating fused topk_xl data into row-major UINT32
 * indices. The chunk_base is ORed into each decoded within-chunk position and
 * must be aligned to K.
 */
ALWI void topk_xl_separate_indices_row_major_init(std::uint32_t chunk_base) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init(chunk_base)));
}

template <std::uint32_t chunk_base_upper16>
ALWI void topk_xl_separate_indices_row_major_init_upper(std::uint32_t chunk_base_low16) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init_upper<chunk_base_upper16>(
        chunk_base_low16)));
}

template <std::uint32_t chunk_base_upper16, std::uint32_t chunk_base_lower16>
ALWI void topk_xl_separate_indices_row_major_init_static() {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init_static<
          chunk_base_upper16,
          chunk_base_lower16>()));
}

ALWI void topk_xl_separate_indices_row_major_reinit() {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_reinit()));
}

/**
 * Separates the indices from the fused topk_xl data.
 *
 * This function extracts the 16-bit indices from the fused
 * (bf16 value | u16 index) format in FP32 and stores them
 * separately for non-fused mode operation. The group_id is
 * placed at the bit position configured by group_id_bit_shift
 * in the preceding topk_xl_separate_indices_init call.
 *
 * | Argument   | Description                                                                | Type     | Valid Range |
 * Required |
 * |------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | K          | Number of elements in the bitonic sequence                                 | uint32_t | 512, 1024, or
 * 2048                                    | True     | | group_id   | The group ID encoded into the indices at the
 * configured bit position       | uint32_t | 0 .. 2^16-1                                           | True     | | idst
 * | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size
 * of the DST register buffer | True     |
 */
template <std::uint32_t K, std::uint32_t group_id>
ALWI void topk_xl_separate_indices(std::uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices<K, group_id>(idst)));
}

/**
 * Separates fused topk_xl data into unfused values and true row-major UINT32
 * indices. The fused low-16 tile coordinate is decoded to a row-major
 * within-chunk position, then ORed with the chunk base configured by
 * topk_xl_separate_indices_row_major_init.
 */
template <std::uint32_t K>
ALWI void topk_xl_separate_indices_row_major(std::uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major<K>(idst)));
}

// Loads the chunk-field mask constants for the global splits below; call
// once before topk_xl_separate_indices_row_major_global / _global_base.
ALWI void topk_xl_separate_indices_row_major_global_init() {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_global_init()));
}

/**
 * Fused end-to-end split: runs ONCE per row on the final fused survivor.
 * Each u16 payload carries [chunk_id 15:11 | within-chunk 10:0]; the global
 * index chunk_id * K + row-major(within-chunk) is recovered from the stamp
 * itself — no chunk-base bookkeeping. Requires
 * topk_xl_separate_indices_row_major_global_init. Sound only for rows of
 * <= 32 chunks (the FUSED_E2E factory gate).
 */
template <std::uint32_t K>
ALWI void topk_xl_separate_indices_row_major_global(std::uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_global<K>(idst)));
}

// Segmented fusion: split one fused segment survivor in place, adding
// seg_base (= segment_index * 32 * K, power-of-two aligned) to every decoded
// index. Same init as the plain global split.
template <std::uint32_t K>
ALWI void topk_xl_separate_indices_row_major_global_base(std::uint32_t idst, std::uint32_t seg_base) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_global_base<K>(idst, seg_base)));
}

template <std::uint32_t K>
ALWI void topk_xl_separate_indices_row_major_advance_chunk_base() {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_advance_chunk_base<K>()));
}
// END TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split compute API.

}  // namespace ckernel
