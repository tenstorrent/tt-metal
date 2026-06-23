// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
#include "experimental/llk_math_eltwise_unary_sfpu_topk_xl.h"
#endif
#endif
#ifdef TRISC_PACK
#ifdef ARCH_BLACKHOLE
#include "experimental/llk_math_eltwise_unary_sfpu_topk_xl.h"
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
template <uint32_t K>
ALWI void topk_xl_local_sort(uint32_t idst, bool ascending) {
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_local_sort<K, APPROX>(idst, ascending)));
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
template <uint32_t K, bool fused = true>
ALWI void topk_xl_merge(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_merge<K, APPROX, fused>(idst)));
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
template <uint32_t K, bool fused = true>
ALWI void topk_xl_rebuild(uint32_t idst, bool ascending) {
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_rebuild<K, APPROX, fused>(idst, ascending)));
}

/**
 * Initialize TopK-XL SFPU state.
 *
 * Must be called once before topk_xl_local_sort/topk_xl_merge/topk_xl_rebuild.
 * Programs all of:
 *   * ADDR_MOD_1..7 for the bitonic load/store strides (incl. the +24 / +40 /
 *     +48 stride-folding slots that the inner loops depend on),
 *   * the math-thread MOP Expander with the merge body template (`fused`
 *     selects the body length: 16 for fused, 18 for unfused),
 *   * the SFPU index-tracking config in unfused mode.
 *
 * Because every merge/rebuild/local_sort relies on the ADDR_MOD programming
 * above, callers should call this exactly once at the top of a query — and
 * once again at the fused → unfused mode switch in the extended 256K path.
 * The hot loop must not re-call this per stage.
 */
template <uint32_t K, bool fused = true>
ALWI void topk_xl_init() {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_init<K, APPROX, fused>()));
}

/**
 * Initialize unpack/math state for topk_xl_copy_tile.
 */
ALWI void topk_xl_copy_tile_init(uint32_t cbid, uint32_t call_line = __builtin_LINE()) {
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
template <uint32_t K>
ALWI void topk_xl_copy_tile(
    uint32_t in_cb_id, uint32_t dst_start_tile_index, uint32_t in_tile_index_base, uint32_t num_elements) {
    constexpr uint32_t elements_per_tile = TILE_R_DIM * TILE_C_DIM;
    if constexpr (K <= elements_per_tile) {
        UNPACK((llk_unpack_topk_xl_copy_one_tile_unpack(in_cb_id, in_tile_index_base, num_elements)));
        MATH((llk_math_topk_xl_copy_one_tile_math(in_cb_id, dst_start_tile_index, num_elements)));
    } else {
        const uint32_t n1 = num_elements < elements_per_tile ? num_elements : elements_per_tile;
        const uint32_t n2 = num_elements > elements_per_tile ? (num_elements - elements_per_tile) : 0;

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
ALWI void topk_xl_add_lsb_indices_init() { MATH((llk_math_eltwise_unary_sfpu_topk_xl_add_lsb_indices_init<APPROX>())); }

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
template <uint32_t K, uint32_t core_id>
ALWI void topk_xl_add_lsb_indices(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_add_lsb_indices<K, APPROX, core_id>(idst)));
}

/**
 * Initializes the state for removing 16 MSB values from the topk_xl data.
 *
 * Programs ADDR_MOD_0 with the +2 hi16 / lo16 stride. Runs on PACK (TRISC2)
 * because in the extended 256K path the value half is packed out first, the
 * hi16 half is then overwritten with zero, and the indices are packed out
 * second; owning the overwrite on PACK lets the value-pack and the
 * zero-overwrite overlap with MATH's final merge tail.
 */
ALWI void topk_xl_remove_msb_values_init() {
    PACK((llk_math_eltwise_unary_sfpu_topk_xl_remove_msb_values_init<false>()));
}

/**
 * Removes MSB values from the topk_xl data, leaving only the indices.
 *
 * This function strips the upper 16 bits (values) from the fused
 * (bf16 value | u16 index) format in FP32, leaving only the
 * indices in the lower 16 bits.
 *
 * | Argument   | Description                                                                | Type     | Valid Range |
 * Required |
 * |------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | K          | Number of elements in the bitonic sequence                                 | uint32_t | 512, 1024, or
 * 2048                                    | True     | | idst       | The index of the tile in DST register buffer to
 * perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
template <uint32_t K>
ALWI void topk_xl_remove_msb_values(uint32_t idst) {
    PACK((llk_math_eltwise_unary_sfpu_topk_xl_remove_msb_values<K, false>(idst)));
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
ALWI void topk_xl_separate_indices_init(uint32_t group_id_bit_shift) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_init<false>(group_id_bit_shift)));
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
ALWI void topk_xl_separate_indices_row_major_init(uint32_t chunk_base) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init<false>(chunk_base)));
}

template <uint32_t chunk_base_upper16>
ALWI void topk_xl_separate_indices_row_major_init_upper(uint32_t chunk_base_low16) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init_upper<false, chunk_base_upper16>(
        chunk_base_low16)));
}

template <uint32_t chunk_base_upper16, uint32_t chunk_base_lower16>
ALWI void topk_xl_separate_indices_row_major_init_static() {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init_static<
          false,
          chunk_base_upper16,
          chunk_base_lower16>()));
}

ALWI void topk_xl_separate_indices_row_major_reinit() {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_reinit<false>()));
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
template <uint32_t K, uint32_t group_id>
ALWI void topk_xl_separate_indices(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices<K, false, group_id>(idst)));
}

/**
 * Separates fused topk_xl data into unfused values and true row-major UINT32
 * indices. The fused low-16 tile coordinate is decoded to a row-major
 * within-chunk position, then ORed with the chunk base configured by
 * topk_xl_separate_indices_row_major_init.
 */
template <uint32_t K>
ALWI void topk_xl_separate_indices_row_major(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major<K, false>(idst)));
}

template <uint32_t K>
ALWI void topk_xl_separate_indices_row_major_advance_chunk_base() {
    MATH((llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_advance_chunk_base<K, false>()));
}
// END TOPK_LARGE_INDICES ADDITION: row-major UINT32 index split compute API.

}  // namespace ckernel
