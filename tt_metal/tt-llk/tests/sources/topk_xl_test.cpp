// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPU topk_xl test: bitonic sort / merge / rebuild top-K for K = 512, 1024, 2048.
// Blackhole-only.
//
// This test covers every LLK referenced by the Metal topk_xl headers
// (llk_api/experimental/llk_{unpack_A,math}_topk_xl_copy_api.h and
// llk_math_eltwise_unary_sfpu_topk_xl.h), which wrap the llk_lib entry points:
//   * ckernel::_llk_unpack_topk_xl_copy_init_ / _llk_unpack_topk_xl_copy_
//   * ckernel::_llk_math_topk_xl_copy_init_ / _llk_math_topk_xl_copy_
//   * ckernel::sfpu::_topk_xl_init_ / _topk_xl_local_sort_ / _topk_xl_merge_ /
//     _topk_xl_rebuild_ / _topk_xl_add_lsb_indices_(_init_) /
//     _topk_xl_separate_indices_row_major_(_init_static_ / _reinit_ /
//     _advance_chunk_base_) / _topk_xl_separate_indices_(_init_) /
//     _topk_xl_remove_msb_values_(_init_)
//
// The shared core is copy_sort (copy -> add_lsb_indices -> init(fused) ->
// local_sort); TOPK_XL_INDEX_OP picks the terminal index step. Op 0 mirrors the
// topk_large_indices compute kernel
// (ttnn/.../topk_large_indices/device/kernels/compute.cpp); ops 1 and 2 cover the
// two LLKs that the op does not use.
//
// Fused word: bf16 value (bits 31:16) | u16 index (bits 15:0). Dest is fp32,
// full sync; K=2048 fills tiles 0-7.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Shared compile-time derived constants:
//   TOPK_XL_K:             512 | 1024 | 2048
//   TOPK_XL_NUM_CHUNKS:    number of K-element windows per row (row-major only)
//   TOPK_XL_TAIL_ELEMENTS: valid element count of the last chunk (1 .. K)
//   TOPK_XL_NUM_ROWS:      number of independent top-K problems
//   TOPK_XL_INDEX_OP:      terminal index step:
//     0 row-major   separate_indices_row_major per chunk, then merge/rebuild the
//                   chunks into slot0 -> value region + row-major u32 index region
//     1 separate    generic separate_indices<group_id> -> value region [value|0]
//                   plus index region [group_id<<shift | raw], raw being the
//                   add_lsb tile coordinate, undecoded. Single chunk.
//     2 remove_msb  zero the bf16 value half in place -> [0|raw], the fused region
//                   only. Issued from PACK, where the compute API puts it. Single
//                   chunk.
//   TOPK_XL_GROUP_ID / TOPK_XL_GROUP_SHIFT : generic separate_indices params
//   TOPK_XL_CORE_ID:         add_lsb_indices core_id, index bits [15:11] (0 .. 31)
//   TOPK_XL_ASCENDING:       rebuild direction (false descending, true ascending)
//   TOPK_XL_FUSED_REDUCE:    false unfused merge/rebuild (op path), true fused
//   TOPK_XL_CHUNK_BASE_MODE: three ways to init chunk_base:
//                            0 init_static<hi,lo> | 1 init_upper<hi>(lo) | 2 init(runtime)
//   TOPK_XL_CHUNK_BASE:      starting chunk_base (must be a multiple of K)

constexpr std::uint32_t ELEMENTS_PER_TILE = ckernel::TILE_R_DIM * ckernel::TILE_C_DIM;
constexpr std::uint32_t TILES_PER_SEQ     = (TOPK_XL_K + ELEMENTS_PER_TILE - 1) / ELEMENTS_PER_TILE;
constexpr std::uint32_t SLOT0             = 0;

constexpr bool INDEX_OP_ROW_MAJOR  = (TOPK_XL_INDEX_OP == 0);
constexpr bool INDEX_OP_SEPARATE   = (TOPK_XL_INDEX_OP == 1);
constexpr bool INDEX_OP_REMOVE_MSB = (TOPK_XL_INDEX_OP == 2);

constexpr bool FUSED_REDUCE = TOPK_XL_FUSED_REDUCE;
// Fused END-TO-END (the topk_large_indices wide-row family): runtime chunk-id
// stamp in index bits [15:11], fused merge/rebuild, one row-major GLOBAL split
// at the end (plain, or with a segment base for segmented fusion).
constexpr bool FUSED_E2E = TOPK_XL_FUSED_E2E;

// Second merge operand. `_topk_xl_merge_` reads it at a fixed distance from the
// first: 64 dest units (one tile) per sequence-tile when fused, 128 (value +
// index region) when unfused, so the slot stride follows the mode.
constexpr std::uint32_t SLOT1 = (FUSED_REDUCE || FUSED_E2E) ? TILES_PER_SEQ : (2 * TILES_PER_SEQ);

// A lone chunk has nothing to merge with, but the row-major path still rebuilds it;
// the fused path merges/rebuilds only when there is a second operand.
// Both TRISCs use this: MATH issues the rebuild, UNPACK the SrcB dummy valid feeding it.
// Fused variants leave TOPK_XL_INDEX_OP at its 0 default and ignore it, hence the !FUSED_REDUCE.
constexpr bool REBUILD_LONE_CHUNK = !FUSED_REDUCE && !FUSED_E2E && INDEX_OP_ROW_MAJOR && TOPK_XL_NUM_CHUNKS == 1;

constexpr std::uint32_t CHUNK_BASE_HI16 = (TOPK_XL_CHUNK_BASE >> 16) & 0xFFFF;
constexpr std::uint32_t CHUNK_BASE_LO16 = TOPK_XL_CHUNK_BASE & 0xFFFF;

// Active element count of chunk `c` (last chunk is the tail).
inline constexpr std::uint32_t chunk_active_elements(std::uint32_t c)
{
    return (c == TOPK_XL_NUM_CHUNKS - 1) ? TOPK_XL_TAIL_ELEMENTS : TOPK_XL_K;
}

inline constexpr std::uint32_t tile_active_elements(std::uint32_t active, std::uint32_t t)
{
    return (t == 0) ? ((active < ELEMENTS_PER_TILE) ? active : ELEMENTS_PER_TILE) : ((active > ELEMENTS_PER_TILE) ? (active - ELEMENTS_PER_TILE) : 0);
}

// Global index of input tile `t` of chunk `c` in row `r`.
inline constexpr std::uint32_t input_tile_index(std::uint32_t r, std::uint32_t c, std::uint32_t t)
{
    return ((r * TOPK_XL_NUM_CHUNKS + c) * TILES_PER_SEQ) + t;
}

#ifdef LLK_TRISC_UNPACK

#include "ckernel_template.h" // ckernel_template used by the topk_xl copy MOP below
#include "experimental/llk_unpack_A_topk_xl_copy.h"
#include "llk_unpack_common.h"

// Replicates llk_unpack_topk_xl_copy_one_tile_unpack() from the metal API wrapper:
// program the partial-tile element count then run the TopK-XL copy MOP for one tile.
inline void unpack_copy_one_tile(std::uint32_t l1_tile_address, std::uint32_t src_format, std::uint32_t dst_format, std::uint32_t elements_this_tile)
{
    const std::uint32_t adc_count = (elements_this_tile == 0) ? (ELEMENTS_PER_TILE - 1) : (elements_this_tile - 1);
    TT_SETADCXX(p_setadc::UNP_A, adc_count, 0x0);
    ckernel::_llk_unpack_topk_xl_copy_(l1_tile_address, src_format, dst_format, elements_this_tile);
}

// Replicates topk_xl_copy_tile<K>() unpack half: 1 tile for K<=1024, 2 tiles for K=2048.
// Marked noinline to ensure K=2048 stays under the TRISC code budget. Emitting one body
// per call site becomes problematic because of loop unrolling.
__attribute__((noinline)) void unpack_copy_tile(RUNTIME_PARAMETERS params, std::uint32_t r, std::uint32_t c, std::uint32_t src_format, std::uint32_t dst_format)
{
    const std::uint32_t active = chunk_active_elements(c);
    for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
    {
        unpack_copy_one_tile(L1_ADDRESS(params.buffer_A[input_tile_index(r, c, t)]), src_format, dst_format, tile_active_elements(active, t));
    }
    // Restore the unpacker element count to a full face row (mirrors the trailing
    // TTI_SETADCXX in topk_xl_copy_tile()).
    TTI_SETADCXX(p_setadc::UNP_A, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t src_format = formats.unpack_A_src;
    const std::uint32_t dst_format = formats.unpack_A_dst;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        src_format, src_format, dst_format, dst_format, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES /* unpA_num_faces */, TILE_NUM_FACES /* unpB_num_faces */);
    ckernel::_llk_unpack_topk_xl_copy_init_(src_format, dst_format);
    for (std::uint32_t r = 0; r < TOPK_XL_NUM_ROWS; r++)
    {
        for (std::uint32_t c = 0; c < TOPK_XL_NUM_CHUNKS; c++)
        {
            unpack_copy_tile(params, r, c, src_format, dst_format); // chunk 0 -> slot0, the rest -> slot1
            _llk_unpack_set_srcb_dummy_valid_();                    // local_sort
            if (c > 0)
            {
                _llk_unpack_set_srcb_dummy_valid_(); // rebuild(slot0)
            }
        }
        if constexpr (REBUILD_LONE_CHUNK)
        {
            _llk_unpack_set_srcb_dummy_valid_();
        }
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

// TRISC1 code region overflows by well over 4K under the default -O3.
#pragma GCC optimize("O2")

#include "experimental/llk_math_eltwise_unary_datacopy_topk_xl_copy.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

using namespace ckernel;

template <std::uint32_t K, bool fused>
inline void topk_xl_init()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_init_<K, fused>();
}

template <std::uint32_t K>
inline void topk_xl_local_sort(std::uint32_t dst_index, bool ascending)
{
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_local_sort_<K>, dst_index, VectorMode::RC_custom, dst_index, ascending);
}

template <std::uint32_t K, bool fused>
inline void topk_xl_merge(std::uint32_t dst_index)
{
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_merge_<K, fused>, dst_index, VectorMode::RC_custom, dst_index);
}

template <std::uint32_t K, bool fused>
inline void topk_xl_rebuild(std::uint32_t dst_index, bool ascending)
{
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_rebuild_<K, fused>, dst_index, VectorMode::RC_custom, dst_index, ascending);
}

inline void topk_xl_add_lsb_indices_init()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_add_lsb_indices_init_();
}

template <std::uint32_t K, std::uint32_t core_id>
inline void topk_xl_add_lsb_indices(std::uint32_t dst_index)
{
    static_assert(core_id < 32, "core_id occupies index bits [15:11]");
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_add_lsb_indices_<K, core_id>, dst_index, VectorMode::RC_custom);
}

// --- Row-major index split (topk_large_indices op path) ---
template <std::uint32_t chunk_base_upper16, std::uint32_t chunk_base_lower16>
inline void topk_xl_separate_indices_row_major_init_static()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_separate_indices_row_major_init_static_<chunk_base_upper16, chunk_base_lower16>();
}

// Same chunk_base latch, runtime value. The flavor a caller uses when the base
// is only known at runtime.
inline void topk_xl_separate_indices_row_major_init(std::uint32_t chunk_base)
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_separate_indices_row_major_init_(chunk_base);
}

// Hybrid flavor: high half static, low half runtime.
template <std::uint32_t chunk_base_upper16>
inline void topk_xl_separate_indices_row_major_init_upper(std::uint32_t chunk_base_low16)
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_separate_indices_row_major_init_upper_<chunk_base_upper16>(chunk_base_low16);
}

inline void topk_xl_separate_indices_row_major_reinit()
{
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    ckernel::sfpu::_topk_xl_separate_indices_row_major_reinit_();
}

template <std::uint32_t K>
inline void topk_xl_separate_indices_row_major(std::uint32_t dst_index)
{
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_separate_indices_row_major_<K>, dst_index, VectorMode::RC_custom);
}

template <std::uint32_t K>
inline void topk_xl_separate_indices_row_major_advance_chunk_base()
{
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    ckernel::sfpu::_topk_xl_separate_indices_row_major_advance_chunk_base_<K>();
}

// Generic separate_indices (keeps the tile coordinate and prepends group_id).
inline void topk_xl_separate_indices_init(std::uint32_t group_id_bit_shift)
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>();
    ckernel::sfpu::_topk_xl_separate_indices_init_(group_id_bit_shift);
}

template <std::uint32_t K, std::uint32_t group_id>
inline void topk_xl_separate_indices(std::uint32_t dst_index)
{
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_separate_indices_<K, group_id>, dst_index, VectorMode::RC_custom);
}

// Shared core: copy the chunk into `slot`, stamp indices, fused local-sort.
// Marked noinline to avoid overflowing the code region.
template <std::uint32_t K>
__attribute__((noinline)) void copy_sort(std::uint32_t slot, std::uint32_t active_elements, bool ascending, std::uint32_t dst_format)
{
    ckernel::_llk_math_topk_xl_copy_init_(dst_format);
    for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
    {
        ckernel::_llk_math_topk_xl_copy_(slot + t, dst_format, tile_active_elements(active_elements, t));
    }

    topk_xl_add_lsb_indices_init();
    topk_xl_add_lsb_indices<K, TOPK_XL_CORE_ID>(slot);

    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(slot, ascending);
}

// Fused-e2e variant of copy_sort: the chunk id is stamped at RUNTIME into
// index bits [15:11] (the topk_large_indices compute kernels' path).
template <std::uint32_t K>
__attribute__((noinline)) void copy_sort_rt(std::uint32_t slot, std::uint32_t active_elements, bool ascending, std::uint32_t chunk_id, std::uint32_t dst_format)
{
    ckernel::_llk_math_topk_xl_copy_init_(dst_format);
    for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
    {
        ckernel::_llk_math_topk_xl_copy_(slot + t, dst_format, tile_active_elements(active_elements, t));
    }

    topk_xl_add_lsb_indices_init();
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_add_lsb_indices_rt_<K>, slot, VectorMode::RC_custom, chunk_id);

    topk_xl_init<K, true>();
    topk_xl_local_sort<K>(slot, ascending);
}

inline void topk_xl_separate_indices_row_major_global_init()
{
    ckernel::sfpu::_topk_xl_separate_indices_row_major_global_init_();
}

template <std::uint32_t K>
inline void topk_xl_separate_indices_row_major_global(std::uint32_t dst_index)
{
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_separate_indices_row_major_global_<K>, dst_index, VectorMode::RC_custom);
}

template <std::uint32_t K>
inline void topk_xl_separate_indices_row_major_global_base(std::uint32_t dst_index, std::uint32_t seg_base)
{
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::_topk_xl_separate_indices_row_major_global_base_<K>, dst_index, VectorMode::RC_custom, seg_base);
}

// Row-major process_chunk: copy_sort then split into unfused values + row-major
// uint32 indices, ready for the merge tree.
template <std::uint32_t K>
__attribute__((noinline)) void process_chunk_math(std::uint32_t slot, std::uint32_t active_elements, bool ascending, std::uint32_t dst_format)
{
    copy_sort<K>(slot, active_elements, ascending, dst_format);
    topk_xl_separate_indices_row_major_reinit();
    topk_xl_separate_indices_row_major<K>(slot);
    topk_xl_separate_indices_row_major_advance_chunk_base<K>();
}

// Init + (optional) merge of slot1 into slot0 + rebuild, in either fused mode.
// `fused` selects the whole merge/rebuild code family: operand distance, MOP body
// length and iteration count all differ (see topk_mop_config / _topk_xl_merge_).
// `_topk_xl_merge_` always keeps the max half, so TOPK_XL_ASCENDING changes only
// the order the surviving top-K is rebuilt into, not which elements survive.
template <std::uint32_t K, bool fused>
__attribute__((noinline)) void merge_and_rebuild(bool do_merge)
{
    topk_xl_init<K, fused>();
    if (do_merge)
    {
        topk_xl_merge<K, fused>(SLOT0);
    }
    topk_xl_rebuild<K, fused>(SLOT0, TOPK_XL_ASCENDING);
}

// Save the starting chunk_base through the requested init flavor.
// All three save into LREG12, but the value is split differently.
inline void topk_xl_chunk_base_init()
{
    if constexpr (TOPK_XL_CHUNK_BASE_MODE == 0)
    {
        topk_xl_separate_indices_row_major_init_static<CHUNK_BASE_HI16, CHUNK_BASE_LO16>();
    }
    else if constexpr (TOPK_XL_CHUNK_BASE_MODE == 1)
    {
        topk_xl_separate_indices_row_major_init_upper<CHUNK_BASE_HI16>(CHUNK_BASE_LO16);
    }
    else
    {
        topk_xl_separate_indices_row_major_init(TOPK_XL_CHUNK_BASE);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t math_format = formats.math;

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);

    for (std::uint32_t r = 0; r < TOPK_XL_NUM_ROWS; r++)
    {
        _llk_math_wait_for_dest_available_<dest_sync>();

        if constexpr (FUSED_E2E)
        {
            // Fused end-to-end: runtime chunk-id stamps, fused merge/rebuild,
            // then ONE global split -- plain or with a segment base.
            copy_sort_rt<TOPK_XL_K>(SLOT0, chunk_active_elements(0), false /* ascending */, 0, math_format);
            for (std::uint32_t c = 1; c < TOPK_XL_NUM_CHUNKS; c++)
            {
                copy_sort_rt<TOPK_XL_K>(SLOT1, chunk_active_elements(c), true /* ascending */, c, math_format);
                merge_and_rebuild<TOPK_XL_K, true /* fused */>(true /* do_merge */);
            }
            topk_xl_separate_indices_row_major_global_init();
            if constexpr (TOPK_XL_SEG_BASE != 0)
            {
                topk_xl_separate_indices_row_major_global_base<TOPK_XL_K>(SLOT0, TOPK_XL_SEG_BASE);
            }
            else
            {
                topk_xl_separate_indices_row_major_global<TOPK_XL_K>(SLOT0);
            }
        }
        else if constexpr (FUSED_REDUCE)
        {
            // Fused reduction: chunks stay in the fused [value|index] form all the
            // way through merge/rebuild, and the index split happens once at the end.
            copy_sort<TOPK_XL_K>(SLOT0, chunk_active_elements(0), false /* ascending */, math_format);

            for (std::uint32_t c = 1; c < TOPK_XL_NUM_CHUNKS; c++)
            {
                copy_sort<TOPK_XL_K>(SLOT1, chunk_active_elements(c), true /* ascending */, math_format);
                merge_and_rebuild<TOPK_XL_K, true /* fused */>(true /* do_merge */);
            }

            topk_xl_separate_indices_init(TOPK_XL_GROUP_SHIFT);
            topk_xl_separate_indices<TOPK_XL_K, TOPK_XL_GROUP_ID>(SLOT0);
        }
        else if constexpr (INDEX_OP_ROW_MAJOR)
        {
            topk_xl_chunk_base_init();

            // chunk 0 -> slot0, local-sort descending.
            process_chunk_math<TOPK_XL_K>(SLOT0, chunk_active_elements(0), false /* ascending */, math_format);

            for (std::uint32_t c = 1; c < TOPK_XL_NUM_CHUNKS; c++)
            {
                // chunk c -> slot1, local-sort ascending, then merge into slot0.
                process_chunk_math<TOPK_XL_K>(SLOT1, chunk_active_elements(c), true /* ascending */, math_format);
                merge_and_rebuild<TOPK_XL_K, false /* fused */>(true /* do_merge */);
            }
            if constexpr (REBUILD_LONE_CHUNK)
            {
                merge_and_rebuild<TOPK_XL_K, false /* fused */>(false /* do_merge */);
            }
        }
        else
        {
            // Single-chunk terminal ops: copy_sort, then the index step. For separate
            // the split is here (MATH). For remove_msb the value-half zero runs on
            // PACK (as the op does), so MATH just leaves the fused [value|index] in slot0.
            static_assert(INDEX_OP_ROW_MAJOR || TOPK_XL_NUM_CHUNKS == 1, "terminal index ops (INDEX_OP 1/2) are single-chunk only");
            copy_sort<TOPK_XL_K>(SLOT0, chunk_active_elements(0), false /* fused */, math_format);

            if constexpr (INDEX_OP_SEPARATE)
            {
                topk_xl_separate_indices_init(TOPK_XL_GROUP_SHIFT);
                topk_xl_separate_indices<TOPK_XL_K, TOPK_XL_GROUP_ID>(SLOT0);
            }
        }

        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "sfpu/experimental/ckernel_sfpu_topk_xl.h"

// remove_msb_values on PACK: verbatim reproduction of the Metal wrapper
// llk_math_eltwise_unary_sfpu_topk_xl_remove_msb_values, which Compute API invokes through
// PACK(...). Zeros the bf16 value half of the fused Dest words in place: [0 | index].
template <std::uint32_t K>
inline void pack_remove_msb_values(std::uint32_t dst_index)
{
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index + get_dest_buffer_base());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH | p_stall::PACK);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);

    // The SFPU drain on the way in and the pack drain on the way out both live
    // inside `_topk_xl_remove_msb_values_`. The LLK static_asserts SyncFull.
    ckernel::sfpu::_topk_xl_remove_msb_values_<K, dest_sync>();
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t pack_src_format = formats.pack_src;
    const std::uint32_t pack_dst_format = formats.pack_dst; // UInt32: raw 32-bit words.

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        pack_src_format, pack_dst_format, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(pack_dst_format, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();

    if constexpr (INDEX_OP_REMOVE_MSB)
    {
        ckernel::sfpu::_topk_xl_remove_msb_values_init_();
    }

    // remove_msb: the in-place fused region [0|index] (TILES_PER_SEQ). Otherwise the
    // value region then the index region (2*TILES_PER_SEQ).
    constexpr std::uint32_t RESULT_TILES_PER_ROW = INDEX_OP_REMOVE_MSB ? TILES_PER_SEQ : (2 * TILES_PER_SEQ);

    for (std::uint32_t r = 0; r < TOPK_XL_NUM_ROWS; r++)
    {
        _llk_packer_wait_for_math_done_();

        if constexpr (INDEX_OP_REMOVE_MSB)
        {
            // Zero the value half on PACK, then pack the fused region as [0|index].
            // Includes the trailing SFPU drain before the pack below.
            pack_remove_msb_values<TOPK_XL_K>(SLOT0);
        }

        std::uint32_t res = r * RESULT_TILES_PER_ROW;
        // Value / fused region of slot0: Dest tiles [SLOT0 .. SLOT0 + TILES_PER_SEQ - 1].
        for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
        {
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(SLOT0 + t, L1_ADDRESS(params.buffer_Res[res++]));
        }
        if constexpr (!INDEX_OP_REMOVE_MSB)
        {
            // Index region of slot0: Dest tiles [SLOT0 + TILES_PER_SEQ .. +2*TILES_PER_SEQ - 1].
            for (std::uint32_t t = 0; t < TILES_PER_SEQ; t++)
            {
                _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(SLOT0 + TILES_PER_SEQ + t, L1_ADDRESS(params.buffer_Res[res++]));
            }
        }

        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif // LLK_TRISC_PACK
