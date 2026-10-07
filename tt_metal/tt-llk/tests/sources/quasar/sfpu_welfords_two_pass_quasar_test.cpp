// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

using namespace ckernel;
#include "params.h" // TWO_PASS_*, IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en

// Two-pass statistics. STREAM sends the tiles through Dest once per pass, one block per section, so the
// state crosses section handoffs; COMBINE and SWITCH run in-place blocks within one section.

constexpr std::uint32_t two_pass_tiles_per_block(std::uint32_t tile_cnt)
{
    return (TWO_PASS_MODE == 0 && TWO_PASS_TILES_PER_BLOCK > 0) ? TWO_PASS_TILES_PER_BLOCK : tile_cnt;
}

constexpr std::uint32_t TWO_PASS_SWEEPS = TWO_PASS_MODE == 0 ? 2 : 1;

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();

    const std::uint32_t tiles_per_block = two_pass_tiles_per_block(params.TILE_CNT);
    const std::uint32_t num_blocks      = params.TILE_CNT / tiles_per_block;

    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src);

    _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(static_cast<DataFormat>(formats.unpack_A_dst));
    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(
        ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), ckernel::DEFAULT_TENSOR_SHAPE, tiles_per_block);
    for (std::uint32_t sweep = 0; sweep < TWO_PASS_SWEEPS; ++sweep)
    {
        for (std::uint32_t block = 0; block < num_blocks; ++block)
        {
            _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(block * tiles_per_block /*l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
            _llk_unpack_dest_dvalid_section_done_<dest_sync>();
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_sfpu/ckernel_sfpu_welfords.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

namespace
{

std::uint32_t fp32_reciprocal_bits(std::uint32_t n)
{
    return __builtin_bit_cast(std::uint32_t, 1.0f / static_cast<float>(n));
}

std::uint32_t fp32_bits(std::uint32_t n)
{
    return __builtin_bit_cast(std::uint32_t, static_cast<float>(n));
}

struct RowWindow
{
    std::uint32_t start;
    std::uint32_t num;
};

RowWindow tile_window(std::uint32_t tile, std::uint32_t last_tile)
{
    if (TWO_PASS_PARTIAL_LAST_TILE && tile == last_tile)
    {
        return {TWO_PASS_START_ROW, TWO_PASS_NUM_ROWS};
    }
    return {0, TILE_R_DIM};
}

template <bool INITIALIZE_ANCHOR>
void pass_one(std::uint32_t dst_index, RowWindow w)
{
    SFPU_UNARY_CALL(
        dest_sync,
        is_fp32_dest_acc_en,
        _two_pass_update_shifted_rows_,
        (false /* accumulate_m2 */, INITIALIZE_ANCHOR, TWO_PASS_DUAL),
        dst_index,
        VectorMode::RC_custom,
        w.start,
        w.num);
}

void pass_two(std::uint32_t dst_index, RowWindow w)
{
    SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, _two_pass_update_rows_, (TWO_PASS_DUAL), dst_index, VectorMode::RC_custom, w.start, w.num);
}

// Both passes over tiles [first_tile, first_tile + count) already in Dest.
std::uint32_t two_pass_in_place(std::uint32_t dst_base, std::uint32_t first_tile, std::uint32_t count, std::uint32_t last_tile)
{
    std::uint32_t rows = 0;
    for (std::uint32_t t = first_tile; t < first_tile + count; ++t)
    {
        const RowWindow w = tile_window(t, last_tile);
        if (t == first_tile)
        {
            pass_one<true>(dst_base + t, w);
        }
        else
        {
            pass_one<false>(dst_base + t, w);
        }
        rows += w.num;
    }
    _two_pass_finish_shifted_mean_<TWO_PASS_DUAL>(fp32_reciprocal_bits(rows));
    for (std::uint32_t t = first_tile; t < first_tile + count; ++t)
    {
        pass_two(dst_base + t, tile_window(t, last_tile));
    }
    return rows;
}

void finalize(std::uint32_t dst_index, std::uint32_t rows)
{
    const std::uint32_t recip = fp32_reciprocal_bits(rows);
    if constexpr (TWO_PASS_FINALIZE == 0)
    {
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            _two_pass_store_mean_var_to_dst_row_,
            (TWO_PASS_DUAL, true /* store_mean */),
            dst_index,
            VectorMode::RC_custom,
            recip);
    }
    else if constexpr (TWO_PASS_FINALIZE == 1)
    {
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            _two_pass_store_mean_var_to_dst_raw_group_,
            (TWO_PASS_DUAL),
            dst_index,
            VectorMode::RC_custom,
            TWO_PASS_GROUP_A,
            recip);
    }
    else if constexpr (TWO_PASS_FINALIZE == 2)
    {
        SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, _two_pass_store_split_mean_var_to_dst_row_, (TWO_PASS_DUAL), dst_index, VectorMode::RC_custom, recip);
    }
    else if constexpr (TWO_PASS_FINALIZE == 3)
    {
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            _two_pass_store_combined_mean_var_to_dst_raw_group_,
            (TWO_PASS_DUAL, TWO_PASS_AVERAGE_VARIANCE),
            dst_index,
            VectorMode::RC_custom,
            TWO_PASS_GROUP_A,
            recip);
    }
    else
    {
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            _two_pass_store_mean_var_to_dst_row_,
            (TWO_PASS_DUAL, false /* store_mean */),
            dst_index,
            VectorMode::RC_custom,
            recip);
    }
}

// LREG7 is cleared between each store and load, so the split output proves both round trips.
void anchor_round_trip(std::uint32_t dst_index)
{
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(dest_sync, is_fp32_dest_acc_en, _two_pass_store_anchor_to_dst_, dst_index, VectorMode::RC_custom);
    _two_pass_zero_<p_sfpu::LREG7>();
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(dest_sync, is_fp32_dest_acc_en, _two_pass_load_anchor_from_dst_, dst_index, VectorMode::RC_custom);
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(dest_sync, is_fp32_dest_acc_en, _two_pass_store_anchor_to_state_dst_, dst_index + 1, VectorMode::RC_custom);
    _two_pass_zero_<p_sfpu::LREG7>();
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(dest_sync, is_fp32_dest_acc_en, _two_pass_load_anchor_from_state_dst_, dst_index + 1, VectorMode::RC_custom);
}

} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::SFPU>();

    const DataFormat math_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(math_format, math_format);

    _llk_math_eltwise_sfpu_init_();
    _two_pass_clear_stats_();

    const std::uint32_t tiles_per_block = two_pass_tiles_per_block(params.TILE_CNT);
    const std::uint32_t num_blocks      = params.TILE_CNT / tiles_per_block;
    const std::uint32_t last_tile       = params.TILE_CNT - 1;
    const std::uint32_t dst             = params.DST_INDEX;

    if constexpr (TWO_PASS_MODE == 0)
    {
        std::uint32_t rows = 0;
        for (std::uint32_t tile = 0; tile < params.TILE_CNT; ++tile)
        {
            rows += tile_window(tile, last_tile).num;
        }

        for (std::uint32_t sweep = 0; sweep < TWO_PASS_SWEEPS; ++sweep)
        {
            for (std::uint32_t block = 0; block < num_blocks; ++block)
            {
                for (std::uint32_t i = 0; i < tiles_per_block; ++i)
                {
                    const std::uint32_t tile = block * tiles_per_block + i;
                    const RowWindow w        = tile_window(tile, last_tile);
                    if (sweep == 1)
                    {
                        pass_two(dst + i, w);
                    }
                    else if (tile == 0)
                    {
                        pass_one<true>(dst + i, w);
                    }
                    else
                    {
                        pass_one<false>(dst + i, w);
                    }
                }

                if (block == num_blocks - 1)
                {
                    if (sweep == 0)
                    {
                        _two_pass_finish_shifted_mean_<TWO_PASS_DUAL, TWO_PASS_RETAIN_ANCHOR>(fp32_reciprocal_bits(rows));
                        if constexpr (TWO_PASS_RETAIN_ANCHOR)
                        {
                            anchor_round_trip(dst + 0);
                        }
                    }
                    else
                    {
                        finalize(dst + TWO_PASS_FINAL_DST, rows);
                    }
                }

                _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
            }
        }
    }
    else if constexpr (TWO_PASS_MODE == 1)
    {
        std::uint32_t total_rows = 0;
        for (std::uint32_t first = 0; first < params.TILE_CNT; first += TWO_PASS_BLOCK_TILES)
        {
            const std::uint32_t block_rows = two_pass_in_place(dst, first, TWO_PASS_BLOCK_TILES, last_tile);
            total_rows += block_rows;
            if (first == 0)
            {
                SFPU_UNARY_CALL(
                    dest_sync, is_fp32_dest_acc_en, _two_pass_store_mean_m2_to_dst_, (TWO_PASS_DUAL), dst + TWO_PASS_STATE_DST, VectorMode::RC_custom);
            }
            else
            {
                SFPU_UNARY_CALL(
                    dest_sync,
                    is_fp32_dest_acc_en,
                    _two_pass_combine_block_to_dst_,
                    (TWO_PASS_DUAL),
                    dst + TWO_PASS_STATE_DST,
                    VectorMode::RC_custom,
                    fp32_reciprocal_bits(total_rows),
                    fp32_bits(block_rows));
            }
        }
        // Combine leaves the merged state in LREG4/LREG5 with LREG6 cleared.
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            _two_pass_store_mean_var_to_dst_row_,
            (false /* dual_m2 */, true /* store_mean */),
            dst + TWO_PASS_FINAL_DST,
            VectorMode::RC_custom,
            fp32_reciprocal_bits(total_rows));
        _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
    }
    else
    {
        static_assert(TWO_PASS_MODE != 2 || !TWO_PASS_DUAL, "group switching is single-accumulator only");
        const std::uint32_t rows_a = two_pass_in_place(dst, 0, TWO_PASS_BLOCK_TILES, last_tile);
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            _two_pass_switch_group_,
            (false),
            dst + TWO_PASS_STATE_DST,
            VectorMode::RC_custom,
            TWO_PASS_GROUP_A,
            TWO_PASS_GROUP_B);
        two_pass_in_place(dst, TWO_PASS_BLOCK_TILES, TWO_PASS_BLOCK_TILES, last_tile);
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            _two_pass_switch_group_,
            (false),
            dst + TWO_PASS_STATE_DST,
            VectorMode::RC_custom,
            TWO_PASS_GROUP_B,
            TWO_PASS_GROUP_A);
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            _two_pass_store_mean_var_to_dst_raw_group_,
            (false),
            dst + TWO_PASS_STATE_DST,
            VectorMode::RC_custom,
            TWO_PASS_GROUP_A,
            fp32_reciprocal_bits(rows_a));
        _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
    }

    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
}

#endif

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();

    const std::uint32_t tiles_per_block = two_pass_tiles_per_block(params.TILE_CNT);
    const std::uint32_t num_blocks      = params.TILE_CNT / tiles_per_block;

    ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(
        ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces), L1_ADDRESS(params.buffer_Res[0]), formats.pack_dst);

    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), ckernel::DEFAULT_TENSOR_SHAPE, tiles_per_block);
    for (std::uint32_t sweep = 0; sweep < TWO_PASS_SWEEPS; ++sweep)
    {
        for (std::uint32_t block = 0; block < num_blocks; ++block)
        {
            _llk_pack_(params.DST_INDEX, block * tiles_per_block /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
            _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
}
#endif
