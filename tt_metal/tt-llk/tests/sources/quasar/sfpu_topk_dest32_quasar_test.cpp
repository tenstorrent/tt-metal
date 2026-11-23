// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TopK SFPU test for Quasar on a 32-bit Dest: the unstable, fused-key and rank-stamped networks.
//
// sfpu_topk_quasar_test.cpp runs the network on a 16-bit Dest with Int16 index tiles datacopied
// through SrcA. A 16-bit integer cannot land in a 32-bit Dest that way on Quasar, so this test
// unpacks straight to Dest instead: Float32 value tiles (bf16 values widened exactly, so the low 16
// bits are free for the stable modes' tags) and Int32 index tiles. UNP_DEST cannot transpose, so
// the host supplies the tiles already transposed to the column-major order the network sorts in.
//
// One 2-tile slab per tile row (W == 128, a single TopK iteration): local sort, merge and the
// final rebuild, as on the last iteration of sfpu_topk_quasar_test.cpp. The packed / tagged words
// never round-trip L1. Pack writes the first value tile (Float32) and the first index tile (Int32)
// of each slab to buffer_Res.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

enum class Stage : int
{
    Values  = 0,
    Indices = 1
};

constexpr int NUM_STAGES                     = 2;
constexpr int NUM_TILES_PER_STAGE            = 2;
constexpr std::uint32_t TOPK_VALUE_FORMAT    = ckernel::to_underlying(DataFormat::Float32);
constexpr std::uint32_t TOPK_INDEX_FORMAT    = ckernel::to_underlying(DataFormat::Int32);
constexpr std::uint32_t STAGE_FORMATS[]      = {TOPK_VALUE_FORMAT, TOPK_INDEX_FORMAT};
constexpr std::uint32_t RESULT_TILES_PER_ROW = NUM_STAGES; // one value tile and one index tile (K == 32)

// ============================================================================
// UNPACK TRISC
// ============================================================================

#ifdef LLK_TRISC_UNPACK

#include "ckernel_buf_desc.h"
#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand_to_dest.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t FULL_RT_DIM = params.FULL_RT_DIM;
    const std::uint32_t FULL_CT_DIM = params.FULL_CT_DIM;
    const Operand& buffer_A         = params.buffer_A;
#endif
    const int NUM_VALUE_TILES_PER_ROW = FULL_CT_DIM / NUM_STAGES;

    constexpr std::uint8_t bfd_unpack = ckernel::trisc::BFD_ID_UNP0;

    // The to-dest placers carry no synchronization; dest-dvalid brackets each slab.
    _llk_unpack_dest_init_();
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    for (std::uint32_t tile_row = 0; tile_row < FULL_RT_DIM; ++tile_row)
    {
        for (Stage stage : {Stage::Values, Stage::Indices})
        {
            const int stage_index         = to_underlying(stage);
            const std::uint32_t l1_format = STAGE_FORMATS[stage_index];

            ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(ckernel::DEFAULT_TENSOR_SHAPE, L1_ADDRESS(buffer_A[0]), l1_format);
            _llk_unpack_configure_unary_<p_unpacr::UNP_DEST>(static_cast<DataFormat>(l1_format));
            _llk_unpack_unary_operand_to_dest_init_(bfd_unpack);

            const int first_tile_index = tile_row * FULL_CT_DIM + stage_index * NUM_VALUE_TILES_PER_ROW;
            _llk_unpack_unary_operand_to_dest_tile_(first_tile_index, stage_index * NUM_TILES_PER_STAGE);
            _llk_unpack_unary_operand_to_dest_tile_(first_tile_index + 1, stage_index * NUM_TILES_PER_STAGE + 1);
        }
        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
    }
}

#endif // LLK_TRISC_UNPACK

// ============================================================================
// MATH TRISC
// ============================================================================

#ifdef LLK_TRISC_MATH

#include "cfg_defines.h"
#include "ckernel_sfpu.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "llk_sfpu/ckernel_sfpu_topk.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t FULL_RT_DIM = params.FULL_RT_DIM;
#endif
    static_assert(TOPK_NUM_ITERATIONS == 1, "the 32-bit Dest TopK test covers one 2-tile slab per tile row (W == 128)");
    static_assert(TOPK_K == 32, "the 32-bit Dest TopK test packs one value tile and one index tile per tile row");
    static_assert(is_fp32_dest_acc_en, "the 32-bit Dest TopK test needs dest_acc");
    static_assert(!TOPK_STABLE_SORT, "Quasar has no comparator-stable TopK network");
    static_assert(!(TOPK_FUSED_STABLE && TOPK_RANK_STAMPED), "fused and rank-stamped modes are mutually exclusive");
    static_assert(TOPK_RANK_STAMPED || TOPK_TAG_BITS == 16, "TOPK_TAG_BITS only applies to the rank-stamped mode");

    constexpr bool APPROX             = false;
    constexpr bool TOPK_LARGEST       = (TOPK_SORT_DIRECTION == 0); // 0 = Descending / largest-first
    constexpr std::uint32_t dst_index = 0;
    const int end_phase               = TOPK_LOGK - 1;

    const DataFormat value_format = static_cast<DataFormat>(TOPK_VALUE_FORMAT);
    _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(value_format, value_format);

    // Base SFPU init, ADDR_MOD_6, and the mode's index-tracking bit and constants (topk_tile_init).
    SFPU_UNARY_INIT_FN(topk_local_sort, sfpu::topk_init, (APPROX, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TAG_BITS));

    set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    for (std::uint32_t tile_row = 0; tile_row < FULL_RT_DIM; ++tile_row)
    {
        if constexpr (TOPK_FUSED_STABLE)
        {
            // topk_fuse_tile: pack [value | index'] keys once per fresh slab.
            SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_topk_fuse, (APPROX, TOPK_LARGEST), dst_index, VectorMode::RC_custom);
        }
        if constexpr (TOPK_RANK_STAMPED)
        {
            // topk_stamp_local_positions: tag the value words with their sequence positions.
            SFPU_UNARY_CALL(
                dest_sync, is_fp32_dest_acc_en, calculate_topk_stamp_local_positions, (APPROX, TOPK_LARGEST, TOPK_TAG_BITS), dst_index, VectorMode::RC_custom);
        }

        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            calculate_bitonic_topk_phases_steps,
            (APPROX, is_fp32_dest_acc_en, false /*STABLE_SORT*/, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED),
            dst_index,
            VectorMode::RC_custom,
            TOPK_SORT_DIRECTION,
            end_phase,
            0 /*start_phase*/,
            0 /*end_step*/,
            0 /*start_step*/);

        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            calculate_bitonic_topk_merge,
            (APPROX,
             is_fp32_dest_acc_en,
             TOPK_SORT_DIRECTION,
             false /*STABLE_SORT*/,
             TOPK_FUSED_STABLE,
             TOPK_RANK_STAMPED,
             ckernel::sfpu::TopkTieOrder::Unset,
             TOPK_TAG_BITS),
            dst_index,
            VectorMode::RC_custom,
            0 /*m_iter*/,
            TOPK_K);

        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            calculate_bitonic_topk_rebuild,
            (APPROX, is_fp32_dest_acc_en, false /*STABLE_SORT*/, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED),
            dst_index,
            VectorMode::RC_custom,
            TOPK_SORT_DIRECTION,
            0 /*m_iter*/,
            TOPK_K,
            TOPK_LOGK,
            1 /*skip_second*/);

        if constexpr (TOPK_RANK_STAMPED)
        {
            // topk_strip_rank_tags: exact value words for the pack.
            ckernel::sfpu::_topk_strip_rank_tags_<TOPK_TAG_BITS>(0 /*dst_tile_index*/);
        }
        if constexpr (TOPK_FUSED_STABLE)
        {
            // topk_defuse_tile: split the keys back into value words and Int32 index words.
            SFPU_UNARY_CALL(
                dest_sync,
                is_fp32_dest_acc_en,
                calculate_topk_defuse,
                (APPROX, TOPK_LARGEST, p_sfpu::sfpmem::INT32),
                dst_index,
                VectorMode::RC_custom,
                2 /*num_tiles*/);
        }

        wait_sfpu_idle();
        _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
    }
}

#endif // LLK_TRISC_MATH

// ============================================================================
// PACK TRISC
// ============================================================================

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const std::uint32_t FULL_RT_DIM = params.FULL_RT_DIM;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    constexpr std::uint8_t bfd_pack = ckernel::trisc::BFD_ID_PACK0;

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    for (std::uint32_t tile_row = 0; tile_row < FULL_RT_DIM; ++tile_row)
    {
        for (Stage stage : {Stage::Values, Stage::Indices})
        {
            const int stage_index         = to_underlying(stage);
            const std::uint32_t l1_format = STAGE_FORMATS[stage_index];
            const std::uint32_t l1_tile   = tile_row * RESULT_TILES_PER_ROW + stage_index;

            ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(ckernel::DEFAULT_TENSOR_SHAPE, buffer_Res[l1_tile] / 16, l1_format);
            _llk_pack_init_(bfd_pack, ckernel::DEFAULT_TENSOR_SHAPE, 1 /*num_tiles*/);
            _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(l1_format), ckernel::ReluConfig::none());
            _llk_pack_(stage_index * NUM_TILES_PER_STAGE, 0 /*start_l1_tile_idx*/, ckernel::DEFAULT_TENSOR_SHAPE);
        }
        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
    }
}

#endif // LLK_TRISC_PACK
