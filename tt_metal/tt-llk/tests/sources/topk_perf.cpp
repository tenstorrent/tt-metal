// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Perf twin of topk_test.cpp: the generic bitonic TopK pipeline (transposed unpack, datacopies, local sort, merge and rebuild per
// 2-tile slab, pack), reported per value tile of the row; TOPK_PERF_PHASE selects the SFPU calls a tile-pair step issues.

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

enum class Stage : int
{
    Values  = 0,
    Indices = 1
};

constexpr int NUM_STAGES         = 2;
constexpr int NUM_TILES_PER_STAGE = 2;

constexpr int PH_FULL = 0, PH_SORT = 1, PH_MERGE = 2, PH_REBUILD = 3, PH_COPY = 4, PH_FUSE = 5;
constexpr bool RUN_SORT    = (TOPK_PERF_PHASE == PH_FULL || TOPK_PERF_PHASE == PH_SORT);
constexpr bool RUN_MERGE   = (TOPK_PERF_PHASE == PH_FULL || TOPK_PERF_PHASE == PH_MERGE);
constexpr bool RUN_REBUILD = (TOPK_PERF_PHASE == PH_FULL || TOPK_PERF_PHASE == PH_REBUILD);
constexpr bool RUN_FUSE    = (TOPK_PERF_PHASE == PH_FULL || TOPK_PERF_PHASE == PH_FUSE);
constexpr bool DROP_COPY   = TOPK_PERF_DROP_COPY && (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE);

static_assert(PERF_RUN_TYPE != PerfRunType::L1_CONGESTION, "topk_perf has no L1_CONGESTION branch");

// Tile-pair steps per tile row: the sum over the iterations of NUM_VALUE_TILES / 2^(i+1).
inline std::uint32_t steps_per_row(std::uint32_t num_value_tiles)
{
    std::uint32_t steps = 0;
    for (std::uint32_t it = 0; it < TOPK_NUM_ITERATIONS; ++it)
    {
        steps += num_value_tiles / ((1u << it) * NUM_TILES_PER_STAGE);
    }
    return steps;
}

// ============================================================================
#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR   = params.LOOP_FACTOR;
    const int NUM_ROWS                = params.FULL_RT_DIM;
    const int NUM_VALUE_TILES_PER_ROW = params.FULL_CT_DIM / NUM_STAGES;

    const std::uint32_t unpack_src_data_types[NUM_STAGES] = {formats.unpack_A_src, ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t unpack_dst_data_types[NUM_STAGES] = {formats.unpack_A_dst, ckernel::to_underlying(DataFormat::UInt16)};

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            unpack_src_data_types[0], unpack_src_data_types[0], unpack_dst_data_types[0], unpack_dst_data_types[0], FACE_R_DIM, FACE_R_DIM, 4, 4);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            1, 1, ckernel::DEFAULT_TENSOR_SHAPE, unpack_src_data_types[0], unpack_dst_data_types[0]);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!DROP_COPY)
            {
                // 4 datacopies per step (2 stages x 2 tiles), one valid per face each.
                const std::uint32_t steps = steps_per_row(NUM_VALUE_TILES_PER_ROW) * NUM_ROWS * LOOP_FACTOR;
                _perf_unpack_loop_set_valid<true, is_fp32_dest_acc_en>(4 * 4 * steps);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE) // L1_TO_L1, UNPACK_ISOLATE
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int row = 0; row < NUM_ROWS; ++row)
                {
                    for (std::uint32_t it = 0; it < TOPK_NUM_ITERATIONS; ++it)
                    {
                        const int distance = (1 << it);
                        const int pairs    = NUM_VALUE_TILES_PER_ROW / (distance * NUM_TILES_PER_STAGE);
                        for (int pair = 0; pair < pairs; ++pair)
                        {
                            for (Stage stage : {Stage::Values, Stage::Indices})
                            {
                                const int si                = static_cast<int>(stage);
                                const std::uint32_t src_fmt = unpack_src_data_types[si];
                                const std::uint32_t dst_fmt = unpack_dst_data_types[si];
                                _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                                    src_fmt, dst_fmt, 16 * 16 * 4);
                                _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                                    (it == 0) ? 1 : 0, (it == 0) ? 1 : 0, ckernel::DEFAULT_TENSOR_SHAPE, src_fmt, dst_fmt);
                                const int first_tile  = row * params.FULL_CT_DIM + si * NUM_VALUE_TILES_PER_ROW + pair * (distance * NUM_TILES_PER_STAGE);
                                const int second_tile = first_tile + distance;
                                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                                    PERF_ADDRESS(PERF_INPUT_A, first_tile), src_fmt, dst_fmt);
                                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                                    PERF_ADDRESS(PERF_INPUT_A, second_tile), src_fmt, dst_fmt);
                            }
                        }
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif // LLK_TRISC_UNPACK

// ============================================================================
#ifdef LLK_TRISC_MATH
#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"

using namespace ckernel;

#define DST_SYNC_MODE  dest_sync
#define DST_ACCUM_MODE is_fp32_dest_acc_en
#include "llk_sfpu/ckernel_sfpu_topk.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#undef DST_SYNC_MODE
#undef DST_ACCUM_MODE

constexpr bool APPROX              = false;
constexpr bool NETWORK_STABLE_SORT = TOPK_STABLE_SORT && !TOPK_FUSED_STABLE;
constexpr bool TOPK_LARGEST        = (TOPK_SORT_DIRECTION == 0);
constexpr auto TOPK_TIE_ORDER      = TOPK_LARGEST ? ckernel::sfpu::TopkTieOrder::Descending : ckernel::sfpu::TopkTieOrder::Ascending;
constexpr std::uint32_t dst_index  = 0;
constexpr VectorMode vector_mode   = VectorMode::RC_custom;

inline std::uint32_t rebuild_direction()
{
    if constexpr (TOPK_PERF_RUNTIME_DIR)
    {
        volatile std::uint32_t dir = TOPK_SORT_DIRECTION; // opaque to the compiler, as a runtime argument is
        return dir;
    }
    return TOPK_SORT_DIRECTION;
}

template <bool TILE0_SORTED>
inline void issue_local_sort(const int end_phase)
{
    if constexpr (TILE0_SORTED)
    {
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            calculate_bitonic_topk_local_sort,
            (APPROX, is_fp32_dest_acc_en, NETWORK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER),
            dst_index,
            vector_mode,
            TOPK_SORT_DIRECTION,
            end_phase,
            0 /* start_phase */,
            0 /* end_step */,
            0 /* start_step */,
            1u /* tile0_sorted */);
    }
    else
    {
        SFPU_UNARY_CALL(
            dest_sync,
            is_fp32_dest_acc_en,
            calculate_bitonic_topk_phases_steps,
            (APPROX, is_fp32_dest_acc_en, NETWORK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER),
            dst_index,
            vector_mode,
            TOPK_SORT_DIRECTION,
            end_phase,
            0 /* start_phase */,
            0 /* end_step */,
            0 /* start_step */);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR   = params.LOOP_FACTOR;
    const int NUM_ROWS                = params.FULL_RT_DIM;
    const int NUM_VALUE_TILES_PER_ROW = params.FULL_CT_DIM / NUM_STAGES;

    static_assert(!(TOPK_FUSED_STABLE && TOPK_STABLE_SORT), "fused and comparator stable modes are mutually exclusive");
    static_assert(!TOPK_FUSED_STABLE || is_fp32_dest_acc_en, "fused stable topk requires 32-bit DEST (dest_acc)");
    static_assert(!(TOPK_RANK_STAMPED && TOPK_STABLE_SORT), "rank-stamped and comparator stable modes are mutually exclusive");
    static_assert(!(TOPK_RANK_STAMPED && TOPK_FUSED_STABLE), "rank-stamped and fused-key modes are mutually exclusive");
    static_assert(!TOPK_RANK_STAMPED || is_fp32_dest_acc_en, "rank-stamped stable topk requires 32-bit DEST (dest_acc)");
    static_assert(TOPK_RANK_STAMPED || TOPK_TAG_BITS == 16, "TOPK_TAG_BITS only applies to the rank-stamped mode");
    const int end_phase = TOPK_LOGK - 1;

    const std::uint32_t math_data_types[NUM_STAGES] = {formats.math, ckernel::to_underlying(DataFormat::UInt16)};

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_data_types[0], math_data_types[0]);
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();
        if constexpr (TOPK_FUSED_STABLE)
        {
            ckernel::sfpu::_init_topk_fused_();
        }
        else if constexpr (TOPK_RANK_STAMPED)
        {
            ckernel::sfpu::_init_topk_rank_stamped_<TOPK_TAG_BITS>();
        }
        else
        {
            ckernel::sfpu::_init_topk();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            // idle
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            const std::uint32_t steps = steps_per_row(NUM_VALUE_TILES_PER_ROW) * NUM_ROWS * LOOP_FACTOR;
            for (std::uint32_t s = 0; s < steps; ++s)
            {
                for (int t = 0; t < 4; ++t)
                {
                    _perf_math_loop_clear_valid<true, true>(4);
                }
            }
        }
        else // MATH_ISOLATE, L1_TO_L1
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int row = 0; row < NUM_ROWS; ++row)
                {
                    for (std::uint32_t it = 0; it < TOPK_NUM_ITERATIONS; ++it)
                    {
                        const int distance         = (1 << it);
                        const int pairs            = NUM_VALUE_TILES_PER_ROW / (distance * NUM_TILES_PER_STAGE);
                        const bool last_iteration  = (it == (TOPK_NUM_ITERATIONS - 1));
                        const bool first_iteration = (it == 0);
                        for (int pair = 0; pair < pairs; ++pair)
                        {
                            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                            {
                                _llk_math_wait_for_dest_available_<dest_sync>();
                            }
                            if constexpr (!DROP_COPY)
                            {
                                for (Stage stage : {Stage::Values, Stage::Indices})
                                {
                                    const int si                    = static_cast<int>(stage);
                                    const std::uint32_t math_format = math_data_types[si];
                                    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(math_format);
                                    _llk_math_eltwise_unary_datacopy_init_wrapper_<
                                        DataCopyType::A2D,
                                        is_fp32_dest_acc_en,
                                        BroadcastType::NONE,
                                        false,
                                        PackMode::Default>(4, math_format);
                                    const int first_tile_in_pair_idx = si * NUM_TILES_PER_STAGE;
                                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                                        first_tile_in_pair_idx, math_format, math_format);
                                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                                        first_tile_in_pair_idx + 1, math_format, math_format);
                                }
                            }

                            if constexpr (RUN_FUSE && TOPK_FUSED_STABLE)
                            {
                                SFPU_UNARY_CALL(dest_sync, is_fp32_dest_acc_en, calculate_topk_fuse, (APPROX, TOPK_LARGEST), dst_index, vector_mode);
                            }
                            if constexpr (RUN_FUSE && TOPK_RANK_STAMPED)
                            {
                                SFPU_UNARY_CALL(
                                    dest_sync, is_fp32_dest_acc_en, calculate_topk_stamp_local_positions, (APPROX, TOPK_LARGEST, TOPK_TAG_BITS), dst_index, vector_mode);
                            }

                            if (first_iteration)
                            {
                                if constexpr (RUN_FUSE && NETWORK_STABLE_SORT)
                                {
                                    SFPU_UNARY_CALL(
                                        dest_sync, is_fp32_dest_acc_en, calculate_topk_canonicalize_negzero, (APPROX, is_fp32_dest_acc_en), dst_index, vector_mode);
                                }
                                if constexpr (RUN_SORT)
                                {
                                    issue_local_sort<TOPK_PERF_TILE0_SORTED>(end_phase);
                                }
                            }
                            else
                            {
                                if constexpr (RUN_REBUILD)
                                {
                                    SFPU_UNARY_CALL(
                                        dest_sync,
                                        is_fp32_dest_acc_en,
                                        calculate_bitonic_topk_rebuild,
                                        (APPROX, is_fp32_dest_acc_en, NETWORK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER),
                                        dst_index,
                                        vector_mode,
                                        rebuild_direction(),
                                        it,
                                        TOPK_K,
                                        TOPK_LOGK,
                                        0 /*skip_second*/);
                                }
                            }

                            if constexpr (RUN_MERGE)
                            {
                                SFPU_UNARY_CALL(
                                    dest_sync,
                                    is_fp32_dest_acc_en,
                                    calculate_bitonic_topk_merge,
                                    (APPROX, is_fp32_dest_acc_en, TOPK_SORT_DIRECTION, NETWORK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER, TOPK_TAG_BITS),
                                    dst_index,
                                    vector_mode,
                                    it,
                                    TOPK_K);
                            }

                            if (last_iteration)
                            {
                                if constexpr (RUN_REBUILD)
                                {
                                    SFPU_UNARY_CALL(
                                        dest_sync,
                                        is_fp32_dest_acc_en,
                                        calculate_bitonic_topk_rebuild,
                                        (APPROX, is_fp32_dest_acc_en, NETWORK_STABLE_SORT, TOPK_FUSED_STABLE, TOPK_RANK_STAMPED, TOPK_TIE_ORDER),
                                        dst_index,
                                        vector_mode,
                                        rebuild_direction(),
                                        it,
                                        TOPK_K,
                                        TOPK_LOGK,
                                        1 /*skip_second*/);
                                }
                                if constexpr (RUN_FUSE && TOPK_RANK_STAMPED)
                                {
                                    ckernel::sfpu::_topk_strip_rank_tags_<TOPK_TAG_BITS>(0);
                                    ckernel::sfpu::_topk_uint16_move_dest_tile_to_pack_half_(2);
                                }
                                if constexpr (RUN_FUSE && TOPK_FUSED_STABLE)
                                {
                                    SFPU_UNARY_CALL(
                                        dest_sync,
                                        is_fp32_dest_acc_en,
                                        calculate_topk_defuse,
                                        (APPROX, TOPK_LARGEST, ckernel::sfpu::TOPK_SFPSTORE_MODE_PACK_UINT16),
                                        dst_index,
                                        vector_mode,
                                        2 /*num_tiles*/);
                                }
                            }

                            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                            {
                                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                            }
                        }
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif // LLK_TRISC_MATH

// ============================================================================
#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR              = params.LOOP_FACTOR;
    const int NUM_ROWS                           = params.FULL_RT_DIM;
    const int NUM_VALUE_TILES_PER_ROW            = params.FULL_CT_DIM / NUM_STAGES;
    const int NUM_TILES_IN_RESULT_BUFFER_PER_ROW = (TOPK_K / ckernel::TILE_C_DIM) * NUM_STAGES;

    const std::uint32_t pack_src_data_types[NUM_STAGES] = {formats.pack_src, ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t pack_dst_data_types[NUM_STAGES] = {formats.pack_dst, ckernel::to_underlying(DataFormat::UInt16)};

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_dest_init_wrapper_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>();
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(pack_src_data_types[0], pack_dst_data_types[0], 16 * 16 * 4);
        _llk_pack_init_wrapper_<PackMode::Default, false>(pack_dst_data_types[0]);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int row = 0; row < NUM_ROWS; ++row)
                {
                    for (std::uint32_t it = 0; it < TOPK_NUM_ITERATIONS; ++it)
                    {
                        const int distance        = (1 << it);
                        const int pairs           = NUM_VALUE_TILES_PER_ROW / (distance * NUM_TILES_PER_STAGE);
                        const bool last_iteration = (it == (TOPK_NUM_ITERATIONS - 1));
                        for (int pair = 0; pair < pairs; ++pair)
                        {
                            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                            {
                                _llk_packer_wait_for_math_done_();
                            }
                            for (Stage stage : {Stage::Values, Stage::Indices})
                            {
                                const int si                        = static_cast<int>(stage);
                                const std::uint32_t pack_src_format = pack_src_data_types[si];
                                const std::uint32_t pack_dst_format = pack_dst_data_types[si];
                                _llk_pack_reconfig_data_format_wrapper_<is_fp32_dest_acc_en, false>(
                                    pack_src_format, pack_dst_format, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4, false, false, 1);
                                _llk_pack_init_wrapper_<PackMode::Default, false>(pack_dst_format);
                                const int tile_dest_offset = si * NUM_TILES_PER_STAGE;
                                if (last_iteration)
                                {
                                    const int tile_L1 = row * NUM_TILES_IN_RESULT_BUFFER_PER_ROW + si;
                                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile_dest_offset, PERF_ADDRESS(PERF_OUTPUT, tile_L1));
                                }
                                else
                                {
                                    const int tile_L1 = row * params.FULL_CT_DIM + si * NUM_VALUE_TILES_PER_ROW + pair * (distance * NUM_TILES_PER_STAGE);
                                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile_dest_offset, PERF_ADDRESS(PERF_INPUT_A, tile_L1));
                                }
                            }
                            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                            {
                                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                            }
                        }
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif // LLK_TRISC_PACK
