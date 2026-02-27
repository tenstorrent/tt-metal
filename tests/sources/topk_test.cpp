// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TopK SFPU Test - Bitonic Merge-Based Iterative TopK Algorithm
//
// This test implements a bitonic merge-based topk algorithm that processes matrices
// row-by-row to find the top-K values and their corresponding indices.
//
// Algorithm Overview:
// ------------------
//
// 1. Initial Setup:
//    - Matrix is split into value tiles and index tiles
//    - Number of iterations = log2(num_value_tiles)
//    - Each iteration processes pairs of tiles at increasing distances
//
// 2. Iteration Structure:
//    For iteration i from 0 to TOPK_NUM_ITERATIONS-1:
//      - distance = 2^i (controls which tiles are paired)
//      - Iteration 0: Pairs (0,1), (2,3), (4,5), ... with distance=1
//      - Iteration 1: Pairs (0,2), (4,6), ... with distance=2
//      - Iteration 2: Pairs (0,4), ... with distance=4
//      - etc.
//
// 3. Operations Per Iteration:
//    a) UNPACK: Load tile pairs from L1 to DEST registers
//       - Transpose tiles in iteration 0 to convert to column-wise format
//       - Load based on current distance calculation
//
//    b) MATH (SFPU operations):
//       Iteration 0:
//         * local_sort - bitonic sort on adjacent tile pairs
//         * merge(m_iter=0) - bitonic merge
//       Iterations 1 through N-2:
//         * rebuild(m_iter, skip_second=0) - rebuild merged tiles from previous iteration
//         * merge(m_iter) - bitonic merge
//       Last iteration (N-1):
//         * rebuild(m_iter, skip_second=0) - rebuild merged tiles from previous iteration
//         * merge(m_iter) - bitonic merge
//         * rebuild(m_iter, skip_second=1) - final rebuild to extract top-K
//
//    c) PACK: Write results back to L1
//       - Final iteration: writes only first tile (top-K) to output buffer
//       - Non-final iterations: writes both tiles in pair back to L1 for next iteration
//
// 4. Example for 32x256 matrix (4 value tiles, Wt=4, TOPK_NUM_ITERATIONS=2):
//    - Iteration 0:
//      * Unpack pairs (0,1) and (2,3) with transpose
//      * local_sort → merge(m=0)
//      * Pack both tiles in each pair back to L1
//    - Iteration 1 (last):
//      * Unpack pairs (0,2) [no transpose]
//      * rebuild(m=1, skip_second=0) → merge(m=1) → rebuild(m=1, skip_second=1)
//      * Pack only first tile (top-K result) to output buffer
//
// 5. Example for 32x512 matrix (8 value tiles, Wt=8, TOPK_NUM_ITERATIONS=3):
//    - Iteration 0:
//      * Unpack pairs (0,1), (2,3), (4,5), (6,7) with transpose
//      * local_sort → merge(m=0)
//      * Pack both tiles in each pair back to L1
//    - Iteration 1:
//      * Unpack pairs (0,2), (4,6) [no transpose]
//      * rebuild(m=1, skip_second=0) → merge(m=1)
//      * Pack both tiles in each pair back to L1
//    - Iteration 2 (last):
//      * Unpack pair (0,4) [no transpose]
//      * rebuild(m=2, skip_second=0) → merge(m=2) → rebuild(m=2, skip_second=1)
//      * Pack only first tile (top-K result) to output buffer
//
// Key Design Points:
// -----------------
// - Iteration 0 uses local_sort instead of rebuild (tiles not yet merged)
// - Iterations 1+ begin with rebuild to process merged tiles from previous iteration
// - Final iteration adds extra rebuild(skip_second=1) to extract top-K values
// - Non-final iterations pack both tiles; final iteration packs only first tile of values and indices to output buffer
// - Processes value and index tiles as separate stages with same operations
// - Each row (tile height elements high) is processed independently

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// ============================================================================
// Stage Definitions
// ============================================================================

enum class Stage : int
{
    Values  = 0, // Stage for processing value tiles.
    Indices = 1  // Stage for processing index tiles.
};

constexpr int NUM_STAGES = 2;

// Two stages, one for values and one for indices since they have different source and destination formats.
// We unpack two tiles at a time, and for local sort we need two value tiles and two corresponding index tiles.
const int NUM_TILES_PER_STAGE = 2;

// ============================================================================
// UNPACK TRISC
// ============================================================================

#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    // Each tile-row is processed separately so we do the topk pipeline on each tile-row independently.
    const int NUM_TOPK_PIPELINE_EXECUTIONS = params->FULL_RT_DIM;

    const int NUM_INDEX_TILES_PER_ROW = params->FULL_CT_DIM / NUM_STAGES;
    const int NUM_VALUE_TILES_PER_ROW = params->FULL_CT_DIM / NUM_STAGES;

    // We pack the result with index tiles right after value tiles.
    const int NUM_TILES_IN_RESULT_BUFFER_PER_ROW = (TOPK_K / ckernel::TILE_C_DIM) * NUM_STAGES;

    // Data formats.
    const std::uint32_t unpack_src_data_types[NUM_STAGES] = {formats.unpack_A_src, ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t unpack_dst_data_types[NUM_STAGES] = {formats.unpack_A_dst, ckernel::to_underlying(DataFormat::UInt16)};

    for (int current_tile_row = 0; current_tile_row < NUM_TOPK_PIPELINE_EXECUTIONS; ++current_tile_row) // Iterates over tile_rows.
    {
        for (std::uint32_t current_iteration = 0; current_iteration < TOPK_NUM_ITERATIONS; ++current_iteration) // Iterates over topk pipelines.
        {
            // If we have 16 value tiles per row, log16 = 4 iterations:
            // We have 8 pairs in 0th iteration. ( 16 / (1 * 2))
            // We have 4 pairs in 1st iteration. ( 16 / (2 * 2))
            // We have 2 pairs in 2nd iteration. ( 16 / (4 * 2))
            // We have 1 pair in 3rd iteration. ( 16 / (8 * 2))
            // distance between tiles that we are comparing/merging in current iteration. It doubles every iteration since we are merging two by two tiles.
            const int distance_between_corresponding_tiles      = (1 << current_iteration);
            const int number_of_tile_pairs_in_current_iteration = (NUM_VALUE_TILES_PER_ROW / (distance_between_corresponding_tiles * NUM_TILES_PER_STAGE));

            for (int current_tile_pair_idx = 0; current_tile_pair_idx < number_of_tile_pairs_in_current_iteration;
                 ++current_tile_pair_idx) // Iterates over tiles in current topk pipeline operation.
            {
                // We do two stages of unpacking since values and indices have different source and destination formats.
                // If we have 8 tiles per row, that's 4 tiles of values and 4 tiles of indices.
                // In the first iteration we compare 0th and 1st tile for values and 4th and 5th tile which hold corresponding indices.
                for (Stage stage : {Stage::Values, Stage::Indices})
                {
                    const int stage_index                 = static_cast<int>(stage);
                    const std::uint32_t unpack_src_format = unpack_src_data_types[stage_index];
                    const std::uint32_t unpack_dst_format = unpack_dst_data_types[stage_index];

                    const bool first_hardware_configuration =
                        (current_tile_row == 0 && current_iteration == 0 && current_tile_pair_idx == 0 && stage == Stage::Values);

                    if (first_hardware_configuration)
                    {
                        // Configure is only done in the first iteration of global algorithm, after that we reconfigure.
                        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                            unpack_src_format,
                            unpack_src_format,
                            unpack_dst_format,
                            unpack_dst_format,
                            FACE_R_DIM,
                            FACE_R_DIM,
                            4 /* num_faces */,
                            4 /* num_faces */);
                    }
                    else
                    {
                        // We need to use reconfigure API to avoid race condition between hardware configuration in the second stage and unpacking in the first
                        // stage.
                        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, false /* to_from_int8 */>(
                            unpack_src_format, unpack_dst_format, 16 * 16 * 4 /* tile_size */);
                    }

                    // only do face level transpose in the first iteration to turn in into column-wise format.
                    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        /* transpose_of_faces */ (current_iteration == 0) ? 1 : 0,
                        /* within_face_16x16_transpose */ (current_iteration == 0) ? 1 : 0,
                        /* face_r_dim     */ FACE_R_DIM,
                        /* num_faces      */ 4,
                        unpack_src_format,
                        unpack_dst_format);

                    const int tile_row_offset = current_tile_row * params->FULL_CT_DIM;
                    const int tile_pair_offset =
                        current_tile_pair_idx * (distance_between_corresponding_tiles *
                                                 NUM_TILES_PER_STAGE); // offset to get to the correct tile pair we are processing in current iteration.
                    const int stage_offset =
                        stage_index * NUM_VALUE_TILES_PER_ROW; // since first half of the tiles are value tiles and second half are index tiles.

                    // Unpack first tile in pair.
                    const int first_tile_index = tile_row_offset + stage_offset + tile_pair_offset;

                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(params->buffer_A[first_tile_index]), unpack_src_format, unpack_dst_format);

                    // Unpack second tile in pair.
                    const int second_tile_index =
                        first_tile_index +
                        distance_between_corresponding_tiles; // since we are processing pairs of tiles that are distance_between_corresponding_tiles apart.

                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(params->buffer_A[second_tile_index]), unpack_src_format, unpack_dst_format);

                } // Stage loop.
            } // Pipeline loop.
        } // iteration loop.
    } // current_tile_row loop
} // Kernel function.
#endif // LLK_TRISC_UNPACK

// ============================================================================
// UNPACK MATH
// ============================================================================

#ifdef LLK_TRISC_MATH
#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_transpose_dest.h"

using namespace ckernel;

// Define DST_SYNC_MODE and DST_ACCUM_MODE so LLK SFPU params helpers compile in this TU.
// This must be done BEFORE including the TopK LLK API header.
#define DST_SYNC_MODE  dest_sync
#define DST_ACCUM_MODE is_fp32_dest_acc_en
#include "llk_math_eltwise_unary_sfpu_topk.h"
#undef DST_SYNC_MODE
#undef DST_ACCUM_MODE

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    // Each tile-row is processed separately so we do the topk pipeline on each tile-row independently.
    const int NUM_TOPK_PIPELINE_EXECUTIONS = params->FULL_RT_DIM;

    const int NUM_INDEX_TILES_PER_ROW = params->FULL_CT_DIM / NUM_STAGES;
    const int NUM_VALUE_TILES_PER_ROW = params->FULL_CT_DIM / NUM_STAGES;

    // We pack the result with index tiles right after value tiles.
    const int NUM_TILES_IN_RESULT_BUFFER_PER_ROW = (TOPK_K / ckernel::TILE_C_DIM) * NUM_STAGES;

    const bool is_int_fpu_en = false;

    /* TOPK api constants. */
    constexpr bool APPROX             = false;
    constexpr std::uint32_t dst_index = 0;             // base DEST index for the 4-tile group.
    const int end_phase               = TOPK_LOGK - 1; // same as other TopK call sites.
    constexpr int start_phase         = 0;
    constexpr int end_step            = 0;
    constexpr int start_step          = 0;
    constexpr int vector_mode         = (int)VectorMode::RC_custom;

    const std::uint32_t math_data_types[NUM_STAGES] = {formats.math, ckernel::to_underlying(DataFormat::UInt16)};

    // Datacopy Initialization.
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();

    // After Datacopy, we do topk SFPU.
    // These two calls are essentially the same as calling ckernel::llk_math_eltwise_unary_sfpu_topk_init<APPROX>(); from metal.
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();
    ckernel::sfpu::_init_topk();

    for (int current_tile_row = 0; current_tile_row < NUM_TOPK_PIPELINE_EXECUTIONS; ++current_tile_row) // Iterates over tile_rows.
    {
        for (std::uint32_t current_iteration = 0; current_iteration < TOPK_NUM_ITERATIONS; ++current_iteration) // Iterates over topk pipelines.
        {
            const int distance_between_corresponding_tiles      = (1 << current_iteration);
            const int number_of_tile_pairs_in_current_iteration = (NUM_VALUE_TILES_PER_ROW / (distance_between_corresponding_tiles * NUM_TILES_PER_STAGE));
            const bool last_iteration                           = (current_iteration == (TOPK_NUM_ITERATIONS - 1));
            const bool first_iteration                          = (current_iteration == 0);

            for (int current_tile_pair_idx = 0; current_tile_pair_idx < number_of_tile_pairs_in_current_iteration;
                 ++current_tile_pair_idx) // Iterates over tiles in current topk pipeline operation.
            {
                _llk_math_wait_for_dest_available_<dest_sync>();

                // Datacopy.
                for (Stage stage : {Stage::Values, Stage::Indices})
                {
                    const int stage_index           = static_cast<int>(stage);
                    const std::uint32_t math_format = math_data_types[stage_index];

                    const bool first_hardware_configuration =
                        (current_tile_row == 0 && current_iteration == 0 && current_tile_pair_idx == 0 && stage == Stage::Values);

                    // We do configure in the first iteration, and then reconfigure in the following iterations to avoid race condition between math and
                    // datacopy.
                    if (first_hardware_configuration)
                    {
                        _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
                    }
                    else
                    {
                        // We need to use reconfigure API to avoid race condition between hardware configuration in the second stage and math in the first
                        // stage.
                        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false /* to_from_int8 */>(math_format);
                    }

#ifdef ARCH_BLACKHOLE
                    _llk_math_eltwise_unary_datacopy_init_<
                        DataCopyType::A2D,
                        is_fp32_dest_acc_en,
                        BroadcastType::NONE,
                        false, // tilize
                        false  // is_int_fpu_en
                        >(
                        /*num_rows_per_matrix=*/4,
                        /*math_format=*/math_format);
#else
                    _llk_math_eltwise_unary_datacopy_init_<
                        DataCopyType::A2D,
                        is_fp32_dest_acc_en,
                        BroadcastType::NONE,
                        false // is_int_fpu_en
                        >(
                        /*num_rows_per_matrix=*/4,
                        /*math_format=*/math_format);
#endif

                    const int first_tile_in_pair_idx = stage_index * NUM_TILES_PER_STAGE;

                    // Datacopy first tile in pair:
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        /*dst_tile_index=*/first_tile_in_pair_idx, math_format, math_format);

                    const int second_tile_in_pair_idx = first_tile_in_pair_idx + 1;
                    // Datacopy second tile in pair:
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        /*dst_tile_index=*/second_tile_in_pair_idx, math_format, math_format);

                } // Stage loop.

                // Pick the first operation.
                if (first_iteration)
                {
                    // same as calling ckernel::llk_math_eltwise_unary_sfpu_topk_local_sort from metal.
                    _llk_math_eltwise_unary_sfpu_params_<APPROX>(
                        ckernel::sfpu::calculate_bitonic_topk_phases_steps<APPROX, is_fp32_dest_acc_en, TOPK_STABLE_SORT>,
                        dst_index,
                        vector_mode,
                        TOPK_SORT_DIRECTION,
                        end_phase,
                        start_phase,
                        end_step,
                        start_step);
                }
                else
                {
                    // Same as calling ckernel::llk_math_eltwise_unary_sfpu_topk_rebuild from metal.
                    _llk_math_eltwise_unary_sfpu_params_<APPROX>(
                        ckernel::sfpu::calculate_bitonic_topk_rebuild<APPROX, is_fp32_dest_acc_en, TOPK_STABLE_SORT>,
                        dst_index,
                        vector_mode,
                        TOPK_SORT_DIRECTION,
                        current_iteration,
                        TOPK_K,
                        TOPK_LOGK,
                        0 /* skip_second */);
                }

                // Always a second operation.
                _llk_math_eltwise_unary_sfpu_params_<APPROX>(
                    ckernel::sfpu::calculate_bitonic_topk_merge<APPROX, is_fp32_dest_acc_en, TOPK_SORT_DIRECTION, TOPK_STABLE_SORT>,
                    dst_index,
                    vector_mode,
                    current_iteration,
                    TOPK_K);

                // Additional last operation.
                if (last_iteration)
                {
                    // Same as calling ckernel::llk_math_eltwise_unary_sfpu_topk_rebuild from metal.
                    _llk_math_eltwise_unary_sfpu_params_<APPROX>(
                        ckernel::sfpu::calculate_bitonic_topk_rebuild<APPROX, is_fp32_dest_acc_en, TOPK_STABLE_SORT>,
                        dst_index,
                        vector_mode,
                        TOPK_SORT_DIRECTION,
                        current_iteration,
                        TOPK_K,
                        TOPK_LOGK,
                        1 /* skip_second */);
                }

                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            } // Within pipeline loop.
        } // iteration loop.
    } // current_tile_row loop
}
#endif // LLK_TRISC_MATH

// ============================================================================
// PACK TRISC
// ============================================================================

#ifdef LLK_TRISC_PACK
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
#ifdef RUNTIME_FORMATS
    const volatile FormatConfig& formats = params->formats;
#endif
    // Each tile-row is processed separately so we do the topk pipeline on each tile-row independently.
    const int NUM_TOPK_PIPELINE_EXECUTIONS = params->FULL_RT_DIM;

    const int NUM_INDEX_TILES_PER_ROW = params->FULL_CT_DIM / NUM_STAGES;
    const int NUM_VALUE_TILES_PER_ROW = params->FULL_CT_DIM / NUM_STAGES;

    // We pack the result with index tiles right after value tiles.
    const int NUM_TILES_IN_RESULT_BUFFER_PER_ROW = (TOPK_K / ckernel::TILE_C_DIM) * NUM_STAGES;

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#else
    _llk_pack_dest_init_<dest_sync, false, false>();
#endif

    const std::uint32_t pack_src_data_types[NUM_STAGES] = {formats.pack_src, ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t pack_dst_data_types[NUM_STAGES] = {formats.pack_dst, ckernel::to_underlying(DataFormat::UInt16)};

    for (int current_tile_row = 0; current_tile_row < NUM_TOPK_PIPELINE_EXECUTIONS; ++current_tile_row) // Iterates over tile_rows.
    {
        for (std::uint32_t current_iteration = 0; current_iteration < TOPK_NUM_ITERATIONS; ++current_iteration) // Iterates over topk pipelines.
        {
            const int distance_between_corresponding_tiles_in_current_math = (1 << current_iteration);
            const int number_of_tile_pairs_in_current_iteration =
                (NUM_VALUE_TILES_PER_ROW / (distance_between_corresponding_tiles_in_current_math * NUM_TILES_PER_STAGE));
            const bool last_iteration = (current_iteration == (TOPK_NUM_ITERATIONS - 1));

            for (int current_tile_pair_idx = 0; current_tile_pair_idx < number_of_tile_pairs_in_current_iteration;
                 ++current_tile_pair_idx) // Iterates over tiles in current topk pipeline operation.
            {
                _llk_packer_wait_for_math_done_();

                for (Stage stage : {Stage::Values, Stage::Indices})
                {
                    const int stage_index               = static_cast<int>(stage);
                    const std::uint32_t pack_src_format = pack_src_data_types[stage_index];
                    const std::uint32_t pack_dst_format = pack_dst_data_types[stage_index];

                    const bool first_hardware_configuration =
                        (current_tile_row == 0 && current_iteration == 0 && current_tile_pair_idx == 0 && stage == Stage::Values);

                    if (first_hardware_configuration)
                    {
#ifdef ARCH_BLACKHOLE
                        _llk_pack_hw_configure_<
                            is_fp32_dest_acc_en,
                            false,  // untilize
                            false>( // tilize
                            pack_src_format,
                            pack_dst_format,
                            16 * 16 * 4);
#else
                        _llk_pack_hw_configure_<
                            is_fp32_dest_acc_en,
                            false>( // untilize
                            pack_src_format,
                            pack_dst_format,
                            16 * 16 * 4);
#endif
                    }
                    else
                    {
                        // We need to use reconfigure API to avoid race condition between hardware configuration in the second stage and pack in the first
                        // stage.
#ifdef ARCH_BLACKHOLE
                        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en, false /* is_tile_dim_reconfig_en */>(
                            pack_src_format,
                            pack_dst_format,
                            16 * 16 * 4,
                            FACE_R_DIM,
                            TILE_C_DIM,
                            4 /* num_faces */,
                            false /* partial_face */,
                            false /* narrow_tile */,
                            1 /* num_tiles */);
#else
                        _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en, false /* is_tile_dim_reconfig_en */>(
                            pack_src_format, pack_dst_format, 16 * 16 * 4, FACE_R_DIM, 4 /* num_faces */, false /* partial_face */, false /* narrow_tile */);
#endif
                    }

                    _llk_pack_init_<false, false>(pack_dst_format);

                    const int tile_dest_offset = stage_index * NUM_TILES_PER_STAGE;

                    if (last_iteration)
                    {
                        const int tile_row_offset = current_tile_row * NUM_TILES_IN_RESULT_BUFFER_PER_ROW;
                        const int tile_L1_offset  = tile_row_offset + stage_index;

                        // Pack only the first tile from the pair in the last iteration since after final merge/rebuild,
                        // the result is in the first tile of each pair (DEST indices 0 and 2 for values and indices respectively).
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(tile_dest_offset, L1_ADDRESS(params->buffer_Res[tile_L1_offset]));
                    }
                    else
                    {
                        const int tile_row_offset = current_tile_row * params->FULL_CT_DIM;
                        const int tile_pair_offset =
                            current_tile_pair_idx * (distance_between_corresponding_tiles_in_current_math *
                                                     NUM_TILES_PER_STAGE); // offset to get to the correct tile pair we are processing in current iteration.
                        const int stage_offset =
                            stage_index * NUM_VALUE_TILES_PER_ROW; // since first half of the tiles are value tiles and second half are index tiles.
                        const int tile_L1_offset = tile_row_offset + stage_offset + tile_pair_offset;

                        // Pack both tiles in the pair back to L1 for next iteration.
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(tile_dest_offset, L1_ADDRESS(params->buffer_A[tile_L1_offset]));
                    }

                } // Stage loop.
                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        } // Iteration loop.
    } // current_tile_row loop
}
#endif // LLK_TRISC_PACK
