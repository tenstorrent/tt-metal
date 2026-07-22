// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TopK SFPU Test for Quasar Architecture
//
// Implements the bitonic merge-based iterative topk algorithm using Quasar LLK APIs.
// See tests/sources/topk_test.cpp for the full algorithm description.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

// ============================================================================
// Stage Definitions
// ============================================================================

enum class Stage : int
{
    Values  = 0,
    Indices = 1
};

constexpr int NUM_STAGES                  = 2;
constexpr int NUM_TILES_PER_STAGE         = 2;
constexpr std::uint32_t TOPK_INDEX_FORMAT = ckernel::to_underlying(DataFormat::Int16);

// PACK -> UNPACK L1 write-back semaphore (free t6 semaphore; MATH_PACK uses index 1).
// In iterations >= 1 the unpacker re-reads tiles from buffer_A that the packer wrote
// back during the previous iteration. The math<->pack dest semaphore does not order
// pack's L1 writes against unpack's L1 reads, so without this semaphore the unpack
// thread runs ahead and reads stale tiles (the same race as tt-llk issue #1344 on BH).
constexpr std::uint8_t TOPK_L1_WB_SEM = 2;

// ============================================================================
// UNPACK TRISC
// ============================================================================

#ifdef LLK_TRISC_UNPACK

#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const int NUM_TOPK_PIPELINE_EXECUTIONS = params.FULL_RT_DIM;
    const int NUM_VALUE_TILES_PER_ROW      = params.FULL_CT_DIM / NUM_STAGES;

    const std::uint32_t unpack_src_data_types[NUM_STAGES] = {formats.unpack_A_src, TOPK_INDEX_FORMAT};
    const std::uint32_t unpack_dst_data_types[NUM_STAGES] = {formats.unpack_A_dst, TOPK_INDEX_FORMAT};

    // Dvalid setup: FPU -> SFPU -> PACK
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    const std::uint32_t buf_desc_id = 0;

    // The L1 base is stable; tile offsets are passed to execute.
    // Keep unpack configuration per stage because values use
    // formats.unpack_A_* while indices use TOPK_INDEX_FORMAT
    // (Quasar Int16 transport for the uint16 index payload).
    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_A[0]);
    bd_val.f.x_dim             = FACE_C_DIM;
    bd_val.f.y_dim             = FACE_R_DIM;
    bd_val.f.z_dim             = 4;

    tdma_descriptor_t td_val;
    td_val.buf_desc    = bd_val;
    td_val.buf_desc_id = buf_desc_id;

    for (int current_tile_row = 0; current_tile_row < NUM_TOPK_PIPELINE_EXECUTIONS; ++current_tile_row)
    {
        const int tile_row_offset = current_tile_row * params.FULL_CT_DIM;

        for (std::uint32_t current_iteration = 0; current_iteration < TOPK_NUM_ITERATIONS; ++current_iteration)
        {
            const int distance         = (1 << current_iteration);
            const int num_pairs        = (NUM_VALUE_TILES_PER_ROW / (distance * NUM_TILES_PER_STAGE));
            const bool first_iteration = (current_iteration == 0);

            for (int current_tile_pair_idx = 0; current_tile_pair_idx < num_pairs; ++current_tile_pair_idx)
            {
                const int tile_pair_offset = current_tile_pair_idx * (distance * NUM_TILES_PER_STAGE);

                if (!first_iteration)
                {
                    // Each source tile of this pair was packed back to buffer_A by one
                    // pair of the previous iteration; pack posts TOPK_L1_WB_SEM once per
                    // pair. Consume two posts before issuing any UNPACR for this pair.
                    // STALL_SYNC holds the SEMGET, STALL_UNPACK/TDMA hold the unpackers.
                    for (int t = 0; t < NUM_TILES_PER_STAGE; ++t)
                    {
                        TTI_SEMWAIT(
                            ckernel::p_stall::STALL_SYNC | ckernel::p_stall::STALL_UNPACK | ckernel::p_stall::STALL_TDMA,
                            ckernel::p_stall::STALL_ON_ZERO,
                            0,
                            ckernel::trisc::semaphore::t6_sem(TOPK_L1_WB_SEM));
                        TTI_SEMGET(0, ckernel::trisc::semaphore::t6_sem(TOPK_L1_WB_SEM));
                    }
                }

                for (Stage stage : {Stage::Values, Stage::Indices})
                {
                    const int stage_index                 = to_underlying(stage);
                    const std::uint32_t unpack_src_format = unpack_src_data_types[stage_index];
                    const std::uint32_t unpack_dst_format = unpack_dst_data_types[stage_index];

                    const int stage_offset = stage_index * NUM_VALUE_TILES_PER_ROW;

                    td_val.buf_desc.f.format = static_cast<std::uint8_t>(unpack_src_format);
                    td_val.reg_data_format   = static_cast<std::uint8_t>(unpack_dst_format);

                    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
                    _llk_unpack_configure_unary_<p_unpacr::UNP_A>(td_val);

                    if (first_iteration)
                    {
                        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, true /*transpose*/, is_fp32_dest_acc_en>(
                            buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, 1);
                    }
                    else
                    {
                        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, is_fp32_dest_acc_en>(
                            buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, 1);
                    }

                    const int first_tile_index  = tile_row_offset + stage_offset + tile_pair_offset;
                    const int second_tile_index = first_tile_index + distance;
                    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(first_tile_index, ckernel::DEFAULT_TENSOR_SHAPE);
                    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(second_tile_index, ckernel::DEFAULT_TENSOR_SHAPE);

                } // Stage loop.
            } // Pipeline loop.
        } // Iteration loop.
    } // current_tile_row loop.
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
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_sfpu/ckernel_sfpu_topk.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const int NUM_TOPK_PIPELINE_EXECUTIONS = params.FULL_RT_DIM;
    const int NUM_VALUE_TILES_PER_ROW      = params.FULL_CT_DIM / NUM_STAGES;

    constexpr bool APPROX             = false;
    constexpr std::uint32_t dst_index = 0;
    const int end_phase               = TOPK_LOGK - 1;
    constexpr int start_phase         = 0;
    constexpr int end_step            = 0;
    constexpr int start_step          = 0;

    // Dest dvalid sync chain: FPU (datacopy) -> SFPU (topk) -> PACK. Math thread owns
    // both the FPU and SFPU clients.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    const std::uint32_t math_data_types[NUM_STAGES] = {formats.math, TOPK_INDEX_FORMAT};

    // Initialize topk SFPU: base init, ADDR_MOD_6, and the index-tracking LaneConfig bit.
    SFPU_UNARY_INIT_FN(topk_local_sort, sfpu::topk_init, (APPROX));

    // Datacopy 4 tiles (2 value + 2 index) from SRC to DEST; the FPU dvalid
    // wait gates the writes until pack releases the bank.
    const std::uint32_t num_rows = 4 * FACE_R_DIM;

    // The index datacopy intentionally leaves ALU_FORMAT_SPEC_REG set to Int16.
    // Restore the value format before the SFPU network; the TopK value
    // SFPLOAD/SFPSTORE paths also use explicit FP16B as a guard.
    const DataFormat value_math_format = static_cast<DataFormat>(formats.math);

    for (int current_tile_row = 0; current_tile_row < NUM_TOPK_PIPELINE_EXECUTIONS; ++current_tile_row)
    {
        for (std::uint32_t current_iteration = 0; current_iteration < TOPK_NUM_ITERATIONS; ++current_iteration)
        {
            const int distance    = (1 << current_iteration);
            const int num_pairs   = (NUM_VALUE_TILES_PER_ROW / (distance * NUM_TILES_PER_STAGE));
            const bool last_iter  = (current_iteration == (TOPK_NUM_ITERATIONS - 1));
            const bool first_iter = (current_iteration == 0);

            for (int current_tile_pair_idx = 0; current_tile_pair_idx < num_pairs; ++current_tile_pair_idx)
            {
                // The datacopy addrmod/MOP depend only on num_rows (constant), not on the
                // per-stage format, so configure once per pair. The SFPU topk ops below may
                // reprogram bank0, so this stays inside the pipeline loop.
                _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(num_rows, 1);

                for (Stage stage : {Stage::Values, Stage::Indices})
                {
                    const int stage_index        = static_cast<int>(stage);
                    const DataFormat math_format = static_cast<DataFormat>(math_data_types[stage_index]);

                    _configure_alu_formats_<false /*EN_IMPLIED_MATH_FORMAT*/, is_fp32_dest_acc_en>(
                        math_format, math_format, false /*en_int32_dest_format*/, DataFormat::Invalid /*no dest-format override*/);

                    const int first_tile_in_pair_idx  = stage_index * NUM_TILES_PER_STAGE;
                    const int second_tile_in_pair_idx = first_tile_in_pair_idx + 1;

                    _llk_math_eltwise_unary_datacopy_(first_tile_in_pair_idx);
                    _llk_math_eltwise_unary_datacopy_(second_tile_in_pair_idx);
                }

                // All 4 tiles are in dest: release the FPU dvalid to the SFPU client.
                _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();

                _configure_alu_formats_<false /*EN_IMPLIED_MATH_FORMAT*/, is_fp32_dest_acc_en>(
                    value_math_format, value_math_format, false /*en_int32_dest_format*/, DataFormat::Invalid /*no dest-format override*/);

                // Run the topk stages. RC_custom issues one SFPU call per tile (no per-face walk).
                if (first_iter)
                {
                    SFPU_UNARY_CALL(
                        dest_sync,
                        is_fp32_dest_acc_en,
                        calculate_bitonic_topk_phases_steps,
                        (APPROX, is_fp32_dest_acc_en, TOPK_STABLE_SORT),
                        dst_index,
                        VectorMode::RC_custom,
                        TOPK_SORT_DIRECTION,
                        end_phase,
                        start_phase,
                        end_step,
                        start_step);
                }
                else
                {
                    SFPU_UNARY_CALL(
                        dest_sync,
                        is_fp32_dest_acc_en,
                        calculate_bitonic_topk_rebuild,
                        (APPROX, is_fp32_dest_acc_en, TOPK_STABLE_SORT),
                        dst_index,
                        VectorMode::RC_custom,
                        TOPK_SORT_DIRECTION,
                        current_iteration,
                        TOPK_K,
                        TOPK_LOGK,
                        0 /* skip_second */);
                }

                // Always merge
                SFPU_UNARY_CALL(
                    dest_sync,
                    is_fp32_dest_acc_en,
                    calculate_bitonic_topk_merge,
                    (APPROX, is_fp32_dest_acc_en, TOPK_SORT_DIRECTION, TOPK_STABLE_SORT),
                    dst_index,
                    VectorMode::RC_custom,
                    current_iteration,
                    TOPK_K);

                // Final iteration: extra rebuild
                if (last_iter)
                {
                    SFPU_UNARY_CALL(
                        dest_sync,
                        is_fp32_dest_acc_en,
                        calculate_bitonic_topk_rebuild,
                        (APPROX, is_fp32_dest_acc_en, TOPK_STABLE_SORT),
                        dst_index,
                        VectorMode::RC_custom,
                        TOPK_SORT_DIRECTION,
                        current_iteration,
                        TOPK_K,
                        TOPK_LOGK,
                        1 /* skip_second */);
                }

                // Release the SFPU dvalid to the PACK client.
                wait_sfpu_idle();
                _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
            } // Pipeline loop.
        } // Iteration loop.
    } // current_tile_row loop.
}

#endif // LLK_TRISC_MATH

// ============================================================================
// PACK TRISC
// ============================================================================

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const int NUM_TOPK_PIPELINE_EXECUTIONS       = params.FULL_RT_DIM;
    const int NUM_VALUE_TILES_PER_ROW            = params.FULL_CT_DIM / NUM_STAGES;
    const int NUM_TILES_IN_RESULT_BUFFER_PER_ROW = (TOPK_K / ckernel::trisc::TILE_C_DIM) * NUM_STAGES;

    const std::uint32_t buf_desc_id = 8;

    // Dest dvalid sync chain: FPU (datacopy) -> SFPU (topk) -> PACK.
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    // Init the PACK -> UNPACK L1 write-back semaphore (see TOPK_L1_WB_SEM).
    // Max pending posts = pairs in one iteration (<= 8 for [32,1024]); 15 is the HW max.
    TTI_SEMINIT(15, 0, 0, ckernel::trisc::semaphore::t6_sem(TOPK_L1_WB_SEM));

    const std::uint32_t pack_src_data_types[NUM_STAGES] = {formats.pack_src, TOPK_INDEX_FORMAT};
    const std::uint32_t pack_dst_data_types[NUM_STAGES] = {formats.pack_dst, TOPK_INDEX_FORMAT};

    // Tile dims and buf_desc_id are stable; only the format and L1 address change
    // per stage. Keep the descriptor here and update those fields inside the loop.
    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc         = buffer_descriptor_u {0};
    tdma_desc.buf_desc.f.x_dim = FACE_C_DIM;
    tdma_desc.buf_desc.f.y_dim = FACE_R_DIM;
    tdma_desc.buf_desc.f.z_dim = 4;
    tdma_desc.buf_desc_id      = buf_desc_id;

    for (int current_tile_row = 0; current_tile_row < NUM_TOPK_PIPELINE_EXECUTIONS; ++current_tile_row)
    {
        for (std::uint32_t current_iteration = 0; current_iteration < TOPK_NUM_ITERATIONS; ++current_iteration)
        {
            const int distance   = (1 << current_iteration);
            const int num_pairs  = (NUM_VALUE_TILES_PER_ROW / (distance * NUM_TILES_PER_STAGE));
            const bool last_iter = (current_iteration == (TOPK_NUM_ITERATIONS - 1));

            for (int current_tile_pair_idx = 0; current_tile_pair_idx < num_pairs; ++current_tile_pair_idx)
            {
                _llk_pack_init_(buf_desc_id, ckernel::DEFAULT_TENSOR_SHAPE, 1);

                // PACR stalls on the SFPU dvalid; no explicit math-done wait needed.
                for (Stage stage : {Stage::Values, Stage::Indices})
                {
                    const int stage_index               = static_cast<int>(stage);
                    const std::uint32_t pack_src_format = pack_src_data_types[stage_index];
                    const std::uint32_t pack_dst_format = pack_dst_data_types[stage_index];

                    const int tile_dest_offset = stage_index * NUM_TILES_PER_STAGE;

                    // Configure the per-stage format and L1 address for packing.
                    tdma_desc.buf_desc.f.format = static_cast<std::uint8_t>(pack_dst_format);

                    if (last_iter)
                    {
                        const int tile_row_offset        = current_tile_row * NUM_TILES_IN_RESULT_BUFFER_PER_ROW;
                        const int tile_L1_offset         = tile_row_offset + stage_index;
                        tdma_desc.buf_desc.f.l1_addr_16B = params.buffer_Res[tile_L1_offset] / 16;
                    }
                    else
                    {
                        const int tile_row_offset        = current_tile_row * params.FULL_CT_DIM;
                        const int tile_pair_offset       = current_tile_pair_idx * (distance * NUM_TILES_PER_STAGE);
                        const int stage_offset           = stage_index * NUM_VALUE_TILES_PER_ROW;
                        const int tile_L1_offset         = tile_row_offset + stage_offset + tile_pair_offset;
                        tdma_desc.buf_desc.f.l1_addr_16B = params.buffer_A[tile_L1_offset] / 16;
                    }

                    tdma_desc.reg_data_format = static_cast<std::uint8_t>(pack_src_format);
                    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

                    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
                    _llk_pack_(tile_dest_offset, 0, ckernel::DEFAULT_TENSOR_SHAPE);

                } // Stage loop.

                // Clear the PACK dvalid, zero the consumed bank, and flip the packer dest bank.
                _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();

                if (!last_iter)
                {
                    // Both tiles of this pair (value + index) are now committed to L1
                    // (section_done stalls on PACK). Release the unpack thread to re-read them.
                    ckernel::trisc::t6_semaphore_post<p_stall::PACK>(TOPK_L1_WB_SEM);
                }
            } // Pipeline loop.
        } // Iteration loop.
    } // current_tile_row loop.
}

#endif // LLK_TRISC_PACK
