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

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// ============================================================================
// Stage Definitions
// ============================================================================

enum class Stage : int
{
    Values  = 0,
    Indices = 1
};

constexpr int NUM_STAGES          = 2;
constexpr int NUM_TILES_PER_STAGE = 2;

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

    // Workaround: Use the values format for the indices stage too. Indices 0..127 are
    // exactly representable in Float16_b, so we sidestep the UINT16/LO16 round-trip path
    // (mode 0b0110) which has a dest-cell layout mismatch on Quasar.
    const std::uint32_t unpack_src_data_types[NUM_STAGES] = {formats.unpack_A_src, formats.unpack_A_src};
    const std::uint32_t unpack_dst_data_types[NUM_STAGES] = {formats.unpack_A_dst, formats.unpack_A_dst};

    // Dvalid setup: UNPACK -> FPU -> SFPU -> PACK
    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});

    const std::uint32_t buf_desc_id = 0;

    for (int current_tile_row = 0; current_tile_row < NUM_TOPK_PIPELINE_EXECUTIONS; ++current_tile_row)
    {
        for (std::uint32_t current_iteration = 0; current_iteration < TOPK_NUM_ITERATIONS; ++current_iteration)
        {
            const int distance         = (1 << current_iteration);
            const int num_pairs        = (NUM_VALUE_TILES_PER_ROW / (distance * NUM_TILES_PER_STAGE));
            const bool first_iteration = (current_iteration == 0);

            for (int current_tile_pair_idx = 0; current_tile_pair_idx < num_pairs; ++current_tile_pair_idx)
            {
                for (Stage stage : {Stage::Values, Stage::Indices})
                {
                    const int stage_index                 = static_cast<int>(stage);
                    const std::uint32_t unpack_src_format = unpack_src_data_types[stage_index];
                    const std::uint32_t unpack_dst_format = unpack_dst_data_types[stage_index];

                    const int tile_row_offset   = current_tile_row * params.FULL_CT_DIM;
                    const int tile_pair_offset  = current_tile_pair_idx * (distance * NUM_TILES_PER_STAGE);
                    const int stage_offset      = stage_index * NUM_VALUE_TILES_PER_ROW;
                    const int first_tile_index  = tile_row_offset + stage_offset + tile_pair_offset;
                    const int second_tile_index = first_tile_index + distance;

                    // Configure buffer descriptor for this stage's format
                    buffer_descriptor_u bd_val = {0};
                    bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_A[first_tile_index]);
                    bd_val.f.format            = static_cast<std::uint8_t>(unpack_src_format);
                    bd_val.f.x_dim             = FACE_C_DIM;
                    bd_val.f.y_dim             = FACE_R_DIM;
                    bd_val.f.z_dim             = 4;

                    tdma_descriptor_t td_val;
                    td_val.buf_desc        = bd_val;
                    td_val.buf_desc_id     = buf_desc_id;
                    td_val.reg_data_format = static_cast<std::uint8_t>(unpack_dst_format);
                    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);
                    _llk_unpack_configure_unary_<p_unpacr::UNP_A>(td_val);

                    // Init and unpack first tile.
                    // Transpose only in iteration 0.
                    if (first_iteration)
                    {
                        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, true /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, 1);
                    }
                    else
                    {
                        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, 1);
                    }
                    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(0);

                    // Reconfigure L1 address for second tile and unpack.
                    bd_val.f.l1_addr_16B = L1_ADDRESS(params.buffer_A[second_tile_index]);
                    td_val.buf_desc      = bd_val;
                    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);

                    if (first_iteration)
                    {
                        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, true /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, 1);
                    }
                    else
                    {
                        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, 1);
                    }
                    _llk_unpack_unary_operand_<p_unpacr::UNP_A>(0);

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

const bool is_int_fpu_en = false;

#include "cfg_defines.h"
#include "ckernel_sfpu.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "llk_math_eltwise_unary_sfpu_topk.h"
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

    // Semaphore sync for MATH <-> PACK back-pressure across iterations
    _llk_math_pack_sync_init_<dest_sync>();

    // Configure math hardware for the value format
    {
        DataFormat src_format = static_cast<DataFormat>(formats.math);
        _llk_math_srcAB_hw_configure_<false /*IMPLIED_MATH_FORMAT*/, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>(src_format, src_format);
    }

    // Initialize topk SFPU
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();
    ckernel::sfpu::_init_topk();

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
                // Wait for PACK to finish previous iteration before writing to DEST
                _llk_math_wait_for_dest_available_();

                // Datacopy 4 tiles (2 value + 2 index) from SRC to DEST
                const std::uint32_t num_rows = 4 * FACE_R_DIM;

                for (Stage stage : {Stage::Values, Stage::Indices})
                {
                    const int stage_index = static_cast<int>(stage);

                    // Initialize datacopy for this stage
                    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(num_rows, 1);

                    const int first_tile_in_pair_idx  = stage_index * NUM_TILES_PER_STAGE;
                    const int second_tile_in_pair_idx = first_tile_in_pair_idx + 1;

                    _llk_math_eltwise_unary_datacopy_(num_rows, first_tile_in_pair_idx);
                    _llk_math_eltwise_unary_datacopy_(num_rows, second_tile_in_pair_idx);
                }

                // SFPU operations
                if (first_iter)
                {
                    ckernel::llk_math_eltwise_unary_sfpu_topk_local_sort<APPROX, is_fp32_dest_acc_en, TOPK_STABLE_SORT>(
                        dst_index, TOPK_SORT_DIRECTION, end_phase, start_phase, end_step, start_step);
                }
                else
                {
                    ckernel::llk_math_eltwise_unary_sfpu_topk_rebuild<APPROX, is_fp32_dest_acc_en, TOPK_STABLE_SORT>(
                        dst_index, TOPK_SORT_DIRECTION, current_iteration, TOPK_K, TOPK_LOGK, 0 /* skip_second */);
                }

                // Always merge
                ckernel::llk_math_eltwise_unary_sfpu_topk_merge<APPROX, is_fp32_dest_acc_en, TOPK_SORT_DIRECTION, TOPK_STABLE_SORT>(
                    dst_index, current_iteration, TOPK_K);

                // Final iteration: extra rebuild
                if (last_iter)
                {
                    ckernel::llk_math_eltwise_unary_sfpu_topk_rebuild<APPROX, is_fp32_dest_acc_en, TOPK_STABLE_SORT>(
                        dst_index, TOPK_SORT_DIRECTION, current_iteration, TOPK_K, TOPK_LOGK, 1 /* skip_second */);
                }

                // Semaphore post + bank flip (matching Blackhole pattern - no dvalid)
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
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

    // No dvalid for diagnostic — pure semaphore sync.
    // Initialize packer dest base to bank 0 so the first pack doesn't read from
    // whatever bank a previous kernel binary left the packer pointing at on the
    // shared simulator session.
    _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();

    // Workaround: Use the values format for the indices stage too (matches unpack side).
    // See unpack TRISC for rationale.
    const std::uint32_t pack_src_data_types[NUM_STAGES] = {formats.pack_src, formats.pack_src};
    const std::uint32_t pack_dst_data_types[NUM_STAGES] = {formats.pack_dst, formats.pack_dst};

    for (int current_tile_row = 0; current_tile_row < NUM_TOPK_PIPELINE_EXECUTIONS; ++current_tile_row)
    {
        for (std::uint32_t current_iteration = 0; current_iteration < TOPK_NUM_ITERATIONS; ++current_iteration)
        {
            const int distance   = (1 << current_iteration);
            const int num_pairs  = (NUM_VALUE_TILES_PER_ROW / (distance * NUM_TILES_PER_STAGE));
            const bool last_iter = (current_iteration == (TOPK_NUM_ITERATIONS - 1));

            for (int current_tile_pair_idx = 0; current_tile_pair_idx < num_pairs; ++current_tile_pair_idx)
            {
                // Wait for MATH to finish this iteration before packing
                _llk_packer_wait_for_math_done_();

                for (Stage stage : {Stage::Values, Stage::Indices})
                {
                    const int stage_index               = static_cast<int>(stage);
                    const std::uint32_t pack_src_format = pack_src_data_types[stage_index];
                    const std::uint32_t pack_dst_format = pack_dst_data_types[stage_index];

                    const int tile_dest_offset = stage_index * NUM_TILES_PER_STAGE;

                    // Configure buffer descriptor for packing
                    buffer_descriptor_u bd_val = {0};
                    bd_val.f.format            = static_cast<std::uint8_t>(pack_dst_format);
                    bd_val.f.x_dim             = FACE_C_DIM;
                    bd_val.f.y_dim             = FACE_R_DIM;
                    bd_val.f.z_dim             = 4;

                    if (last_iter)
                    {
                        const int tile_row_offset = current_tile_row * NUM_TILES_IN_RESULT_BUFFER_PER_ROW;
                        const int tile_L1_offset  = tile_row_offset + stage_index;
                        bd_val.f.l1_addr_16B      = params.buffer_Res[tile_L1_offset] / 16;
                    }
                    else
                    {
                        const int tile_row_offset  = current_tile_row * params.FULL_CT_DIM;
                        const int tile_pair_offset = current_tile_pair_idx * (distance * NUM_TILES_PER_STAGE);
                        const int stage_offset     = stage_index * NUM_VALUE_TILES_PER_ROW;
                        const int tile_L1_offset   = tile_row_offset + stage_offset + tile_pair_offset;
                        bd_val.f.l1_addr_16B       = params.buffer_A[tile_L1_offset] / 16;
                    }

                    tdma_descriptor_t tdma_desc;
                    tdma_desc.buf_desc        = bd_val;
                    tdma_desc.buf_desc_id     = buf_desc_id;
                    tdma_desc.reg_data_format = static_cast<std::uint8_t>(pack_src_format);
                    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

                    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
                    _llk_pack_init_(buf_desc_id, 1);
                    _llk_pack_(tile_dest_offset, 0);

                } // Stage loop.

                // Pure semaphore sync: zeros bank, decrements semaphore, flips bank, updates packer dest
                _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
            } // Pipeline loop.
        } // Iteration loop.
    } // current_tile_row loop.
}

#endif // LLK_TRISC_PACK
