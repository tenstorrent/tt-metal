// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const Operand& buffer_A         = params.buffer_A;
    const Operand& buffer_B         = params.buffer_B;
#endif
    constexpr std::uint32_t SELECTED_UNPACKER = unpack_to_dest ? p_unpacr::UNP_DEST : p_unpacr::UNP_A;
    const ckernel::TensorShape tensor_shape_A = TENSOR_SHAPE_FROM_PARAMS(params);

    {
        ZONE_SCOPED("INIT")
        if constexpr (unpack_to_dest)
        {
            // Only the end-to-end path uses the unpack→pack dest-dvalid
            // handshake. Isolates deliberately have no consumer, so clear the
            // persisted wait mask rather than leaving UNP_DEST blocked.
            if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
            {
                set_up_unpack_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
            }
            else
            {
                set_up_zero_dest_dvalid_handshake_for_unpack();
            }
        }
        else
        {
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::UNPACK>();
        }

        unsigned l1_addr_16B;
        if constexpr (UNPACKER_ENGINE_SEL == p_unpacr::UNP_B)
        {
            l1_addr_16B = L1_ADDRESS(buffer_B[0]);
        }
        else
        {
            l1_addr_16B = L1_ADDRESS(buffer_A[0]);
        }

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(tensor_shape_A, l1_addr_16B, formats.unpack_A_src);

        if constexpr (unpack_to_dest)
        {
            _llk_unpack_configure_unary_<SELECTED_UNPACKER>(static_cast<DataFormat>(formats.unpack_A_dst));
            // Unpack one tile row at a time for double-buffering with packer (SyncHalf).
            // Writing all tiles at once would cause _llk_pack_dest_dvalid_section_done_'s
            // ZEROACC to wipe subsequent tile rows after packing the first one.
            _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(
                ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), tensor_shape_A, BLOCK_CT_DIM);
        }
        else
        {
            if constexpr (is_fp32_dest_acc_en)
            {
                _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(
                    static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_A_dst));
            }
            else
            {
                _llk_unpack_configure_unary_<SELECTED_UNPACKER>(static_cast<DataFormat>(formats.unpack_A_dst));
            }
            _llk_unpack_unary_operand_init_<SELECTED_UNPACKER, false /*transpose*/, is_fp32_dest_acc_en>(
                ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), tensor_shape_A, TILE_CNT);
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                // Math consumes every tile in every block row.
                const std::uint32_t src_handshake_iters = LOOP_FACTOR * TILE_CNT;
                if constexpr (is_fp32_dest_acc_en)
                {
                    // 32-bit datacopy uses ELWADD and consumes both SrcA
                    // and the dummy SrcB produced by unpack.
                    _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(src_handshake_iters);
                }
                else
                {
                    _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(src_handshake_iters);
                }
            }
        }
        else if constexpr (unpack_to_dest)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
                {
                    _llk_unpack_unary_operand_<SELECTED_UNPACKER>(block_rt * BLOCK_CT_DIM, tensor_shape_A);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_unpack_unary_operand_<SELECTED_UNPACKER>(0 /*l1_tile_idx*/, tensor_shape_A);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR     = params.LOOP_FACTOR;
    const std::uint32_t num_faces       = params.num_faces;
    const std::uint32_t TEST_FACE_R_DIM = params.TEST_FACE_R_DIM;
#endif
    if constexpr (!unpack_to_dest)
    {
        {
            ZONE_SCOPED("INIT")
            // PACK_ISOLATE and L1_CONGESTION measure pack without the
            // FPU→PACK dest-dvalid handshake (WH/BH style).
            if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE && PERF_RUN_TYPE != PerfRunType::L1_CONGESTION)
            {
                set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
            }

            configure_math_hardware_for_float32_int32_or_default<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en>(
                static_cast<DataFormat>(formats.math), static_cast<DataFormat>(formats.pack_src));

            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(
                num_faces * TEST_FACE_R_DIM /*num_rows_per_matrix*/, 1 /*num_matrices*/);
            PROFILER_SYNC();
        }
        {
            ZONE_SCOPED("TILE_LOOP")
            if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
            {
            }
            else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
            {
                const std::uint32_t src_handshake_iters = LOOP_FACTOR * BLOCK_RT_DIM * BLOCK_CT_DIM;
                if constexpr (is_fp32_dest_acc_en)
                {
                    _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(src_handshake_iters);
                }
                else
                {
                    _perf_math_loop_clear_valid<true /*clear_a*/, false /*clear_b*/>(src_handshake_iters);
                }
            }
            else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
                    {
                        for (std::uint32_t block_ct = 0; block_ct < BLOCK_CT_DIM; block_ct++)
                        {
                            _llk_math_eltwise_unary_datacopy_(block_ct);
                        }
                    }
                }
            }
            else
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
                {
                    for (std::uint32_t block_rt = 0; block_rt < BLOCK_RT_DIM; block_rt++)
                    {
                        for (std::uint32_t block_ct = 0; block_ct < BLOCK_CT_DIM; block_ct++)
                        {
                            _llk_math_eltwise_unary_datacopy_(block_ct);
                        }
                        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
                    }
                }
            }
            PROFILER_SYNC();
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "cpack_common.h"
#include "llk_bfd_alloc.h"
#include "llk_pack_common.h"
#include "llk_pack_untilize.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_Res       = params.buffer_Res;
#endif
    const ckernel::TensorShape tensor_shape = TENSOR_SHAPE_FROM_PARAMS(params);
    {
        ZONE_SCOPED("INIT")
        // Match WH/BH PACK_ISOLATE and L1_CONGESTION: no math↔pack handshake;
        // pack from whatever is in dest.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            if constexpr (unpack_to_dest)
            {
                set_up_unpack_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
            }
            else
            {
                set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::PACK>();
            }
        }

        if (tensor_shape.face_r_dim < ckernel::pack::PACR_STRIDE_OFFSET_ROWS)
        {
            // PACR_STRIDE quirk: tiny-tiles index L1 rows as tiles, so the BD is built with y_dim = 1.
            ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0, ckernel::trisc::L1AccessMode::Strided>(
                tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        }
        else
        {
            ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0, ckernel::trisc::L1AccessMode::Continuous>(
                tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        }

        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        if (tensor_shape.total_num_faces() == NUM_FACES)
        {
            _llk_pack_untilize_init_<FULL_CT_DIM, BLOCK_CT_DIM>(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), tensor_shape);
        }
        else
        {
            _llk_pack_untilize_strided_init_<FULL_CT_DIM, BLOCK_CT_DIM>(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), tensor_shape);
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        // _llk_pack_untilize_ packs one block ct_dim of tiles (one tile row) at a time.
        const std::uint32_t y_stride_external = FULL_CT_DIM * tensor_shape.num_faces_r_dim * tensor_shape.face_r_dim;

        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            // No dest-dvalid section_done: WH/BH isolate packs without math handshake.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t y = 0; y < BLOCK_RT_DIM; y++)
                {
                    // Each tile row is produced in alternating banks (SyncHalf).
                    // Read from dest_idx 0; section_done zeroes the current bank
                    // and switches the packer to the other bank.
                    if (tensor_shape.total_num_faces() == NUM_FACES)
                    {
                        _llk_pack_untilize_(0 /*dest_idx*/, y * y_stride_external);
                    }
                    else
                    {
                        _llk_pack_untilize_strided_<FULL_CT_DIM>(
                            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), tensor_shape, y * y_stride_external, 0 /*src_tile_idx*/);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t y = 0; y < BLOCK_RT_DIM; y++)
                {
                    if (tensor_shape.total_num_faces() == NUM_FACES)
                    {
                        _llk_pack_untilize_(0 /*dest_idx*/, y * y_stride_external);
                    }
                    else
                    {
                        _llk_pack_untilize_strided_<FULL_CT_DIM>(
                            ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), tensor_shape, y * y_stride_external, 0 /*src_tile_idx*/);
                    }
                    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
