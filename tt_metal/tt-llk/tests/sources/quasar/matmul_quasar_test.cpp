// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"
#include "sfpu_stub.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "ckernel_buf_desc.h"
#include "llk_unpack_matmul.h"
#include "params.h"

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t CT_DIM      = params.CT_DIM;
    const std::uint32_t RT_DIM      = params.RT_DIM;
    const std::uint32_t KT_DIM      = params.KT_DIM;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_A         = params.buffer_A;
    const Operand& buffer_B         = params.buffer_B;
    const ckernel::TensorShape tensor_shape_A =
        ckernel::make_tensor_shape(params.in0_face_r_dim, params.in0_face_c_dim, params.num_faces_r_dim_A, params.num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape_B =
        ckernel::make_tensor_shape(params.in1_face_r_dim, params.in1_face_c_dim, params.num_faces_r_dim_B, params.num_faces_c_dim_B);
#else
    constexpr ckernel::TensorShape tensor_shape_A = ckernel::make_tensor_shape(in0_face_r_dim, in0_face_c_dim, num_faces_r_dim_A, num_faces_c_dim_A);
    constexpr ckernel::TensorShape tensor_shape_B = ckernel::make_tensor_shape(in1_face_r_dim, in1_face_c_dim, num_faces_r_dim_B, num_faces_c_dim_B);
#endif
    // An output taller than one Dest section runs as blocks of rt_block rows, one section each.
    const std::uint32_t rt_block   = matmul_rows_per_dest_section<dest_sync, is_fp32_dest_acc_en>(RT_DIM, CT_DIM);
    const std::uint32_t num_blocks = RT_DIM / rt_block;

    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    // Matmul's hw startup takes the operands reversed: in0 (A) feeds SrcB and in1 (B) feeds SrcA.
    // An MxFp4 operand starts at its Float16_b table format; matmul's init moves it to the 2x register format.
    _llk_unpack_configure_binary_<ckernel::p_unpacr::UNP_A, ckernel::p_unpacr::UNP_B>(
        static_cast<DataFormat>(formats.unpack_B_src) == DataFormat::MxFp4 ? DataFormat::Float16_b : static_cast<DataFormat>(formats.unpack_B_dst),
        static_cast<DataFormat>(formats.unpack_A_src) == DataFormat::MxFp4 ? DataFormat::Float16_b : static_cast<DataFormat>(formats.unpack_A_dst));
    {
        START_PERF_MEASURE("INIT")
        // Matmul flips the unpacker roles: _llk_unpack_matmul_init_ arg0 drives UNPACR1/SrcB, arg1 drives
        // UNPACR0/SrcA -- so operand A is recorded under Unp1 and operand B under Unp0 (matches product).
        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP1>(tensor_shape_A, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        constexpr std::uint8_t bfd_a = ckernel::trisc::BFD_ID_UNP1;
        ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_UNP0>(tensor_shape_B, L1_ADDRESS(buffer_B[0]), formats.unpack_B_src);
        constexpr std::uint8_t bfd_b = ckernel::trisc::BFD_ID_UNP0;

        _llk_unpack_matmul_init_<UNPACK_TRANSPOSE_FACES>(bfd_a, bfd_b, CT_DIM, rt_block, KT_DIM, tensor_shape_A, tensor_shape_B);
        // As llk_unpack_AB_matmul_init: an MxFp4 operand unpacks to its 2x register format.
        if (static_cast<DataFormat>(formats.unpack_B_src) == DataFormat::MxFp4)
        {
            _llk_unpack_reconfig_data_format_src_<ckernel::p_unpacr::UNP_A, false>(formats.unpack_B_src, formats.unpack_B_dst);
        }
        if (static_cast<DataFormat>(formats.unpack_A_src) == DataFormat::MxFp4)
        {
            _llk_unpack_reconfig_data_format_src_<ckernel::p_unpacr::UNP_B, false>(formats.unpack_A_src, formats.unpack_A_dst);
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_matmul_mock<PERF_RUN_TYPE>(LOOP_FACTOR * num_blocks, rt_block, KT_DIM, CT_DIM);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                // Each block is the next rt_block rows of A; B is shared by every block.
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    for (std::uint32_t j = 0; j < KT_DIM; j++)
                    {
                        _llk_unpack_matmul_(CT_DIM, rt_block, KT_DIM, block * rt_block * KT_DIM + j, j * CT_DIM, tensor_shape_A, tensor_shape_B);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    // From L1, so run_kernel keeps nothing live for this call.
    extern const volatile struct RuntimeParams __runtime_args_start[];
    const std::uint32_t unpack_A_src = __runtime_args_start[0].formats.unpack_A_src;
    const std::uint32_t unpack_B_src = __runtime_args_start[0].formats.unpack_B_src;
#else
    const std::uint32_t unpack_A_src = formats.unpack_A_src;
    const std::uint32_t unpack_B_src = formats.unpack_B_src;
#endif
    // As llk_unpack_AB_matmul_uninit: an MxFp4 operand's unpacker goes back to the Float16_b table format.
    if (static_cast<DataFormat>(unpack_B_src) == DataFormat::MxFp4)
    {
        _llk_unpack_reconfig_data_format_src_<ckernel::p_unpacr::UNP_A, false>(unpack_B_src, static_cast<std::uint32_t>(DataFormat::Float16_b));
    }
    if (static_cast<DataFormat>(unpack_A_src) == DataFormat::MxFp4)
    {
        _llk_unpack_reconfig_data_format_src_<ckernel::p_unpacr::UNP_B, false>(unpack_A_src, static_cast<std::uint32_t>(DataFormat::Float16_b));
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t CT_DIM      = params.CT_DIM;
    const std::uint32_t RT_DIM      = params.RT_DIM;
    const std::uint32_t KT_DIM      = params.KT_DIM;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const ckernel::TensorShape tensor_shape_A =
        ckernel::make_tensor_shape(params.in0_face_r_dim, params.in0_face_c_dim, params.num_faces_r_dim_A, params.num_faces_c_dim_A);
    const ckernel::TensorShape tensor_shape_B =
        ckernel::make_tensor_shape(params.in1_face_r_dim, params.in1_face_c_dim, params.num_faces_r_dim_B, params.num_faces_c_dim_B);
#else
    constexpr ckernel::TensorShape tensor_shape_A = ckernel::make_tensor_shape(in0_face_r_dim, in0_face_c_dim, num_faces_r_dim_A, num_faces_c_dim_A);
    constexpr ckernel::TensorShape tensor_shape_B = ckernel::make_tensor_shape(in1_face_r_dim, in1_face_c_dim, num_faces_r_dim_B, num_faces_c_dim_B);
#endif
    // An output taller than one Dest section runs as blocks of rt_block rows, one section each.
    const std::uint32_t rt_block   = matmul_rows_per_dest_section<dest_sync, is_fp32_dest_acc_en>(RT_DIM, CT_DIM);
    const std::uint32_t num_blocks = RT_DIM / rt_block;
    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1 || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
    {
        _llk_math_pack_sync_init_<dest_sync>();
    }
    DataFormat math_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
    {
        START_PERF_MEASURE("INIT")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Math isolate keeps the FPU→PACK dest-dvalid producer setup.
            set_up_fpu_to_pack_dest_dvalid_chain<dest_dvalid_client::FPU>();
        }

        // matmul_block_init's ALU config, before the matmul init.
        _configure_alu_cfg_<AluCfgType::DEFAULT, is_fp32_dest_acc_en>(math_format, math_format);

        // ENABLE_2X_FORMAT enables the 2x-packed FP4 matmul path (8 MVMULs per tile vs 16, K-dim
        // halved per MVMUL via the SrcA 2x sub-datum expansion). Set when SrcA/SrcB are
        // configured as MxFp4_2x_A or MxFp4_2x_B.
        // ENABLE_DIRECT_INDEXING selects the DI variant (MVMULDI with explicit indices) vs
        // the auto-increment-addr_mod MVMUL variant.
        _llk_math_matmul_init_<(ckernel::MathFidelity)MATH_FIDELITY, ENABLE_DIRECT_INDEXING, ENABLE_2X_FORMAT>(
            math_format, math_format, CT_DIM, rt_block, tensor_shape_A, tensor_shape_B);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_matmul_mock<PERF_RUN_TYPE>(LOOP_FACTOR * num_blocks, rt_block, KT_DIM, CT_DIM);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR * num_blocks; loop++)
            {
                for (std::uint32_t i = 0; i < KT_DIM; i++)
                {
                    _llk_math_matmul_block_(CT_DIM, rt_block);
                }
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                // Stands in for the idle pack's release, so MATH_PACK never fills up.
                ckernel::trisc::t6_semaphore_get(ckernel::trisc::semaphore::MATH_PACK);
            }
        }
        else
        {
            // One Dest section per block of rt_block x CT_DIM output tiles.
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR * num_blocks; loop++)
            {
                _llk_math_wait_for_dest_available_();
                for (std::uint32_t i = 0; i < KT_DIM; i++)
                {
                    _llk_math_matmul_block_(CT_DIM, rt_block);
                }
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
}

#endif

#ifdef LLK_TRISC_PACK

#include "ckernel_buf_desc.h"
#include "llk_pack.h"
#include "params.h"

// Out of line and after run_kernel, so adding the zone leaves run_kernel's code unchanged.
__attribute__((noipa, section(".text.uninit_zone"))) void uninit_zone();

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t CT_DIM      = params.CT_DIM;
    const std::uint32_t RT_DIM      = params.RT_DIM;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const Operand& buffer_Res       = params.buffer_Res;
    const ckernel::TensorShape output_shape =
        ckernel::make_tensor_shape(params.in0_face_r_dim, params.in1_face_c_dim, params.num_faces_r_dim_A, params.num_faces_c_dim_B);
#else
    constexpr ckernel::TensorShape output_shape = ckernel::make_tensor_shape(in0_face_r_dim, in1_face_c_dim, num_faces_r_dim_A, num_faces_c_dim_B);
#endif
    // An output taller than one Dest section runs as blocks of rt_block rows, one section each.
    const std::uint32_t rt_block   = matmul_rows_per_dest_section<dest_sync, is_fp32_dest_acc_en>(RT_DIM, CT_DIM);
    const std::uint32_t num_blocks = RT_DIM / rt_block;
    // compute_kernel_hw_startup work: production runs it before the op's init, so it stays out of INIT.
    // matmul_block_init has no pack part on Quasar: the one-tile pack MOP is llk_pack_init's, and pack_block sets the count.
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    ckernel::trisc::program_bfd<ckernel::trisc::BFD_ID_PACK0>(output_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
    _llk_pack_init_(ckernel::trisc::BFD_ID_PACK0, output_shape);
    _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
    {
        START_PERF_MEASURE("INIT")
        // PACK_ISOLATE packs without a math↔pack handshake.
        // Explicitly clear wait_mask — CFG can persist across run-types in the same session.
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE || PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    _llk_pack_(0 /*start_math_dest_tile_idx*/, block * rt_block * CT_DIM /*start_l1_tile_idx*/, output_shape, rt_block * CT_DIM);
                    // Stands in for the idle math's post, so the section done's SEMGET never underflows.
                    ckernel::trisc::t6_semaphore_post(ckernel::trisc::semaphore::MATH_PACK);
                    _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                for (std::uint32_t block = 0; block < num_blocks; block++)
                {
                    _llk_packer_wait_for_math_done_();
                    _llk_pack_(0 /*start_math_dest_tile_idx*/, block * rt_block * CT_DIM /*start_l1_tile_idx*/, output_shape, rt_block * CT_DIM);
                    _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
    uninit_zone();
}

void uninit_zone()
{
    ZONE_SCOPED("UNINIT")
}

#endif
