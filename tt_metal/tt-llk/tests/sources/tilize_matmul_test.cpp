// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Standard tilize→matmul baseline test.
// Flow: standard_tilize(kt_dim tiles of row-major activation) → pack to L1
//       → matmul(tilized_act × weights) → pack result
// Purpose: verify test infrastructure (buffer layout, golden, matmul params)
//          before swapping in fast-tilize.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// buffer_A = row-major activation (kt_dim tiles wide, 1 tile tall)
// buffer_B = weights (kt_dim tiles, already tilized)
// buffer_Res[0..kt_dim-1] = tilized activation intermediate
// buffer_Res[kt_dim] = matmul output (1 tile)

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t KT_DIM = params.BLOCK_CT_DIM;

    // Initialize FPU_SFPU semaphore (sem 0) — needed on ttsim for standard tilize datacopy
    TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_0);

    // === Phase 0: hw_configure ===
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);

    // === Phase 1: Standard tilize (activation row-major → tilized) ===
    _llk_unpack_tilize_init_(formats.unpack_A_src, formats.unpack_A_dst, 0, FACE_R_DIM, false);

    for (std::uint32_t t = 0; t < KT_DIM; t++)
    {
        _llk_unpack_tilize_(L1_ADDRESS(params.buffer_A[0]), t, formats.unpack_A_src, formats.unpack_A_dst, 0, FACE_R_DIM, 4, false);
    }

    // Wait for pack to finish writing tilized tiles to L1
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    // === Phase 2: Matmul (tilized activation × weights) ===
#ifdef ARCH_BLACKHOLE
    _llk_unpack_tilize_uninit_(formats.unpack_A_dst, 4, FACE_R_DIM);
#else
    _llk_unpack_tilize_uninit_(formats.unpack_A_dst, FACE_R_DIM);
#endif

    _llk_unpack_AB_matmul_init_<>(0, 1, 1, KT_DIM, FACE_R_DIM, FACE_R_DIM, 4, 4, false, false);

    for (std::uint32_t k = 0; k < KT_DIM; k++)
    {
        _llk_unpack_AB_matmul_<>(
            L1_ADDRESS(params.buffer_Res[0]), // tilized activations (base_address_a)
            L1_ADDRESS(params.buffer_B[0]),   // weights (base_address_b)
            k,
            k,
            params.TILE_SIZE_UNPACK_A, // tile_size_a
            params.TILE_SIZE_UNPACK_B, // tile_size_b
            false,
            false,
            1,
            1,
            KT_DIM);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_matmul.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t KT_DIM = params.BLOCK_CT_DIM;

    // === Phase 0: init (order matches matmul_unpack_tilize_test: datacopy_init first) ===
    // Initialize FPU_SFPU semaphore — needed by tilize datacopy MOP on BH
    TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_0);

#ifdef ARCH_BLACKHOLE
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, true, false>(4, formats.math);
#else
    _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(4, formats.math);
#endif
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    for (std::uint32_t t = 0; t < KT_DIM; t++)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, false>(0, formats.math, formats.math);
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }

    // === Phase 2: Matmul ===
    _llk_math_matmul_init_<MathFidelity::HiFi4, 0>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, 0, 1, 1);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    for (std::uint32_t k = 0; k < KT_DIM; k++)
    {
        _llk_math_matmul_<MathFidelity::HiFi4, 0>(0, 1, 1);
    }
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t KT_DIM = params.BLOCK_CT_DIM;

    // Initialize FPU_SFPU semaphore (sem 0) — needed on ttsim for standard tilize datacopy
    TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_0);

    // === Phase 0: pack init (tilize mode on BH) ===
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#ifdef ARCH_BLACKHOLE
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, true>(
        formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM), FACE_R_DIM, TILE_C_DIM, 4, false);
    _llk_pack_init_<false, false, true>(formats.pack_dst);
#else
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(
        formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM), FACE_R_DIM, TILE_C_DIM, 4, false);
    _llk_pack_init_<false, false>(formats.pack_dst);
#endif

    // === Phase 1: Pack tilized tiles to L1 (buffer_Res[0..KT_DIM-1]) ===
    for (std::uint32_t t = 0; t < KT_DIM; t++)
    {
        _llk_packer_wait_for_math_done_();
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(0, L1_ADDRESS(params.buffer_Res[t]));
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }

    // Signal unpack that tilized data is in L1
    t6_semaphore_post<>(semaphore::PACK_DONE);

    // === Phase 2: Reinit pack for standard (non-tilize) mode, then pack matmul result ===
#ifdef ARCH_BLACKHOLE
    _llk_pack_init_<false, false, false>(formats.pack_src, formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4, false, false, 1);
#endif

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(0, L1_ADDRESS(params.buffer_Res[KT_DIM]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
