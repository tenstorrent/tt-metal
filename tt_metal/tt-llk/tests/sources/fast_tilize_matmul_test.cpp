// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Minimal tilize→matmul repro for BH fast-tilize conv2d PCC issue.
// Flow: fast_tilize(KT_DIM tiles of activation) → matmul(activation × weight) → pack result

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
std::uint32_t tile_size                = 128;

// buffer_A = row-major activation (KT_DIM tiles wide, 1 tile tall)
// buffer_B = weights (KT_DIM tiles, already tilized)
// buffer_Res[0..KT_DIM-1] = tilized activation (intermediate)
// buffer_Res[KT_DIM] = matmul output (1 tile)

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t KT_DIM = params.BLOCK_CT_DIM;

    // Initialize semaphores — needed on ttsim where firmware doesn't pre-init
    TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_0); // sem 0: FPU_SFPU datacopy

    // === Phase 0: hw_configure ===
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);

    // === Phase 1: Fast-tilize (activation row-major → tilized) ===
    // Base address is programmed inside _llk_unpack_fast_tilize_block_ via
    // _llk_unpack_configure_single_address_ (respects current cfg context).
    _llk_unpack_fast_tilize_init_(formats.unpack_A_dst, KT_DIM, KT_DIM <= 1 ? 1 : 4);
    _llk_unpack_fast_tilize_reinit_xdim_(KT_DIM);
    _llk_unpack_fast_tilize_block_(L1_ADDRESS(params.buffer_A[0]), 0, formats.unpack_A_src, KT_DIM, 1, KT_DIM, 4, 0);

    _llk_unpack_fast_tilize_uninit_<is_fp32_dest_acc_en>();

    // Wait for tilize pack done before starting matmul
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);
    t6_semaphore_get<>(semaphore::PACK_DONE);

    // === Phase 2: Matmul (tilized activation × weights) ===
    _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, false>(formats.unpack_A_src, formats.unpack_A_dst, tile_size);
    _llk_unpack_reconfig_data_format_srcb_impl_<is_fp32_dest_acc_en, false>(formats.unpack_B_src, formats.unpack_B_dst, tile_size);
#ifdef ARCH_BLACKHOLE
    _llk_unpack_tilize_uninit_(formats.unpack_A_dst, 4, FACE_R_DIM);
#else
    _llk_unpack_tilize_uninit_(formats.unpack_A_dst, FACE_R_DIM);
#endif
    _llk_unpack_AB_matmul_init_<>(0, 1, 1, KT_DIM, FACE_R_DIM, FACE_R_DIM, 4, 4, false, false);

    for (std::uint32_t k = 0; k < KT_DIM; k++)
    {
        _llk_unpack_AB_matmul_<>(
            L1_ADDRESS(params.buffer_Res[0]), // tilized activations → base_address_a
            L1_ADDRESS(params.buffer_B[0]),   // weights → base_address_b
            k,
            k,
            128, // tile_size_a: 32x32 bf16 tile = 2048 bytes = 128 × 16B
            128, // tile_size_b: same
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

    // Initialize semaphores — needed on ttsim where firmware doesn't pre-init
    TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_0); // sem 0: FPU_SFPU datacopy

    // === Phase 0: sync_init + hw_configure ===
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    // === Phase 1: Fast-tilize math ===
    _llk_math_fast_tilize_init_<is_fp32_dest_acc_en>(formats.math, 4);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_fast_tilize_block_<is_fp32_dest_acc_en>(0, formats.math, 4, 1, 4);
    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Uninit: restore math state for matmul
    _llk_math_fast_tilize_uninit_<is_fp32_dest_acc_en>(formats.math);

    // === Phase 2: Matmul ===
    _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(formats.math);
    _llk_math_reconfig_data_format_srcb_<is_fp32_dest_acc_en, false>(formats.math);
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
#include "llk_pack_fast_tilize.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t KT_DIM = params.BLOCK_CT_DIM;

    // Initialize semaphores — needed on ttsim where firmware doesn't pre-init
    TTI_SEMINIT(1, 0, p_stall::SEMAPHORE_0);      // sem 0: FPU_SFPU datacopy
    TTI_SEMINIT(2, 0, 1 << semaphore::PACK_DONE); // sem 4: pack→unpack barrier

    // === Phase 0: pack dest_init + hw_configure ===
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(
        formats.pack_src, formats.pack_dst, SCALE_DATUM_SIZE(formats.pack_dst, TILE_C_DIM * TILE_R_DIM), FACE_R_DIM, TILE_C_DIM, 4, false);

    // === Phase 1: Fast-tilize pack ===
    _llk_pack_fast_tilize_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>(0, formats.pack_dst, KT_DIM, 4);

    _llk_packer_wait_for_math_done_();
    _llk_pack_fast_tilize_block_(0, L1_ADDRESS(params.buffer_Res[0]), KT_DIM, 1, 4);
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Uninit: restore pack state for matmul
    _llk_pack_fast_tilize_uninit_<DstSync::SyncHalf, is_fp32_dest_acc_en>(formats.pack_dst, FACE_R_DIM, 4);

    // Signal unpack that phase 1 pack is done
    t6_semaphore_post<>(semaphore::PACK_DONE);

    // Re-init pack for standard (non-tilize) mode
    _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, tile_size);
#ifdef ARCH_BLACKHOLE
    _llk_pack_init_<false, false, false>(formats.pack_dst);
#endif

    // === Phase 2: Matmul pack ===
    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(0, L1_ADDRESS(params.buffer_Res[KT_DIM]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
