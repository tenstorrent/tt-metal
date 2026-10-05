// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// K splits over CUSTOM_MM_NUM_CALLS back-to-back calls that accumulate into one DEST; 1 is the single-call kernel.
// Call c reads its activation tiles after the previous call's and its own meta buffer, padded to a fixed slot.
constexpr std::uint32_t num_calls           = CUSTOM_MM_NUM_CALLS;
constexpr std::uint32_t kt_per_call         = KT_DIM / num_calls;
constexpr std::uint32_t meta_bytes_per_call = (kt_per_call * CT_DIM + 8) * sizeof(std::uint32_t);
static_assert(kt_per_call * num_calls == KT_DIM && kt_per_call % 2 == 0, "each call needs an even share of the K tiles");
static_assert(num_calls == 1 || CT_DIM > 1, "split accumulation (CT_DIM 1) finalizes per call");

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_face_compressed_mm.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_B_src,
        formats.unpack_A_src,
        formats.unpack_B_dst,
        formats.unpack_A_dst,
        params.in1_face_r_dim,
        params.in0_face_r_dim,
        params.num_faces_B,
        params.num_faces_A,
        params.TILE_SIZE_UNPACK_B,
        params.TILE_SIZE_UNPACK_A);

    if constexpr (num_calls > 1)
    {
        // A both-bank SrcB clear only corrupts the SrcA writes it overlaps while unpacker 0's last SrcA clear value is
        // non-zero, and that value outlives kernels.
        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 0, 0, p_unpacr_nop::CLR_SRC_1, p_unpacr_nop::CLR_SRC);
    }

    _llk_unpack_AB_face_compressed_mm_init_<false /* transpose */, true /* clear_src */>(params.in0_face_r_dim);

    // An activation tile is 2 faces x in0_face_r_dim rows x 16 Float16_b datums = 4 * in0_face_r_dim 16-byte words.
    for (std::uint32_t call = 0; call < num_calls; call++)
    {
        const std::uint32_t address_b = L1_ADDRESS(params.buffer_A[0]) + call * kt_per_call * 4 * params.in0_face_r_dim;
        const std::uint32_t address_meta = params.buffer_C[0] + call * meta_bytes_per_call;
        // Even calls with a successor skip their trailing context poll, so both ends of call run back to back.
        if (call % 2 == 0 && call + 1 < num_calls)
        {
            _llk_unpack_AB_face_compressed_mm_<CT_DIM, true /* finalize */, true /* chained */>(address_b, address_meta, kt_per_call);
        }
        else
        {
            _llk_unpack_AB_face_compressed_mm_<CT_DIM, true /* finalize */>(address_b, address_meta, kt_per_call);
        }
        if constexpr (CUSTOM_MM_REARM)
        {
            // A -inf SrcA clear after every call, as a max-reduce between calls would; it waits for its bank.
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1 /* Stall_Clr_Cntrl */, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
        }
    }

    _llk_unpack_AB_face_compressed_mm_uninit_(params.num_faces_B);
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_face_compressed_mm.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    _llk_math_face_compressed_mm_init_<CT_DIM>(params.in0_face_r_dim);

    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

    for (std::uint32_t call = 0; call < num_calls; call++)
    {
        if constexpr (num_calls > 1)
        {
            // Hold math back so the unpacker reaches call c + 1 while math still holds call c's banks.
            ckernel::wait(2000);
        }
        _llk_math_face_compressed_mm_<CT_DIM, true>(params.buffer_C[0] + call * meta_bytes_per_call, params.in0_face_r_dim, 0, kt_per_call);
    }

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
        formats.pack_src, formats.pack_dst, params.TILE_SIZE_PACK, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, true);

    _llk_pack_init_<PackMode::Default, false /*zero_output*/, false /*skip_addrmod_config*/, true /*skip_packer_strides*/, true /*mutex_ADC*/>(
        formats.pack_src, params.in0_face_r_dim, TILE_C_DIM, params.num_faces, 1 /*num_tiles*/, false /*skip_bh_tilize_workaround*/);
    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2);

    _llk_packer_wait_for_math_done_();

    for (std::uint32_t i = 0; i < CT_DIM; i++)
    {
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default, true /* mutex_ADC */>(i, L1_ADDRESS(params.buffer_Res[i]));
    }

    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
}

#endif
