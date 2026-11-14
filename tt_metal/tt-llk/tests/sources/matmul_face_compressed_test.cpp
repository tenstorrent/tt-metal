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
// More than one call is for test_matmul_face_compressed_multi_call, which exercises the call boundary: math is held
// back before each call so the unpacker enters call c + 1 while call c's last faces are still in flight, and the
// per-call face formats put the bfp2 <-> bfp4 switch on that boundary. The result must not depend on that timing.
// Call c's activation tiles follow the previous call's, its meta stream starts CUSTOM_MM_META_WORDS_PER_CALL words
// after the previous one (the test computes that stride and pads every call's meta to it), and its weight faces
// are addressed absolutely by the address words inside its own meta.
constexpr std::uint32_t num_calls           = CUSTOM_MM_NUM_CALLS;
constexpr std::uint32_t kt_per_call         = KT_DIM / num_calls;
constexpr std::uint32_t meta_words_per_call = CUSTOM_MM_META_WORDS_PER_CALL;
static_assert(kt_per_call * num_calls == KT_DIM && kt_per_call % 2 == 0, "each call needs an even share of the K tiles");
static_assert(num_calls == 1 || meta_words_per_call > 0, "a multi-call kernel needs the per-call meta stride");
// Cycles math waits before each call so the unpacker runs ahead and reaches the next call's first face while math
// still holds the previous call's last banks.
constexpr std::uint32_t math_hold_back_cycles = 2000;

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

    _llk_unpack_AB_face_compressed_mm_init_<false>(params.in0_face_r_dim);

    // An activation tile is 2 faces x in0_face_r_dim rows x 16 Float16_b datums = 4 * in0_face_r_dim 16-byte words.
    auto address_b    = [&params](std::uint32_t call) { return L1_ADDRESS(params.buffer_A[0]) + call * kt_per_call * 4 * params.in0_face_r_dim; };
    auto address_meta = [&params](std::uint32_t call) { return params.buffer_C[0] + call * meta_words_per_call * sizeof(std::uint32_t); };

    if constexpr (num_calls == 1)
    {
        _llk_unpack_AB_face_compressed_mm_<CT_DIM, true /*clear_src*/, true /*finalize*/>(address_b(0), address_meta(0), KT_DIM);
    }
    else
    {
        // SrcB is cleared once, ahead of the first call, with the instruction clear_src would issue: the activation
        // fills the same rows in every call. Issued here rather than through clear_src so the loop needs only two
        // instantiations of the execute, which is what fits the unpack code region. Only the last call finalizes, so
        // a ct_dim == 1 split accumulation is merged once, after all of K.
        TTI_UNPACR_NOP(
            SrcB,
            0 /*Stream_Id*/,
            0 /*Msg_Clr_Cnt*/,
            0 /*Set_Dvalid*/,
            0 /*Clr_to1_fmt_Ctrl*/,
            0 /*Stall_Clr_Cntrl*/,
            1 /*Bank_Clr_Ctrl*/,
            0 /*Src_ClrVal_Ctrl*/,
            p_unpacr_nop::CLR_SRC);
        for (std::uint32_t call = 0; call + 1 < num_calls; call++)
        {
            _llk_unpack_AB_face_compressed_mm_<CT_DIM, false /*clear_src*/, false /*finalize*/>(address_b(call), address_meta(call), kt_per_call);
        }
        _llk_unpack_AB_face_compressed_mm_<CT_DIM, false /*clear_src*/, true /*finalize*/>(address_b(num_calls - 1), address_meta(num_calls - 1), kt_per_call);
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

    if constexpr (num_calls == 1)
    {
        _llk_math_face_compressed_mm_<CT_DIM, true /*finalize*/>(params.buffer_C[0], params.in0_face_r_dim, 0 /*dst_index*/, KT_DIM);
    }
    else
    {
        for (std::uint32_t call = 0; call < num_calls; call++)
        {
            // Hold math back before each call so the unpacker runs ahead and reaches call c + 1 while math still holds
            // call c's last banks: call c's last face is then still waiting for its bank when call c + 1's first face
            // arrives in the other format. Without the delay math keeps up and the boundary is not contended.
            ckernel::wait(math_hold_back_cycles);
            const std::uint32_t address_meta = params.buffer_C[0] + call * meta_words_per_call * sizeof(std::uint32_t);
            if (call + 1 < num_calls)
            {
                _llk_math_face_compressed_mm_<CT_DIM, false /*finalize*/>(address_meta, params.in0_face_r_dim, 0 /*dst_index*/, kt_per_call);
            }
            else
            {
                _llk_math_face_compressed_mm_<CT_DIM, true /*finalize*/>(address_meta, params.in0_face_r_dim, 0 /*dst_index*/, kt_per_call);
            }
        }
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
