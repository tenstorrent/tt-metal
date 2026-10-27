// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reduction kernel for the face_compressed_mm split-K call-boundary hang. Same structure as
// matmul_face_compressed_test.cpp's multi-call path, with timing knobs (FCMM_PERTURB) and
// kernel-level reduction modes (FCMM_KERNEL_MODE).

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

constexpr std::uint32_t num_calls           = CUSTOM_MM_NUM_CALLS;
constexpr std::uint32_t kt_per_call         = KT_DIM / num_calls;
constexpr std::uint32_t meta_words_per_call = CUSTOM_MM_META_WORDS_PER_CALL;
static_assert(kt_per_call * num_calls == KT_DIM && kt_per_call % 2 == 0, "each call needs an even share of the K tiles");
static_assert(num_calls == 1 || meta_words_per_call > 0, "a multi-call kernel needs the per-call meta stride");

#define FCMM_NOPS(n)                                                         \
    do                                                                       \
    {                                                                        \
        _Pragma("GCC unroll 256") for (std::uint32_t _i = 0; _i < (n); ++_i) \
        {                                                                    \
            TTI_NOP;                                                         \
        }                                                                    \
    } while (0)

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_AB_face_compressed_mm.h"
#include "llk_unpack_common.h"

// Kernel-level entry barriers under test (modes 3/8/9 live in the LLK header):
//   1  non-clearing UNPACR_NOP that waits like an UNPACR (own-bank wait), no CLR_SRC
//   2  STALLWAIT: hold THCON/CFG/UNPACK/SYNC instructions until both unpackers are idle
#if FCMM_KERNEL_MODE == 1
#define FCMM_ENTRY_BARRIER() TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_NOP)
#elif FCMM_KERNEL_MODE == 2
#define FCMM_ENTRY_BARRIER() TTI_STALLWAIT(p_stall::STALL_THCON | p_stall::STALL_CFG | p_stall::STALL_UNPACK | p_stall::STALL_SYNC, p_stall::UNPACK)
#else
#define FCMM_ENTRY_BARRIER() \
    do                       \
    {                        \
    } while (0)
#endif

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

    auto address_b    = [&params](std::uint32_t call) { return L1_ADDRESS(params.buffer_A[0]) + call * kt_per_call * 4 * params.in0_face_r_dim; };
    auto address_meta = [&params](std::uint32_t call) { return params.buffer_C[0] + call * meta_words_per_call * sizeof(std::uint32_t); };

    if constexpr (num_calls == 1)
    {
        _llk_unpack_AB_face_compressed_mm_<CT_DIM, true /*clear_src*/, true /*finalize*/>(address_b(0), address_meta(0), KT_DIM);
    }
#if FCMM_KERNEL_MODE >= 30
    else
    {
        // Fully hand-rolled: call 0's entry writes copied from the LLK execute (ctx bases, Y offsets, SrcB
        // base), then raw UNPACR streams: B, FCMM_NFACES faces in ctx 1 (bfp4), [NOPs], FCMM_NFACES faces in
        // ctx 0 (bfp2). No semaphores, no finalize; the math side (mode 30) issues one MVMUL per face.
        static_assert(num_calls == 2, "hand-rolled modes need exactly two calls");
        volatile std::uint32_t* cfg       = get_cfg_pointer();
        const std::uint32_t* pre_meta_ptr = reinterpret_cast<std::uint32_t*>(address_meta(0)) + (kt_per_call * CT_DIM + 4) / 5;
        TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
        cfg[THCON_SEC0_REG3_Base_address_ADDR32]       = pre_meta_ptr[1] & 0x00FFFFFF;
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = pre_meta_ptr[2] & 0x00FFFFFF;
        t6_mutex_acquire(mutex::THREAD2_ADC);
        TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::Y));
        TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
        TT_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, p_setadc::THREAD_OVRD_MATH, pre_meta_ptr[1] >> 24);
        TT_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, p_setadc::THREAD_OVRD_PACK, pre_meta_ptr[2] >> 24);
        t6_mutex_release(mutex::THREAD2_ADC);
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b(0);
#if FCMM_KERNEL_MODE == 33
        // both contexts bfp4 (format identical, only the context / counter set switches)
        cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx0_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp4_b));
        cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx0_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp4_b));
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = pre_meta_ptr[2] & 0x00FFFFFF;
        t6_mutex_acquire(mutex::THREAD2_ADC);
        TT_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, p_setadc::THREAD_OVRD_MATH, pre_meta_ptr[2] >> 24);
        t6_mutex_release(mutex::THREAD2_ADC);
#endif
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        TTI_UNPACR_COMMON(SrcB, 0b00'00'10'10, 1); // B
        for (std::uint32_t f = 0; f < FCMM_NFACES; ++f)
        {
#if FCMM_KERNEL_MODE == 38
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00, 0, 1, 1); // '0' first: the other direction
#else
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00, 1, 2, 1); // '1' bfp4
#endif
        }
        FCMM_NOPS(FCMM_UNP_ENTRY_NOPS);
        if constexpr (FCMM_UNP_WAIT > 0)
        {
            ckernel::wait(FCMM_UNP_WAIT); // RISC-side gap; the Tensix stream is empty meanwhile
        }
#if FCMM_KERNEL_MODE == 31
        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC); // the S the LLK would emit in-call
#elif FCMM_KERNEL_MODE == 32
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK); // drain only, no clear
#endif
        for (std::uint32_t f = 0; f < FCMM_NFACES; ++f)
        {
#if FCMM_KERNEL_MODE == 35
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00, 0, 2, 1); // ctx0 (bfp2) with the bfp4 counter set
#elif FCMM_KERNEL_MODE == 36
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00, 1, 1, 1); // ctx1 (bfp4) with the bfp2 counter set
#elif FCMM_KERNEL_MODE == 37 || FCMM_KERNEL_MODE == 38
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00, 1, 2, 1); // '1': no switch (37) or the 0 -> 1 direction (38)
#else
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00, 0, 1, 1); // '0' bfp2
#endif
        }
        // drain before the uninit's cfg / counter writes (the LLK's finalize block does this for the real op)
        TTI_STALLWAIT(p_stall::STALL_THCON | p_stall::STALL_CFG | p_stall::STALL_UNPACK | p_stall::STALL_SYNC, p_stall::UNPACK);
    }
#elif FCMM_KERNEL_MODE >= 20
    else
    {
        // Hand-rolled second call: call 0 through the LLK (sets every context base / Y offset), then the raw
        // UNPACR stream of a 4-face call with no entry writes and no handshake. FCMM_CALL1_CTX picks the face
        // context (0 bfp2, 1 bfp4). Modes: 20 = B then faces (as the LLK emits), 21 = faces then B,
        // 23 = CLR_SRC stall, B, faces, 24 = B, CLR_SRC stall, faces, 25 = B then faces, no finalize.
        static_assert(num_calls == 2, "hand-rolled modes need exactly two calls");
        TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
        _llk_unpack_AB_face_compressed_mm_<CT_DIM, false /*clear_src*/, false /*finalize*/>(address_b(0), address_meta(0), kt_per_call);
        FCMM_NOPS(FCMM_UNP_ENTRY_NOPS);
#define FCMM_FACE()                                                                                 \
    do                                                                                              \
    {                                                                                               \
        if constexpr (FCMM_CALL1_CTX == 0)                                                          \
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00, 0, 1, 1); /* '0' */ \
        else                                                                                        \
            TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00, 1, 2, 1); /* '1' */ \
    } while (0)
#define FCMM_B() TTI_UNPACR_COMMON(SrcB, 0b00'00'10'10, 1)
#define FCMM_S() TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC)
#if FCMM_KERNEL_MODE == 20 || FCMM_KERNEL_MODE == 25
        FCMM_B();
        FCMM_FACE();
        FCMM_FACE();
        FCMM_FACE();
        FCMM_FACE();
#elif FCMM_KERNEL_MODE == 21
        FCMM_FACE();
        FCMM_FACE();
        FCMM_FACE();
        FCMM_FACE();
        FCMM_B();
#elif FCMM_KERNEL_MODE == 23
        FCMM_S();
        FCMM_B();
        FCMM_FACE();
        FCMM_FACE();
        FCMM_FACE();
        FCMM_FACE();
#elif FCMM_KERNEL_MODE == 24
        FCMM_B();
        FCMM_S();
        FCMM_FACE();
        FCMM_FACE();
        FCMM_FACE();
        FCMM_FACE();
#endif
#if FCMM_KERNEL_MODE != 25
        // the LLK's ct_dim == 1 finalize: both sources zeroed and valid for the math merge
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
#endif
    }
#else
    else
    {
        TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
        for (std::uint32_t call = 0; call + 1 < num_calls; call++)
        {
            if (call > 0)
            {
#if FCMM_UNP_ENTRY_POS == 0
                FCMM_NOPS(FCMM_UNP_ENTRY_NOPS);
#endif
#if FCMM_ENTRY_STALL
                TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::CLR_SRC);
#endif
                FCMM_ENTRY_BARRIER();
            }
            _llk_unpack_AB_face_compressed_mm_<CT_DIM, false /*clear_src*/, false /*finalize*/>(address_b(call), address_meta(call), kt_per_call);
#if FCMM_UNP_ENTRY_POS == 1
            FCMM_NOPS(FCMM_UNP_ENTRY_NOPS);
#endif
        }
#if FCMM_UNP_ENTRY_POS == 0
        FCMM_NOPS(FCMM_UNP_ENTRY_NOPS);
#endif
#if FCMM_ENTRY_STALL
        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::CLR_SRC);
#endif
        FCMM_ENTRY_BARRIER();
        _llk_unpack_AB_face_compressed_mm_<CT_DIM, false /*clear_src*/, true /*finalize*/>(address_b(num_calls - 1), address_meta(num_calls - 1), kt_per_call);
    }
#endif

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
#if FCMM_KERNEL_MODE >= 30
    else
    {
        // one MVMUL per face, SrcA cleared per face, SrcB held (never cleared), no finalize
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
        const std::uint32_t single_row = (params.in0_face_r_dim == 1) ? 1 : 0;
        for (std::uint32_t call = 0; call < 2; call++)
        {
            if constexpr (FCMM_MATH_HOLD > 0)
            {
                ckernel::wait(FCMM_MATH_HOLD);
            }
            for (std::uint32_t f = 0; f < FCMM_NFACES; ++f)
            {
                if (call == 1 && f + 1 == FCMM_NFACES)
                {
                    TT_MVMUL(p_setrwc::CLR_AB, single_row, ADDR_MOD_0, 0); // release the SrcB bank at the very end
                }
                else
                {
                    TT_MVMUL(p_setrwc::CLR_A, single_row, ADDR_MOD_0, 0);
                }
            }
        }
    }
#else
    else
    {
        for (std::uint32_t call = 0; call < num_calls; call++)
        {
            if constexpr (FCMM_MATH_HOLD > 0)
            {
                ckernel::wait(FCMM_MATH_HOLD);
            }
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
#endif

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
