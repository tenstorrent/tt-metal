// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Parallel FPU matmul (TRISC0-2 -> buffer_C) + isolated SrcS SFPU add (TRISC3 -> buffer_Res).
// FPU path uses Dest dvalid {FPU, PACK}. SFPU path uses UNP_S/SrcS/PACK1 only (no Dest).

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

constexpr std::uint32_t buf_desc_id_src_a = 29;
constexpr std::uint32_t buf_desc_id_src_b = 30;
constexpr std::uint32_t buf_desc_id_dst   = 31;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
    set_ttsync_enables<TRACK_ALL>(ckernel::TRISC_ID);

    buffer_descriptor_u bd_src_a = {0};
    bd_src_a.f.l1_addr_16B       = L1_ADDRESS(params.buffer_A[0]);
    bd_src_a.f.format            = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_src_a.f.lmt_addr_16B      = 0;
    bd_src_a.f.x_dim             = FACE_C_DIM;
    bd_src_a.f.y_dim             = FACE_R_DIM;
    bd_src_a.f.z_dim             = params.num_faces_A;

    buffer_descriptor_u bd_src_b = {0};
    bd_src_b.f.l1_addr_16B       = L1_ADDRESS(params.buffer_B[0]);
    bd_src_b.f.format            = static_cast<std::uint8_t>(formats.unpack_B_src);
    bd_src_b.f.lmt_addr_16B      = 0;
    bd_src_b.f.x_dim             = FACE_C_DIM;
    bd_src_b.f.y_dim             = FACE_R_DIM;
    bd_src_b.f.z_dim             = params.num_faces_B;

    _configure_buf_desc_table_(buf_desc_id_src_a, bd_src_a);
    _configure_buf_desc_table_(buf_desc_id_src_b, bd_src_b);
    _llk_unpack_configure_binary_<p_unpacr::UNP_B, p_unpacr::UNP_A>(
        static_cast<DataFormat>(formats.unpack_A_dst), static_cast<DataFormat>(formats.unpack_B_dst));

    _llk_unpack_matmul_init_<UNPACK_TRANSPOSE_FACES>(buf_desc_id_src_a, buf_desc_id_src_b, params.CT_DIM, params.RT_DIM, params.KT_DIM);

    for (std::uint32_t j = 0; j < params.KT_DIM; j++)
    {
        _llk_unpack_matmul_(params.CT_DIM, params.RT_DIM, params.KT_DIM, j, j * params.CT_DIM);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    _llk_math_srcAB_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*EN_INT32_MATH_FORMAT*/>(
        static_cast<DataFormat>(formats.math), static_cast<DataFormat>(formats.math));
    _llk_math_matmul_init_<(ckernel::MathFidelity)MATH_FIDELITY, ENABLE_DIRECT_INDEXING, ENABLE_2X_FORMAT>(params.CT_DIM, params.RT_DIM);

    for (std::uint32_t i = 0; i < params.KT_DIM; i++)
    {
        _llk_math_matmul_block_(params.CT_DIM, params.RT_DIM);
    }
    _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_srcs.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const volatile FormatConfig& formats = params.formats;
#endif

    const bool PARAM_SRCS_32BIT_MODE                = _is_srcs_32bit_mode_(static_cast<DataFormat>(formats.unpack_S_dst));
    constexpr std::uint32_t PARAM_SRCS_XDIM         = srcs_dims::XDIM;
    constexpr std::uint32_t PARAM_SRCS_ZDIM         = srcs_dims::ZDIM;
    const std::uint32_t PARAM_SRCS_YDIM             = srcs_dims::ydim(PARAM_SRCS_32BIT_MODE);
    const std::uint32_t PARAM_SRCS_SLICE_COUNT      = srcs_dims::slice_count(PARAM_SRCS_32BIT_MODE);
    constexpr std::uint32_t PARAM_SRCS_INSTRN_COUNT = 1;

    constexpr std::uint32_t buf_desc_id_unpack_0 = 0;
    constexpr std::uint32_t buf_desc_id_unpack_1 = 1;
    constexpr std::uint32_t buf_desc_id_pack     = 8;

    buffer_descriptor_u bd_unpack_0 = {0};
    buffer_descriptor_u bd_unpack_1 = {0};
    buffer_descriptor_u bd_pack     = {0};

    bd_unpack_0.f.l1_addr_16B = L1_ADDRESS(params.buffer_S[0]);
    bd_unpack_0.f.format      = static_cast<std::uint8_t>(formats.unpack_S_src);
    bd_unpack_0.f.x_dim       = PARAM_SRCS_XDIM;
    bd_unpack_0.f.y_dim       = PARAM_SRCS_YDIM;
    bd_unpack_0.f.z_dim       = PARAM_SRCS_ZDIM;
    _configure_buf_desc_table_(buf_desc_id_unpack_0, bd_unpack_0);
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(static_cast<DataFormat>(formats.unpack_S_dst));

    // FormatConfig has no unpack_T_* fields, so T is unpacked with S's format. Callers must keep
    // stimuli_T_format == stimuli_S_format or T will be read with the wrong format.
    bd_unpack_1.f.l1_addr_16B = L1_ADDRESS(params.buffer_T[0]);
    bd_unpack_1.f.format      = static_cast<std::uint8_t>(formats.unpack_S_src);
    bd_unpack_1.f.x_dim       = PARAM_SRCS_XDIM;
    bd_unpack_1.f.y_dim       = PARAM_SRCS_YDIM;
    bd_unpack_1.f.z_dim       = PARAM_SRCS_ZDIM;
    _configure_buf_desc_table_(buf_desc_id_unpack_1, bd_unpack_1);
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(static_cast<DataFormat>(formats.unpack_S_dst));

    bd_pack.f.l1_addr_16B = L1_ADDRESS(params.buffer_Res[0]);
    bd_pack.f.format      = static_cast<std::uint8_t>(formats.pack_S_dst);
    bd_pack.f.x_dim       = PARAM_SRCS_XDIM;
    bd_pack.f.y_dim       = PARAM_SRCS_YDIM;
    bd_pack.f.z_dim       = PARAM_SRCS_ZDIM;
    _configure_buf_desc_table_(buf_desc_id_pack, bd_pack);
    _llk_pack_hw_configure_<p_pacr::PACK1, false /*EN_32BIT_DEST*/>(static_cast<DataFormat>(formats.pack_S_src), ckernel::ReluConfig::none());

    cfg[DISABLE_IMPLIED_SRCS_FORMAT_ADDR32 + TRISC_ID] = !IMPLIED_MATH_FORMAT;

    const int in0_base = ckernel::math::SFPU_SRCS_BASE_ADDR;
    const int in1_base = ckernel::math::SFPU_SRCS_BASE_ADDR + PARAM_SRCS_YDIM;
    const int out_base = ckernel::math::SFPU_SRCS_BASE_ADDR + 2 * PARAM_SRCS_YDIM;

    const int num_sfpu_iterations      = PARAM_SRCS_YDIM >> 1;
    const std::uint32_t replay_buf_len = num_sfpu_iterations * 4;
    load_replay_buf( // TODO: Replace with SFPI call (#1637)
        0 /*start*/,
        replay_buf_len,
        false /*exec_while_loading*/,
        0 /*set_mutex*/,
        0 /*last*/,
        [in0_base, in1_base, out_base, num_sfpu_iterations]
        {
#pragma GCC unroll 4
            for (int d = 0; d < num_sfpu_iterations; d++)
            {
                TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in0_base + (d << 1));
                TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in1_base + (d << 1));
                TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0x0);
                TT_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, out_base + (d << 1));
            }
        });

    _llk_pack_srcs_config_for_tile_<PARAM_SRCS_INSTRN_COUNT>(PARAM_SRCS_32BIT_MODE);
    _llk_math_eltwise_sfpu_init_();

    for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
    {
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, i * PARAM_SRCS_SLICE_COUNT);

        _llk_pack_srcs_<PARAM_SRCS_INSTRN_COUNT>(buf_desc_id_pack, i * PARAM_SRCS_SLICE_COUNT);

        // No auto-loops for unpacker due to HW bug for binary unpacking. Issue #1635: https://github.com/tenstorrent/tt-llk/issues/1635
        // Preload the unpacker pipeline so that SFPU is not starved
        constexpr int preload_count = 3;
        // Guard the unsigned drain loop below: PARAM_SRCS_SLICE_COUNT - preload_count wraps if slice_count <= preload_count.
        // PARAM_SRCS_SLICE_COUNT is a runtime value (not constexpr), so this is a runtime assert.
        LLK_ASSERT(PARAM_SRCS_SLICE_COUNT > preload_count, "slice count must exceed preload_count or the drain loop bound underflows");
#pragma GCC unroll preload_count
        for (std::uint32_t j = 0; j < preload_count; j++)
        {
            TT_UNPACR2_TILE_INC(0b1 /*SrcS tile inc*/, 0b0 /*no L1 inc*/, buf_desc_id_unpack_0, 0b0 /*no dvalid*/);
            TT_UNPACR2_TILE_INC(0b0 /*no SrcS tile inc*/, 0b1 /*L1 inc*/, buf_desc_id_unpack_1, 0b1 /*Set dvalid*/);
        }

        for (std::uint32_t slice = 0; slice < PARAM_SRCS_SLICE_COUNT - preload_count; slice++)
        {
            TT_UNPACR2_TILE_INC(0b1 /*SrcS tile inc*/, 0b0 /*no L1 inc*/, buf_desc_id_unpack_0, 0b0 /*no dvalid*/);
            TT_UNPACR2_TILE_INC(0b0 /*no SrcS tile inc*/, 0b1 /*L1 inc*/, buf_desc_id_unpack_1, 0b1 /*Set dvalid*/);
            TT_REPLAY(0, replay_buf_len, 0, 0, 0, 0);
            _llk_math_eltwise_sfpu_srcs_clear_vlds_<true /*SRCS_RD_DONE*/, true /*SRCS_WR_DONE*/>();
        }

#pragma GCC unroll preload_count
        for (std::uint32_t j = 0; j < preload_count; j++)
        {
            TT_REPLAY(0, replay_buf_len, 0, 0, 0, 0);
            _llk_math_eltwise_sfpu_srcs_clear_vlds_<true /*SRCS_RD_DONE*/, true /*SRCS_WR_DONE*/>();
        }
    }

    wait_unpack_idle();
    wait_sfpu_idle();
    wait_pack_idle();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_matmul.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});

    buffer_descriptor_u bd_dst = {0};
    bd_dst.f.l1_addr_16B       = L1_ADDRESS(params.buffer_C[0]);
    bd_dst.f.lmt_addr_16B      = 0;
    bd_dst.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_dst.f.x_dim             = FACE_C_DIM;
    bd_dst.f.y_dim             = FACE_R_DIM;
    bd_dst.f.z_dim             = params.num_faces;

    _configure_buf_desc_table_(buf_desc_id_dst, bd_dst);
    _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
    _llk_pack_matmul_init_(buf_desc_id_dst, params.RT_DIM, params.CT_DIM, 1 /*num_subblocks_c_dim*/);

    _llk_pack_matmul_(0 /*start_math_dest_tile_idx*/, 0 /*start_l1_tile_idx*/);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}

#endif
