// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The compute API's constant-output reconfigure (pack_api_shim.h) of one output, before and after every other writer of the
// packer fields it writes (Blackhole); the steps (RL_STEPS) are named in test_pack_reconfig_live.py. A reconfigure step ('R')
// packs its own four tiles of the input to its own four result tiles; the writer steps pack nothing.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
constexpr std::uint32_t NT = 4;
} // namespace

#ifdef LLK_TRISC_UNPACK

#include "llk_fp32_dest_acc.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const Operand& buffer_A = params.buffer_A;
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4), formats.unpack_A_src, formats.unpack_A_dst);
    std::uint32_t r = 0;
    for (int s = 0; s < RL_NSTEPS; s++)
    {
        const char step = RL_STEPS[s];
        if (step == 'R')
        {
            for (std::uint32_t t = 0; t < NT; t++)
            {
                _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
                    L1_ADDRESS(buffer_A[r * NT + t]), formats.unpack_A_src, formats.unpack_A_dst);
            }
            r++;
        }
        else if (step == 'M')
        {
            // the compute API's reconfig_data_format to another input and back (unpack side)
            constexpr std::uint32_t alt = ckernel::to_underlying(DataFormat::Float16);
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE>(alt, alt, 2048);
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE>(
                formats.unpack_A_src, formats.unpack_A_dst, SCALE_DATUM_SIZE(formats.unpack_A_src, TILE_C_DIM * TILE_R_DIM));
        }
        else if (step == 'S')
        {
            if constexpr (is_fp32_dest_acc_en)
            {
                _llk_set_fp32_dest_acc_<ThreadId::UnpackThreadId>(false);
                _llk_set_fp32_dest_acc_<ThreadId::UnpackThreadId>(true);
            }
        }
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_fp32_dest_acc.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        ckernel::DEFAULT_TENSOR_SHAPE, formats.math);
    for (int s = 0; s < RL_NSTEPS; s++)
    {
        const char step = RL_STEPS[s];
        if (step == 'R')
        {
            _llk_math_wait_for_dest_available_<dest_sync>();
            for (std::uint32_t t = 0; t < NT; t++)
            {
                _llk_math_eltwise_unary_datacopy_wrapper_<DataCopyType::A2D, dest_sync, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
                    t, formats.math, formats.math, ckernel::DEFAULT_TENSOR_SHAPE);
            }
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
        else if (step == 'M')
        {
            // the compute API's reconfig_data_format to another input and back (math side)
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en>(ckernel::to_underlying(DataFormat::Float16));
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en>(formats.math);
        }
        else if (step == 'S')
        {
            if constexpr (is_fp32_dest_acc_en)
            {
                _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(false);
                _llk_set_fp32_dest_acc_<ThreadId::MathThreadId>(true);
            }
        }
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "experimental/llk_fast_untilize_common.h"
#include "experimental/llk_pack_fast_tilize.h"
#include "experimental/llk_pack_fast_untilize.h"
#include "llk_fp32_dest_acc.h"
#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "llk_pack_rows.h"
#include "llk_pack_untilize.h"
#include "pack_api_shim.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(ARCH_BLACKHOLE)
    const Operand& buffer_Res = params.buffer_Res;
    constexpr std::uint32_t cb  = PACK_API_SHIM_OUT;
    const std::uint32_t page    = (L1_ADDRESS(buffer_Res[1]) - L1_ADDRESS(buffer_Res[0])) << 4;
    const std::uint32_t src     = formats.pack_src;
    const std::uint32_t dst     = formats.pack_dst;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(src, dst, SCALE_DATUM_SIZE(dst, TILE_C_DIM * TILE_R_DIM));
    pack_api_shim::cb_bind(cb, L1_ADDRESS(buffer_Res[0]), page / 16, RL_NR * NT);
    pack_api_shim::pack_init(cb);
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();

    std::uint32_t r = 0;
    for (int s = 0; s < RL_NSTEPS; s++)
    {
        switch (RL_STEPS[s])
        {
            case 'R':
            {
                _llk_packer_wait_for_math_done_();
#if PACK_API_SHIM_PR
                pack_api_shim::pack_reconfig_data_format<is_fp32_dest_acc_en, cb>();
#else
                pack_api_shim::pack_reconfig_data_format<is_fp32_dest_acc_en>(cb);
#endif
                // the default program back after another pack program, the strides left to the reconfigure
                _llk_pack_init_<PackMode::Default, false, false, true>(src, FACE_R_DIM, TILE_C_DIM, 4, 1, false);
                for (std::uint32_t t = 0; t < NT; t++)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en, PackMode::Default>(t, L1_ADDRESS(buffer_Res[r * NT + t]));
                }
                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                r++;
                break;
            }
            case 'U': // pack untilize
                _llk_pack_untilize_init_wrapper_<NT, NT>(src, dst, FACE_R_DIM, 4);
                _llk_init_packer_dest_offset_registers_<dest_sync>();
                _llk_pack_untilize_uninit_wrapper_(src, FACE_R_DIM);
                _llk_init_packer_dest_offset_registers_<dest_sync>();
                break;
            case 'F': // fast untilize, its pack uninit called
                ckernel::_llk_pack_fast_untilize_init_<ckernel::FAST_UNTILIZE_MAX_UNIT_DIM, NT>(src, dst);
                ckernel::_llk_pack_fast_untilize_uninit_(src);
                _llk_init_packer_dest_offset_registers_<dest_sync>();
                break;
            case 'T': // fast tilize
                _llk_pack_fast_tilize_init_<dest_sync, is_fp32_dest_acc_en>(is_fp32_dest_acc_en, dst, 2, 4, src);
                _llk_pack_fast_tilize_uninit_<dest_sync, is_fp32_dest_acc_en>(dst, FACE_R_DIM, 4, src);
                break;
            case 'W': // pack_rows
                _llk_pack_rows_init_(16);
                _llk_pack_rows_uninit_();
                break;
            case 'S': // the run-time FP32 DEST switch, off and back on (compute API: llk_pack_wait_fp32_dest_acc)
                if constexpr (is_fp32_dest_acc_en)
                {
                    pack_api_shim::pack_wait_fp32_dest_acc(false);
                    pack_api_shim::pack_wait_fp32_dest_acc(true);
                }
                break;
            case 'Z': // the experimental compute API's direct W stride writes (sdpa.h, face_compressed_mm.h): halve, then restore
#ifdef LLK_PACK_LIVE_OUTPUT
                ckernel::packer::llk_pack_live_forget();
#endif
                cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>((TILE_NUM_FACES / 2) * FACE_C_DIM * FACE_R_DIM * 2);
#ifdef LLK_PACK_LIVE_OUTPUT
                ckernel::packer::llk_pack_live_forget();
#endif
                cfg_reg_rmw_tensix<PCK0_ADDR_CTRL_ZW_REG_0_Wstride_RMW>(TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * 2);
                break;
            case 'I': // a default pack init with its strides
                _llk_pack_init_<PackMode::Default, false, false, false>(src, FACE_R_DIM, TILE_C_DIM, 4, 1, false);
                break;
            default: // 'M': unpack and math only
                break;
        }
    }
#endif
}

#endif
