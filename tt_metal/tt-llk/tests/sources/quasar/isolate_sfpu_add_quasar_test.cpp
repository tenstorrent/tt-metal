// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Isolated SFPU add (binary): UNPACK2 (UNP_S) x2 -> SrcS -> SFPU -> PACK1 -> L1.
// Two input operands are unpacked into adjacent SrcS slices (slice 0 and slice 1),
// added by the SFPU, and the result is stored to slice 2 then packed to L1.
// All logic runs in LLK_TRISC_ISOLATE_SFPU; UNPACK, MATH, PACK are stubbed.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

#ifdef LLK_TRISC_UNPACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
}

#endif

#ifdef LLK_TRISC_MATH

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

#include "cfg_defines.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
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
    const std::uint32_t num_tiles = params.TILE_CNT;

    // -------------------------------------------------------------------------
    // Data format inference and dimensions
    // -------------------------------------------------------------------------

    const bool PARAM_SRCS_32BIT_MODE                = _is_srcs_32bit_mode_(static_cast<DataFormat>(formats.unpack_S_dst));
    constexpr std::uint32_t PARAM_SRCS_XDIM         = srcs_dims::XDIM;
    constexpr std::uint32_t PARAM_SRCS_ZDIM         = srcs_dims::ZDIM;
    const std::uint32_t PARAM_SRCS_YDIM             = srcs_dims::ydim(PARAM_SRCS_32BIT_MODE);
    const std::uint32_t PARAM_SRCS_SLICE_COUNT      = srcs_dims::slice_count(PARAM_SRCS_32BIT_MODE);
    constexpr std::uint32_t PARAM_SRCS_INSTRN_COUNT = 1;

    // -------------------------------------------------------------------------
    // Buffer descriptors and HW setup
    // -------------------------------------------------------------------------

    constexpr std::uint32_t buf_desc_id_unpack_0 = 0; // First input operand (buffer_A)
    constexpr std::uint32_t buf_desc_id_unpack_1 = 1; // Second input operand (buffer_B)
    constexpr std::uint32_t buf_desc_id_pack     = 8;

    buffer_descriptor_u bd_unpack_0 = {0};
    tdma_descriptor_t td_unpack_0;
    buffer_descriptor_u bd_unpack_1 = {0};
    tdma_descriptor_t td_unpack_1;
    buffer_descriptor_u bd_pack = {0};
    tdma_descriptor_t td_pack;

    // Unpack BD 0: L1 input A -> SrcS slice 0
    bd_unpack_0.f.l1_addr_16B   = L1_ADDRESS(params.buffer_A[0]);
    bd_unpack_0.f.format        = static_cast<std::uint8_t>(formats.unpack_S_src);
    bd_unpack_0.f.x_dim         = PARAM_SRCS_XDIM;
    bd_unpack_0.f.y_dim         = PARAM_SRCS_YDIM;
    bd_unpack_0.f.z_dim         = PARAM_SRCS_ZDIM;
    td_unpack_0.buf_desc        = bd_unpack_0;
    td_unpack_0.buf_desc_id     = buf_desc_id_unpack_0;
    td_unpack_0.reg_data_format = static_cast<std::uint8_t>(formats.unpack_S_dst);
    _configure_buf_desc_table_(td_unpack_0.buf_desc_id, td_unpack_0.buf_desc);
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(td_unpack_0);

    // Unpack BD 1: L1 input B -> SrcS slice 1
    bd_unpack_1.f.l1_addr_16B   = L1_ADDRESS(params.buffer_B[0]);
    bd_unpack_1.f.format        = static_cast<std::uint8_t>(formats.unpack_S_src);
    bd_unpack_1.f.x_dim         = PARAM_SRCS_XDIM;
    bd_unpack_1.f.y_dim         = PARAM_SRCS_YDIM;
    bd_unpack_1.f.z_dim         = PARAM_SRCS_ZDIM;
    td_unpack_1.buf_desc        = bd_unpack_1;
    td_unpack_1.buf_desc_id     = buf_desc_id_unpack_1;
    td_unpack_1.reg_data_format = static_cast<std::uint8_t>(formats.unpack_S_dst);
    _configure_buf_desc_table_(td_unpack_1.buf_desc_id, td_unpack_1.buf_desc);
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(td_unpack_1);

    // Pack BD: SrcS slice 2 -> L1 output
    bd_pack.f.l1_addr_16B   = L1_ADDRESS(params.buffer_Res[0]);
    bd_pack.f.format        = static_cast<std::uint8_t>(formats.pack_S_dst);
    bd_pack.f.x_dim         = PARAM_SRCS_XDIM;
    bd_pack.f.y_dim         = PARAM_SRCS_YDIM;
    bd_pack.f.z_dim         = PARAM_SRCS_ZDIM;
    td_pack.buf_desc        = bd_pack;
    td_pack.buf_desc_id     = buf_desc_id_pack;
    td_pack.reg_data_format = static_cast<std::uint8_t>(formats.pack_S_src);
    _configure_buf_desc_table_(td_pack.buf_desc_id, td_pack.buf_desc);
    _llk_pack_hw_configure_<p_pacr::PACK1>(td_pack);

    // Implied math format disable for SrcS and sfpmem mod selection
    cfg[DISABLE_IMPLIED_SRCS_FORMAT_ADDR32 + TRISC_ID] = !IMPLIED_MATH_FORMAT;

    // -------------------------------------------------------------------------
    // SFPU configuration and execution
    // -------------------------------------------------------------------------

    // SrcS slice layout: slice 0 = in0, slice 1 = in1, slice 2 = out.
    // Each slice is PARAM_SRCS_YDIM rows apart in the SFPU address space.
    const int in0_base = ckernel::math::SFPU_SRCS_BASE_ADDR;
    const int in1_base = ckernel::math::SFPU_SRCS_BASE_ADDR + PARAM_SRCS_YDIM;
    const int out_base = ckernel::math::SFPU_SRCS_BASE_ADDR + 2 * PARAM_SRCS_YDIM;

    // Load replay buffer
    const int num_sfpu_iterations      = PARAM_SRCS_YDIM >> 1; // Divide by 2 since SFSPU operates on 2 rows at a time
    const std::uint32_t replay_buf_len = num_sfpu_iterations * 4;
    load_replay_buf( // TODO: Replace with SFPI call when SFPI is supported for Quasar: https://github.com/tenstorrent/tt-llk/issues/1637
        0,
        replay_buf_len,
        false,
        0,
        0,
        // Lambda function to load replay buffer
        [in0_base, in1_base, out_base, num_sfpu_iterations]
        {
#pragma GCC unroll 4
            for (int d = 0; d < num_sfpu_iterations; d++)
            {
                TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in0_base + (d << 1));
                TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, in1_base + (d << 1));
                // Add LREG0 + LREG1, store result in LREG2
                TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0x0);
                // Store result back to output slice
                TT_SFPSTORE(p_sfpu::LREG2, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, out_base + (d << 1));
            }
        });

    _llk_pack_srcs_config_for_tile_<PARAM_SRCS_INSTRN_COUNT>(PARAM_SRCS_32BIT_MODE);
    _llk_math_eltwise_unary_sfpu_init_();

    for (std::uint32_t i = 0; i < num_tiles; ++i)
    {
        TT_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_S, i * PARAM_SRCS_SLICE_COUNT);

        _llk_pack_srcs_<PARAM_SRCS_INSTRN_COUNT>(buf_desc_id_pack, i * PARAM_SRCS_SLICE_COUNT);

        // No auto-loops for unpacker due to HW bug for binary unpacking. Issue #1635: https://github.com/tenstorrent/tt-llk/issues/1635
        // Preload the unpacker pipeline so that SFPU is not starved
        constexpr int preload_count = 3;
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
            _llk_math_eltwise_unary_sfpu_srcs_clear_vlds_<0x1, 0x1>(); // Clears dvalid for SFPU read and write
        }

        // Remaining SFPU iterations with no unpacker instructions (since they are preloaded)
#pragma GCC unroll preload_count
        for (std::uint32_t j = 0; j < preload_count; j++)
        {
            TT_REPLAY(0, replay_buf_len, 0, 0, 0, 0);
            _llk_math_eltwise_unary_sfpu_srcs_clear_vlds_<0x1, 0x1>(); // Clears dvalid for SFPU read and write
        }
    }

    // Wait for all operations to complete
    wait_sfpu_idle();
    wait_unpack_idle();
    wait_pack_idle();
}

#endif

#ifdef LLK_TRISC_PACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
}

#endif
