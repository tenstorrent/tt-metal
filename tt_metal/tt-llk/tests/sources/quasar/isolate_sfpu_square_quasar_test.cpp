// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Isolated SFPU square: UNPACK2 (UNP_S) -> SrcS -> SFPU -> PACK1 -> L1.
// All logic runs in LLK_TRISC_ISOLATE_SFPU; UNPACK, MATH, PACK are stubbed.
// SrcS dvalid is controlled by UNPACR2/PACR1 TILE_INC (SetDatValid/ClrDatValid in llk_srcs_tdma.h).

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
#include "llk_srcs_tdma.h"
#include "params.h"
#include "sfpu/ckernel_sfpu_square.h"

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

    const bool PARAM_SRCS_32BIT_MODE =
        static_cast<DataFormat>(formats.unpack_S_dst) == DataFormat::Float32 || static_cast<DataFormat>(formats.unpack_S_dst) == DataFormat::Int32;
    constexpr std::uint32_t PARAM_SRCS_XDIM         = 16; // datums per row of SrcS slice
    constexpr std::uint32_t PARAM_SRCS_ZDIM         = 1;
    constexpr std::uint32_t PARAM_SRCS_YDIM_BASE    = 8; // rows per slice if SrcS were 16-bit columns
    const std::uint32_t PARAM_SRCS_YDIM             = PARAM_SRCS_32BIT_MODE ? (PARAM_SRCS_YDIM_BASE / 2) : PARAM_SRCS_YDIM_BASE;
    const std::uint32_t PARAM_SRCS_SLICE_COUNT      = (32 * 32) / (PARAM_SRCS_XDIM * PARAM_SRCS_YDIM * PARAM_SRCS_ZDIM);
    constexpr std::uint32_t PARAM_SRCS_INSTRN_COUNT = 1;

    // -------------------------------------------------------------------------
    // Buffer descriptor and HW setup
    // -------------------------------------------------------------------------

    constexpr std::uint32_t buf_desc_id_unpack = 0;
    constexpr std::uint32_t buf_desc_id_pack   = 8;

    buffer_descriptor_u bd_unpack = {0};
    tdma_descriptor_t td_unpack;
    buffer_descriptor_u bd_pack = {0};
    tdma_descriptor_t td_pack;

    // Unpack BD: L1 input -> SrcS
    bd_unpack.f.l1_addr_16B   = L1_ADDRESS(params.buffer_A[0]);
    bd_unpack.f.format        = static_cast<std::uint8_t>(formats.unpack_S_src);
    bd_unpack.f.x_dim         = PARAM_SRCS_XDIM;
    bd_unpack.f.y_dim         = PARAM_SRCS_YDIM;
    bd_unpack.f.z_dim         = PARAM_SRCS_ZDIM;
    td_unpack.buf_desc        = bd_unpack;
    td_unpack.buf_desc_id     = buf_desc_id_unpack;
    td_unpack.reg_data_format = static_cast<std::uint8_t>(formats.unpack_S_dst);
    _configure_buf_desc_table_(td_unpack.buf_desc_id, td_unpack.buf_desc);
    _llk_unpack_configure_unary_<p_unpacr::UNP_S>(td_unpack);

    // Pack BD: SrcS -> L1 output
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
    const std::uint32_t sfpmem_mod                     = IMPLIED_MATH_FORMAT ? p_sfpu::sfpmem::DEFAULT : _sfpu_sfpmem_type_(formats.unpack_S_dst);

    // -------------------------------------------------------------------------
    // SFPU configuration and execution
    // -------------------------------------------------------------------------

    // If SrcS is 32-bit, we need 16 slices (unpack/pack) per tile
    if (PARAM_SRCS_32BIT_MODE)
    {
        _llk_unpack_srcs_config_<PARAM_SRCS_INSTRN_COUNT, 16>();
        _llk_pack_srcs_config_<PARAM_SRCS_INSTRN_COUNT, 16>();
    }
    else
    {
        _llk_unpack_srcs_config_<PARAM_SRCS_INSTRN_COUNT, 8>();
        _llk_pack_srcs_config_<PARAM_SRCS_INSTRN_COUNT, 8>();
    }
    _llk_math_eltwise_unary_sfpu_init_();

    const int num_sfpu_iterations = PARAM_SRCS_YDIM >> 1; // SFP_ROWS == 2
    for (std::uint32_t i = 0; i < num_tiles; ++i)
    {
        // Unpack/Pack calls can be moved outside the loop by incorporating the loop into the auto-loop registers
        // Keeping them here for now since num_tiles is not a compile-time constant
        _llk_unpack_srcs_<PARAM_SRCS_INSTRN_COUNT>(buf_desc_id_unpack, i * PARAM_SRCS_SLICE_COUNT); // Sets dvalid for SFPU to read

        // Pack is placed before SFPU because SFPU loop fills up and clogs the instruction buffer leading to hangs
        _llk_pack_srcs_<PARAM_SRCS_INSTRN_COUNT>(buf_desc_id_pack, i * PARAM_SRCS_SLICE_COUNT); // Sets dvalid for SFPU to write

        for (std::uint32_t slice = 0; slice < PARAM_SRCS_SLICE_COUNT; slice++)
        {
            // Square is inlined instead of _calculate_square_ / _calculate_square_sfp_rows_: those
            // helpers assume fixed operand addresses (and default sfpmem); this kernel needs explicit
            // SrcS load/store bases and sfpmem_mod. Passing addresses into calculate_* will land in a
            // follow-up PR handled in https://github.com/tenstorrent/tt-llk/issues/1353.
            const int load_base_addr  = ckernel::math::SFPU_SRCS_BASE_ADDR;                       // First slice of SrcS
            const int store_base_addr = ckernel::math::SFPU_SRCS_BASE_ADDR + 2 * PARAM_SRCS_YDIM; // Third slice of SrcS

#pragma GCC unroll 8
            for (int d = 0; d < num_sfpu_iterations; d++)
            {
                TT_SFPLOAD(p_sfpu::LREG0, sfpmem_mod, ADDR_MOD_7, 0, load_base_addr + (d << 1));
                // Multiply LREG0 * LREG0, store result in LREG0
                TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
                // Store result back to destination
                TT_SFPSTORE(p_sfpu::LREG0, sfpmem_mod, ADDR_MOD_7, 0, store_base_addr + (d << 1));
            }

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
