// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Isolated SFPU square: UNPACK2 (UNP_S) -> SrcS -> SFPU -> PACK1 -> L1.
// All logic runs in LLK_TRISC_ISOLATE_SFPU; UNPACK, MATH, PACK are stubbed.
// SrcS dvalid is controlled by UNPACR2/PACR1 TILE_INC (SetDatValid/ClrDatValid in llk_srcs.h).

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
#include "llk_sfpu_srcs.h"
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
    // Buffer descriptors
    // -------------------------------------------------------------------------

    constexpr std::uint32_t buf_desc_id_unpack = 0;
    constexpr std::uint32_t buf_desc_id_pack   = 8;

    // Unpack BD: L1 input -> SrcS
    tdma_descriptor_t td_unpack =
        ckernel::trisc::construct_srcs_tdma_desc(L1_ADDRESS(params.buffer_A[0]), formats.unpack_S_src, buf_desc_id_unpack, formats.unpack_S_dst);
    _configure_buf_desc_table_(td_unpack.buf_desc_id, td_unpack.buf_desc);

    // Pack BD: SrcS -> L1 output
    tdma_descriptor_t td_pack =
        ckernel::trisc::construct_srcs_tdma_desc(L1_ADDRESS(params.buffer_Res[0]), formats.pack_S_dst, buf_desc_id_pack, formats.pack_S_src);
    _configure_buf_desc_table_(td_pack.buf_desc_id, td_pack.buf_desc);

    // -------------------------------------------------------------------------
    // SFPU configuration and execution
    // -------------------------------------------------------------------------

    _llk_sfpu_srcs_init_<1>(td_unpack, td_pack, IMPLIED_MATH_FORMAT);

    // Square is a single SFPMUL, so one slice is 3 instructions per SFPU pass.
    // Passing addresses into calculate_* will land in a follow-up PR handled in https://github.com/tenstorrent/tt-llk/issues/1353.
    constexpr std::uint32_t instrns_per_pass = 3;
    const std::uint32_t replay_buf_len       = instrns_per_pass * (srcs_dims::ydim(_is_srcs_32bit_mode_(static_cast<DataFormat>(formats.unpack_S_dst))) >> 1);

    _llk_sfpu_srcs_<1>(
        num_tiles,
        td_unpack,
        td_pack,
        replay_buf_len,
        [](const int load_base_addr, const int store_base_addr, const int num_sfpu_iterations)
        {
#pragma GCC unroll 8
            for (int d = 0; d < num_sfpu_iterations; d++)
            {
                TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, load_base_addr + (d << 1));
                // Multiply LREG0 * LREG0, store result in LREG0
                TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
                // Store result back to destination
                TT_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, store_base_addr + (d << 1));
            }
        });

    _llk_sfpu_srcs_done_();
}

#endif

#ifdef LLK_TRISC_PACK

#include "params.h"

void run_kernel(RUNTIME_PARAMETERS /*params*/)
{
}

#endif
