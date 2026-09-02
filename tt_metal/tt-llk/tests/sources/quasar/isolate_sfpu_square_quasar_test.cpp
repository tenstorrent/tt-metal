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

#include "llk_sfpu/ckernel_sfpu_square.h"
#include "llk_sfpu_srcs_api.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const volatile FormatConfig& formats = params.formats;
#endif

    llk_sfpu_srcs_unary_init(
        L1_ADDRESS(params.buffer_A[0]),
        static_cast<DataFormat>(formats.unpack_S_src),
        static_cast<DataFormat>(formats.unpack_S_dst),
        L1_ADDRESS(params.buffer_Res[0]),
        static_cast<DataFormat>(formats.pack_S_src),
        static_cast<DataFormat>(formats.pack_S_dst),
        IMPLIED_MATH_FORMAT);

    // SFPU load reads what UNP_S wrote; store writes what PACK1 will read.
    const std::uint32_t load_sfpmem  = _sfpu_sfpmem_type_(static_cast<DataFormat>(formats.unpack_S_dst));
    const std::uint32_t store_sfpmem = _sfpu_sfpmem_type_(static_cast<DataFormat>(formats.pack_S_src));

    llk_sfpu_srcs_unary(
        params.TILE_CNT,
        static_cast<DataFormat>(formats.unpack_S_dst),
        [load_sfpmem, store_sfpmem](const int load_base_addr, const int store_base_addr, const int num_sfpu_iterations)
        { _calculate_square_(load_base_addr, store_base_addr, num_sfpu_iterations, load_sfpmem, store_sfpmem); });

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
