// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"
#include "tensix.h"
#include "tensix_types.h"

constexpr std::uint32_t GATHER_ELEM_CNT  = 32;
constexpr std::uint32_t GATHER_ROW_ELEMS = 16;

using namespace ckernel;
using ckernel::trisc::bd_table;

constexpr std::uint32_t SRC_BD_SLOT = 17;
constexpr std::uint32_t RES_BD_SLOT = 29;

inline void program_bd_slot(std::uint32_t slot, std::uint32_t l1_addr_16B, std::uint32_t format, std::uint32_t x_dim, std::uint32_t y_dim)
{
    buffer_descriptor_u bd_val;
    bd_val.words[0] = bd_val.words[1] = bd_val.words[2] = bd_val.words[3] = 0;

    bd_val.f.l1_addr_16B = l1_addr_16B;
    bd_val.f.format      = static_cast<std::uint8_t>(format);
    bd_val.f.x_dim       = x_dim;
    bd_val.f.y_dim       = y_dim;
    bd_val.f.z_dim       = 1;

    bd_table[slot].words[0] = bd_val.words[0];
    bd_table[slot].words[1] = bd_val.words[1];
    bd_table[slot].words[2] = bd_val.words[2];
}

#ifdef LLK_TRISC_UNPACK

#include "ckernel_vector.h"
#include "llk_unpack_gather_compress.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_A = params.buffer_A; // gather source: GATHER_ELEM_CNT contiguous elements
    const Operand& buffer_B = params.buffer_B; // gather indices: GATHER_ELEM_CNT bytes
#endif

    (void)vsetvl<E8, M2, GATHER_ELEM_CNT>();
    vector_load<V0>(reinterpret_cast<const std::uint8_t*>(buffer_B[0]));
    auto unpack_arg_mbox = (std::uint8_t*)UNPACKER_ARG_MAILBOX_BASE1;
    vector_store<V0>(unpack_arg_mbox);

    program_bd_slot(SRC_BD_SLOT, L1_ADDRESS(buffer_A[0]), formats.unpack_B_src, GATHER_ROW_ELEMS, 1 /*y_dim*/);

    _llk_unpack_gather_compress_init_(GatherCompressModeSelect::GATHER, formats.unpack_B_dst, SRC_BD_SLOT);
    _llk_unpack_gather_compress_();
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_gather_compress.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
    set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
    _llk_math_gather_compress_init_<GATHER_ELEM_CNT>();
    _llk_math_gather_compress_(0 /*dst_index*/);
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_Res = params.buffer_Res; // gather result: GATHER_ELEM_CNT contiguous elements
#endif

    set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
    program_bd_slot(RES_BD_SLOT, L1_ADDRESS(buffer_Res[0]), formats.pack_dst, GATHER_ROW_ELEMS, GATHER_ELEM_CNT / GATHER_ROW_ELEMS);
    cfg_rmw(THCON_PACKER0_REG0_IN_DATA_FORMAT_RMW, static_cast<std::uint8_t>(formats.pack_src));
    TTI_PACR0_TILE(0, 0, RES_BD_SLOT, 1);
    wait_pack_idle();
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

void run_kernel(RUNTIME_PARAMETERS params)
{
    (void)params;
}

#endif
