// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "profiler.h"

std::uint32_t unp_cfg_context              = 0;
std::uint32_t pack_sync_tile_dst_ptr       = 0;
std::uint32_t math_sync_tile_dst_index     = 0;
static constexpr ckernel::DstSync DST_SYNC = ckernel::DstSync::SyncHalf;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);

    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */,
        0 /* within_face_16x16_transpose */,
        ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, TILE_NUM_FACES),
        formats.unpack_A_src,
        formats.unpack_A_dst);

    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_A[0]), formats.unpack_A_src, formats.unpack_A_dst);

    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        L1_ADDRESS(params.buffer_B[0]), formats.unpack_A_src, formats.unpack_A_dst);
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "sfpu/ckernel_sfpu_binary_bcast.h"

using namespace ckernel;
using namespace ckernel::sfpu;

static constexpr auto BCAST_DIM             = static_cast<BroadcastType>(BCAST_DIM_VAL);
static constexpr std::uint32_t INPUT_TILE_A = INPUT_TILE_A_VAL;
static constexpr std::uint32_t INPUT_TILE_B = INPUT_TILE_A + 1;
static constexpr std::uint32_t RESULT_TILE  = INPUT_TILE_A + 2;

namespace
{

// Test-only compiler-flow counterpart.  Addressing, transpose, replay and
// stores remain architectural LLK operations; the arithmetic island is
// expressed through typed SFPI values so GCC owns its dataflow and scheduling.
// The implementation deliberately lives in this fixture rather than either
// production ckernel_sfpu_binary_bcast.h.
template <BinaryOp BINOP>
sfpi_inline sfpi::vFloat generated_binary(sfpi::vFloat data, sfpi::vFloat bcast)
{
    if constexpr (BINOP == BinaryOp::ADD)
    {
        return data + bcast;
    }
    else if constexpr (BINOP == BinaryOp::SUB)
    {
        return data - bcast;
    }
    else
    {
        static_assert(BINOP == BinaryOp::MUL);
        return data * bcast;
    }
}

template <BinaryOp BINOP, sfpi::LRegs Data0, sfpi::LRegs Data1, sfpi::LRegs Data2, sfpi::LRegs Data3, sfpi::LRegs Bcast>
sfpi_inline void generated_binary_lregs()
{
    const sfpi::vFloat bcast = sfpi::l_reg[Bcast];
    const sfpi::vFloat data0 = sfpi::l_reg[Data0];
    const sfpi::vFloat data1 = sfpi::l_reg[Data1];
    const sfpi::vFloat data2 = sfpi::l_reg[Data2];
    const sfpi::vFloat data3 = sfpi::l_reg[Data3];
    sfpi::l_reg[Data0]       = generated_binary<BINOP>(data0, bcast);
    sfpi::l_reg[Data1]       = generated_binary<BINOP>(data1, bcast);
    sfpi::l_reg[Data2]       = generated_binary<BINOP>(data2, bcast);
    sfpi::l_reg[Data3]       = generated_binary<BINOP>(data3, bcast);
}

template <sfpi::LRegs Data0, sfpi::LRegs Data1, sfpi::LRegs Data2, sfpi::LRegs Data3>
sfpi_inline void keep_generated_outputs_live_through_raw_consumers()
{
    sfpi::l_reg[Data0].in_use();
    sfpi::l_reg[Data1].in_use();
    sfpi::l_reg[Data2].in_use();
    sfpi::l_reg[Data3].in_use();
}

template <BinaryOp BINOP>
sfpi_inline void generated_col_band(
    std::uint32_t bcast_addr,
    std::uint32_t left_addr,
    std::uint32_t right_addr,
    std::uint32_t data_base,
    std::uint32_t out_base)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    const std::uint32_t slot0      = left_addr;
    const std::uint32_t slot1      = left_addr + ODD_COLS_OFFSET;
    const std::uint32_t slot2      = right_addr;
    const std::uint32_t slot3      = right_addr + ODD_COLS_OFFSET;

    TT_SFPLOAD(LREG_BCAST, IM, ADDR_MOD_7, bcast_addr);
    lltt::replay(REPLAY_SLOT_BROADCAST, REPLAY_LEN_BROADCAST);
    _broadcast_stage3_with_data_prefetch_(data_base + slot0, data_base + slot1, data_base + slot2, data_base + slot3);

    generated_binary_lregs<
        BINOP,
        sfpi::LRegs::LReg1,
        sfpi::LRegs::LReg3,
        sfpi::LRegs::LReg4,
        sfpi::LRegs::LReg5,
        sfpi::LRegs::LReg0>();

    TT_SFPSTORE(LREG_DATA0, IM, ADDR_MOD_7, out_base + slot0);
    TT_SFPSTORE(LREG_DATA1, IM, ADDR_MOD_7, out_base + slot1);
    TT_SFPSTORE(LREG_DATA2, IM, ADDR_MOD_7, out_base + slot2);
    TT_SFPSTORE(LREG_DATA3, IM, ADDR_MOD_7, out_base + slot3);
    keep_generated_outputs_live_through_raw_consumers<
        sfpi::LRegs::LReg1, sfpi::LRegs::LReg3, sfpi::LRegs::LReg4, sfpi::LRegs::LReg5>();
}

template <BinaryOp BINOP>
sfpi_inline void generated_col_full_tile(std::uint32_t data_tile, std::uint32_t bcast_tile, std::uint32_t out_tile)
{
    const std::uint32_t data_base  = data_tile * DEST_TILE_SIZE_RAW;
    const std::uint32_t bcast_base = bcast_tile * DEST_TILE_SIZE_RAW;
    const std::uint32_t out_base   = out_tile * DEST_TILE_SIZE_RAW;
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; ++band)
    {
        const std::uint32_t off = band * ROW_BAND_STRIDE;
        generated_col_band<BINOP>(bcast_base + FACE0_BASE + off, FACE0_BASE + off, FACE1_BASE + off, data_base, out_base);
    }
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; ++band)
    {
        const std::uint32_t off = band * ROW_BAND_STRIDE;
        generated_col_band<BINOP>(bcast_base + FACE2_BASE + off, FACE2_BASE + off, FACE3_BASE + off, data_base, out_base);
    }
}

template <BinaryOp BINOP>
sfpi_inline void generated_row_band(std::uint32_t data_base, std::uint32_t out_base, std::uint32_t face_base, std::uint32_t band_off)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    const std::uint32_t src         = data_base + face_base + band_off;
    const std::uint32_t dst         = out_base + face_base + band_off;
    TT_SFPLOAD(p_sfpu::LREG0, IM, ADDR_MOD_7, src + COL_GROUP_OFFSETS[0]);
    TT_SFPLOAD(p_sfpu::LREG1, IM, ADDR_MOD_7, src + COL_GROUP_OFFSETS[1]);
    TT_SFPLOAD(p_sfpu::LREG2, IM, ADDR_MOD_7, src + COL_GROUP_OFFSETS[2]);
    TT_SFPLOAD(p_sfpu::LREG3, IM, ADDR_MOD_7, src + COL_GROUP_OFFSETS[3]);
    TTI_SFPTRANSP(0, 0, 0, 0);
    generated_binary_lregs<
        BINOP,
        sfpi::LRegs::LReg0,
        sfpi::LRegs::LReg1,
        sfpi::LRegs::LReg2,
        sfpi::LRegs::LReg3,
        sfpi::LRegs::LReg4>();
    TTI_SFPTRANSP(0, 0, 0, 0);
    TT_SFPSTORE(p_sfpu::LREG0, IM, ADDR_MOD_7, dst + COL_GROUP_OFFSETS[0]);
    TT_SFPSTORE(p_sfpu::LREG1, IM, ADDR_MOD_7, dst + COL_GROUP_OFFSETS[1]);
    TT_SFPSTORE(p_sfpu::LREG2, IM, ADDR_MOD_7, dst + COL_GROUP_OFFSETS[2]);
    TT_SFPSTORE(p_sfpu::LREG3, IM, ADDR_MOD_7, dst + COL_GROUP_OFFSETS[3]);
    keep_generated_outputs_live_through_raw_consumers<
        sfpi::LRegs::LReg0, sfpi::LRegs::LReg1, sfpi::LRegs::LReg2, sfpi::LRegs::LReg3>();
}

template <BinaryOp BINOP>
sfpi_inline void generated_row_full_tile(std::uint32_t data_tile, std::uint32_t bcast_tile, std::uint32_t out_tile)
{
    constexpr InstrModLoadStore IM = InstrModLoadStore::DEFAULT;
    const std::uint32_t data_base  = data_tile * DEST_TILE_SIZE_RAW;
    const std::uint32_t bcast_base = bcast_tile * DEST_TILE_SIZE_RAW;
    const std::uint32_t out_base   = out_tile * DEST_TILE_SIZE_RAW;
    for (std::uint32_t i = 0; i < 4; ++i)
    {
        TT_SFPLOAD(p_sfpu::LREG4 + i, IM, ADDR_MOD_7, bcast_base + COL_GROUP_OFFSETS[i]);
    }
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; ++band)
    {
        generated_row_band<BINOP>(data_base, out_base, FACE0_BASE, band * ROW_BAND_STRIDE);
    }
    for (std::uint32_t band = 0; band < NUM_ROW_BANDS_PER_FACE_HALF; ++band)
    {
        generated_row_band<BINOP>(data_base, out_base, FACE2_BASE, band * ROW_BAND_STRIDE);
    }
}

template <BinaryOp BINOP, BroadcastType DIM>
sfpi_inline void generated_binary_bcast_full_tile(std::uint32_t data_tile, std::uint32_t bcast_tile, std::uint32_t out_tile)
{
    if constexpr (DIM == BroadcastType::COL)
    {
        generated_col_full_tile<BINOP>(data_tile, bcast_tile, out_tile);
    }
    else
    {
        generated_row_full_tile<BINOP>(data_tile, bcast_tile, out_tile);
    }
}

} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false /* is_int_fpu_en */, PackMode::Default>(
        TILE_NUM_FACES, formats.math);
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_math_eltwise_binary_sfpu_init_<SfpuType::add1>();

    _llk_math_wait_for_dest_available_<DST_SYNC>();

    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        INPUT_TILE_A, formats.math, formats.math);
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        INPUT_TILE_B, formats.math, formats.math);

    _llk_math_eltwise_sfpu_start_(0);

    _sfpu_binary_bcast_init_<BCAST_DIM>();

    static_assert(BINARY_BCAST_IMPL <= 1, "Unknown binary-broadcast implementation selector");
    {
        START_PERF_MEASURE("BINARY_BCAST_BODY")
        if constexpr (BINARY_BCAST_IMPL == 0)
        {
            _calculate_sfpu_binary_bcast_full_tile_<SFPU_BINARY_OPERATION, BCAST_DIM>(INPUT_TILE_A, INPUT_TILE_B, RESULT_TILE);
        }
        else
        {
            generated_binary_bcast_full_tile<SFPU_BINARY_OPERATION, BCAST_DIM>(INPUT_TILE_A, INPUT_TILE_B, RESULT_TILE);
        }
    }

    _llk_math_eltwise_sfpu_done_();

    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, TILE_NUM_FACES);
    _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();

    _llk_packer_wait_for_math_done_();

    static constexpr std::uint32_t RESULT_TILE = INPUT_TILE_A_VAL + 2;
    _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(RESULT_TILE, L1_ADDRESS(params.buffer_Res[0]));

    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
}

#endif
