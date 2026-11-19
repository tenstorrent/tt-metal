// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Experiment for tt-metal#64475, not meant for CI: does a PACK-thread SFPLOADMACRO store lose a MATH-thread FPU write
// to the same half of Dst? Built on the single-thread probe in amahmud/bh-dst-fpu-sfpu-collision-repro.
//
// MATH writes one tile per block into tile 0 of its Dst half (SyncHalf double buffering) with hand-issued ELWADD,
// MOVA2D or MOVB2D. While it writes block b + 1, PACK, which holds block b in the other half, rewrites rows 64..255 of
// MATH's half with their own values. Those rows are never written by MATH, so the rewrites cannot change any data: a
// wrong output value is a lost FPU write. MATH's writes are held until PACK's rewrite stream is running.
//
// params.DST_INDEX:
//   bit 0      remap_addrs            bit 1      swizzle_32b
//   bits 2..3  PACK stream: 0 none, 1 SFPLOAD + SFPSTORE, 2 SFPLOADMACRO with a cycle-counted store,
//              3 SFPLOADMACRO whose store waits for one more SFPU instruction (an SFPNOP issued after the gap)
//   bits 4..6  store delay (stream 2)
//   bits 7..10 Tensix NOPs after each PACK element
//   bits 11..12 Tensix NOPs after each MATH write
//   bits 13..15 Tensix NOPs before MATH's first write
//   bits 16..17 MATH writer: 0 ELWADD, 1 MOVA2D, 2 MOVB2D
// Wall-clock timestamps go to L1: MATH's writes issued at 0x110000[block], PACK's stream drained at 0x111000[block].

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

namespace
{
constexpr std::uint32_t STREAM_NONE   = 0;
constexpr std::uint32_t STREAM_PLAIN  = 1;
constexpr std::uint32_t STREAM_CYCLES = 2;
constexpr std::uint32_t STREAM_INSTR  = 3;
constexpr std::uint32_t WRITER_MOVA2D = 1;
constexpr std::uint32_t WRITER_MOVB2D = 2;
constexpr std::uint8_t TOKEN          = ckernel::semaphore::FPU_SFPU;

inline std::uint32_t mode_stream(const std::uint32_t mode)
{
    return (mode >> 2) & 3;
}

inline std::uint32_t mode_writer(const std::uint32_t mode)
{
    return (mode >> 16) & 3;
}
} // namespace

#ifdef LLK_TRISC_UNPACK

#include "llk_lib_unpack_wrappers.h"
#include "llk_unpack_A.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats   = params.formats;
    const std::uint32_t mode      = static_cast<std::uint32_t>(params.DST_INDEX);
    const std::uint32_t num_tiles = params.NUM_BLOCKS * params.NUM_TILES_IN_BLOCK;
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
    if (mode_writer(mode) == WRITER_MOVB2D)
    {
        _llk_unpack_AB_init_<BroadcastType::NONE>(ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4));
        for (std::uint32_t i = 0; i < num_tiles; ++i)
        {
            _llk_unpack_AB_<BroadcastType::NONE>(L1_ADDRESS(params.buffer_A[i]), L1_ADDRESS(params.buffer_B[i]));
        }
        return;
    }
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
        0, 0, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4), formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t i = 0; i < num_tiles; ++i)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "llk_lib_unpack_wrappers.h"
#include "llk_math_common.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    const std::uint32_t mode    = static_cast<std::uint32_t>(params.DST_INDEX);
    const std::uint32_t knop    = (mode >> 11) & 3;
    const std::uint32_t lead    = (mode >> 13) & 7;
    const std::uint32_t writer  = mode_writer(mode);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, PackMode::Default>(
        ckernel::DEFAULT_TENSOR_SHAPE, formats.math);
    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_reconfig_remap_(false);
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_remap_addrs_RMW>((mode & 1) ? 1 : 0);
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_swizzle_32b_RMW>((mode & 2) ? 1 : 0);
    t6_semaphore_init(TOKEN, 0, 1);
    if (writer == WRITER_MOVB2D)
    {
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 4},
            .dest = {.incr = 4},
        }
            .set(ADDR_MOD_4);
    }
    const std::uint32_t per_face = (writer == WRITER_MOVB2D) ? 4 : 2;
    volatile std::uint32_t* ts   = reinterpret_cast<volatile std::uint32_t*>(0x110000);

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        if (block > 0)
        {
            // PACK posts once its rewrite stream for this half is running. The SEMGET is sync class, so it and the
            // writes behind it wait for the post.
            t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(TOKEN);
            t6_semaphore_get(TOKEN);
        }
        for (std::uint32_t k = 0; k < lead; ++k)
        {
            TTI_NOP;
        }
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
        // The writes below advance the Dst counter; start every tile at row 0 of the half.
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        for (std::uint32_t f = 0; f < 4; ++f)
        {
            for (std::uint32_t i = 0; i < per_face; ++i)
            {
                if (writer == WRITER_MOVB2D)
                {
                    TTI_MOVB2D(0, 0, ADDR_MOD_4, p_movb2d::MOV_4_ROWS, 0);
                }
                else if (writer == WRITER_MOVA2D)
                {
                    TTI_MOVA2D(p_mov::DEST_NORM, 0, ADDR_MOD_2, p_mova2d::MOV_8_ROWS, 0);
                }
                else
                {
                    TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
                }
                for (std::uint32_t k = 0; k < knop; ++k)
                {
                    TTI_NOP;
                }
            }
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
        }
        tensix_sync();
        ts[block] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
    // PACK's stream for the last block posts too.
    t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(TOKEN);
    t6_semaphore_get(TOKEN);
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

using namespace ckernel;

namespace
{
constexpr std::uint32_t SFPMOD     = is_fp32_dest_acc_en ? 3 : 2;
constexpr std::uint32_t FIRST_ROW  = 64;  // tile 0 belongs to MATH
constexpr std::uint32_t ROW_SPAN   = 192; // rows 64..255: inside the half for both Dst widths
constexpr std::uint32_t REPS       = 2;
constexpr std::uint32_t POST_AFTER = 8; // elements issued before MATH is let go

inline void rewrite_stream(const std::uint32_t stream, const std::uint32_t base, const std::uint32_t gap)
{
    std::uint32_t element = 0;
    for (std::uint32_t rep = 0; rep < REPS; ++rep)
    {
        for (std::uint32_t a = 0; a < ROW_SPAN; a += 2, ++element)
        {
            if (element == POST_AFTER)
            {
                t6_semaphore_post(TOKEN);
            }
            if (stream == STREAM_PLAIN)
            {
                TT_SFPLOAD(p_sfpu::LREG1, SFPMOD, ADDR_MOD_7, base + a);
                TT_SFPSTORE(p_sfpu::LREG1, SFPMOD, ADDR_MOD_7, base + a);
            }
            else
            {
                TT_SFPLOADMACRO((0 << 2) | p_sfpu::LREG1, SFPMOD, ADDR_MOD_7, base + a);
            }
            for (std::uint32_t k = 0; k < gap; ++k)
            {
                TTI_NOP;
            }
            if (stream == STREAM_INSTR)
            {
                TTI_SFPNOP;
            }
        }
    }
}
} // namespace

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    const std::uint32_t mode    = static_cast<std::uint32_t>(params.DST_INDEX);
    const std::uint32_t stream  = mode_stream(mode);
    const std::uint32_t delay   = (mode >> 4) & 7;
    const std::uint32_t gap     = (mode >> 7) & 15;
    volatile std::uint32_t* ts  = reinterpret_cast<volatile std::uint32_t*>(0x111000);
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
    if (stream == STREAM_CYCLES || stream == STREAM_INSTR)
    {
        // Macro 0: only the store sub-unit, SFPSTORE of the loaded LReg to the loaded address with the load's Mod0.
        const std::uint32_t store_bits = (((stream == STREAM_INSTR) ? 1 : delay) << 3) | 3;
        TTI_SFPLOADI(p_sfpu::LREG0, 10 /*lower 16 bits*/, 0);
        TT_SFPLOADI(p_sfpu::LREG0, 8 /*upper 16 bits*/, store_bits << 8);
        TTI_SFPCONFIG(0, 4 + 0, 0);
        if (stream == STREAM_INSTR)
        {
            TTI_SFPCONFIG(0x810, 8, 1); // the store sub-unit counts SFPU instructions
        }
        else
        {
            TTI_SFPCONFIG(0x010, 8, 1); // every delay counts cycles
        }
        TTI_SFPNOP;
    }
    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();
        // PACK holds the half MATH just finished; MATH writes its next block into the other one.
        const std::uint32_t math_half = (dest_offset_id == 0) ? DEST_REGISTER_HALF_SIZE : 0;
        if (stream == STREAM_NONE)
        {
            t6_semaphore_post(TOKEN);
        }
        else
        {
            rewrite_stream(stream, math_half + FIRST_ROW, gap);
            for (std::uint32_t k = 0; k < 8; ++k)
            {
                TTI_SFPNOP;
            }
        }
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU);
        tensix_sync();
        ts[block] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>(0, L1_ADDRESS(params.buffer_Res[block]));
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
