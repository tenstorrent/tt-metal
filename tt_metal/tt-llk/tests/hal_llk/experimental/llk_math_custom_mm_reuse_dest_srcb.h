// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_custom_mm_reuse_dest_srcb.h. Same instruction sequence
// and access kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "experimental/llk_custom_mm_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/fpu.h"
#include "hal/move.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod

using namespace ckernel;

namespace hal_port
{

// TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + srcb_row, ADDR_MOD_2, mov_rows, dest_row): Dest rows -> SrcB rows.
template <std::uint32_t SrcBRow, std::uint32_t DestRow, std::uint16_t Rows>
inline __attribute__((always_inline)) void custom_mm_reuse_movd2b()
{
    namespace move = hal::move;
    constexpr move::Transfer<move::Dst, move::SrcB> t {
        .source_row = DestRow, .destination_row = p_movd2b::SRC_ZERO_OFFSET + SrcBRow, .address_mode = ADDR_MOD_2, .number_of_rows = move::RowCount {Rows}};
    static_assert(
        t.get_operation() ==
            TT_OP_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + SrcBRow, ADDR_MOD_2, (Rows == 4 ? p_movd2b::MOV_4_ROWS : p_movd2b::MOV_1_ROW), DestRow),
        "MOVD2B encoding");
    TTI_INSN(t.get_operation());
}

// TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, offset)
inline __attribute__((always_inline)) void custom_mm_reuse_set_dest_offset(const std::uint32_t offset)
{
    hal::cfg::write<hal::cfg::Access::TensixCfgUnit, hal::cfg::DestTargetRegCfgMath::Offset, hal::cfg::Sec::S0>(offset);
}

} // namespace hal_port

// Second matmul of a fused chain.  in0 is *not* unpacked from L1 — it is moved
// straight out of DEST into SrcB with MOVD2B.  Only SrcA (the weights) is unpacked.
//
// Source (in0) tile layout in DEST — what `custom_mm<DenseSplitAcc>`
// produces after its finalization step, for an in0 tile of shape [r, 32]:
//     cols  0..15  ->  rows  base + 0  .. base + (r-1)
//     cols 16..31  ->  rows  base + 16 .. base + 16 + (r-1)
//     next tile at base + src_tile_stride   (32 for a Dense layout, else 64)
//
// Output (accumulator) tile layout in DEST — the dense 2x8-row layout shared
// with the SDPA custom matmuls:
//     cols  0..15  ->  rows base + 0 .. base + 7
//     cols 16..31  ->  rows base + 8 .. base + 15
//     next tile at base + 16
// Half the DEST footprint of a `custom_mm` output tile, which is what lets the
// two matmuls' live ranges coexist in half DEST.  Pack it with Zstride = 8 rows
// and Wstride = 16 rows (see `custom_mm_reuse_dest_srcb_pack_init`).
//
// Constraints:
// - in0 (SrcB) tile shape: [{1, 2, 4, 8}, 32]; in1 (SrcA) tile shape: [32, 32]
// - rt_dim: 1
// - nt_dim: 1 to 16
// - kt_dim: any integer from 1 to 256 (inclusive)
// - fidelity: LoFi only

inline void custom_mm_reuse_dest_srcb_configure_addrmod()
{
    using hal::cfg::Sec;
    using hal_port::MathAddrMod;
    using hal_port::set_math_addr_mod;

    set_math_addr_mod<Sec::S0, MathAddrMod {.srca_incr = 16, .dest_incr = 8}>(); // Next face in the output width dim (cols 16..31)
    // Next face in the inner (K) dim: SrcB rows 8..15; Dest returns to the tile base
    set_math_addr_mod<Sec::S1, MathAddrMod {.srca_incr = 16, .srcb_incr = 8, .dest_cr = 1}>();
    set_math_addr_mod<Sec::S2, MathAddrMod {}>(); // NOP — used by the DEST -> SrcB moves
    // Next output tile
    set_math_addr_mod<Sec::S3, MathAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_incr = 8, .dest_c2cr = 1}>();
}

// FPU half of the math replay buffer: [replay_buf_offset, replay_buf_offset + 16).
constexpr std::uint32_t CUSTOM_MM_REUSE_FPU_REPLAY_LEN = 16;
constexpr std::uint32_t CUSTOM_MM_REUSE_REPLAY_LEN     = 4;
// Top-anchored so custom_mm's program, which sits at the window base, cannot overwrite
// it.  That is what lets this one load before the first matmul instead of between them.
constexpr std::uint32_t CUSTOM_MM_REUSE_REPLAY_OFFSET = ckernel::math::replay_buf_offset + CUSTOM_MM_REUSE_FPU_REPLAY_LEN - CUSTOM_MM_REUSE_REPLAY_LEN;
// custom_mm's longest program is a full in1 tile with 8-row in0 faces.
static_assert(
    CUSTOM_MM_REUSE_REPLAY_OFFSET >= ckernel::math::replay_buf_offset + ckernel::custom_mm_replay_buf_len(8, ckernel::MAX_NUM_FACES),
    "replay program overlaps custom_mm's at the window base");
static_assert(
    CUSTOM_MM_REUSE_REPLAY_OFFSET + CUSTOM_MM_REUSE_REPLAY_LEN <= ckernel::math::replay_buf_offset + CUSTOM_MM_REUSE_FPU_REPLAY_LEN,
    "replay program runs past the FPU half of the math replay window");

inline void custom_mm_reuse_dest_srcb_configure_mop()
{
    // One output tile = 4 MVMULs (2 SrcA faces along K x 2 along N), all
    // accumulating into the 16 DEST rows of that tile.
    using hal::fpu::MatrixMultiply;
    using hal::fpu::SourceRelease;
    hal::replay::record<hal::replay::BufferRange {CUSTOM_MM_REUSE_REPLAY_OFFSET, CUSTOM_MM_REUSE_REPLAY_LEN}>(
        []
        {
            hal::fpu::run<MatrixMultiply {.address_modifier = ADDR_MOD_0, .release = SourceRelease::None}>(); // K 0..15  x N 0..15
            hal::fpu::run<MatrixMultiply {.address_modifier = ADDR_MOD_1, .release = SourceRelease::None}>(); // K 0..15  x N 16..31
            hal::fpu::run<MatrixMultiply {.address_modifier = ADDR_MOD_0, .release = SourceRelease::None}>(); // K 16..31 x N 0..15
            hal::fpu::run<MatrixMultiply {.address_modifier = ADDR_MOD_3, .release = SourceRelease::SrcA}>(); // K 16..31 x N 16..31
        });
}

// Loads the replay program.  May run before the first matmul: custom_mm reloads only
// the base of the window, which this program sits clear of.
inline void _llk_math_custom_mm_reuse_dest_srcb_replay_init_()
{
    custom_mm_reuse_dest_srcb_configure_mop();
}

// custom_mm rewrites all eight ADDR_MODs each invocation, so this must follow its last
// math instruction.  Set load_replay = false when _replay_init_ has already run.
template <bool load_replay = true>
inline void _llk_math_custom_mm_reuse_dest_srcb_init_()
{
    custom_mm_reuse_dest_srcb_configure_addrmod();
    if constexpr (load_replay)
    {
        custom_mm_reuse_dest_srcb_configure_mop();
    }
}

// src_index / dst_index are bank-relative DEST *row* offsets (not tile slots).
//
// Preconditions:
// - the unpacker has issued a zero-and-set-dvalid on SrcB
// - DEST at dst_index is zero
// - for in0_tile_r_dim < 4, rows of the producing custom_mm's output above the tile
//   height should be zero (its clear_src=false power optimization can leave stale
//   products there; MOV_4_ROWS drags two such rows into SrcB for in0_tile_r_dim == 2 —
//   benign today because the packer reads only in0_tile_r_dim rows per face, but the
//   cross-op coupling is real)
template <std::uint32_t in0_tile_r_dim>
inline void _llk_math_custom_mm_reuse_dest_srcb_(
    const std::uint32_t src_index, const std::uint32_t dst_index, const std::uint32_t kt_dim, const std::uint32_t nt_dim, const std::uint32_t src_tile_stride)
{
    static_assert(
        in0_tile_r_dim == 1 || in0_tile_r_dim == 2 || in0_tile_r_dim == 4 || in0_tile_r_dim == 8,
        "custom_mm_reuse_dest_srcb: in0 tile height must be 1, 2, 4 or 8");

    const std::uint32_t dest_buffer_base = get_dest_buffer_base();
    namespace sync                       = hal::sync;
    using hal_port::custom_mm_reuse_movd2b;
    // TTI_STALLWAIT(STALL_MATH, WAIT_SFPU | MATH | SRCB_VLD)
    sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SfpuIdle | sync::StallCondition::MathIdle | sync::StallCondition::SrcBValid>();

    for (std::uint32_t i = 0; i < kt_dim; i++)
    {
        hal_port::custom_mm_reuse_set_dest_offset(src_index + i * src_tile_stride + dest_buffer_base);
        hal_port::reset_counters_abd_f(); // math::reset_counters(p_setrwc::SET_ABD_F)

        // DEST -> SrcB.  MVMULs 1/2 read SrcB rows 0..7 (K cols 0..15) and
        // MVMULs 3/4 read rows 8..15 (K cols 16..31), so source face0 lands at
        // SrcB row 0 and source face1 (DEST row +16) at SrcB row 8.
        if constexpr (in0_tile_r_dim == 8)
        {
            custom_mm_reuse_movd2b<0, 0, 4>();
            custom_mm_reuse_movd2b<4, 4, 4>();
            custom_mm_reuse_movd2b<8, 16, 4>();
            custom_mm_reuse_movd2b<12, 20, 4>();
        }
        else if constexpr (in0_tile_r_dim == 1)
        {
            custom_mm_reuse_movd2b<0, 0, 1>();
            custom_mm_reuse_movd2b<8, 16, 1>();
        }
        else
        {
            // r_dim 2 or 4: one 4-row move per face.  Rows r..3 of a custom_mm
            // output face are zero (its in0 unpack clears the SrcB rows above
            // the tile height), so over-copying them is harmless.
            custom_mm_reuse_movd2b<0, 0, 4>();
            custom_mm_reuse_movd2b<8, 16, 4>();
        }

        hal_port::custom_mm_reuse_set_dest_offset(dst_index + dest_buffer_base);
        for (std::uint32_t j = 0; j < nt_dim; j++)
        {
            hal::replay::run<hal::replay::BufferRange {CUSTOM_MM_REUSE_REPLAY_OFFSET, CUSTOM_MM_REUSE_REPLAY_LEN}>();
        }
    }

    // TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD)
    namespace rwc = hal::counters::rwc;
    constexpr rwc::MathCounter reset_abd_release_b {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::B};
    static_assert(reset_abd_release_b.operation() == TT_OP_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD), "SETRWC encoding");
    rwc::set<reset_abd_release_b>();
}
