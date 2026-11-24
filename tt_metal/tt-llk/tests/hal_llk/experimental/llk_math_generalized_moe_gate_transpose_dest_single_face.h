// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_generalized_moe_gate_transpose_dest_single_face.h. Same
// Tensix instruction sequence as production.

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/replay.h"
#include "hal/src.h"
#include "hal/sync.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod

using namespace ckernel;

namespace hal_port
{
namespace generalized_moe_gate_transpose
{
namespace move = hal::move;

// TTI_MOVD2B(0, SrcBRow, ADDR_MOD_3, Rows == 4 ? MOV_4_ROWS : MOV_1_ROW, DestRow)
template <std::uint32_t SrcBRow, std::uint32_t DestRow, std::uint16_t Rows>
inline __attribute__((always_inline)) void movd2b()
{
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = DestRow, .destination_row = SrcBRow, .address_mode = ADDR_MOD_3, .number_of_rows = move::RowCount {Rows}}>();
}

// TTI_MOVB2D(0, SrcBRow, AddrMod, MOV_1_ROW, DestRow)
template <std::uint32_t SrcBRow, std::uint8_t AddrMod, std::uint32_t DestRow>
inline __attribute__((always_inline)) void movb2d_1row()
{
    move::run<move::Transfer<move::SrcB, move::Dst> {
        .source_row = SrcBRow, .destination_row = DestRow, .address_mode = AddrMod, .number_of_rows = move::rows::One}>();
}

// TTI_TRNSPSRCB
inline __attribute__((always_inline)) void transpose_srcb()
{
    hal::src<hal::SrcB>.transpose();
}

// TT_OP_TRNSPSRCB
inline constexpr std::uint32_t TRNSPSRCB = hal::src<hal::SrcB>.transpose_operation();

// The 8 MOVB2D(SrcB row 16 + 2i -> DEST row Base + i) of one transposed run; the last one advances by ADDR_MOD_2.
template <std::uint32_t Base>
inline __attribute__((always_inline)) void movb2d_run()
{
    movb2d_1row<16, ADDR_MOD_3, Base + 0>();
    movb2d_1row<18, ADDR_MOD_3, Base + 1>();
    movb2d_1row<20, ADDR_MOD_3, Base + 2>();
    movb2d_1row<22, ADDR_MOD_3, Base + 3>();
    movb2d_1row<24, ADDR_MOD_3, Base + 4>();
    movb2d_1row<26, ADDR_MOD_3, Base + 5>();
    movb2d_1row<28, ADDR_MOD_3, Base + 6>();
    movb2d_1row<30, ADDR_MOD_2, Base + 7>();
}

// TTI_STALLWAIT(STALL_MATH, WAIT_SFPU | MATH | SRCB_VLD)
inline __attribute__((always_inline)) void wait_sfpu_math_srcb()
{
    namespace sync = hal::sync;
    sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SfpuIdle | sync::StallCondition::MathIdle | sync::StallCondition::SrcBValid>();
}
} // namespace generalized_moe_gate_transpose
} // namespace hal_port

namespace ckernel
{

constexpr std::uint32_t transpose_dest_tile_offset = 64; // 1 tile x 64 rows per tile

// Configure address modifiers for single face transpose
template <bool is_32bit>
inline void generalized_moe_gate_transpose_dest_single_face_configure_addrmod()
{
    using hal_port::MathAddrMod;
    using hal_port::set_math_addr_mod;
    using Sec = hal::cfg::Sec;

    set_math_addr_mod<Sec::S2, MathAddrMod {.dest_incr = transpose_dest_tile_offset}>();
    set_math_addr_mod<Sec::S3, MathAddrMod {}>();
}

template <std::uint32_t num_tiles = 1, bool is_32bit>
inline void generalized_moe_gate_transpose_dest_single_face_step0_configure_mop()
{
    namespace mop    = hal::mop;
    namespace replay = hal::replay;
    using namespace hal_port::generalized_moe_gate_transpose;

    static_assert(!is_32bit, "32-bit is not supported for single face transpose");
    // For 16-bit data, simple single-pass transpose
    // Load replay buffer with the transpose sequence for one 16x16 face (face 0: rows 0-15)
    replay::record<replay::BufferRange {ckernel::math::replay_buf_offset, 16}>(
        []
        {
            // Move 8 rows from DEST to SrcB interleaved (1 row at a time)
            movd2b<16, 0, 1>();
            movd2b<18, 1, 1>();
            movd2b<20, 2, 1>();
            movd2b<22, 3, 1>();
            movd2b<24, 4, 1>();
            movd2b<26, 5, 1>();
            movd2b<28, 6, 1>();
            movd2b<30, 7, 1>();

            // Move 8 rows from SrcB back to DEST interleaved (1 row at a time)
            movb2d_run<0>();
        });

    constexpr std::uint32_t d2b_instr = replay::get_operation(replay::BufferRange {ckernel::math::replay_buf_offset, 8});
    constexpr std::uint32_t b2d_instr = replay::get_operation(replay::BufferRange {ckernel::math::replay_buf_offset + 8, 8});

    // ckernel_template(num_tiles, 1, d2b_instr, TT_OP_TRNSPSRCB) + set_end_op(b2d_instr)
    mop::program(mop::loop(num_tiles, mop::loop(1, d2b_instr, TRNSPSRCB), b2d_instr));
}

template <std::uint32_t num_tiles = 1, bool is_32bit>
inline void generalized_moe_gate_transpose_dest_single_face_step1_configure_mop()
{
    namespace mop    = hal::mop;
    namespace replay = hal::replay;
    using namespace hal_port::generalized_moe_gate_transpose;

    static_assert(!is_32bit, "32-bit is not supported for single face transpose");
    // For 16-bit data, simple single-pass transpose
    // Load replay buffer with the transpose sequence for one 16x16 face (face 0: rows 0-15)
    constexpr replay::BufferRange range {ckernel::math::replay_buf_offset, 11};
    replay::record<range>(
        []
        {
            // Move 4 rows from DEST to SrcB (4 rows at a time)
            // This will place the first 2 rows in the first 2 columns, and the last 2 rows in the last 2 columns
            movd2b<16, 0, 4>();
            movd2b<28, 0, 4>();

            transpose_srcb();

            // Move 8 rows from SrcB back to DEST interleaved (1 row at a time)
            movb2d_run<0>();
        });

    // ckernel_template(num_tiles, 1, replay_instr)
    mop::program(mop::loop(num_tiles, mop::loop(1, replay::get_operation(range))));
}

// step1_hi: clone of step1 with two knobs:
//   d2b_dst  = MOVD2B source DEST row (which 4 groups to read; post-step0 group g is at row g).
//   b2d_base = MOVB2D output DEST row base (where to write the resulting run; step1 uses 0).
// The output base lets the LOW half write its run to rows 8-15 (b2d_base=8) so it does NOT clobber
// the post-step0 groups 4-7 sitting at rows 4-7 — which the HIGH half (d2b_dst=4) then reads.
template <std::uint32_t num_tiles, bool is_32bit, std::uint32_t d2b_dst, std::uint32_t b2d_base>
inline void generalized_moe_gate_transpose_dest_single_face_step1_hi_configure_mop()
{
    namespace mop    = hal::mop;
    namespace replay = hal::replay;
    using namespace hal_port::generalized_moe_gate_transpose;

    static_assert(!is_32bit, "32-bit is not supported for single face transpose");
    constexpr replay::BufferRange range {ckernel::math::replay_buf_offset, 11};
    replay::record<range>(
        []
        {
            movd2b<16, d2b_dst, 4>();
            movd2b<28, d2b_dst, 4>();

            transpose_srcb();

            movb2d_run<b2d_base>();
        });

    // ckernel_template(num_tiles, 1, replay_instr)
    mop::program(mop::loop(num_tiles, mop::loop(1, replay::get_operation(range))));
}

// Plain (non-transposed) FPU copy of 4 DEST rows [src..src+3] -> [dst..dst+3], across the 3 data
// regions (scores/indices/bias, via num_tiles + the ADDR_MOD_2 base advance). Used to stash/restore
// data in rows 8-15 — which the SFPU merge cannot address (SFPU offsets >=8 wrap) but the FPU can —
// during the ungrouped two-half assembly. src/dst must be 4-row aligned (0,4,8,12).
template <std::uint32_t num_tiles, bool is_32bit, std::uint32_t src, std::uint32_t dst, std::uint32_t srcb = 16>
inline void generalized_moe_gate_copy4rows_configure_mop()
{
    namespace mop    = hal::mop;
    namespace replay = hal::replay;
    using namespace hal_port::generalized_moe_gate_transpose;

    static_assert(!is_32bit, "32-bit is not supported");
    // srcb selects the 4-row SrcB scratch window (16/20/24/28). Back-to-back copy4rows calls use
    // DISJOINT srcb windows so a later MOVB2D cannot read a previous copy's SrcB leftover.
    constexpr replay::BufferRange range {ckernel::math::replay_buf_offset, 5};
    replay::record<range>(
        []
        {
            movd2b<srcb, src, 4>();                       // DEST rows src..+3 -> SrcB srcb..+3
            movb2d_1row<srcb + 0, ADDR_MOD_3, dst + 0>(); // SrcB srcb -> DEST dst+0 (no transpose)
            movb2d_1row<srcb + 1, ADDR_MOD_3, dst + 1>();
            movb2d_1row<srcb + 2, ADDR_MOD_3, dst + 2>();
            movb2d_1row<srcb + 3, ADDR_MOD_2, dst + 3>(); // ADDR_MOD_2 advances base by 64
        });
    // ckernel_template(num_tiles, 1, replay_instr)
    mop::program(mop::loop(num_tiles, mop::loop(1, replay::get_operation(range))));
}

template <std::uint32_t num_tiles = 1, bool is_32bit>
inline void generalized_moe_gate_transpose_dest_single_face_step2_configure_mop()
{
    namespace mop    = hal::mop;
    namespace replay = hal::replay;
    using namespace hal_port::generalized_moe_gate_transpose;

    static_assert(!is_32bit, "32-bit is not supported for single face transpose");
    constexpr replay::BufferRange range {ckernel::math::replay_buf_offset, 4};
    replay::record<range>(
        []
        {
            // Move 8 rows from DEST to SrcB (4 rows at a time)
            movd2b<16, 0, 4>();
            movd2b<20, 4, 4>();

            transpose_srcb();

            // Move 1 row from SrcB back to DEST
            movb2d_1row<16, ADDR_MOD_2, 0>();
        });

    // ckernel_template(num_tiles, 1, replay_instr)
    mop::program(mop::loop(num_tiles, mop::loop(1, replay::get_operation(range))));
}

// Initialize for single face transpose
template <bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_common_init_()
{
    generalized_moe_gate_transpose_dest_single_face_configure_addrmod<is_32bit>();
}

// Initialize for single face transpose
template <bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init_()
{
    generalized_moe_gate_transpose_dest_single_face_step0_configure_mop<4, is_32bit>();
    // Transpose-dest is a data-movement op -> PRESERVE. Route through the tracker (not a raw write) so a
    // later matmul/eltwise-binary re-establishes DEFAULT instead of skipping and inheriting this keep flag.
    hal_port::_configure_preserve_zero_flag_state_();
}

// Initialize for single face transpose
template <bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init_()
{
    generalized_moe_gate_transpose_dest_single_face_step1_configure_mop<3, is_32bit>();
}

// Initialize for single face transpose
template <bool is_32bit = false, std::uint32_t num_tiles = 3>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init_()
{
    // num_tiles=3: transpose scores(0) + idx(1) + BIAS(2). The bias (tile 2) MUST be math->standard'd too,
    // else the combine's bias round-trip packs a math-layout bias -> 2-period-corrupted sort key (the 256
    // output path never reads bias, so num_tiles=2 was enough there; the >256 combine merge sorts by bias).
    static_assert(num_tiles == 2 || num_tiles == 3, "step2 transposes scores/indices and optionally bias");
    generalized_moe_gate_transpose_dest_single_face_step2_configure_mop<num_tiles, is_32bit>();
}

// copy4rows init/runner.
template <std::uint32_t src = 0, std::uint32_t dst = 0, bool is_32bit = false, std::uint32_t srcb = 16>
inline void _llk_math_generalized_moe_gate_copy4rows_init_()
{
    generalized_moe_gate_copy4rows_configure_mop<3, is_32bit, src, dst, srcb>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_copy4rows_()
{
    static_assert(!(is_32bit || is_fp32_dest_acc_en), "32-bit / fp32 dest accum not supported");
    hal_port::reset_counters_abd_f();
    hal_port::generalized_moe_gate_transpose::wait_sfpu_math_srcb();
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}

// step1_hi init/runner — tunable knobs (d2b_dst, b2d_base) for the high-group experiment.
template <std::uint32_t d2b_dst = 0, std::uint32_t b2d_base = 24, bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init_()
{
    generalized_moe_gate_transpose_dest_single_face_step1_hi_configure_mop<3, is_32bit, d2b_dst, b2d_base>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_()
{
    static_assert(!(is_32bit || is_fp32_dest_acc_en), "32-bit and fp32 dest accum enable are not supported for single face transpose");
    hal_port::reset_counters_abd_f();
    hal_port::generalized_moe_gate_transpose::wait_sfpu_math_srcb();
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_()
{
    static_assert(!(is_32bit || is_fp32_dest_acc_en), "32-bit and fp32 dest accum enable are not supported for single face transpose");
    hal_port::reset_counters_abd_f();

    // Wait for SFPU and SrcB to be available
    hal_port::generalized_moe_gate_transpose::wait_sfpu_math_srcb();

    // Run the 16-bit single-face transpose MOP
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}

// Perform in-place transpose on face 0 (rows 0-15) of a single tile in DEST
// dst_index: The tile index in DEST register buffer (0, 1, 2, ...)
//            The function transposes face 0 of the specified tile
template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_()
{
    static_assert(!(is_32bit || is_fp32_dest_acc_en), "32-bit and fp32 dest accum enable are not supported for single face transpose");

    hal_port::reset_counters_abd_f();

    // Wait for SFPU and SrcB to be available
    hal_port::generalized_moe_gate_transpose::wait_sfpu_math_srcb();

    // Run the 16-bit single-face transpose MOP
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}

// Perform in-place transpose on face 0 (rows 0-15) of a single tile in DEST
// dst_index: The tile index in DEST register buffer (0, 1, 2, ...)
//            The function transposes face 0 of the specified tile
template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_()
{
    namespace rwc = hal::counters::rwc;

    static_assert(!(is_32bit || is_fp32_dest_acc_en), "32-bit and fp32 dest accum enable are not supported for single face transpose");

    hal_port::reset_counters_abd_f();

    // Wait for SFPU and SrcB to be available
    hal_port::generalized_moe_gate_transpose::wait_sfpu_math_srcb();

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    // TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_ABD)
    rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::A | rwc::Src::B}>();
}

} // namespace ckernel
