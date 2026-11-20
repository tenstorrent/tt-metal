// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// hifi_mm: the fused row matmul of the Hy4 decode kernels with the multiplier's fidelity phases as a parameter.
//
// Same operands, CB usage, DEST layouts and call shape as custom_mm (api/compute/experimental/custom_mm.h):
//   in0   [{1, 2, 4, 8}, 32] row tiles, unpacked to SrcB, one tile per K tile
//   in1   [32, 32] weight tiles, K-major (or column-major with read_transposed), unpacked to SrcA
//   out   ct_dim [{1, 2, 4, 8}, 32] tiles accumulated in DEST over any number of calls (FP32 DEST as the pieces use
//         it), in any CustomMmDestLayout; the SplitAcc layouts merge their partials when `finalize` is set
// What is added is the template parameter `passes` (and `phase_step`): every (K tile, output tile) pair runs its
// four MVMULs `passes` times on the SAME SrcA / SrcB contents with the multiplier's fidelity phase stepping
// 0, phase_step, 2 phase_step, ... and all passes accumulating into the same DEST rows.
//
//   passes 1                   LoFi: what custom_mm does (the instruction stream is the same)
//   passes 2, phase_step 1     HiFi2 (phases 0, 1)
//   passes 3, phase_step 1     HiFi3 (phases 0, 1, 2)
//   passes 4, phase_step 1     HiFi4 (phases 0, 1, 2, 3)
//   passes 2, phase_step 2     phases 0 and 2
// What each phase multiplies was measured with fused/fidelity/tests/test_probe.py and is written up in
// fused/fidelity/GUIDE.md: phase bit 0 selects the low bits of SrcA (the weight), phase bit 1 the low bits of SrcB
// (the row operand); LoFi multiplies the top 5 significant bits of the weight by the top 7 of the row operand.
//
// The SplitAcc layouts and their merge. With a SplitAcc layout the four MVMULs of a tile write four different DEST
// row groups (the second K face of the tile accumulates in rows 8.. and 24..), which lets the FPU run them back to
// back: about 1 ns per MVMUL against 2 ns with the plain layouts, where every second MVMUL waits for the row group
// it shares (fused/fidelity/tests/test_cost.py). custom_mm's `finalize` merges the two accumulators by moving the
// second one through SrcB (MOVD2B of the upper 16 bits of the FP32 DEST datum), which cuts it to BF16: 8
// significant bits, 2e-3 relative rms on the sum whatever the fidelity (tests/test_dense.py). hifi_mm keeps that
// `finalize` for compatibility and adds an exact one: call every block with finalize = false and then
// hifi_mm_merge<ct, layout>(), which adds the two accumulators of every output tile in the SFPU in FP32.
//
// Only the math thread differs from custom_mm: the unpack sequence (one SrcA tile per output tile, one SrcB tile per
// K tile, data valid set once each) and the packer strides do not depend on how often the math thread reads the
// source registers, so the unpack and pack halves below call the tree's custom_mm LLKs unchanged. The math half is
// written on the open primitives (MVMUL, address mods, the MOP template, the replay buffer):
//   * custom_mm's MOP is  outer 1 x inner ct_dim  of [3 MVMUL | last MVMUL]; here it is  outer ct_dim x inner
//     passes: the last MVMUL of a pass returns SrcA, SrcB and DEST to the start of the tile and steps the phase
//     (ADDR_MOD_4 with a fidelity increment), the last MVMUL of the tile's last pass releases SrcA and moves DEST to
//     the next tile (ADDR_MOD_2, phase cleared), the last of the K tile also releases SrcB (ADDR_MOD_3).
//   * the SplitAcc merge runs the earlier passes from the replay buffer, then the last pass and the merge; the phase
//     is cleared by a SETRWC before the moves and adds of the merge.
// 16 token rows in one pass over the weights: hifi_mm16_* below take [16, 32] row tiles (two faces of 16 rows) and
// run eight MVMULs per (K tile, output tile) and pass, rows 0..7 and 8..15 of each face against the same weight
// face. The output tile needs all 32 rows of its DEST slot (Dense or Default layout), there are no split
// accumulators and nothing to merge: the eight MVMULs already write four row groups in turn.
//
// Limits: in1 must be a full 4-face tile (the narrow [32, 16] form of custom_mm is not carried over);
// passes * phase_step <= 4 (the phase counter has 2 bits).

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/custom_mm.h"
#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpi.h"
#endif

namespace hy4fid {

using ckernel::CustomMmDestLayout;

constexpr bool hifi_mm_legal(const std::uint32_t passes, const std::uint32_t phase_step) {
    return passes >= 1 && passes <= 4 && phase_step >= 1 && phase_step <= 3 && (passes - 1) * phase_step <= 3;
}

#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)

// Replay buffer of one output tile's last pass with the SplitAcc merge:
// 4 MVMUL + SETRWC + 2 MOVD2B (4 for 8-row faces) + ZEROSRC + 2 ELWADD.
constexpr std::uint32_t hifi_mm_replay_len(const std::uint32_t operandB_face_r_dim) {
    return operandB_face_r_dim == 8 ? 12 : 10;
}

template <bool transpose, CustomMmDestLayout layout, std::uint32_t phase_step>
inline void hifi_mm_configure_addrmod() {
    using namespace ckernel;
    constexpr bool split_acc = custom_mm_split_acc(layout);
    constexpr bool compact = layout == CustomMmDestLayout::Compact;

    constexpr std::uint8_t MOD_0_SRCA_INCR = transpose ? 32 : 16;
    constexpr std::uint8_t MOD_0_DEST_INCR = compact ? 8 : 16;
    constexpr std::uint8_t MOD_1_SRCA_INCR = transpose ? (64 - 16) : 16;
    constexpr std::uint16_t MOD_1_DEST_INCR = (split_acc || compact) ? (1024 - 8) : (1024 - 16);
    constexpr std::uint8_t MOD_2_DEST_INCR = custom_mm_tile_rows(layout);

    // next face along the output width (face 2 instead of 1 with transpose)
    addr_mod_t{
        .srca = {.incr = MOD_0_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = MOD_0_DEST_INCR, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    // next face along K: SrcB to its second face, DEST back to output face 0 (rows 8 / 24 with split partials)
    addr_mod_t{
        .srca = {.incr = MOD_1_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = MOD_1_DEST_INCR, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    // end of an output tile: next tile in DEST, phase back to 0
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = MOD_2_DEST_INCR, .clr = 0, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_2);

    // end of a K tile: everything back to the start, phase back to 0
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_3);

    // end of a pass: SrcA, SrcB and DEST back to the start of this tile, next phase
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
        .fidelity = {.incr = static_cast<std::uint8_t>(phase_step), .clr = 0},
    }
        .set(ADDR_MOD_4);

    // the merge of the split partials (as custom_mm): SrcB to row 32
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 32, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_5);

    // next face in all three
    addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_6);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_7);
}

inline void hifi_mm_configure_mop(
    const std::uint32_t operandB_face_r_dim, const std::uint32_t ct_dim, const std::uint32_t passes) {
    using namespace ckernel;
    load_replay_buf(ckernel::math::replay_buf_offset, hifi_mm_replay_len(operandB_face_r_dim), [operandB_face_r_dim] {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B F0 x A F0 => rows 0
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // B F0 x A F1 (F2 with transpose) => rows 16
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B F1 x A F2 (F1) => rows 0 (8 with split partials)
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);  // B F1 x A F3 => rows 16 (24); end of a pass

        // the merge of the split partials, after the tile's last pass (custom_mm's sequence)
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_F);  // phase 0 for what follows
        TTI_MOVD2B(0, 32, ADDR_MOD_5, p_movd2a::MOV_4_ROWS, 0 + 8);
        TTI_MOVD2B(0, 16, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 16 + 8);
        if (operandB_face_r_dim == 8) {
            TTI_MOVD2B(0, 0 + 4, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 0 + 8 + 4);
            TTI_MOVD2B(0, 16 + 4, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 16 + 8 + 4);
        }
        TTI_ZEROSRC(0, 1, 0, 1);
        TTI_ELWADD(0, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_6, 0);
        TTI_ELWADD(1, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
    });

    const std::uint32_t mvmul_base = lltt::replay_insn(ckernel::math::replay_buf_offset, 3);
    const std::uint32_t mvmul_end_pass = TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);
    const std::uint32_t mvmul_end_tile = TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0);
    const std::uint32_t mvmul_end_block = TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);

    // One MOP run = one K tile: per output tile (outer) the passes (inner). The template replaces the second
    // instruction of the last inner iteration by the end-of-tile MVMUL, and by the end-of-K-tile one in the last
    // outer iteration.
    ckernel_template tmp = ckernel_template(ct_dim, passes, mvmul_base, mvmul_end_pass);
    tmp.set_last_inner_loop_instr(mvmul_end_tile);
    tmp.set_last_outer_loop_instr(mvmul_end_block);
    tmp.program();
}

template <std::uint32_t passes, std::uint32_t phase_step, bool transpose, CustomMmDestLayout layout>
inline void hifi_mm_math_init(const std::uint32_t operand0, const std::uint32_t operand1, const std::uint32_t ct_dim) {
    using namespace ckernel;
    static_assert(hifi_mm_legal(passes, phase_step), "hifi_mm: phases 0, step, .. must stay below 4");
    // operand0 goes to SrcB and operand1 to SrcA, as in custom_mm
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(get_operand_id(operand0));
    LLK_ASSERT(get_operand_num_faces(get_operand_id(operand1)) == 4, "hifi_mm: in1 must be a [32, 32] tile");
    LLK_ASSERT(ct_dim >= 1 && ct_dim <= CUSTOM_MM_MAX_CT_DIM, "hifi_mm: ct_dim must be 1 to 34");

    hifi_mm_configure_addrmod<transpose, layout, phase_step>();
    hifi_mm_configure_mop(operandB_face_r_dim, ct_dim, passes);
    math::reset_counters(p_setrwc::SET_ABD_F);
    math::_configure_default_zero_flag_state_();
}

// ---------------------------------------------------------------------------------------------------------------
// 16-row tiles: SrcB faces of 16 rows, 8 rows per MVMUL.
template <CustomMmDestLayout layout, std::uint32_t phase_step>
inline void hifi_mm16_configure_addrmod() {
    using namespace ckernel;
    static_assert(
        layout == CustomMmDestLayout::Dense || layout == CustomMmDestLayout::Default,
        "hifi_mm16: a 16-row output tile fills a 32-row DEST slot (Dense or Default layout, no split partials)");
    constexpr std::uint8_t TILE_DEST_INCR = custom_mm_tile_rows(layout);

    // next 8 rows of the same SrcB face, next 8 DEST rows
    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    // next SrcA face along the output width, back to the first 8 rows of the SrcB face (the counter wraps at 64)
    addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 64 - 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    // next face along K: SrcA face 2, SrcB face 1, DEST back to the tile's first rows
    addr_mod_t{
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 1024 - 24, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_5);

    // end of an output tile
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = TILE_DEST_INCR, .clr = 0, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_2);

    // end of a K tile
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_3);

    // end of a pass
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
        .fidelity = {.incr = static_cast<std::uint8_t>(phase_step), .clr = 0},
    }
        .set(ADDR_MOD_4);

    addr_mod_t{
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_7);
}

template <std::uint32_t passes, std::uint32_t phase_step, CustomMmDestLayout layout>
inline void hifi_mm16_math_init(
    const std::uint32_t operand0, const std::uint32_t operand1, const std::uint32_t ct_dim) {
    using namespace ckernel;
    static_assert(hifi_mm_legal(passes, phase_step), "hifi_mm16: phases 0, step, .. must stay below 4");
    LLK_ASSERT(get_operand_face_r_dim(get_operand_id(operand0)) == 16, "hifi_mm16: in0 must be a [16, 32] tile");
    LLK_ASSERT(get_operand_num_faces(get_operand_id(operand1)) == 4, "hifi_mm16: in1 must be a [32, 32] tile");
    LLK_ASSERT(ct_dim >= 1 && ct_dim <= CUSTOM_MM_MAX_CT_DIM, "hifi_mm16: ct_dim must be 1 to 34");

    hifi_mm16_configure_addrmod<layout, phase_step>();
    load_replay_buf(ckernel::math::replay_buf_offset, 7, [] {
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B F0 rows 0..7  x A F0 => rows 0
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // B F0 rows 8..15 x A F0 => rows 8
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B F0 rows 0..7  x A F1 => rows 16
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0);  // B F0 rows 8..15 x A F1 => rows 24
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B F1 rows 0..7  x A F2 => rows 0
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);  // B F1 rows 8..15 x A F2 => rows 8
        TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);  // B F1 rows 0..7  x A F3 => rows 16
    });
    // the eighth MVMUL (B F1 rows 8..15 x A F3 => rows 24) ends the pass, the tile or the K tile
    ckernel_template tmp = ckernel_template(
        ct_dim,
        passes,
        lltt::replay_insn(ckernel::math::replay_buf_offset, 7),
        TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0));
    tmp.set_last_inner_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0));
    tmp.set_last_outer_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0));
    tmp.program();
    math::reset_counters(p_setrwc::SET_ABD_F);
    math::_configure_default_zero_flag_state_();
}

template <CustomMmDestLayout layout>
inline void hifi_mm16_math(const std::uint32_t dst_index, const std::uint32_t kt_dim, const std::uint32_t ct_dim) {
    using namespace ckernel;
    LLK_ASSERT(
        (custom_mm_fits_dest<DST_SYNC_MODE, layout>(dst_index, ct_dim)), "hifi_mm16: the output tiles overrun DEST");
    LLK_ASSERT(kt_dim >= 1 && kt_dim <= CUSTOM_MM_MAX_KT_DIM, "hifi_mm16: kt_dim must be 1 to 256");
    constexpr DstTileShape dst_tile_shape =
        custom_mm_tile_rows(layout) == 32 ? DstTileShape::Tile32x16 : DstTileShape::Tile32x32;
    math::set_dst_write_addr<dst_tile_shape, UnpackDestination::SrcRegs>(dst_index);
    for (std::uint32_t i = 0; i < kt_dim; i++) {
        TTI_MOP(1, 0, 0);
    }
}

// The exact merge of the split accumulators: one SFPU register is 4 DEST rows x 8 lanes (the even or the odd
// columns), sfpi::dst_reg index u = DEST rows 2 u .. (the layout fused/projections/kernels/proj_sfpu.hpp works with).
// Output tile t starts at index TILE_UNITS * t; its output faces are indices 0..3 and 8..11, their split partners
// 4 indices (8 rows) further.
// ROUND_BF16: the sum is also rounded to the nearest BF16 (halves away from zero). Not needed before a pack into a
// BF16 buffer: the packer rounds an FP32 DEST value the same way (measured: fused/fidelity/tests/test_lab.py
// test_packer_bf16; the MoE's outputs are the same with and without it). It is for sums that stay FP32 and must
// carry BF16 values.
template <std::uint32_t CT, std::uint32_t TILE_UNITS, bool ROUND_BF16>
inline void hifi_mm_merge_fn() {
#pragma GCC unroll 34
    for (std::uint32_t t = 0; t < CT; t++) {
#pragma GCC unroll 8
        for (std::uint32_t i = 0; i < 8; i++) {
            const int u = static_cast<int>(TILE_UNITS * t + 8 * (i >> 2) + (i & 3));
            sfpi::vFloat main_acc = sfpi::dst_reg[u];
            sfpi::vFloat split_acc = sfpi::dst_reg[u + 4];
            main_acc = main_acc + split_acc;
            if constexpr (ROUND_BF16) {
                sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(main_acc);
                bits = bits + 0x8000;
                bits = bits >> 16;
                bits = bits << 16;
                main_acc = sfpi::as<sfpi::vFloat>(bits);
            }
            sfpi::dst_reg[u] = main_acc;
        }
    }
}

// The rows of output tiles in a plain layout (no split partners) rounded to the nearest BF16 in place.
template <std::uint32_t CT, std::uint32_t TILE_UNITS>
inline void hifi_mm_round_fn() {
#pragma GCC unroll 34
    for (std::uint32_t t = 0; t < CT; t++) {
#pragma GCC unroll 8
        for (std::uint32_t i = 0; i < 8; i++) {
            const int u = static_cast<int>(TILE_UNITS * t + 8 * (i >> 2) + (i & 3));
            sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::dst_reg[u]));
            bits = bits + 0x8000;
            bits = bits >> 16;
            bits = bits << 16;
            sfpi::dst_reg[u] = sfpi::as<sfpi::vFloat>(bits);
        }
    }
}

template <std::uint32_t passes, CustomMmDestLayout layout, bool finalize>
inline void hifi_mm_math(
    const std::uint32_t operand0,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim) {
    using namespace ckernel;
    static_assert(!finalize || custom_mm_split_acc(layout), "hifi_mm: finalize merges split partials");
    const std::uint32_t operandB_face_r_dim = get_operand_face_r_dim(get_operand_id(operand0));
    LLK_ASSERT(
        (custom_mm_fits_dest<DST_SYNC_MODE, layout>(dst_index, ct_dim)), "hifi_mm: the output tiles overrun DEST");
    LLK_ASSERT(kt_dim >= 1 && kt_dim <= CUSTOM_MM_MAX_KT_DIM, "hifi_mm: kt_dim must be 1 to 256");

    constexpr std::uint32_t tile_rows = custom_mm_tile_rows(layout);
    constexpr DstTileShape dst_tile_shape = tile_rows == 16
                                                ? DstTileShape::Tile16x16
                                                : (tile_rows == 32 ? DstTileShape::Tile32x16 : DstTileShape::Tile32x32);
    math::set_dst_write_addr<dst_tile_shape, UnpackDestination::SrcRegs>(dst_index);

    for (std::uint32_t i = 0; i < kt_dim - 1; i++) {
        TTI_MOP(1, 0, 0);
    }
    if constexpr (finalize) {
        const std::uint32_t replay_len = hifi_mm_replay_len(operandB_face_r_dim);
        for (std::uint32_t c = 0; c < ct_dim; c++) {
            for (std::uint32_t p = 0; p + 1 < passes; p++) {
                lltt::replay(ckernel::math::replay_buf_offset, 4);  // a pass that ends in the phase step
            }
            if (c + 1 < ct_dim) {
                lltt::replay(ckernel::math::replay_buf_offset, replay_len);  // last pass, merge, next tile
            } else {
                // last tile: the merge's last add releases both sources and returns everything to the start
                lltt::replay(ckernel::math::replay_buf_offset, replay_len - 1);
                TTI_ELWADD(3, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
            }
        }
    } else {
        TTI_MOP(1, 0, 0);
    }
}

#endif  // TRISC_MATH && ARCH_BLACKHOLE

}  // namespace hy4fid

namespace ckernel {

#if defined(ARCH_BLACKHOLE)

// The three calls below take custom_mm's arguments (custom_mm.h has the tables) plus, first, `passes` and
// `phase_step`. hifi_mm_block_uninit is custom_mm_block_uninit.

template <
    std::uint32_t passes,
    std::uint32_t phase_step = 1,
    bool transpose = false,
    CustomMmDestLayout layout = CustomMmDestLayout::Default,
    bool fp32_dest_acc_en = DST_ACCUM_MODE,
    bool clear_src = true>
ALWI void hifi_mm_block_init(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t out_cb_id,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_hw_configure<fp32_dest_acc_en>(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_custom_mm_init<transpose, clear_src>(in0_cb_id, in1_cb_id, ct_dim)));

    MATH((llk_math_pack_sync_init<fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<fp32_dest_acc_en>(in0_cb_id, in1_cb_id)));
    MATH((hy4fid::hifi_mm_math_init<passes, phase_step, transpose, layout>(in0_cb_id, in1_cb_id, ct_dim)));

    PACK((llk_pack_dest_init<fp32_dest_acc_en, PackMode::Default>(out_cb_id)));
    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(out_cb_id)));
    PACK((llk_pack_init<PackMode::Default, false /* zero_output */>(out_cb_id)));
    PACK((_llk_pack_custom_mm_init_<layout>(pack_src_format[get_output_id(out_cb_id)])));
}

template <
    std::uint32_t passes,
    std::uint32_t phase_step = 1,
    bool transpose = false,
    CustomMmDestLayout layout = CustomMmDestLayout::Default,
    bool clear_src = true>
ALWI void hifi_mm_block_init_short(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t out_cb_id,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_AB_custom_mm_init<transpose, clear_src>(in0_cb_id, in1_cb_id, ct_dim)));
    MATH((hy4fid::hifi_mm_math_init<passes, phase_step, transpose, layout>(in0_cb_id, in1_cb_id, ct_dim)));
    PACK((_llk_pack_custom_mm_init_<layout>(pack_src_format[get_output_id(out_cb_id)])));
}

template <
    std::uint32_t passes,
    CustomMmDestLayout layout = CustomMmDestLayout::Default,
    bool finalize = custom_mm_split_acc(layout),
    bool read_transposed = false>
ALWI void hifi_mm_block(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t in0_tile_index,
    const std::uint32_t in1_tile_index,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_AB_custom_mm<read_transposed>(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, kt_dim, ct_dim)));
    MATH((hy4fid::hifi_mm_math<passes, layout, finalize>(in0_cb_id, dst_index, kt_dim, ct_dim)));
}

// The exact merge of a SplitAcc layout's accumulators, in place of custom_mm's `finalize`: after the last
// hifi_mm_block of the sums (all of them called with finalize = false) and before the pack or any SFPU finishing.
// The output tiles are those hifi_mm_block wrote from dst_index 0. Another hifi_mm_block may follow without a new
// init (the next sums after a pack): the merge leaves the address mods, the MOP and the replay buffer alone.
// hifi_mm_merge_init once per kernel, after hifi_mm_block_init (it resets the address counters).
ALWI void hifi_mm_merge_init() { MATH((_llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>())); }

template <std::uint32_t ct_dim, CustomMmDestLayout layout, bool round_bf16 = false>
ALWI void hifi_mm_merge() {
    static_assert(custom_mm_split_acc(layout), "hifi_mm_merge: only the SplitAcc layouts have partials to merge");
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        hy4fid::hifi_mm_merge_fn<ct_dim, custom_mm_tile_rows(layout) / 2, round_bf16>, 0, VectorMode::RC_custom)));
}

// `ct_dim` tiles of up to 8 rows from DEST tile 0, spaced `tile_rows` DEST rows apart (64: the standard tile slot),
// rounded to the nearest BF16 in place (halves away from zero: what the packer does when it packs FP32 into a BF16
// buffer). After hifi_mm_merge_init.
template <std::uint32_t ct_dim, std::uint32_t tile_rows = 64>
ALWI void hifi_mm_round_bf16() {
    MATH((_llk_math_eltwise_unary_sfpu_params_(
        hy4fid::hifi_mm_round_fn<ct_dim, tile_rows / 2>, 0, VectorMode::RC_custom)));
}

// 16 token rows: in0 and out are [16, 32] tiles, layout Dense or Default. Otherwise as hifi_mm_block_init / _block
// (no transpose of in1, no finalize: there are no split accumulators). hifi_mm_block_uninit undoes the init.
template <
    std::uint32_t passes,
    std::uint32_t phase_step = 1,
    CustomMmDestLayout layout = CustomMmDestLayout::Dense,
    bool fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void hifi_mm16_block_init(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t out_cb_id,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_hw_configure<fp32_dest_acc_en>(in1_cb_id, in0_cb_id)));
    UNPACK((llk_unpack_AB_custom_mm_init<false, false>(in0_cb_id, in1_cb_id, ct_dim)));

    MATH((llk_math_pack_sync_init<fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<fp32_dest_acc_en>(in0_cb_id, in1_cb_id)));
    MATH((hy4fid::hifi_mm16_math_init<passes, phase_step, layout>(in0_cb_id, in1_cb_id, ct_dim)));

    PACK((llk_pack_dest_init<fp32_dest_acc_en, PackMode::Default>(out_cb_id)));
    PACK((llk_pack_hw_configure<fp32_dest_acc_en>(out_cb_id)));
    PACK((llk_pack_init<PackMode::Default, false /* zero_output */>(out_cb_id)));
    PACK((_llk_pack_custom_mm_init_<layout>(pack_src_format[get_output_id(out_cb_id)])));
}

template <std::uint32_t passes, std::uint32_t phase_step = 1, CustomMmDestLayout layout = CustomMmDestLayout::Dense>
ALWI void hifi_mm16_block_init_short(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t out_cb_id,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_AB_custom_mm_init<false, false>(in0_cb_id, in1_cb_id, ct_dim)));
    MATH((hy4fid::hifi_mm16_math_init<passes, phase_step, layout>(in0_cb_id, in1_cb_id, ct_dim)));
    PACK((_llk_pack_custom_mm_init_<layout>(pack_src_format[get_output_id(out_cb_id)])));
}

template <CustomMmDestLayout layout = CustomMmDestLayout::Dense, bool read_transposed = false>
ALWI void hifi_mm16_block(
    const std::uint32_t in0_cb_id,
    const std::uint32_t in1_cb_id,
    const std::uint32_t in0_tile_index,
    const std::uint32_t in1_tile_index,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1) {
    UNPACK((llk_unpack_AB_custom_mm<read_transposed>(
        in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, kt_dim, ct_dim)));
    MATH((hy4fid::hifi_mm16_math<layout>(dst_index, kt_dim, ct_dim)));
}

template <CustomMmDestLayout layout = CustomMmDestLayout::Default>
ALWI void hifi_mm_block_uninit(const std::uint32_t out_cb_id) {
    custom_mm_block_uninit<layout>(out_cb_id);
}

#endif  // ARCH_BLACKHOLE

}  // namespace ckernel
