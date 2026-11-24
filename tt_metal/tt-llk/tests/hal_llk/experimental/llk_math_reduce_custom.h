// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_reduce_custom.h. Same Tensix instruction sequence as
// production, expressed through the HAL. The helpers in hal_port::reduce_block_max_math are shared with
// llk_math_reduce_runtime_custom.h.

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "hal/counters/rwc.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/replay.h"
#include "hal/src.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_math_reduce.h"
#include "lltt.h"
#include "tensor_shape.h"

using namespace ckernel;

namespace hal_port
{
namespace reduce_block_max_math
{
namespace move = hal::move;
using hal_port::ReduceAddrMod;
using hal_port::set_reduce_addr_mod;
using S = hal::cfg::Sec;

// ADDR_MOD_6: no auto-increment, fidelity cleared. ADDR_MOD_1: SrcA += 16 (CR), SrcB cleared.
// ADDR_MOD_2: Dest += 4 (CR). ADDR_MOD_3: Dest += 20 (CR), jumping to F2.
inline void set_addr_mod_6()
{
    set_reduce_addr_mod<S::S6, ReduceAddrMod {.fid_clr = 1}>();
}

inline void set_addr_mod_1()
{
    set_reduce_addr_mod<S::S1, ReduceAddrMod {.srca_incr = 16, .srca_cr = 1, .srcb_clr = 1}>();
}

inline void set_addr_mod_2()
{
    set_reduce_addr_mod<S::S2, ReduceAddrMod {.dest_incr = 4, .dest_cr = 1}>();
}

inline void set_addr_mod_3()
{
    set_reduce_addr_mod<S::S3, ReduceAddrMod {.dest_incr = 20, .dest_cr = 1}>();
}

// TTI_GMPOOL(CLR_NONE, DIM_16X16, ADDR_MOD_1, INDEX_DIS, 0)
inline constexpr std::uint32_t gmpool = reduce_pool(PoolType::MAX, p_setrwc::CLR_NONE, ADDR_MOD_1, 0);

// TTI_MOVD2B(Lo32 ? DEST_32B_LOW : DEST_NORM, SRC_ROW16_OFFSET, ADDR_MOD_6, MOV_1_ROW, 0)
template <bool Lo32>
inline __attribute__((always_inline)) void movd2b_row0()
{
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row       = 0,
        .destination_row  = p_movd2b::SRC_ROW16_OFFSET,
        .address_mode     = ADDR_MOD_6,
        .number_of_rows   = move::rows::One,
        .source_32bit_low = Lo32}>();
}

// TTI_MOVB2D(Lo32 ? DEST_32B_LOW : DEST_NORM, SRC_ROW16_OFFSET + SrcRow, AddrMod, MOV_4_ROWS, DestRow)
template <bool Lo32, std::uint32_t SrcRow, std::uint8_t AddrMod, std::uint32_t DestRow>
inline __attribute__((always_inline)) void movb2d()
{
    move::run<move::Transfer<move::SrcB, move::Dst> {
        .source_row            = p_movb2d::SRC_ROW16_OFFSET + SrcRow,
        .destination_row       = DestRow,
        .address_mode          = AddrMod,
        .number_of_rows        = move::rows::Four,
        .destination_32bit_low = Lo32}>();
}

// TTI_TRNSPSRCB
inline __attribute__((always_inline)) void transpose_srcb()
{
    hal::src<hal::SrcB>.transpose();
}

// TTI_SETRWC(CLR_B, 0, 0, 0, 0, SET_ABD): clear B valid bits and all address counters
inline __attribute__((always_inline)) void clear_b_reset_abd()
{
    rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::B}>();
}

// Two GMPOOLs to pool F0 and F1 (or F2 and F3) together
inline __attribute__((always_inline)) void two_gmpools()
{
    TTI_INSN(gmpool);
    TTI_INSN(gmpool);
}

// Records the replay buffer at slots 0.. (15 instructions, or 9 for a non-FP32 single face-row).
template <bool is_fp32_dest_acc_en>
inline void record_replay(const ckernel::TensorShape &tensor_shape)
{
    namespace replay = hal::replay;
    if (tensor_shape.num_faces_r_dim == 1)
    {
        // Single face-row (16x32 tiny tile): only F0&F1 exist. Reduce them, transpose once, no F2 jump.
        if constexpr (is_fp32_dest_acc_en)
        {
            // FP32 path records the same 15-instruction layout as num_faces=4 (2 GMPOOLs + hi/lo
            // transpose + CLR_B). Only the execute-side replay differs (single face-row, done below).
            replay::record<replay::BufferRange {0, 15}>(
                []
                {
                    two_gmpools();

                    // Move high 16 bits from DEST row 0 to SrcB rows 16 - 31 and transpose
                    movd2b_row0<false>();
                    transpose_srcb();
                    // Move high 16 bits back to Dest
                    movb2d<false, 0, ADDR_MOD_6, 0>();
                    movb2d<false, 4, ADDR_MOD_6, 4>();
                    movb2d<false, 8, ADDR_MOD_6, 8>();
                    movb2d<false, 12, ADDR_MOD_6, 12>();
                    // Move low 16 bits to SrcB rows 16 - 31 and transpose
                    movd2b_row0<true>();
                    transpose_srcb();
                    // Move low 16 bits from SrcB rows 16 - 31 to DEST rows 0, 4, 8, 12.
                    // ADDR_MOD_2 increments CR_D and Dest counter val by 4. No F2 for a single face-row, so
                    // the last write uses ADDR_MOD_2 (no +20 jump); CLR_B resets the counters afterwards.
                    movb2d<true, 0, ADDR_MOD_2, 0>();
                    movb2d<true, 4, ADDR_MOD_2, 0>();
                    movb2d<true, 8, ADDR_MOD_2, 0>();
                    movb2d<true, 12, ADDR_MOD_2, 0>();
                    clear_b_reset_abd();
                });
        }
        else
        {
            // Non-FP32 single face-row: 2 GMPOOLs + one transpose block (6 instrs) + CLR_B = 9 instructions.
            replay::record<replay::BufferRange {0, 9}>(
                []
                {
                    two_gmpools();

                    // Move row 0 from DEST to SrcB with offset of 16 rows and transpose
                    movd2b_row0<false>();
                    transpose_srcb();
                    // Move the reduced row from SrcB to DEST in 4-row chunks (rows 0, 4, 8, 12).
                    // ADDR_MOD_2 increments CR_D and Dest counter val by 4. No F2 jump (no ADDR_MOD_3): the
                    // last write uses ADDR_MOD_2 and CLR_B resets the counters afterwards.
                    movb2d<false, 0, ADDR_MOD_2, 0>();
                    movb2d<false, 4, ADDR_MOD_2, 0>();
                    movb2d<false, 8, ADDR_MOD_2, 0>();
                    movb2d<false, 12, ADDR_MOD_2, 0>();
                    clear_b_reset_abd();
                });
        }
        return;
    }

    // See _llk_math_reduce_max_row_ for a full algorithm explanation
    // Put the following 15 instructions in a REPLAY buffer
    replay::record<replay::BufferRange {0, 15}>(
        []
        {
            two_gmpools();

            // The following instructions are going to transpose the whole tile.

            // Move row 0 from DEST to SrcB with offset of 16 rows and transpose
            movd2b_row0<false>();
            transpose_srcb();
            // Move row 0 from SrcB to DEST in 4-row chunks
            // ADDR_MOD_2 increments CR_D and Dest counter val by 4, so that's why DEST location is '0', not '0, 4, 8, 12'.
            movb2d<false, 0, ADDR_MOD_2, 0>();
            movb2d<false, 4, ADDR_MOD_2, 0>();
            movb2d<false, 8, ADDR_MOD_2, 0>();
            // ADDR_MOD_3 increments CR_D and Dest counter val by 20, to point to F2.
            movb2d<false, 12, ADDR_MOD_3, 0>();

            // Move row 32 (F2R0) from DEST to SrcB with offset of 16 rows and transpose
            movd2b_row0<false>();
            transpose_srcb();
            // Move row 32 from SrcB to DEST in 4-row chunks
            movb2d<false, 0, ADDR_MOD_6, 0>();
            movb2d<false, 4, ADDR_MOD_6, 4>();
            movb2d<false, 8, ADDR_MOD_6, 8>();
            movb2d<false, 12, ADDR_MOD_6, 12>();

            clear_b_reset_abd();
        });
}

// Reduce F0 and F1 column-wise, doesn't change DEST counter: TT_OP_REPLAY(0, 2, 0, 0)
inline constexpr std::uint32_t replay_gmpools = hal::replay::get_operation<hal::replay::BufferRange {0, 2}>();
// Clear A valid bit to get another operand tile (scaler face stays the same): TT_OP_SETRWC(CLR_A, 0, 0, 0, 0, SET_ABD)
inline constexpr std::uint32_t clear_a_reset_abd =
    rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::A}.operation();
// Increment DEST counter by 8 (CR) to step towards F2 (DEST row 32): TT_OP_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)
// HAL-GAP: rwc encodes a lone Dest AdvanceAndReset as INCRWC; production's SETRWC encoding is kept.
inline constexpr std::uint32_t dest_cr_plus_8 = TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);

// The per-tile MOP: start REPLAY(0, 2); for two face-rows, inner 4x Dest += 8 then REPLAY(0, 2) and CLR_A;
// for one face-row, inner NOP then CLR_A.
inline void program_mop(const std::uint32_t outer_loop, const ckernel::TensorShape &tensor_shape)
{
    using namespace hal::mop;
    if (tensor_shape.num_faces_r_dim == 1)
    {
        // No F2 jump for a single face-row
        program(loop(outer_loop, replay_gmpools, loop(1, TT_OP_NOP), clear_a_reset_abd, TT_OP_NOP));
        return;
    }
    // Reduce F2 and F3 column-wise as end op 0, then clear A valid bits
    program(loop(outer_loop, replay_gmpools, loop(4, dest_cr_plus_8), replay_gmpools, clear_a_reset_abd));
}

// Per tile: the MOP issues the GMPOOLs, the transpose is replayed here.
template <bool is_fp32_dest_acc_en>
inline __attribute__((always_inline)) void run_tile(const std::uint32_t dst_index, const ckernel::TensorShape &tensor_shape)
{
    namespace replay = hal::replay;
    // Packer indexes at the 32x32 slot stride regardless of the operand's face count.
    set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    // Run the MOP. For num_faces=4 this reduces both face-rows (F0&F1 then F2&F3); for num_faces=2
    // it reduces the single face-row (F0&F1). The MOP only emits the GMPOOLs; the transpose is
    // replayed below.
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    if (tensor_shape.num_faces_r_dim == 1)
    {
        // Single face-row: transpose only the one recorded face-row. The 2-face record has no
        // ADDR_MOD_3 F2 jump, so no spurious DEST advance occurs.
        if constexpr (is_fp32_dest_acc_en)
        {
            // MOVB2D/D2B depends on the SrcA ALU format; Hi/Lo16 transpose does not work with the
            // Tf32 that the pool wrote, so override SrcA to Tf32 (and disable the zero-flag source)
            // for the replayed transpose, exactly as the num_faces=4 path does below.
            set_disable_implied_srca_fmt<1>();
            _configure_src_zero_flag_(true);
            set_srca_format<DataFormat::Tf32>();

            // Replay the 13 instructions (recorded slots 2-14): hi/lo transpose of the single reduced
            // face-row + CLR_B.
            replay::run<replay::BufferRange {2, 13}>();

            set_disable_implied_srca_fmt<0>();
            _configure_src_zero_flag_(false);
        }
        else
        {
            // Replay the 7 instructions (recorded slots 2-8) to transpose the single reduced
            // face-row. 7th instruction (CLR_B) releases SrcB bank and clears all address counters.
            replay::run<replay::BufferRange {2, 7}>();
        }
        return;
    }

    if constexpr (is_fp32_dest_acc_en)
    {
        set_disable_implied_srca_fmt<1>();
        _configure_src_zero_flag_(true);
        set_srca_format<DataFormat::Tf32>();
    }

    // Replay the 13 instructions to transpose the reduced results
    replay::run<replay::BufferRange {2, 13}>();

    if constexpr (is_fp32_dest_acc_en)
    {
        set_disable_implied_srca_fmt<0>();
        _configure_src_zero_flag_(false);
    }
}

// TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0) then math::reset_counters(SET_ABD_F)
inline void enable_clear_dvalid_and_reset_counters()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());
    reset_counters_abd_f();
}

} // namespace reduce_block_max_math
} // namespace hal_port

/**
 * Configures address modifiers for specialized reduce_max_row operations.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block/tile operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32 (num_faces=4) or 16x32 (num_faces=2, a single face-row)
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * @note ADDR_MOD_3 (the +20 DEST jump into F2) is only consumed by the two-face-row
 *       (num_faces=4) code path. For num_faces=2 there is no F2/F3, so it is unused, but it
 *       is still programmed here so that both code paths share one addrmod configuration.
 *
 * This function should NOT be used as a substitute for native reduce LLK configuration.
 * Use the standard reduce configuration functions for general-purpose reduction operations.
 */
inline void reduce_max_row_configure_addrmod()
{
    namespace rbm = hal_port::reduce_block_max_math;
    rbm::set_addr_mod_6(); // ADDR_MOD_6: default mode used with pool operations (no auto-increment)
    rbm::set_addr_mod_1(); // ADDR_MOD_1: face-to-face advancement used with GMPOOL (SrcA += 16, SrcB cleared)
    rbm::set_addr_mod_2(); // ADDR_MOD_2: sequential 4-row write mode used with MOVB2D (Dest += 4)
    rbm::set_addr_mod_3(); // ADDR_MOD_3: face-boundary jump used with the final MOVB2D in each face (Dest += 20)
}

// Minimal reinit: ADDR_MOD_1 + ADDR_MOD_2 (clobbered by matmul) and ADDR_MOD_6
// (clobbered by sub_exp runtime). Skips ADDR_MOD_3 which is preserved from the
// previous reduce (matmul doesn't touch 3, sub_exp custom doesn't touch 3,
// copy_tile_custom_v2 doesn't touch 3).
inline void reduce_max_row_configure_addrmod_reinit_minimal()
{
    namespace rbm = hal_port::reduce_block_max_math;
    rbm::set_addr_mod_6();
    rbm::set_addr_mod_1();
    rbm::set_addr_mod_2();
}

/**
 * Configures MOP (Macro Operation) for block-based reduce_max_row operations.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * For num_faces=4 the per-tile MOP reduces F0&F1 into DEST rows 0-15, jumps DEST to row 32 via a
 * CR_D+8 SETRWC repeated inner_loop=4 times (4*8=32 = row 32 = F2), then reduces F2&F3, and the
 * transpose (replayed by the execute function) covers both face-rows.
 *
 * For num_faces=2 there is only one face-row (F0&F1): the per-tile MOP reduces F0&F1 only, with no
 * F2 jump and no second reduce, and the recorded transpose block covers the single face-row.
 *
 * This function should NOT be used as a substitute for native reduce LLK MOP configuration.
 * Use the standard reduce MOP configuration with _llk_math_reduce_init_ for general-purpose reduction.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_mop_config_(const ckernel::TensorShape tensor_shape)
{
    // Constraint on the outerloop and innerloop dim
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(!(tensor_shape.num_faces_r_dim == 1 && is_fp32_dest_acc_en), "16x32 reduce_block_max_row not supported in FP32 dest mode yet");

    hal_port::reduce_block_max_math::record_replay<is_fp32_dest_acc_en>(tensor_shape);
    hal_port::reduce_block_max_math::program_mop(block_ct_dim, tensor_shape);
}

/**
 * Reprograms only the MOP registers for reduce_max_row, without re-recording the replay buffer.
 * The replay buffer at positions 0-14 must still contain the reduce GMPOOL+transpose sequence
 * from the original _llk_math_reduce_block_max_row_mop_config_ call.
 * Use when the MOP was clobbered (e.g., by eltwise binary ops) but the replay buffer is intact.
 */
template <std::uint32_t block_ct_dim>
inline void _llk_math_reduce_block_max_row_mop_reprogram_only_(const ckernel::TensorShape tensor_shape)
{
    static_assert(block_ct_dim < 128, "block_ct_dim must be less than 128");
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    hal_port::reduce_block_max_math::program_mop(block_ct_dim, tensor_shape);
}

/**
 * Initializes block-based reduce_max_row operation for processing multiple tiles.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native _llk_math_reduce_init_ LLK.
 * Use the standard _llk_math_reduce_init_<PoolType::MAX, ReduceDim::REDUCE_ROW>() with multiple
 * _llk_math_reduce_() calls in a loop for general-purpose block reduction.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_init_(const ckernel::TensorShape& tensor_shape)
{
    reduce_max_row_configure_addrmod();

    hal_port::reduce_block_max_math::enable_clear_dvalid_and_reset_counters();

    _llk_math_reduce_block_max_row_mop_config_<block_ct_dim, is_fp32_dest_acc_en>(tensor_shape);

    // The reduce runs on the FPU pool (GMPOOL). Restore the operand-driven zero-flag baseline so a
    // preceding datacopy/copy_init that left PRESERVE (keep denormals) does not leak into it, mirroring
    // the standard _llk_math_reduce_init_.
    hal_port::_configure_default_zero_flag_state_();
}

template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_uninit_()
{
}

/**
 * Performs block-based reduce_max_row operation across multiple tiles in the width dimension.
 *
 * This function works with the following assumptions:
 * - Scaler values are 1.0 and are contained inside F0 of the scaler tile
 * - The scaler doesn't change for the duration of the whole block operation
 * - Operand and scaler data format is bfloat16_b
 * - Operand tile size is 32x32
 * - Can work on both 16-bit or 32-bit DEST register modes based on is_fp32_dest_acc_en flag
 * - Does only MAX pool on ROW dimension
 *
 * This function should NOT be used as a substitute for the native _llk_math_reduce_ LLK.
 * Use the standard _llk_math_reduce_<PoolType::MAX, ReduceDim::REDUCE_ROW>() in a loop
 * for general-purpose block reduction across multiple tiles.
 */
template <std::uint32_t block_ct_dim, bool is_fp32_dest_acc_en = false>
inline void _llk_math_reduce_block_max_row_(const std::uint32_t dst_index, const ckernel::TensorShape tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");
    LLK_ASSERT(!(tensor_shape.num_faces_r_dim == 1 && is_fp32_dest_acc_en), "16x32 reduce_block_max_row not supported in FP32 dest mode yet");

    hal_port::reduce_block_max_math::run_tile<is_fp32_dest_acc_en>(dst_index, tensor_shape);
}
