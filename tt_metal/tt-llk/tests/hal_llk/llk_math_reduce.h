// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_math_reduce.h, except SFPU: the SFPLOAD/SFPSTORE lines stay as production
// has them. Same Tensix instruction sequence as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/replay.h"
#include "hal/src.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_math.h"

using namespace ckernel;

namespace hal_port
{

// One ADDR_MOD slot in the units of ckernel::addr_mod_t (every field the reduce sets, including SrcA/SrcB CR).
struct ReduceAddrMod
{
    std::uint32_t srca_incr = 0;
    std::uint32_t srca_clr  = 0;
    std::uint32_t srca_cr   = 0;
    std::uint32_t srcb_incr = 0;
    std::uint32_t srcb_clr  = 0;
    std::uint32_t srcb_cr   = 0;
    std::uint32_t dest_incr = 0;
    std::uint32_t dest_clr  = 0;
    std::uint32_t dest_cr   = 0;
    std::uint32_t fid_incr  = 0;
    std::uint32_t fid_clr   = 0;
};

// addr_mod_t::set(n): three SETC16 words (SrcA|SrcB, Dest|Fidelity, Bias), each fully specified.
template <cfg::Sec S, ReduceAddrMod M>
inline void set_reduce_addr_mod()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, S, M.srca_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, S, M.srca_cr>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, S, M.srca_clr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, S, M.srcb_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, S, M.srcb_cr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, S, M.srcb_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, S, M.dest_incr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, S, M.dest_cr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, S, M.dest_clr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, S, M.fid_incr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, S, M.fid_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::Bias].Incr, S, 0>(), cfg::set<cfg::AddrMod[cfg::Bias].Clear, S, 0>());
}

// p_setrwc::CLR_NONE / CLR_AB as an FPU source release.
constexpr hal::fpu::SourceRelease reduce_release(const std::uint32_t clear_mode)
{
    return clear_mode == p_setrwc::CLR_AB ? hal::fpu::SourceRelease::Both : hal::fpu::SourceRelease::None;
}

// TT_OP_MVMUL(clear_mode, 0, addr_mod, dst)
constexpr std::uint32_t reduce_mvmul(const std::uint32_t clear_mode, const std::uint8_t addr_mod, const std::uint16_t dst)
{
    return hal::fpu::get_operation(hal::fpu::MatrixMultiply {.address_modifier = addr_mod, .dest_row_offset = dst, .release = reduce_release(clear_mode)});
}

// TT_OP_GMPOOL / TT_OP_GAPOOL(clear_mode, DIM_16X16, addr_mod, INDEX_DIS, dst)
constexpr std::uint32_t reduce_pool(const PoolType type, const std::uint32_t clear_mode, const std::uint8_t addr_mod, const std::uint16_t dst)
{
    return hal::fpu::get_operation(hal::fpu::Pool {
        .function         = type == PoolType::MAX ? hal::fpu::PoolFunction::Maximum : hal::fpu::PoolFunction::Sum,
        .indices          = hal::fpu::IndexTracking::Disabled,
        .address_modifier = addr_mod,
        .dest_row_offset  = dst,
        .release          = reduce_release(clear_mode)});
}

// TT_OP_SETRWC(CLR_AB, 0, 0, 0, 0, SET_ABD_F)
inline constexpr std::uint32_t setrwc_clr_ab_abd_f =
    rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::A | rwc::Src::B, .clear_fidelity = true}.operation();
// TT_OP_SETRWC(CLR_AB, 0, 0, 0, 0, SET_B)
inline constexpr std::uint32_t setrwc_clr_ab_b = rwc::MathCounter {.src_b = rwc::Set(0), .flip = rwc::Src::A | rwc::Src::B}.operation();

} // namespace hal_port

// local function declarations
template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_addrmod(const ckernel::TensorShape& tensor_shape);

template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_mop(const ckernel::TensorShape& tensor_shape);

/**
 * @brief Transpose the pooled row result through SrcB so a row reduction lands as a column in the destination register.
 *
 * Only used for MAX pool (GMPOOL does column wise max of SrcA only).
 *
 * @tparam is_int_fpu_en: Cast int32 dest datums to int8 (via SFPU) before moving to SrcB.
 */
template <bool is_int_fpu_en>
inline void reduce_row_perform_transpose()
{
    // The MOVD2B/ELWADD below read the Src zero substitution flag (FlushDenormals = !flag).
    // A datum whose low byte is zero (e.g. bf16 0x4400 = 768.0) would be flushed to 0 mid-reduction,
    // corrupting the sum. Disable the flag (via the math state tracker) around the transpose+add, then
    // return it to the operand driven baseline. WH does the same in its fp32 transpose.
    namespace cfg  = hal::cfg;
    namespace move = hal::move;
    namespace rwc  = hal::counters::rwc;
    namespace sync = hal::sync;

    hal_port::_configure_preserve_zero_flag_state_();

    if constexpr (is_int_fpu_en)
    {
        sync::wait::stall<sync::StallTarget::Sfpu, sync::StallCondition::MathIdle>();
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT8, ADDR_MOD_0, 0);
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, 2);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT8, ADDR_MOD_0, 2);
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SfpuIdle>();
        cfg::write<cfg::Access::TensixCfgUnit, cfg::Fp16aForce::Enable, cfg::Sec::S0, 1>();
    }

    // Move pooled result to SrcB rows 16–31 and transpose
    // TTI_SETRWC(CLR_NONE, CR_D, 0, 0, 0, SET_AB): the CR_D bit has no effect without SET_D, so the HAL leaves it out.
    rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0)}>();
    // TTI_MOVD2B(0, SRC_ROW16_OFFSET, ADDR_MOD_0, MOV_1_ROW, 0)
    constexpr move::Transfer<move::Dst, move::SrcB> dest_row0_to_srcb_row16 {
        .source_row = 0, .destination_row = p_movd2b::SRC_ROW16_OFFSET, .address_mode = ADDR_MOD_0, .number_of_rows = move::rows::One};
    move::run<dest_row0_to_srcb_row16>();
    hal::src<hal::SrcB>.transpose();
    move::run<dest_row0_to_srcb_row16>();

    if constexpr (is_int_fpu_en)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::Fp16aForce::Enable, cfg::Sec::S0, 0>();
    }

    // Add transposed SrcB into Dest through a zeroed SrcA
    // HAL-GAP: rwc::set encodes a lone SrcB AdvanceAndReset as INCRWC; production issues SETRWC, kept as is.
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
    // TTI_ZEROSRC(0, 1, 0, 1): zero the math thread's current SrcA bank
    hal::src<hal::SrcA>.fill<hal::src_ops::BankScope::Current, hal::src_ops::BankOwner::Math, hal::src_ops::Fill::Zero>();
    // TTI_ELWADD(0, 0, SRCB_NO_BCAST, ADDR_MOD_1, 0), twice
    constexpr hal::fpu::Elementwise add_srcb {
        .operation        = hal::fpu::ElementwiseOperation::Add,
        .broadcast        = hal::fpu::SrcBBroadcast::None,
        .dest_write       = hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = ADDR_MOD_1,
        .dest_row_offset  = 0,
        .release          = hal::fpu::SourceRelease::None};
    hal::fpu::run<add_srcb>();
    hal::fpu::run<add_srcb>();

    // Restore the operand-driven baseline for the currently configured formats.
    hal_port::_configure_default_zero_flag_state_();
}

/**
 * @brief Pool one face into the destination register, dispatching to the right pool instruction for the op type.
 *
 * MAX uses GMPOOL; SUM/AVG use GAPOOL directly (LoFi) or the preconfigured MOP followed by a DVALID clear (high fidelity).
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam high_fidelity: Run the multi-phase fidelity MOP instead of a single GAPOOL.
 * @tparam clear_mode: Source-clear mode applied after pooling (p_setrwc::CLR_* value).
 * @tparam index: Destination-register offset (the GMPOOL/GAPOOL dst field) the pooled result is written to.
 */
template <PoolType type, bool high_fidelity, std::uint32_t clear_mode, std::uint32_t index = 0>
inline void reduce_pool_op()
{
    static_assert(clear_mode == p_setrwc::CLR_NONE || clear_mode == p_setrwc::CLR_AB, "reduce pools release no source or both");
    if constexpr (type == PoolType::MAX)
    {
        constexpr std::uint32_t gmpool = hal_port::reduce_pool(PoolType::MAX, clear_mode, ADDR_MOD_0, index);
        TTI_INSN(gmpool);
    }
    else if constexpr (high_fidelity)
    {
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        if constexpr (clear_mode != p_setrwc::CLR_NONE)
        {
            // TTI_CLEARDVALID(CLR_AB, 0)
            hal::src<hal::BothSources>.release<hal::src_ops::BankAdvance::Flip>();
        }
    }
    else
    {
        constexpr std::uint32_t gapool = hal_port::reduce_pool(type, clear_mode, ADDR_MOD_0, index);
        TTI_INSN(gapool);
    }
}

/**
 * @brief Pool one row of faces, clearing SrcA/B between faces and keeping the final result.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam high_fidelity: Run the multi-phase fidelity MOP instead of a single GAPOOL.
 * @param num_faces_c_dim: Number of column faces in the row.
 */
template <PoolType type, bool high_fidelity>
inline void reduce_row_pool_all_faces(const std::uint32_t num_faces_c_dim)
{
    for (std::uint32_t col_num = 0; col_num < num_faces_c_dim - 1; col_num++)
    {
        reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, 0>();
    }
    reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
}

/**
 * @brief Advance the dest counter past the current face row so the next row writes to the correct offset.
 *
 * @param is_narrow_tile: True when tile width < tile height (num_faces_c < num_faces_r), halving the stride.
 */
inline void reduce_row_advance_dest(const bool is_narrow_tile)
{
    namespace rwc = hal::counters::rwc;

    // HAL-GAP: rwc::set encodes a lone Dest AdvanceAndReset as INCRWC; production issues SETRWC, kept as is.
    if (!is_narrow_tile)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    // TTI_SETRWC(CLR_AB, CR_D, 8, 0, 0, SET_BD)
    rwc::set<rwc::MathCounter {.src_b = rwc::Set(0), .dest = rwc::AdvanceAndReset(8), .flip = rwc::Src::A | rwc::Src::B}>();
}

/**
 * @brief Configure the reduce MOP for all paths: REDUCE_ROW SUM/AVG records a dual replay-buffer
 *        MOP, high-fidelity COL/SCALAR programs a multi-phase GAPOOL loop, all other paths need no MOP.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Tile shape determining face count and dest stride.
 */
template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_mop(const ckernel::TensorShape& tensor_shape)
{
    using namespace hal::mop;
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    if constexpr (dim == ReduceDim::REDUCE_ROW && type != PoolType::MAX)
    {
        if (tensor_shape.total_num_faces() == 4)
        {
            constexpr std::uint32_t last_clr    = high_fidelity ? p_setrwc::CLR_NONE : p_setrwc::CLR_AB;
            constexpr std::uint32_t inner_loops = high_fidelity ? to_underlying(math_fidelity) : 1;
            constexpr hal::replay::BufferRange replay_range {ckernel::math::replay_buf_offset /*start*/, 8 /*count*/};

            hal::replay::record<replay_range>(
                []
                {
                    using hal_port::reduce_mvmul;
                    TTI_INSN(reduce_mvmul(p_setrwc::CLR_NONE, ADDR_MOD_0, 0));
                    TTI_INSN(reduce_mvmul(p_setrwc::CLR_NONE, ADDR_MOD_1, 0));
                    TTI_INSN(reduce_mvmul(p_setrwc::CLR_NONE, ADDR_MOD_0, 0));
                    TTI_INSN(reduce_mvmul(p_setrwc::CLR_NONE, ADDR_MOD_2, 0));
                    TTI_INSN(reduce_mvmul(p_setrwc::CLR_NONE, ADDR_MOD_0, 0));
                    TTI_INSN(reduce_mvmul(p_setrwc::CLR_NONE, ADDR_MOD_1, 0));
                    TTI_INSN(reduce_mvmul(p_setrwc::CLR_NONE, ADDR_MOD_0, 0));
                    TTI_INSN(reduce_mvmul(last_clr, ADDR_MOD_3, 0));
                });

            constexpr std::uint32_t replay_op = hal::replay::get_operation<replay_range>();
            if constexpr (high_fidelity)
            {
                program(loop(1, loop(inner_loops, replay_op), hal_port::setrwc_clr_ab_abd_f));
            }
            else
            {
                program(loop(1, loop(inner_loops, replay_op)));
            }
        }
        else
        {
            constexpr std::uint32_t fid_count = (math_fidelity == MathFidelity::LoFi) ? 1 : to_underlying(math_fidelity);
            const hal::replay::BufferRange replay_range {ckernel::math::replay_buf_offset /*start*/, tensor_shape.num_faces_c_dim * 2 * fid_count};

            hal::replay::record(
                replay_range,
                [&tensor_shape]
                {
                    using hal_port::reduce_mvmul;
                    for (std::uint32_t faces_remaining = tensor_shape.num_faces_c_dim; faces_remaining > 0; faces_remaining--)
                    {
                        for (std::uint32_t p = 0; p < fid_count - 1; p++)
                        {
                            TTI_INSN(reduce_mvmul(p_setrwc::CLR_NONE, ADDR_MOD_0, 0));
                        }
                        TTI_INSN(reduce_mvmul(p_setrwc::CLR_NONE, ADDR_MOD_1, 0));

                        for (std::uint32_t p = 0; p < fid_count - 1; p++)
                        {
                            TTI_INSN(reduce_mvmul(p_setrwc::CLR_NONE, ADDR_MOD_0, 8));
                        }
                        if (faces_remaining == 1)
                        {
                            TTI_INSN(reduce_mvmul(p_setrwc::CLR_NONE, ADDR_MOD_3, 8));
                        }
                        else
                        {
                            TTI_INSN(reduce_mvmul(p_setrwc::CLR_AB, ADDR_MOD_2, 8));
                        }
                    }
                });

            program(loop(tensor_shape.num_faces_r_dim, loop(1, hal::replay::get_operation(replay_range)), hal_port::setrwc_clr_ab_b));
        }
    }
    else if constexpr (high_fidelity)
    {
        constexpr std::uint32_t inner_loop_len = to_underlying(math_fidelity);
        constexpr std::uint16_t dst            = (dim == ReduceDim::REDUCE_SCALAR) ? 4 : 0;

        // Always GAPOOL, as production. Production also sets the last-inner-loop op (MopCfg word 8) to the ADDR_MOD_0
        // GAPOOL; with an outer count of 1 the expander never issues it, and the HAL rejects that patch, so word 8 keeps
        // the ADDR_MOD_1 GAPOOL.
        program(loop(1, loop(inner_loop_len, hal_port::reduce_pool(PoolType::SUM, p_setrwc::CLR_NONE, ADDR_MOD_1, dst)))
                    .patch(last_outer, hal_port::reduce_pool(PoolType::SUM, p_setrwc::CLR_NONE, ADDR_MOD_0, dst)));
    }
}

/**
 * @brief Perform a reduction on the math thread, pooling faces into the destination register.
 *
 * REDUCE_ROW SUM/AVG uses MVMUL to multiply each face row by a column-scaler in SrcA, producing the reduced
 * column directly in dest; MAX pools each face row then transposes the result into dest via SrcB.
 * REDUCE_COL pools down rows of faces; REDUCE_SCALAR pools all faces then transposes the partial result
 * into a single column for a final pool.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam is_int_fpu_en: Enable integer FPU datapath (casts int32 dest datums to int8 before moving to SrcB).
 * @param dst_index: Tile index into the destination register.
 * @param tensor_shape: Tensor shape describing tile dimensions.
 * @note Call @ref _llk_math_reduce_init_ with matching template args before this
 *       function, and @ref _llk_math_reduce_uninit_ after it to restore modified state.
 */
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool is_int_fpu_en = false>
inline void _llk_math_reduce_(const std::uint32_t dst_index, const ckernel::TensorShape tensor_shape)
{
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_reduce_", tensor_shape);

    // Supported narrow tiles per BH Tiny Tile Summary: [16]x16 (num_faces=1) and [32]x16 (num_faces=2) only
    LLK_ASSERT(
        !((tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim) /* narrow_tile */ && tensor_shape.total_num_faces() == 4),
        "Reduce narrow tile requires num_faces 1 or 2; num_faces=4 is full-width 32x32");

    namespace rwc  = hal::counters::rwc;
    namespace move = hal::move;
    using Mop      = hal::mop::Runner<hal::mop::MopTemplate::Template1>;

    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr (dim == ReduceDim::REDUCE_ROW)
    {
        const bool is_narrow_tile = tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim;

        if constexpr (type == PoolType::MAX)
        {
            reduce_row_pool_all_faces<type, high_fidelity>(tensor_shape.num_faces_c_dim);
            reduce_row_perform_transpose<is_int_fpu_en>();

            if (tensor_shape.num_faces_r_dim > 1)
            {
                reduce_row_advance_dest(is_narrow_tile);
                reduce_row_pool_all_faces<type, high_fidelity>(tensor_shape.num_faces_c_dim);
                reduce_row_perform_transpose<is_int_fpu_en>();
            }
            // TTI_SETRWC(CLR_AB, 0, 0, 0, 0, SET_BD)
            rwc::set<rwc::MathCounter {.src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::A | rwc::Src::B}>();
        }
        else
        {
            Mop::run();
            // TTI_SETRWC(CLR_NONE, 0, 0, 0, 0, SET_ABD)
            rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0)}>();
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_COL)
    {
        for (std::uint32_t row_num = 0; row_num < tensor_shape.num_faces_r_dim; row_num++)
        {
            // Just pool
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
            if (tensor_shape.num_faces_c_dim > 1)
            {
                // HAL-GAP: rwc::set encodes a lone Dest AdvanceAndReset as INCRWC; production issues SETRWC, kept as is.
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
                // TTI_SETRWC(CLR_AB, CR_D, 8, 0, 0, SET_D)
                rwc::set<rwc::MathCounter {.dest = rwc::AdvanceAndReset(8), .flip = rwc::Src::A | rwc::Src::B}>();

                reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
            }
            // Reset Dest Counter: TTI_SETRWC(CLR_AB, 0, 0, 0, 0, SET_AD)
            rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::A | rwc::Src::B}>();
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        for (std::uint32_t face_num = 0; face_num < static_cast<std::uint32_t>(tensor_shape.total_num_faces() - 1); face_num++)
        {
            // Wait and pool
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, 4>();
        }
        // Wait and pool
        reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 4>();

        // Need row in dest as column in src A
        // TTI_SETRWC(CLR_NONE, CR_D, 0, 0, 0, SET_AB): the CR_D bit has no effect without SET_D, so the HAL leaves it out.
        rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0)}>();

        // copy over from dest to B and do transpose
        // use rows 16 - 31 in src B as scratch
        // TTI_MOVD2B(0, SRC_ROW16_OFFSET, ADDR_MOD_0, MOV_1_ROW, 4)
        move::run<move::Transfer<move::Dst, move::SrcB> {
            .source_row = 4, .destination_row = p_movd2b::SRC_ROW16_OFFSET, .address_mode = ADDR_MOD_0, .number_of_rows = move::rows::One}>();
        hal::src<hal::BothSources>.reset_pipeline_gating(); // TTI_GATESRCRST(0b1, 0b1)
        hal::src<hal::SrcB>.transpose();
        // gate math instructions until src B has been updated
        hal::src<hal::BothSources>.reset_pipeline_gating();
        // copy over all 16 rows from B to A
        // HAL-GAP: hal::move has no SrcB-to-SrcA transfer (MOVB2A); production lines kept.
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
        // gate math instructions until src A has been updated by MOV instructions
        hal::src<hal::BothSources>.reset_pipeline_gating();
        // zero out scratch in dest: TTI_ZEROACC(CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 4)
        hal::dst::run<hal::dst::Zero {.scope = hal::dst::ZeroScope::SingleRow, .index = 4, .width = hal::dst::DestWidth::Bits16, .address_mode = ADDR_MOD_0}>();

        if constexpr (type == PoolType::MAX)
        {
            constexpr std::uint32_t gmpool = hal_port::reduce_pool(PoolType::MAX, p_setrwc::CLR_AB, ADDR_MOD_0, 0);
            TTI_INSN(gmpool);
        }
        else
        {
            if constexpr (high_fidelity)
            {
                constexpr std::uint32_t gapool_phase = hal_port::reduce_pool(type, p_setrwc::CLR_NONE, ADDR_MOD_1, 0);
                for (std::uint32_t i = 0; i < to_underlying(math_fidelity) - 1; i++)
                {
                    TTI_INSN(gapool_phase);
                }
            }
            constexpr std::uint32_t gapool_last = hal_port::reduce_pool(type, p_setrwc::CLR_AB, ADDR_MOD_0, 0);
            TTI_INSN(gapool_last);
        }
    }
}

/**
 * @brief Program address mod registers for the reduce operation.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Tile shape determining dest stride for REDUCE_ROW.
 */
template <PoolType type, ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_addrmod(const ckernel::TensorShape& tensor_shape)
{
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    using hal_port::ReduceAddrMod;
    using hal_port::set_reduce_addr_mod;
    using S = hal::cfg::Sec;

    if constexpr (dim == ReduceDim::REDUCE_ROW && type != PoolType::MAX)
    {
        if (tensor_shape.total_num_faces() == 4)
        {
            set_reduce_addr_mod<S::S0, ReduceAddrMod {.srcb_incr = 8, .dest_incr = 8}>();
            set_reduce_addr_mod<S::S1, ReduceAddrMod {.srcb_incr = 24, .dest_incr = 24}>();
            set_reduce_addr_mod<S::S2, ReduceAddrMod {.srca_incr = 16, .srcb_incr = 16, .srcb_cr = 1, .dest_cr = 1}>();
            set_reduce_addr_mod<
                S::S3,
                ReduceAddrMod {.srca_clr = 1, .srca_cr = 1, .srcb_clr = 1, .srcb_cr = 1, .dest_clr = 1, .dest_cr = 1, .fid_incr = high_fidelity ? 1u : 0u}>();
        }
        else
        {
            if constexpr (high_fidelity)
            {
                set_reduce_addr_mod<S::S0, ReduceAddrMod {.fid_incr = 1}>();
            }

            set_reduce_addr_mod<S::S1, ReduceAddrMod {.srcb_incr = 8, .fid_clr = 1}>();
            set_reduce_addr_mod<S::S2, ReduceAddrMod {.srcb_cr = 1, .fid_clr = 1}>();

            if (tensor_shape.num_faces_c_dim == 2)
            {
                set_reduce_addr_mod<S::S3, ReduceAddrMod {.srcb_cr = 1, .dest_incr = 32, .fid_clr = 1}>();
            }
            else
            {
                set_reduce_addr_mod<S::S3, ReduceAddrMod {.srcb_cr = 1, .dest_incr = 16, .fid_clr = 1}>();
            }
        }
    }
    else
    {
        set_reduce_addr_mod<S::S0, ReduceAddrMod {.fid_clr = 1}>();

        if constexpr (type == PoolType::MAX)
        {
            set_reduce_addr_mod<S::S1, ReduceAddrMod {.srcb_incr = 8, .dest_incr = 8}>();
        }
        else if constexpr (high_fidelity)
        {
            set_reduce_addr_mod<S::S1, ReduceAddrMod {.fid_incr = 1}>();
        }
    }
}

/**
 * @brief Configure the math (FPU) thread for a reduce operation: programs address mods and the MOP.
 *
 * @tparam type: Pooling op, values = <SUM/AVG/MAX>
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam is_fp32_dest_acc_en: Enable FP32 accumulation in the destination register.
 * @tparam math_fidelity: Math fidelity for controlling precision, values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Tile shape describing tile dimensions.
 * @note Call @ref _llk_math_reduce_ with matching template args after this function,
 *       and @ref _llk_math_reduce_uninit_ after the last reduce to restore modified state.
 */
template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void _llk_math_reduce_init_(const ckernel::TensorShape& tensor_shape)
{
    // There is no min-pool instruction - the FPU has GMPOOL (max) and GAPOOL (average/sum).
    static_assert(
        type != PoolType::MIN,
        "The FPU reduce has no MIN: the hardware provides GMPOOL (max) and GAPOOL (average) only. "
        "Use the SFPU reduce instead (ckernel_sfpu_reduce.h::calculate_reduce).");

    reduce_configure_addrmod<type, dim, math_fidelity>(tensor_shape);
    reduce_configure_mop<type, dim, math_fidelity>(tensor_shape);

    namespace cfg = hal::cfg;

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): the whole word, both disable bits
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    hal_port::reset_counters_abd_f();

    // Establish the operand-driven DEFAULT zero-flag state before the reduce's GMPOOLs, mirroring
    // _llk_math_matmul_init_ / _llk_math_eltwise_binary_init_. A preceding copy_init that left
    // PRESERVE (keep denormals) would otherwise leak "keep" into the pool GMPOOL — harmless on HW
    // when fp32 DEST accumulation is enabled (the flag is ignored), but a real invariant violation.
    hal_port::_configure_default_zero_flag_state_();
}

/**
 * @brief Uninitialize after a reduce operation, undoing any init/execute-time workarounds.
 *
 * @note Reverses @ref _llk_math_reduce_init_
 */
inline void _llk_math_reduce_uninit_()
{
}
