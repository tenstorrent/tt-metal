// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_deepseek_moe_gate_eltwise_binary.h. Same Tensix
// instruction sequence as production.

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod

using namespace ckernel;

enum class DeepseekMoeGateEltwiseBinaryMode
{
    COPY,
    RELOAD,
};

namespace hal_port
{
namespace deepseek_moe_gate
{
// Replay range of the DEST -> SrcA face reload (production: replay_buf_offset, 5).
inline constexpr hal::replay::BufferRange REPLAY_RELOAD {ckernel::math::replay_buf_offset, 5};

// TT_OP_MOVA2D(0, 0, ADDR_MOD_1, p_mova2d::MOV_8_ROWS, 0)
inline constexpr std::uint32_t MOVA2D_8_ROWS =
    hal::move::Transfer<hal::move::SrcA, hal::move::Dst> {
        .source_row = 0, .destination_row = 0, .address_mode = ADDR_MOD_1, .number_of_rows = hal::move::rows::Eight}
        .get_operation();

// TT_OP_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB)
inline constexpr std::uint32_t SETRWC_CLR_AB_CR_AB =
    hal::counters::rwc::MathCounter {
        .src_a = hal::counters::rwc::AdvanceAndReset(0),
        .src_b = hal::counters::rwc::AdvanceAndReset(0),
        .flip  = hal::counters::rwc::Src::A | hal::counters::rwc::Src::B}
        .operation();

// TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + Row, ADDR_MOD_1, p_movd2a::MOV_4_ROWS, Row)
template <std::uint32_t Row>
inline __attribute__((always_inline)) void movd2a_4rows()
{
    hal::move::run<hal::move::Transfer<hal::move::Dst, hal::move::SrcA> {
        .source_row = Row, .destination_row = p_mova2d::MATH_HALO_ROWS + Row, .address_mode = ADDR_MOD_1, .number_of_rows = hal::move::rows::Four}>();
}

// TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + Row, ADDR_MOD_1, p_movd2b::MOV_4_ROWS, Row)
template <std::uint32_t Row>
inline __attribute__((always_inline)) void movd2b_4rows()
{
    hal::move::run<hal::move::Transfer<hal::move::Dst, hal::move::SrcB> {
        .source_row = Row, .destination_row = p_movd2b::SRC_ZERO_OFFSET + Row, .address_mode = ADDR_MOD_1, .number_of_rows = hal::move::rows::Four}>();
}
} // namespace deepseek_moe_gate
} // namespace hal_port

// local function declarations
template <EltwiseBinaryType eltwise_binary_type>
inline void deepseek_moe_gate_eltwise_binary_configure_addrmod()
{
    using hal_port::MathAddrMod;
    using hal_port::set_math_addr_mod;
    using Sec = hal::cfg::Sec;

    // Use srcA for data movement
    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        set_math_addr_mod<Sec::S0, MathAddrMod {.srca_incr = 8, .srcb_incr = 8, .dest_incr = 8}>();
        set_math_addr_mod<Sec::S1, MathAddrMod {}>();
    }
}

template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void deepseek_moe_gate_eltwise_binary_reuse_dest_as_src()
{
    namespace sync = hal::sync;
    namespace dmg  = hal_port::deepseek_moe_gate;

    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        // MOVD2A for a whole face assumes unpacker will set a dummy data_valid, so we want to wait on that. MATH: this
        // sequence replays as the MOP's start op right after its end op cleared both banks; the flip must land before
        // SRCA_VLD is sampled.
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle | sync::StallCondition::SrcAValid>();
        dmg::movd2a_4rows<0>();
        dmg::movd2a_4rows<4>();
        dmg::movd2a_4rows<8>();
        dmg::movd2a_4rows<12>();
    }
    else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        // MOVD2B for a whole face assumes unpacker will set a dummy data_valid, so we want to wait on that. MATH: this
        // sequence replays as the MOP's start op right after its end op cleared both banks; the flip must land before
        // SRCB_VLD is sampled.
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::MathIdle | sync::StallCondition::SfpuIdle | sync::StallCondition::SrcBValid>();
        dmg::movd2b_4rows<0>();
        dmg::movd2b_4rows<4>();
        dmg::movd2b_4rows<8>();
        dmg::movd2b_4rows<12>();
    }
}

// NOTE: clear_fp32_dst_acc is accepted but not acted on, and ZERO_ACC_MODE is its unused remnant --
// unlike _llk_math_eltwise_binary_, this path issues no ZEROACC. Both are marked maybe_unused rather
// than removed so the signature keeps matching blaze's, which behaves the same way.
template <EltwiseBinaryType eltwise_binary_type, DstSync Dst, bool is_fp32_dest_acc_en, MathFidelity math_fidelity>
inline void _llk_math_deepseek_moe_gate_eltwise_binary_(const std::uint32_t num_faces, std::uint32_t dst_index, [[maybe_unused]] const bool clear_fp32_dst_acc)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr bool high_fidelity                           = is_high_fidelity(math_fidelity);
    [[maybe_unused]] constexpr std::uint32_t ZERO_ACC_MODE = p_zeroacc::CLR_16;
    static_assert(!(eltwise_binary_type == EltwiseBinaryType::ELWMUL && high_fidelity), "High fidelity is not supported for ELWMUL");

    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
    }
    hal_port::clear_dst_reg_addr();
}

template <EltwiseBinaryType eltwise_binary_type, DeepseekMoeGateEltwiseBinaryMode mode, MathFidelity math_fidelity>
inline void deepseek_moe_gate_eltwise_binary_configure_mop(const std::uint32_t acc_to_dest = 0, const std::uint32_t num_faces = 4)
{
    namespace mop    = hal::mop;
    namespace fpu    = hal::fpu;
    namespace replay = hal::replay;
    namespace dmg    = hal_port::deepseek_moe_gate;

    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(acc_to_dest <= 1, "acc_to_dest is the one-bit ELW dest-accumulate field");
    constexpr bool high_fidelity      = is_high_fidelity(math_fidelity);
    const std::uint8_t addr_mod       = ADDR_MOD_0;
    constexpr std::uint32_t innerloop = 16 >> 3; // 8 rows per eltwise op at a time.
    std::uint32_t outerloop           = num_faces;

    constexpr std::uint32_t dst_tile_offset = 64; // 1 tile x 64 rows per tile
    constexpr std::uint32_t dst_math_offset = 2 * dst_tile_offset;

    // TODO: Probably not worth it to use a replay buffer/mop for this
    // Just hardcode the math ops in _llk_math_deepseek_moe_gate_eltwise_binary_ since we only have 1 face
    // TT_OP_ELW{ADD,SUB,MUL}(0, acc_to_dest (0 for ELWMUL), p_elwise::SRCB_NO_BCAST, addr_mod, dst_math_offset)
    static_assert(
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
            (eltwise_binary_type == EltwiseBinaryType::ELWMUL),
        "eltwise_binary_type must be ELWADD, ELWSUB, or ELWMUL");
    static_assert(!(eltwise_binary_type == EltwiseBinaryType::ELWMUL && high_fidelity), "High fidelity is not supported for ELWMUL");
    constexpr fpu::ElementwiseOperation operation = (eltwise_binary_type == EltwiseBinaryType::ELWADD)   ? fpu::ElementwiseOperation::Add
                                                    : (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ? fpu::ElementwiseOperation::Subtract
                                                                                                         : fpu::ElementwiseOperation::Multiply;
    // acc_to_dest is the raw ELW dest-accumulate bit (0/1, asserted above), as production passes it; ELWMUL passes 0.
    const fpu::DestWriteMode dest_write =
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL) ? fpu::DestWriteMode::Overwrite : static_cast<fpu::DestWriteMode>(acc_to_dest);
    const std::uint32_t math_op = fpu::get_operation(fpu::Elementwise {
        .operation        = operation,
        .broadcast        = fpu::SrcBBroadcast::None,
        .dest_write       = dest_write,
        .address_modifier = addr_mod,
        .dest_row_offset  = dst_math_offset,
        .release          = fpu::SourceRelease::None});

    if constexpr (mode == DeepseekMoeGateEltwiseBinaryMode::RELOAD)
    {
        replay::record<dmg::REPLAY_RELOAD>([] { deepseek_moe_gate_eltwise_binary_reuse_dest_as_src<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(); });
        constexpr std::uint32_t replay_instr = replay::get_operation(dmg::REPLAY_RELOAD);
        // ckernel_template(outerloop, innerloop, math_op) + set_start_op(replay) + set_end_op(SETRWC(CLR_AB, CR_AB, ..., SET_AB))
        mop::program(mop::loop(outerloop, replay_instr, mop::loop(innerloop, math_op), dmg::SETRWC_CLR_AB_CR_AB));
    }
    else
    {
        // ckernel_template(outerloop, innerloop, MOVA2D, math_op) + set_end_op(SETRWC(CLR_AB, CR_AB, ..., SET_AB))
        mop::program(mop::loop(outerloop, mop::loop(innerloop, dmg::MOVA2D_8_ROWS, math_op), dmg::SETRWC_CLR_AB_CR_AB));
    }
}

template <EltwiseBinaryType eltwise_binary_type, DeepseekMoeGateEltwiseBinaryMode mode, MathFidelity math_fidelity>
inline void _llk_math_deepseek_moe_gate_eltwise_binary_init_(const std::uint32_t num_faces, const std::uint32_t acc_to_dest)
{
    namespace cfg = hal::cfg;

    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    deepseek_moe_gate_eltwise_binary_configure_addrmod<eltwise_binary_type>();

    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        deepseek_moe_gate_eltwise_binary_configure_mop<eltwise_binary_type, mode, math_fidelity>(acc_to_dest, num_faces);
    }

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): both fields of the word
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    hal_port::reset_counters_abd_f();
}
