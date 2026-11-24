// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_rmsnorm_bcast_scalar_dest_reuse.h. Same instruction
// sequence and access kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/mop.h"
#include "hal/move.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h" // hal_port::set_math_addr_mod / MathAddrMod

using namespace ckernel;

namespace hal_port::rmsnorm
{
namespace cfg = hal::cfg;

// TT_OP_ELW{ADD,SUB,MUL}(clear_dvalid, acc_to_dest, SRCB_BCAST_ALL, addr_mod, 0). acc_to_dest is a runtime argument
// of the MOP config, as in production.
template <EltwiseBinaryType eltwise_binary_type, hal::fpu::SourceRelease release, std::uint8_t addr_mod>
inline __attribute__((always_inline)) std::uint32_t elw_bcast_scalar(const std::uint32_t acc_to_dest)
{
    constexpr hal::fpu::ElementwiseOperation operation = (eltwise_binary_type == EltwiseBinaryType::ELWADD)   ? hal::fpu::ElementwiseOperation::Add
                                                         : (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ? hal::fpu::ElementwiseOperation::Subtract
                                                                                                              : hal::fpu::ElementwiseOperation::Multiply;
    return hal::fpu::get_operation(hal::fpu::Elementwise {
        .operation        = operation,
        .broadcast        = hal::fpu::SrcBBroadcast::Scalar,
        .dest_write       = acc_to_dest ? hal::fpu::DestWriteMode::Accumulate : hal::fpu::DestWriteMode::Overwrite,
        .address_modifier = addr_mod,
        .dest_row_offset  = 0,
        .release          = release});
}

// addr_mod_t::set() with a runtime Dest increment: the SrcA|SrcB and Bias words are compile-time, the Dest|Fidelity
// word takes the runtime increment (three SETC16, as production).
template <cfg::Sec S, MathAddrMod M>
inline void set_math_addr_mod_dest_incr(const std::uint32_t dest_incr)
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, S, M.srca_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, S, M.srca_clr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, S, M.srcb_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, S, M.srcb_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, S>(dest_incr),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, S, M.dest_cr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, S, M.dest_clr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, S, M.dest_c2cr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, S, M.fid_incr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, S, M.fid_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::Bias].Incr, S, 0>(), cfg::set<cfg::AddrMod[cfg::Bias].Clear, S, 0>());
}

} // namespace hal_port::rmsnorm

// Clear one reused product slot without touching the cross-chunk accumulator.
// mul_reduce_scalar_init must have configured the non-incrementing ADDR_MOD_1.
template <std::uint32_t dst_capacity, bool is_fp32_dest_acc_en>
inline void _llk_math_rmsnorm_clear_product_tile_(const std::uint32_t dst_index)
{
    static_assert(dst_capacity >= 2 && dst_capacity <= 8);
    LLK_ASSERT(dst_index < dst_capacity - 1, "product clear must preserve the accumulator");

    // Finish the preceding chunk's SFPU read before clearing its product slots.
    namespace sync = hal::sync;
    namespace dst  = hal::dst;
    sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SfpuIdle>();
    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    constexpr std::uint32_t tiles_per_bank = is_fp32_dest_acc_en ? 4 : 8;
    const std::uint32_t local_tile         = dst_index & (tiles_per_bank - 1);
    // ZEROACC selects the bank from the destination offset. Face indices must
    // stay bank-relative, including slots 4..6 with FP32/full-dest sync.
    // Keep the offset at the tile start: advancing it face-by-face can trigger
    // Blackhole's CLR_16 bank-selection issue (tt-metal#53693).
#pragma GCC unroll 4
    for (std::uint32_t face = 0; face < 4; ++face)
    {
        // TT_ZEROACC(CLR_16, is_fp32_dest_acc_en, 0, ADDR_MOD_1, get_dest_index_in_faces(local_tile, face))
        dst::run(dst::Zero {
            .scope        = dst::ZeroScope::Face,
            .index        = get_dest_index_in_faces(local_tile, face),
            .width        = is_fp32_dest_acc_en ? dst::DestWidth::Bits32 : dst::DestWidth::Bits16,
            .address_mode = ADDR_MOD_1});
    }
}

template <EltwiseBinaryType eltwise_binary_type, std::uint32_t num_tiles, MathFidelity math_fidelity>
inline void rmsnorm_bcast_scalar_dest_reuse_configure_mop(const std::uint32_t num_faces = 4, const std::uint32_t acc_to_dest = 0)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    using namespace hal::mop;
    using hal::fpu::SourceRelease;
    using hal_port::rmsnorm::elw_bcast_scalar;
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    // Production also computes innerloop / outerloop / addr_mod locals that no instruction uses.

    // Scalar broadcast should not Clear B within a mop.  This is controlled outside of MOP.
    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        const std::uint32_t last = elw_bcast_scalar<eltwise_binary_type, SourceRelease::SrcA, ADDR_MOD_3>(acc_to_dest);
        auto mop                 = loop(
                       num_tiles,
                       loop(
                           num_faces,
                           elw_bcast_scalar<eltwise_binary_type, SourceRelease::None, ADDR_MOD_0>(acc_to_dest),
                           elw_bcast_scalar<eltwise_binary_type, SourceRelease::SrcA, ADDR_MOD_2>(acc_to_dest)))
                       .patch(last_outer, last);
        // With one outer iteration the last_inner op is never issued (production still stores it there).
        if constexpr (num_tiles > 1)
        {
            mop = mop.patch(last_inner, last);
        }
        program(mop);
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        if constexpr (high_fidelity)
        {
            auto mop = loop(
                           num_faces,
                           loop(
                               to_underlying(math_fidelity),
                               elw_bcast_scalar<eltwise_binary_type, SourceRelease::None, ADDR_MOD_0>(0),
                               elw_bcast_scalar<eltwise_binary_type, SourceRelease::None, ADDR_MOD_2>(0)))
                           .patch(last_outer, elw_bcast_scalar<eltwise_binary_type, SourceRelease::SrcA, ADDR_MOD_4>(0));
            // With one outer iteration (num_faces == 1) the last_inner op is never issued (production still stores it there).
            if (num_faces > 1)
            {
                mop = mop.patch(last_inner, elw_bcast_scalar<eltwise_binary_type, SourceRelease::SrcA, ADDR_MOD_3>(0));
            }
            program(mop);
        }
        else
        {
            const std::uint32_t last = elw_bcast_scalar<eltwise_binary_type, SourceRelease::SrcA, ADDR_MOD_3>(0);
            auto mop                 = loop(
                           num_tiles,
                           loop(
                               num_faces,
                               elw_bcast_scalar<eltwise_binary_type, SourceRelease::None, ADDR_MOD_0>(0),
                               elw_bcast_scalar<eltwise_binary_type, SourceRelease::SrcA, ADDR_MOD_2>(0)))
                           .patch(last_outer, last);
            if constexpr (num_tiles > 1)
            {
                mop = mop.patch(last_inner, last);
            }
            program(mop);
        }
    }
}

inline void rmsnorm_bcast_scalar_reuse_dest_as_src()
{
    namespace sync = hal::sync;
    namespace move = hal::move;
    // MOVD2B for a whole face assumes unpacker will set a dummy data_valid, so we want to wait on that
    sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SfpuIdle | sync::StallCondition::MathIdle | sync::StallCondition::SrcBValid>();
    // TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_1, p_movd2b::MOV_1_ROW, 0)
    move::run<move::Transfer<move::Dst, move::SrcB> {
        .source_row = 0, .destination_row = p_movd2b::SRC_ZERO_OFFSET + 0, .address_mode = ADDR_MOD_1, .number_of_rows = move::rows::One}>();
}

/**
 * @brief Output[tile] = SrcA[tile] (+, -, *) scalar for num_tiles tiles, the scalar being element [0] of the DEST
 * tile at src_index, moved into SrcB by MOVD2B and broadcast by the FPU op.
 *
 * @tparam eltwise_binary_type: FPU op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam num_tiles: Tiles the MOP walks per call (matches the unpack init).
 * @tparam clear_dest: ZEROACC every tile of the DEST section between the MOVD2B and the MOP. ELWMUL accumulates
 *         into DEST, so set it unless the output tiles are already zero (and no other tile in the section is live).
 * @param src_index: DEST tile holding the scalar.
 * @param dst_index: DEST tile the first output lands in.
 * @note No op that leaves the Src zero-substitution flag at keep (copy_init / datacopy) may run between
 *       @ref _llk_math_rmsnorm_bcast_scalar_dest_reuse_init_ and this call: the MOP would keep denormal
 *       Src operands it should flush. Checked on entry under LLK asserts.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    std::uint32_t num_tiles,
    DstSync Dst,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool clear_dest = false>
inline void _llk_math_rmsnorm_bcast_scalar_dest_reuse_(std::uint32_t src_index, std::uint32_t dst_index)
{
    namespace dst                  = hal::dst;
    namespace rwc                  = hal::counters::rwc;
    using Runner                   = hal::mop::Runner<hal::mop::MopTemplate::Template1>;
    constexpr bool high_fidelity   = is_high_fidelity(math_fidelity);
    constexpr dst::DestWidth width = is_fp32_dest_acc_en ? dst::DestWidth::Bits32 : dst::DestWidth::Bits16;

    // The init re-established the operand-driven flag; an op issued between it and this call must not have
    // left keep behind, or the MOP keeps the denormal Src operands it should flush (fires only under LLK asserts).
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "rmsnorm_bcast_scalar_dest_reuse: Src zero-substitution flag does not hold the operand-driven value; an op between the init "
        "and this call left it at keep");

    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(src_index);
    rmsnorm_bcast_scalar_reuse_dest_as_src();
    if constexpr (clear_dest)
    {
        if constexpr (Dst == DstSync::SyncFull)
        {
            // TT_ZEROACC(CLR_ALL, is_fp32_dest_acc_en, 0, ADDR_MOD_1, 0)
            dst::run<dst::Zero {.scope = dst::ZeroScope::All, .index = 0, .width = width, .address_mode = ADDR_MOD_1}>();
        }
        else
        {
            static_assert(Dst == DstSync::SyncHalf);
            // TT_ZEROACC(CLR_HALF, is_fp32_dest_acc_en, 0, ADDR_MOD_1, dest_offset_id)
            dst::run(dst::Zero {.scope = dst::ZeroScope::Half, .index = dest_offset_id, .width = width, .address_mode = ADDR_MOD_1});
        }
    }
    hal_port::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr ((eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB))
    {
        // Row and no broadcasted behaves similarly
        Runner::run();
    }
    else if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL)
    {
        if constexpr (high_fidelity)
        {
            for (std::uint32_t tile_num = 0; tile_num < num_tiles; tile_num++)
            {
                Runner::run();
            }
        }
        else
        {
            Runner::run();
        }
    }
    // Manually clear B once mop is done for scaler bcast: TTI_SETRWC(CLR_B, 0, 0, 0, 0, SET_D)
    rwc::set<rwc::MathCounter {.dest = rwc::Set(0), .flip = rwc::Src::B}>();
}

template <EltwiseBinaryType eltwise_binary_type, MathFidelity math_fidelity>
inline void rmsnorm_bcast_scalar_dest_reuse_configure_addrmod(const std::uint32_t num_faces)
{
    namespace cfg = hal::cfg;
    using hal_port::MathAddrMod;
    using hal_port::set_math_addr_mod;
    using hal_port::rmsnorm::set_math_addr_mod_dest_incr;
    constexpr bool high_fidelity               = is_high_fidelity(math_fidelity);
    constexpr std::uint32_t fidelity_increment = high_fidelity ? 1 : 0;
    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        // Use srcA for data movement
        set_math_addr_mod<cfg::Sec::S0, MathAddrMod {.srca_incr = 8, .dest_incr = 8}>(); // ADDR_MOD_0
        set_math_addr_mod<cfg::Sec::S1, MathAddrMod {}>();                               // ADDR_MOD_1

        // addr_mod_t truncates the Dest increment (int16) to its 10-bit field.
        const std::uint32_t face_step = static_cast<std::uint32_t>(static_cast<std::int16_t>(8 + (4 - num_faces) * 16)) & 0x3FF;

        if constexpr (eltwise_binary_type == EltwiseBinaryType::ELWMUL && high_fidelity)
        {
            set_math_addr_mod<cfg::Sec::S2, MathAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_cr = 1, .fid_incr = fidelity_increment}>();    // ADDR_MOD_2
            set_math_addr_mod<cfg::Sec::S3, MathAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_incr = 8, .dest_c2cr = 1, .fid_clr = 1}>();    // ADDR_MOD_3
            set_math_addr_mod_dest_incr<cfg::Sec::S4, MathAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_c2cr = 1, .fid_clr = 1}>(face_step); // ADDR_MOD_4
        }
        else
        {
            // Clear SrcA / SrcB counters to 0
            set_math_addr_mod<cfg::Sec::S2, MathAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_incr = 8}>();    // ADDR_MOD_2
            set_math_addr_mod_dest_incr<cfg::Sec::S3, MathAddrMod {.srca_clr = 1, .srcb_clr = 1}>(face_step); // ADDR_MOD_3
        }
    }
}

/**
 * @brief Initialize the FPU for @ref _llk_math_rmsnorm_bcast_scalar_dest_reuse_: addr-mods and the num_tiles MOP.
 *
 * @tparam eltwise_binary_type: FPU op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam num_tiles: Tiles the MOP walks per call.
 * @param num_faces: Faces per tile (1, 2 or 4).
 * @param acc_to_dest: Dest-accumulate field of the ELWADD/ELWSUB MOP instructions.
 * @note Establishes the operand-driven default Src zero-substitution state, which the execute asserts under
 *       LLK asserts; an op that leaves it at keep (copy_init / datacopy) must be followed by this init again.
 *       Pairs with @ref _llk_unpack_A_rmsnorm_init_ on the unpack thread.
 */
template <EltwiseBinaryType eltwise_binary_type, std::uint32_t num_tiles, MathFidelity math_fidelity>
inline void _llk_math_rmsnorm_bcast_scalar_dest_reuse_init_(const std::uint32_t num_faces, const std::uint32_t acc_to_dest)
{
    namespace cfg = hal::cfg;
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");

    rmsnorm_bcast_scalar_dest_reuse_configure_addrmod<eltwise_binary_type, math_fidelity>(num_faces);

    if constexpr (
        (eltwise_binary_type == EltwiseBinaryType::ELWADD) || (eltwise_binary_type == EltwiseBinaryType::ELWSUB) ||
        (eltwise_binary_type == EltwiseBinaryType::ELWMUL))
    {
        rmsnorm_bcast_scalar_dest_reuse_configure_mop<eltwise_binary_type, num_tiles, math_fidelity>(num_faces, acc_to_dest);
    }

    // TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0): both fields of the word
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::ClrDvalid::SrcA_Disable, cfg::Sec::S0, 0>(), cfg::set<cfg::ClrDvalid::SrcB_Disable, cfg::Sec::S0, 0>());

    hal_port::reset_counters_abd_f(); // math::reset_counters(p_setrwc::SET_ABD_F)

    // ELWADD/ELWMUL/ELWSUB read the Src zero-substitution flag and the addr-mod/MOP setup above does not touch
    // it, so re-establish the operand-driven DEFAULT here -- otherwise a preceding copy_init/datacopy op that
    // left PRESERVE leaks into the MOP (denormal Src results differ). Mirrors _llk_math_eltwise_binary_init_.
    hal_port::_configure_default_zero_flag_state_();
}
