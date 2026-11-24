// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_math_sdpa_custom_mm_reuse_dest_srcb.h. Same instruction
// sequence and access kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "cmath_common.h"
#include "hal/cfg.h"
#include "hal/counters/rwc.h"
#include "hal/dst.h"
#include "hal/fpu.h"
#include "hal/move.h"
#include "hal/replay.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_math_common.h"

using namespace ckernel;

// One ADDR_MOD slot and its addr_mod_t::set(n) equivalent, shared by the four SDPA math headers. The block is identical
// in each of them; the guard keeps a translation unit that includes several to one definition.
#ifndef HAL_PORT_SDPA_ADDR_MOD_DEFINED_
#define HAL_PORT_SDPA_ADDR_MOD_DEFINED_

namespace hal_port
{

// One ADDR_MOD slot, in the units of ckernel::addr_mod_t (Dest increment already in its 10-bit encoding).
struct SdpaAddrMod
{
    std::uint32_t srca_incr = 0;
    std::uint32_t srca_clr  = 0;
    std::uint32_t srcb_incr = 0;
    std::uint32_t srcb_clr  = 0;
    std::uint32_t dest_incr = 0;
    std::uint32_t dest_clr  = 0;
    std::uint32_t dest_cr   = 0;
    std::uint32_t dest_c2cr = 0;
    std::uint32_t fid_incr  = 0;
    std::uint32_t fid_clr   = 0;
};

// addr_mod_t::set(n): three SETC16 words (SrcA|SrcB, Dest|Fidelity, Bias), each fully specified.
template <std::uint8_t Slot, SdpaAddrMod M>
inline void set_sdpa_addr_mod()
{
    namespace cfg        = hal::cfg;
    constexpr cfg::Sec S = static_cast<cfg::Sec>(Slot);
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::SrcA].Incr, S, M.srca_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcA].Clear, S, M.srca_clr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Incr, S, M.srcb_incr>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::SrcB].Clear, S, M.srcb_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::Dest].Incr, S, M.dest_incr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CR, S, M.dest_cr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].Clear, S, M.dest_clr>(),
        cfg::set<cfg::AddrMod[cfg::Dest].CToCR, S, M.dest_c2cr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Incr, S, M.fid_incr>(),
        cfg::set<cfg::AddrMod[cfg::Fidelity].Clear, S, M.fid_clr>());
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<cfg::AddrMod[cfg::Bias].Incr, S, 0>(), cfg::set<cfg::AddrMod[cfg::Bias].Clear, S, 0>());
}

// t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU): STALLWAIT(STALL_SYNC, MATH) then SEMPOST.
inline void sdpa_post_fpu_sfpu()
{
    namespace sync = hal::sync;
    static_assert(ckernel::semaphore::FPU_SFPU == 0, "FPU_SFPU is semaphore 0");
    sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle>();
    sync::semaphore::post<sync::Semaphore::S0>();
}

// TTI_ZEROACC(CLR_ALL / CLR_HALF, IS_FP32_MATH_DEST_EN, 0, ADDR_MOD_1, ...): the SDPA preambles' optional dest clear.
template <DstSync DST, bool IS_FP32_MATH_DEST_EN>
inline void sdpa_clear_dest()
{
    namespace dst                  = hal::dst;
    constexpr dst::DestWidth width = IS_FP32_MATH_DEST_EN ? dst::DestWidth::Bits32 : dst::DestWidth::Bits16;
    if constexpr (DST == DstSync::SyncFull)
    {
        dst::run<dst::Zero {.scope = dst::ZeroScope::All, .index = 0, .width = width, .address_mode = ADDR_MOD_1}>();
    }
    else
    {
        static_assert(DST == DstSync::SyncHalf);
        dst::run(dst::Zero {.scope = dst::ZeroScope::Half, .index = ckernel::dest_offset_id, .width = width, .address_mode = ADDR_MOD_1});
    }
}

} // namespace hal_port
#endif // HAL_PORT_SDPA_ADDR_MOD_DEFINED_

namespace hal_port
{
inline constexpr hal::replay::BufferRange sdpa_custom_mm_reuse_dest_srcb_replay {ckernel::math::replay_buf_offset, 4};

// TTI_MVMUL(clr, 0, addr_mod, 0)
template <hal::fpu::SourceRelease Release, std::uint8_t AddrMod>
inline constexpr hal::fpu::MatrixMultiply sdpa_reuse_mvmul {
    .broadcast = hal::fpu::SrcBRowBroadcast::None, .address_modifier = AddrMod, .dest_row_offset = 0, .release = Release};

// TTI_MOVD2B(0, SRC_ZERO_OFFSET + Row, ADDR_MOD_2, MOV_4_ROWS, Row): Dest rows Row..Row+3 -> SrcB rows Row..Row+3.
template <std::uint32_t Row>
inline constexpr hal::move::Transfer<hal::move::Dst, hal::move::SrcB> sdpa_reuse_d2b {
    .source_row = Row, .destination_row = p_movd2b::SRC_ZERO_OFFSET + Row, .address_mode = ADDR_MOD_2, .number_of_rows = hal::move::rows::Four};
} // namespace hal_port

// sdpa_custom_mm_reuse_dest_srcb
// Custom matmul that uses MOP to loop both srcA and srcB along inner dim. Output height
// and width should be single tile with tile shape [1, 32]. Further work will uplift the
// custom mm to support for tiles along the width.
template <MathFidelity math_fidelity>
inline void sdpa_custom_mm_reuse_dest_srcb_configure_addrmod(
    [[maybe_unused]] const bool transpose,
    [[maybe_unused]] const std::uint32_t kt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    using hal_port::SdpaAddrMod;
    using hal_port::set_sdpa_addr_mod;
    set_sdpa_addr_mod<ADDR_MOD_0, SdpaAddrMod {.srca_incr = 16, .dest_incr = 8}>();
    set_sdpa_addr_mod<ADDR_MOD_1, SdpaAddrMod {.srca_incr = 16, .srcb_incr = 8, .dest_cr = 1}>(); // Return to tile base
    set_sdpa_addr_mod<ADDR_MOD_2, SdpaAddrMod {}>();
    set_sdpa_addr_mod<ADDR_MOD_3, SdpaAddrMod {.srca_clr = 1, .srcb_clr = 1, .dest_incr = 8, .dest_c2cr = 1}>(); // Update to next tile
}

template <MathFidelity math_fidelity>
inline void sdpa_custom_mm_reuse_dest_srcb_configure_mop(
    [[maybe_unused]] bool transpose,
    [[maybe_unused]] const std::uint32_t kt_dim,
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false)
{
    namespace fpu = hal::fpu;
    using fpu::SourceRelease;
    using hal_port::sdpa_reuse_mvmul;
    // Select MOV instruction count based on output tile height (in0_tile_r_dim)
    // m=1/4: 4 MOVs (MOV_4_ROWS), m=8: no tail reduction
    // Replay buffer size: m=8: 8 MVMULs, m=1/4: 8 MVMULs + 4 MOVs + 2 ELWADDs = 14
    // Finalization length (starting from offset 4):
    // m=8: 4 instructions (4 MVMULs), m=1/4: 10 instructions (4 MVMULs + 4 MOVs + 2 ELWADDs)
    // sdpa_custom_mm_reuse_dest_srcb::finalization_len = (in0_tile_r_dim == 8) ? 4 : 10;

    // load_replay_buf(replay_buf_offset, 4, ...): record only.
    hal::replay::record<hal_port::sdpa_custom_mm_reuse_dest_srcb_replay>(
        []
        {
            fpu::run<sdpa_reuse_mvmul<SourceRelease::None, ADDR_MOD_0>>(); // 0
            fpu::run<sdpa_reuse_mvmul<SourceRelease::None, ADDR_MOD_1>>(); // 16
            fpu::run<sdpa_reuse_mvmul<SourceRelease::None, ADDR_MOD_0>>(); // 0 (32)
            fpu::run<sdpa_reuse_mvmul<SourceRelease::SrcA, ADDR_MOD_3>>(); // 16 (48)
        });
}

template <MathFidelity math_fidelity>
inline void _llk_math_sdpa_custom_mm_reuse_dest_srcb_init_(
    const std::uint32_t in0_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in0_tile_c_dim = TILE_C_DIM,
    const std::uint32_t in1_tile_r_dim = TILE_R_DIM,
    const std::uint32_t in1_tile_c_dim = TILE_C_DIM,
    const bool partial_face            = false,
    const std::uint32_t transpose      = 0,
    const std::uint32_t kt_dim         = 1)
{
    sdpa_custom_mm_reuse_dest_srcb_configure_addrmod<math_fidelity>(
        transpose, kt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    sdpa_custom_mm_reuse_dest_srcb_configure_mop<math_fidelity>(
        transpose > 0, kt_dim, in0_tile_r_dim, in0_tile_c_dim, in1_tile_r_dim, in1_tile_c_dim, partial_face);

    // math::reset_counters(p_setrwc::SET_ABD_F);
}

// Simplified implementation: NUM_FIDELITY_PHASES = 0 (no high fidelity mode)
// Runtime parameter signal_output:
//   false (default): Full sdpa_custom_mm_reuse_dest_srcb - MVMULs for all K tiles + finalization
//   true: Partial K accumulation - MVMULs only, NO finalization (for intermediate K subblocks)
template <std::uint32_t output_granularity, std::uint32_t input_granularity = 1>
inline void _llk_math_sdpa_custom_mm_reuse_dest_srcb_(
    std::uint32_t src_index,
    std::uint32_t dst_index,
    [[maybe_unused]] const bool transpose = false,
    const std::uint32_t kt_dim            = 1,
    const std::uint32_t nt_dim            = 1,
    bool signal_output                    = false)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    namespace move = hal::move;
    namespace rwc  = hal::counters::rwc;
    using hal_port::sdpa_reuse_d2b;

    static_assert(output_granularity >= 1, "output_granularity must be >= 1");
    static_assert(input_granularity >= 1, "input_granularity must be >= 1");
    // SFPU_FPU = ckernel::semaphore::UNPACK_MATH_DONE
    constexpr sync::Semaphore SFPU_FPU = static_cast<sync::Semaphore>(ckernel::semaphore::UNPACK_MATH_DONE);
    std::uint32_t dest_buffer_base     = get_dest_buffer_base();
    sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::SfpuIdle | sync::StallCondition::MathIdle | sync::StallCondition::SrcBValid>();
    for (std::uint32_t i = 0; i < kt_dim; i++)
    {
        // TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ...)
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(src_index + i * 8 * 2 + dest_buffer_base);
        // math::reset_counters(p_setrwc::SET_ABD_F)
        hal_port::reset_counters_abd_f();
        if (i % input_granularity == 0)
        {
            // t6_semaphore_wait_on_zero<p_stall::STALL_MATH>(SFPU_FPU)
            sync::wait::semaphore<sync::StallTarget::Math, sync::SemaphoreCondition::WhileZero, SFPU_FPU>();
        }
        move::run<sdpa_reuse_d2b<0>>();
        move::run<sdpa_reuse_d2b<4>>();
        move::run<sdpa_reuse_d2b<8>>();
        move::run<sdpa_reuse_d2b<12>>();
        if (i % input_granularity == input_granularity - 1 || i == kt_dim - 1)
        {
            // t6_semaphore_get<p_stall::MATH>(SFPU_FPU)
            sync::wait::stall<sync::StallTarget::Sync, sync::StallCondition::MathIdle>();
            sync::semaphore::get<SFPU_FPU>();
        }
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgMath::Offset, cfg::Sec::S0>(dst_index + dest_buffer_base);
        if (signal_output && i == kt_dim - 1)
        {
            LLK_ASSERT(nt_dim % output_granularity == 0, "nt_dim must be divisible by output_granularity for FPU->SFPU output signal counts to balance");
            for (std::uint32_t j = 0; j < nt_dim / output_granularity; j++)
            {
                for (std::uint32_t g = 0; g < output_granularity; g++)
                {
                    hal::replay::run<hal_port::sdpa_custom_mm_reuse_dest_srcb_replay>();
                }
                hal_port::sdpa_post_fpu_sfpu();
            }
        }
        else
        {
            for (std::uint32_t j = 0; j < nt_dim; j++)
            {
                hal::replay::run<hal_port::sdpa_custom_mm_reuse_dest_srcb_replay>();
            }
        }
    }
    // TTI_SETRWC(CLR_B, 0, 0, 0, 0, SET_ABD)
    rwc::set<rwc::MathCounter {.src_a = rwc::Set(0), .src_b = rwc::Set(0), .dest = rwc::Set(0), .flip = rwc::Src::B}>();
}
