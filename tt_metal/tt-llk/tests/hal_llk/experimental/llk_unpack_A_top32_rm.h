// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_A_top32_rm.h. Same Tensix instruction sequence and
// access kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

namespace hal_port
{
namespace top32_rm
{
// TTI_UNPACR(SrcA, 0b01'00'01'00, 0, 0, 0, 1, SetDvalid, RAREFYB_DISABLE, 0, 0, 0, 0, 1):
// one face into SrcA, channel-0 and channel-1 Y += 1, explicit context 0.
template <bool SetDvalid>
inline constexpr hal::unpack::DataTransfer UNPACK_FACE {
    .engine     = hal::unpack::Engine::Unpacker0,
    .increments = {.channel0 = {.y = 1}, .channel1 = {.y = 1}},
    .handoff    = SetDvalid ? hal::unpack::SourceHandoff::FlipAndSetDataValid : hal::unpack::SourceHandoff::Keep,
};
} // namespace top32_rm
} // namespace hal_port

// Low-level unpack for Top32 row-major tiles.

template <bool unpack_to_dest = false>
inline void _llk_unpack_A_top32_rm_init_(
    const std::uint32_t within_face_16x16_transpose, const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format)
{
    // Two-context handshake op: an earlier op's init may have left Auto TTSync on, and tt-metal calls no uninit.
    hal_port::unpack_sync_mode<hal_port::UnpackSyncMode::Handshake>();
    namespace cfg = hal::cfg;
    namespace adc = hal::counters::adc;

    // cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(within_face_16x16_transpose)
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0>(within_face_16x16_transpose);

    // TTI_SETADCXX(p_setadc::UNP_A, 16 - 1, 0x0): X start on channel 0, X end on channel 1
    adc::set<adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = 16 - 1}}.for_clients<adc::Client::Unpacker0>()>();

    // set y stride in src to 16 data elements and dst to 16x16 data elements
    const DataFormat dst_format       = static_cast<DataFormat>(unpack_dst_format & 0x3);
    const std::uint32_t unpA_x_stride = dst_format == DataFormat::Float32 ? 4 : dst_format == DataFormat::Float16 ? 2 : 1;

    // cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(16 | (16 << 16)): both halves of the word
    constexpr auto R5 = cfg::Thcon[cfg::Reg5];
    static_assert(R5.Tile_x_dim_cntx0.addr32(cfg::Sec::S0) == THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, "Tile_x_dim word");
    cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R5.Tile_x_dim_cntx0, cfg::Sec::S0, 16>(), cfg::set<R5.Tile_x_dim_cntx1, cfg::Sec::S0, 16>());

    // cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_RMW>(16 * 16 * unpA_x_stride)
    constexpr auto Unp0Ch1 = cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1];
    static_assert(Unp0Ch1.Ystride.addr32(cfg::Sec::S0) == UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, "Unp0 ch1 Y-stride word");
    cfg::write<cfg::Access::TensixCfgUnit, Unp0Ch1.Ystride, cfg::Sec::S0>(16 * 16 * unpA_x_stride);
}

template <bool unpack_to_dest = false>
inline void _llk_unpack_A_top32_rm_(
    const std::uint32_t num_faces, const std::uint32_t address, const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format)
{
    namespace hcfg   = hal::cfg;
    namespace sync   = hal::sync;
    namespace adc    = hal::counters::adc;
    namespace unpack = hal::unpack;
    using hal_port::top32_rm::UNPACK_FACE;

    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");

    // TTI_SETADCXY(UNP_A, 0, 0, 0, 0, 0b1010): Y of both channels
    adc::set<adc::AddrCounter {.channel0 = {.y = 0}, .channel1 = {.y = 0}}.for_clients<adc::Client::Unpacker0>()>();
    // TTI_SETADCZW(UNP_A, 0, 0, 0, 0, 0b1111): Z and W of both channels
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0>()>();

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr *cfg = hcfg::detail::state_cfg_bank(); // get pointer to registers for current state ID

    // Wait for free context
    hal_port::wait_for_next_context(2);

    // Set upk0/1 L1 read addr: whole-word MMIO store, as production.
    constexpr std::uint32_t base_cntx0 = hcfg::Unpacker[0].Cntx[0].Base.addr32(hcfg::Sec::S0);
    constexpr std::uint32_t base_cntx1 = hcfg::Unpacker[0].Cntx[1].Base.addr32(hcfg::Sec::S0);
    static_assert(base_cntx0 == THCON_SEC0_REG3_Base_address_ADDR32 && base_cntx1 == THCON_SEC0_REG3_Base_cntx1_address_ADDR32, "unpacker A base words");
    const std::uint32_t upk0_reg = (unp_cfg_context == 0) ? base_cntx0 : base_cntx1;
    cfg[upk0_reg]                = address;

    // Trisc::SEMPOST for context acquire
    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

    if constexpr (unpack_to_dest)
    {
        if (ckernel::unpacker::is_32bit_input(unpack_src_format, unpack_dst_format))
        {
            hal_port::set_dst_write_addr(unp_cfg_context, unpack_dst_format);
            hal_port::wait_for_dest_available();
        }
    }

    // Stall unpacker until pending CFG writes from Trisc have completed
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

    // Run unpack sequence
    if constexpr (unpack_to_dest)
    {
        for (std::uint32_t face_index = 0; face_index < num_faces; face_index++)
        {
            unpack::run<UNPACK_FACE<false>>();
        }
    }
    else
    {
        // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the clear-source flavours (CLR_SRC_NEGINF / CLR_SRC,
        // SET_DVALID, WAIT_LIKE_UNPACR) have no HAL descriptor, so the production encodings are kept.
        // clear A to -infinity
        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
        // clear B to zero
        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::CLR_SRC);

        for (std::uint32_t face_index = 0; face_index < num_faces - 1; face_index++)
        {
            unpack::run<UNPACK_FACE<false>>();
        }

        unpack::run<UNPACK_FACE<true>>();
    }

    if (unpack_to_dest)
    {
        if (ckernel::unpacker::is_32bit_input(unpack_src_format, unpack_dst_format))
        {
            hal_port::unpack_to_dest_tile_done(unp_cfg_context, unpack_dst_format);
        }
    }

    // T6::SEMGET for context release, after unpack_to_dest_tile_done's config writes so that
    // configure_unpack_AB's wait_for_idle also waits for them
    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

    // Switch unpacker config context
    hal_port::switch_config_context(unp_cfg_context);
}
