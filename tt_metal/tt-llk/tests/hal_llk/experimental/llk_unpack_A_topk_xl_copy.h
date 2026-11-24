// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TopK-XL copy: single UNPACR (unpack_srca_to_dest encoding) per tile run for FP32 and 16-bit
// unpack-to-dest — no MOP wrapper — instead of face-by-face loops from stock _llk_unpack_A_mop_config_.
//
// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_A_topk_xl_copy.h. Same Tensix instruction sequence
// and access kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel::unpacker;

namespace hal_port
{
namespace topk_xl_copy
{
// TT_OP_UNPACR(SrcA, 0, 0, 0, 0, 1, SetDvalid, RAREFYB_DISABLE, 0, 0, 0, 0, 1): no counter increments, explicit context 0.
template <bool SetDvalid>
inline constexpr std::uint32_t UNPACK_SRCA = hal::unpack::get_operation<hal::unpack::DataTransfer {
    .engine  = hal::unpack::Engine::Unpacker0,
    .handoff = SetDvalid ? hal::unpack::SourceHandoff::FlipAndSetDataValid : hal::unpack::SourceHandoff::Keep,
}>();

// HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the clear-source flavours (CLR_SRC_NEGINF / CLR_SRC,
// UNP_ZEROSRC, SET_DVALID, WAIT_LIKE_UNPACR) have no HAL descriptor, so the production encodings are kept.
inline constexpr std::uint32_t CLEAR_SRCA_TO_NEGINF =
    TT_OP_UNPACR_NOP(SrcA, 0, 0, 0, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
inline constexpr std::uint32_t CLEAR_SRCB_DVALID =
    TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
} // namespace topk_xl_copy
} // namespace hal_port

namespace ckernel
{

inline void _llk_unpack_topk_xl_copy_init_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format)
{
    // Two-context handshake op: an earlier op's init may have left Auto TTSync on, and tt-metal calls no uninit.
    hal_port::unpack_sync_mode<hal_port::UnpackSyncMode::Handshake>();
    namespace mop = hal::mop;
    namespace xl  = hal_port::topk_xl_copy;

    if (is_32bit_input(unpack_src_format, unpack_dst_format))
    {
        // ckernel_template(1, 1, unpack_srca_to_dest)
        mop::program(mop::loop(1, mop::loop(1, xl::UNPACK_SRCA<false>)));
    }
    else
    {
        // ckernel_template(1, 1, clear_srcb_dvalid, unpack_srca) + set_start_op(clear_srca_to_neginf)
        mop::program(mop::loop(1, xl::CLEAR_SRCA_TO_NEGINF, mop::loop(1, xl::CLEAR_SRCB_DVALID, xl::UNPACK_SRCA<true>)));
    }
}

// Minimal unpack-to-dest execute path for TopK-XL copy (one TTI_UNPACR per tile).
inline void _llk_unpack_topk_xl_copy_(
    const std::uint32_t address, const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t elements_this_tile)
{
    namespace hcfg = hal::cfg;
    namespace sync = hal::sync;
    namespace adc  = hal::counters::adc;

    // TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111): Z and W of both channels of both unpackers
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();

    volatile std::uint32_t tt_reg_ptr *cfg = hcfg::detail::state_cfg_bank();

    hal_port::wait_for_next_context(2);

    // Whole-word MMIO store, as production.
    constexpr std::uint32_t base_cntx0 = hcfg::Unpacker[0].Cntx[0].Base.addr32(hcfg::Sec::S0);
    constexpr std::uint32_t base_cntx1 = hcfg::Unpacker[0].Cntx[1].Base.addr32(hcfg::Sec::S0);
    static_assert(base_cntx0 == THCON_SEC0_REG3_Base_address_ADDR32 && base_cntx1 == THCON_SEC0_REG3_Base_cntx1_address_ADDR32, "unpacker A base words");
    const std::uint32_t upk0_reg = (unp_cfg_context == 0) ? base_cntx0 : base_cntx1;
    cfg[upk0_reg]                = address;

    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

    if (elements_this_tile == 0)
    {
        // Zero-element tile: clear SrcA to -inf so math datacopy writes full -inf padding.
        // This avoids SETADC underflow behavior and avoids reading source tile data.
        // HAL-GAP: clear-source / zero-source + set-data-valid UNPACR_NOP flavours have no HAL descriptor.
        TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

        sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();
        hal_port::switch_config_context(unp_cfg_context);
        return;
    }

    if (is_32bit_input(unpack_src_format, unpack_dst_format))
    {
        if (elements_this_tile < 1024)
        {
            // clear to -inf first for padding
            // HAL-GAP: clear-source / zero-source + set-data-valid UNPACR_NOP flavours have no HAL descriptor.
            TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, p_unpacr_nop::CLR_SRC_NEGINF, p_unpacr_nop::CLR_SRC);
            TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        }
        hal_port::set_dst_write_addr(unp_cfg_context, unpack_dst_format);
        hal_port::wait_for_dest_available();
    }

    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

    // Run MOP
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    if (is_32bit_input(unpack_src_format, unpack_dst_format))
    {
        hal_port::unpack_to_dest_tile_done(unp_cfg_context, unpack_dst_format);
    }

    // T6::SEMGET for context release, after unpack_to_dest_tile_done's config writes so that
    // configure_unpack_AB's wait_for_idle also waits for them
    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

    hal_port::switch_config_context(unp_cfg_context);
}

} // namespace ckernel
