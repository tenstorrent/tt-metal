// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_A_custom.h. Same instruction sequence and access
// kinds as production, expressed through the HAL.

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
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

inline void _llk_unpack_A_custom_(const std::uint32_t address)
{
    namespace hcfg = hal::cfg;
    namespace sync = hal::sync;
    namespace adc  = hal::counters::adc;
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
    // Clear z/w start counters: TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();

    // Should be propagated to the unpacker as parameter if we wish to use this LLK standalone
    // cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(1); // Disable haloize mode

    // Program srcA base address
    volatile std::uint32_t tt_reg_ptr *cfg = hcfg::detail::state_cfg_bank(); // get pointer to registers for current state ID

    // Wait for free context
    hal_port::wait_for_next_context(2);

    // Whole-word MMIO store, as production.
    constexpr std::uint32_t base_cntx0 = hcfg::Unpacker[0].Cntx[0].Base.addr32(hcfg::Sec::S0);
    constexpr std::uint32_t base_cntx1 = hcfg::Unpacker[0].Cntx[1].Base.addr32(hcfg::Sec::S0);
    static_assert(base_cntx0 == THCON_SEC0_REG3_Base_address_ADDR32 && base_cntx1 == THCON_SEC0_REG3_Base_cntx1_address_ADDR32, "unpacker A base words");
    const std::uint32_t upk0_reg = (unp_cfg_context == 0) ? base_cntx0 : base_cntx1;
    cfg[upk0_reg]                = address;

    // Trisc::SEMPOST for context acquire
    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

    // Stall unpacker until pending CFG writes from Trisc have completed
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();
    // TTI_SETADCXX(p_setadc::UNP_A, 1023, 0x0)
    adc::set<adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = 1023}}.for_clients<adc::Client::Unpacker0>()>();
    // TTI_UNPACR(SrcA, 0 /*Z inc*/, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Set Dvalid*/, RAREFYB_DISABLE, 0, 0, 0, 0, 1)
    hal::unpack::run<hal::unpack::DataTransfer {.engine = hal::unpack::Engine::Unpacker0, .handoff = hal::unpack::SourceHandoff::FlipAndSetDataValid}>();
    // T6::SEMGET for context release
    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

    // Switch unpacker config context
    hal_port::switch_config_context(unp_cfg_context);
}
