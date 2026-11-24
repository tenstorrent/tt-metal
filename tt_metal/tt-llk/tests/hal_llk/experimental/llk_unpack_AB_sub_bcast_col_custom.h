// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_sub_bcast_col_custom.h. Same instruction sequence
// and access kinds as production, expressed through the HAL.

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
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// Custom init for the blocked sub+bcast(col) unpack flow.
// @param tensor_shape Shape of the operand tile (2 faces for 16x32 tiny tiles, 4 faces for full 32x32 tiles).
inline void _llk_unpack_AB_sub_bcast_col_init_custom_(const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    // Two-context handshake op: an earlier op's init may have left Auto TTSync on, and tt-metal calls no uninit.
    hal_port::unpack_sync_mode<hal_port::UnpackSyncMode::Handshake>();
    namespace cfg = hal::cfg;
    namespace adc = hal::counters::adc;
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg2].Haloize_mode, cfg::Sec::S0, 0>(); // transpose within the face

    // Force both unpackers to unpack all faces of the operand. Full 32x32 tile = 4 faces = 1024
    // datums, 16x32 tiny tile = 2 faces = 512 datums.
    const std::uint32_t x_end = tensor_shape.total_tensor_size() - 1;
    // TT_SETADCXX(p_setadc::UNP0, x_end, 0x0); TT_SETADCXX(p_setadc::UNP1, x_end, 0x0)
    adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = x_end}}.for_clients<adc::Client::Unpacker0>());
    adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = x_end}}.for_clients<adc::Client::Unpacker1>());
}

// Custom blocked unpack: one SrcB tile + ct_dim SrcA tiles.
inline void _llk_unpack_AB_sub_bcast_col_custom_(const std::uint32_t address_a, const std::uint32_t address_b, const std::uint32_t ct_dim = 1)
{
    namespace hcfg   = hal::cfg;
    namespace sync   = hal::sync;
    namespace adc    = hal::counters::adc;
    namespace unpack = hal::unpack;

    // reset counters: TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();

    // Program srcA and srcB base addresses
    volatile std::uint32_t tt_reg_ptr* cfg = hcfg::detail::state_cfg_bank(); // get pointer to registers for current state ID

    // Wait for free context
    hal_port::wait_for_next_context(2);

    // Validate and configure addresses
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);

    // Trisc::SEMPOST for context acquire
    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

    // Stall unpacker until pending CFG writes from Trisc have completed
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

    // UNPACR(Src, ADDRMOD_CH1Y_0_CH1Z_0_CH0Y_0_CH0Z_0, 0, 0, 0, 1, 1 /* dvalid */, RAREFYB_DISABLE, 0, 0, 0, 0, 1)
    constexpr unpack::DataTransfer unpack_srcb {.engine = unpack::Engine::Unpacker1, .handoff = unpack::SourceHandoff::FlipAndSetDataValid};
    constexpr unpack::DataTransfer unpack_srca {.engine = unpack::Engine::Unpacker0, .handoff = unpack::SourceHandoff::FlipAndSetDataValid};

    // Unpack srcB once (it will be reused via broadcast for all srcA tiles)
    unpack::run<unpack_srcb>();

    // Unpack srcA tiles sequentially
    for (std::uint32_t i = 0; i < ct_dim; i++)
    {
        unpack::run<unpack_srca>();
        // TTI_INCADCZW(p_setadc::UNP_A, 0, 0, 1, 0): srcA ch0_w += 1
        adc::set<adc::AddrCounter {.action = adc::Action::Increment, .channel0 = {.w = 1}}.for_clients<adc::Client::Unpacker0>()>();
    }

    // T6::SEMGET for context release
    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

    // Switch unpacker config context
    hal_port::switch_config_context(unp_cfg_context);
}

inline void _llk_unpack_AB_sub_bcast_col_uninit_custom_()
{
    // No state to restore: the multi-face unpack X span set in the custom init is transient — every
    // generic unpack op re-establishes its own X span (config_unpacker_x_end / SETADCXX) in its init.
}
