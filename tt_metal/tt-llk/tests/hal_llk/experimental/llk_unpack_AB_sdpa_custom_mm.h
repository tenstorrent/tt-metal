// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_sdpa_custom_mm.h. Same instruction sequence and
// access kinds as production, expressed through the HAL. The block loop itself is _llk_unpack_AB_custom_mm_run_ from
// experimental/llk_unpack_AB_custom_mm.h, which belongs to the custom_mm work package and is included unchanged.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "experimental/llk_unpack_AB_custom_mm.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

template <bool read_transposed = false, bool configure_mask_extent = false>
inline void _llk_unpack_AB_sdpa_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t base_address_mask,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim              = 1,
    const bool mask_chunk                   = false,
    const std::uint32_t operandB_face_r_dim = 8,
    const std::uint32_t row_stride          = 0)
{
    namespace hcfg = hal::cfg;
    namespace adc  = hal::counters::adc;
    namespace sync = hal::sync;

    const std::uint32_t in1_row = row_stride ? row_stride : kt_dim;
    if constexpr (read_transposed)
    {
        LLK_ASSERT(in1_row >= kt_dim, "row_stride must contain the contraction tiles");
    }
    volatile std::uint32_t* cfg         = hcfg::detail::state_cfg_bank();
    const std::uint32_t block_increment = read_transposed ? in1_row * tile_size_a : tile_size_a;
    const std::uint32_t inner_increment = read_transposed ? -(((ct_dim - 1) * in1_row) - 1) * tile_size_a : tile_size_a;

    const std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    const std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b;

    // Wait for all contexts to be free
    hal_port::wait_for_next_context(1);
    hal_port::reset_config_context();

    if (mask_chunk)
    {
        // Unpacker 1 context-1 base address: whole-word MMIO store into the state-CFG bank, as production.
        constexpr std::uint32_t base_b_cntx1 = hcfg::Unpacker[1].Cntx[1].Base.addr32(hcfg::Sec::S0);
        static_assert(base_b_cntx1 == THCON_SEC1_REG3_Base_cntx1_address_ADDR32, "unpacker 1 context-1 base word");
        cfg[base_b_cntx1] = base_address_mask;
        if constexpr (configure_mask_extent)
        {
            LLK_ASSERT(ct_dim * 2 <= 64, "mask_chunk requires ct_dim*2 <= 64 SrcB rows");
            // TT_SETADCXX(UNP_B, ct_dim * 2 * FACE_C_DIM - 1, 0x0)
            adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = ct_dim * 2 * FACE_C_DIM - 1}}.for_clients<adc::Client::Unpacker1>());
        }
        sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();
        // TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcB, 0b00000000, 1 /*CfgContextId*/, 1 /*SetDatValid*/)
        hal::unpack::run<hal::unpack::DataTransfer {
            .engine  = hal::unpack::Engine::Unpacker1,
            .context = hal::unpack::ContextSelection::explicit_context(1, 0),
            .handoff = hal::unpack::SourceHandoff::FlipAndSetDataValid,
        }>();
        if constexpr (configure_mask_extent)
        {
            // TT_SETADCXX(UNP_B, operandB_face_r_dim * FACE_C_DIM - 1, 0x0)
            adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = operandB_face_r_dim * FACE_C_DIM - 1}}.for_clients<adc::Client::Unpacker1>());
        }
    }

    _llk_unpack_AB_custom_mm_run_(cfg, address_a, address_b, tile_size_b, block_increment, inner_increment, kt_dim);
}
