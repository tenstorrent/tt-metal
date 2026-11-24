// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_fast_untilize.h. Same Tensix instruction sequence
// and access kinds as production, expressed through the HAL, except the HAL-GAPs marked below.
//
// BH Fast-Untilize Unpack.
//
// This is the standard unpack_A face stream specialized for fast_untilize. It
// still emits the same real SrcA face payload as regular unpack_A, but the
// common 16-bit DEST path skips the generic unpack_A zero-SrcB dvalid sideband.
// Native fp32 DEST is the exception: math uses ELWADD as SrcA + zero-SrcB, so
// unpack emits one valid zero SrcB face alongside each real SrcA face. Compressed
// BFP inputs use a unit_dim=1 MOP internally because each tile must be addressed
// explicitly to skip exponent sections, while the block-level chunk size still
// follows the 2/3/4-tile fast-untilize decomposition.

#pragma once

#include <cstdint>

#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/nop.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

namespace hal_port
{
namespace fast_untilize_unpack
{

namespace cfg = hal::cfg;

// TT_OP_UNPACR(SrcA, 0b1 /*CH0_Z += 1*/, 0, 0, 0, 1, 1, RAREFYB_DISABLE, 0, 0, 0, 0, 1): one face into SrcA, handed to math.
inline constexpr std::uint32_t UNPACK_SRCA = hal::unpack::get_operation(hal::unpack::DataTransfer {
    .engine     = hal::unpack::Engine::Unpacker0,
    .increments = {.channel0 = {.z = 1}},
    .context    = hal::unpack::ContextSelection::explicit_context(0, 0),
    .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
});
static_assert(UNPACK_SRCA == TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1), "fast-untilize UNPACR encoding");

// HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the zero-source + set-data-valid flavor (UNP_ZEROSRC,
// SET_DVALID) has no HAL descriptor, so the production encoding is kept.
inline constexpr std::uint32_t UNPACK_SRCB_SET_DVALID =
    TT_OP_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

// Unpacker 0 base address of config context 0 and 1 (whole words, MMIO-stored like production's cfg[] writes).
inline constexpr std::uint32_t BASE_CNTX0_ADDR32 = cfg::Unpacker[0].Cntx[0].Base.addr32(cfg::Sec::S0);
inline constexpr std::uint32_t BASE_CNTX1_ADDR32 = cfg::Unpacker[0].Cntx[1].Base.addr32(cfg::Sec::S0);
static_assert(cfg::Unpacker[0].Cntx[0].Base.shamt(cfg::Sec::S0) == 0 && cfg::Unpacker[0].Cntx[0].Base.width == 32, "context-0 base is a whole word");
static_assert(cfg::Unpacker[0].Cntx[1].Base.shamt(cfg::Sec::S0) == 0 && cfg::Unpacker[0].Cntx[1].Base.width == 32, "context-1 base is a whole word");
static_assert(BASE_CNTX0_ADDR32 == THCON_SEC0_REG3_Base_address_ADDR32, "context-0 base A word");
static_assert(BASE_CNTX1_ADDR32 == THCON_SEC0_REG3_Base_cntx1_address_ADDR32, "context-1 base A word");
static_assert(cfg::Scratch::val.addr32(cfg::Sec::S0) == SCRATCH_SEC0_val_ADDR32, "scratch word");

// TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111): Z and W of both channels of both unpackers.
inline __attribute__((always_inline)) void reset_zw_counters()
{
    namespace adc = hal::counters::adc;
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
}

} // namespace fast_untilize_unpack
} // namespace hal_port

namespace ckernel
{

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_fast_untilize_mop_config_(const std::uint32_t unit_dim)
{
    namespace mop = hal::mop;
    namespace fu  = hal_port::fast_untilize_unpack;

    LLK_ASSERT(unit_dim >= 1 && unit_dim <= 4, "fast_untilize unpack supports unit_dim 1, 2, 3, or 4");

    const std::uint32_t outerloop     = unit_dim * 4;
    constexpr std::uint32_t innerloop = 1;

    // One MOP covers every face in the chunk: unit_dim tiles x 4 faces/tile.
    // The 16-bit DEST path copies SrcA to DEST with MOVA2D, so the loop body
    // only needs real SrcA UNPACRs. The fp32 DEST path copies with ELWADD,
    // which is a binary op and therefore needs a dummy zero SrcB dvalid for
    // every real SrcA face.
    // Explicit NOP end ops keep production's MOP_CFG stores (end_op0/end_op1 = NOP) unchanged.
    if constexpr (is_fp32_dest_acc_en)
    {
        // Native fp32 DEST math uses ELWADD as a SrcA->DEST copy, which needs
        // a valid zero SrcB face alongside each real SrcA face.
        mop::program(mop::loop(outerloop, mop::loop(innerloop, fu::UNPACK_SRCA, fu::UNPACK_SRCB_SET_DVALID), TT_OP_NOP, TT_OP_NOP));
    }
    else
    {
        mop::program(mop::loop(outerloop, mop::loop(innerloop, fu::UNPACK_SRCA), TT_OP_NOP, TT_OP_NOP));
    }
}

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_fast_untilize_init_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t init_unit_dim)
{
    // State setup of _llk_unpack_A_init_ without its Auto TTSync request: _llk_unpack_fast_untilize_block_ keeps the
    // two-context handshake.
    _llk_unpack_A_init_state_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, unpack_src_format, unpack_dst_format);
    _llk_unpack_fast_untilize_mop_config_<is_fp32_dest_acc_en>(init_unit_dim);
}

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_fast_untilize_reinit_unit_dim_(const std::uint32_t unit_dim)
{
    _llk_unpack_fast_untilize_mop_config_<is_fp32_dest_acc_en>(unit_dim);
}

inline void _llk_unpack_fast_untilize_block_(const std::uint32_t address, const std::uint32_t unit_dim)
{
    namespace sync = hal::sync;
    namespace fu   = hal_port::fast_untilize_unpack;

    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
    LLK_ASSERT(unit_dim >= 1 && unit_dim <= 4, "fast_untilize unpack supports unit_dim 1, 2, 3, or 4");

    // Reset both SrcA/SrcB Z/W counters before each block so the MOP is
    // stateless across calls. SrcB is included here because fp32 DEST mode uses
    // the same MOP shape with a zero-SrcB dvalid sideband.
    fu::reset_zw_counters();

    volatile std::uint32_t tt_reg_ptr* cfg = hal::cfg::detail::state_cfg_bank();
    hal_port::wait_for_next_context(2);

    const std::uint32_t upk0_reg = (unp_cfg_context == 0) ? fu::BASE_CNTX0_ADDR32 : fu::BASE_CNTX1_ADDR32;
    cfg[upk0_reg]                = address;

    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();
    // The base-address store above is a RISCV write to Tensix backend config,
    // and the following MOP expands to UNPACR instructions that consume it.
    // STALL_UNPACK blocks those UNPACRs; TRISC_CFG is STALLWAIT C10, which waits
    // until the current TRISC's config/GPR/TDMA write requests are processed.
    // ISA: BlackholeA0/TensixTile/TensixCoprocessor/STALLWAIT.md,
    // BlackholeA0/TensixTile/BabyRISCV/MemoryOrdering.md.
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();
    hal_port::switch_config_context(unp_cfg_context);
}

inline void _llk_unpack_fast_untilize_bfp_block_(const std::uint32_t address, const std::uint32_t tile_stride_16B, const std::uint32_t unit_dim)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    namespace fu   = hal_port::fast_untilize_unpack;

    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address + tile_stride_16B * (unit_dim - 1)), "L1 address must be in valid L1 memory region");
    LLK_ASSERT(tile_stride_16B > 0, "fast_untilize BFP tile stride must be greater than zero");
    LLK_ASSERT(unit_dim >= 2 && unit_dim <= 4, "fast_untilize BFP unpack supports unit_dim 2, 3, or 4");

    volatile std::uint32_t tt_reg_ptr* cfg_regs = cfg::detail::state_cfg_bank();
    hal_port::wait_for_next_context(2);

    const bool use_context_0     = unp_cfg_context == 0;
    const std::uint32_t upk0_reg = use_context_0 ? fu::BASE_CNTX0_ADDR32 : fu::BASE_CNTX1_ADDR32;
    cfg_regs[upk0_reg]           = address;

    // SCRATCH_SEC0_val is not context-double-buffered and is consumed by the per-tile
    // CFGSHIFTMASK updates below. Program it with an ordered Tensix WRCFG (as in
    // llk_unpack_tilize.h) rather than a raw MMIO store, so it is stream-ordered after a
    // prior call's still-in-flight CFGSHIFTMASK that reads it (a trailing TRISC_CFG stall
    // only orders the write before the next run, not against that prior consumer).
    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TMP0>(), LOWER_HALFWORD(tile_stride_16B));
    hal::scalar::set_high(hal::gpr<p_gpr_unpack::TMP0>(), UPPER_HALFWORD(tile_stride_16B));
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Scratch::val, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_unpack::TMP0>());

    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();
    // Order the RISCV MMIO write to the unpack base address before any UNPACR from the MOP
    // can start (the scratch stride above is already ordered via WRCFG).
    // ISA: STALLWAIT C10 plus the RISCV-store-to-Tensix-instruction ordering rule.
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

    // Compressed BFP tiles are not laid out as one dense multi-tile payload:
    // each tile has its own exponent section. Run the same per-tile face MOP,
    // then advance the unpack base register by the CB page stride before the
    // next tile. Non-BFP input can use one wider base address for the chunk.
    for (std::uint32_t tile = 0; tile < unit_dim; tile++)
    {
        fu::reset_zw_counters();
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

        if (tile + 1 < unit_dim)
        {
            // HAL-GAP: no HAL API for CFGSHIFTMASK (base += SCRATCH_SEC0_val); production encoding kept.
            if (use_context_0)
            {
                TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32);
            }
            else
            {
                TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            }
            // CFGSHIFTMASK updates the unpack base register for the next tile.
            // Keep the scheduling bubble local to the address update so the next
            // loop iteration can issue its UNPACR MOP without reasoning about the
            // two-cycle config write. ISA: CFGSHIFTMASK.md.
            hal::nop::thread();
        }
    }

    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();
    hal_port::switch_config_context(unp_cfg_context);
}

inline void _llk_unpack_fast_untilize_uninit_()
{
}

} // namespace ckernel
