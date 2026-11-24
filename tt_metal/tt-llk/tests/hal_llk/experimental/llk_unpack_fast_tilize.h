// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_fast_tilize.h. Same Tensix instruction sequence and
// access kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "hal/cfg.h"
#include "hal/mop.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_unpack_common.h"
#include "llk_unpack_tilize.h"

namespace hal_port
{
namespace fast_tilize
{

namespace cfg = hal::cfg;

// TT_OP_UNPACR_COMMON(SrcA, 0b00'01'00'01 /*CH1_Z += 1, CH0_Z += 1*/, set_dvalid): explicit context 0, Last set.
constexpr std::uint32_t unpack_read(const hal::unpack::SourceHandoff handoff)
{
    return hal::unpack::get_operation(hal::unpack::DataTransfer {
        .engine     = hal::unpack::Engine::Unpacker0,
        .increments = {.channel0 = {.z = 1}, .channel1 = {.z = 1}},
        .context    = hal::unpack::ContextSelection::explicit_context(0, 0),
        .handoff    = handoff,
    });
}

static_assert(unpack_read(hal::unpack::SourceHandoff::Keep) == TT_OP_UNPACR_COMMON(SrcA, 0b00'01'00'01, 0), "fast-tilize UNPACR encoding");
static_assert(unpack_read(hal::unpack::SourceHandoff::FlipAndSetDataValid) == TT_OP_UNPACR_COMMON(SrcA, 0b00'01'00'01, 1), "fast-tilize UNPACR encoding");

// The CFG words fast-tilize saves, rewrites and restores (each anchored by the field at bit 0 of the word).
inline constexpr auto Unp0Ch1 = cfg::Unpacker[0].AddrCtrl[cfg::UnpackerReg::Reg1];
inline constexpr auto R2      = cfg::Thcon[cfg::Reg2];
inline constexpr auto R5      = cfg::Thcon[cfg::Reg5];
inline constexpr auto TileDsc = cfg::Thcon[cfg::Reg0].TileDescriptor;
static_assert(Unp0Ch1.Zstride.addr32(cfg::Sec::S0) == UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32, "Unp0 ch1 Z-stride word");
static_assert(R5.Tile_x_dim_cntx0.addr32(cfg::Sec::S0) == THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, "Tile_x_dim word");
static_assert(TileDsc.YDim.addr32(cfg::Sec::S0) == THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, "TileDescriptor Y/Z-dim word");
static_assert(R2.Out_data_format.addr32(cfg::Sec::S0) == THCON_SEC0_REG2_Out_data_format_ADDR32, "Out_data_format word");

} // namespace fast_tilize
} // namespace hal_port

// ============================================================================
// BH Fast-Tilize Unpack
//
// Software tilization via UNPACR address modes. Processes 4 tiles at a time
// (unit_dim=4). Each UNPACR reads 128 datums (4 tile widths) into 8 SrcA rows.
// CH1_Z stride = 256 bytes (8 contiguous SrcA rows per read).
// MASK_LOOP MOP with zmask=0x80808080 fires dvalid every 8th read
// (32 reads total, 4 dvalids per unit).
// ============================================================================

inline void _llk_unpack_fast_tilize_mop_config_()
{
    using namespace hal::mop;
    using hal::unpack::SourceHandoff;
    // addr_mode: CH0_Z+=1 (next L1 row), CH1_Z+=1 (next SrcA dest with gap)
    // ckernel_unpack_template(unpackB = false, unpackHalo = false, A0 = read without dvalid, A1-3 = NOP,
    // skipA = read with dvalid, B/skipB = NOP). The HAL leaves the unused InsnA1-3/InsnB/SkipB words unwritten
    // (the expander does not read them without unpackB/unpackHalo).
    constexpr std::uint32_t read          = hal_port::fast_tilize::unpack_read(SourceHandoff::Keep);                // A0: read, no dvalid
    constexpr std::uint32_t read_and_pass = hal_port::fast_tilize::unpack_read(SourceHandoff::FlipAndSetDataValid); // skipA: read WITH dvalid
    program(foreach_mask_bit(if_clear(read), otherwise(read_and_pass)));
}

// BH fast-tilize: block height is always 1 (one row of tiles per call).
// Multiple rows are handled by the caller, looping over rows and calling
// the block function once per chunk per row.
//
// init_unit_dim: unit_dim of the first chunk (= decompose_row(ct_dim)[0]).
// Initialising X to the first chunk's width avoids one reinit_xdim call per row.
// Formula: ct_dim > 5 ? 4 : ct_dim == 5 ? 2 : ct_dim.
inline void _llk_unpack_fast_tilize_init_(const std::uint32_t unpack_dst_format, const std::uint32_t ct_dim, const std::uint32_t init_unit_dim)
{
    // Two-context handshake op: an earlier op's init may have left Auto TTSync on, and tt-metal calls no uninit.
    hal_port::unpack_sync_mode<hal_port::UnpackSyncMode::Handshake>();
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    namespace adc  = hal::counters::adc;
    namespace ft   = hal_port::fast_tilize;
    using cfg::Access;
    using cfg::GprTransferSize;
    using cfg::Sec;
    using cfg::WrcfgCompletion;

    // Context-safe writes only: Tile_x_dim (WRCFG below writes the full 32-bit word,
    // covering cntx0 low-16 and cntx1 high-16), TileDescriptor (shared across contexts),
    // Zstride (RMW on shared reg), and SETADCXX (thread-scoped counter, not per
    // cfg context - see ISA SETADCXX). Per-call context switching happens in
    // _llk_unpack_fast_tilize_block_.

    // Save state
    cfg::read<Access::TensixCfgUnit, ft::Unp0Ch1.Zstride, Sec::S0>(hal::gpr<p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0>());
    cfg::read<Access::TensixCfgUnit, ft::R5.Tile_x_dim_cntx0, Sec::S0>(hal::gpr<p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1>());
    cfg::read<Access::TensixCfgUnit, ft::TileDsc.YDim, Sec::S0>(hal::gpr<p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2>());
    // Save the unpacker Out_data_format word so we can restore it in uninit if the
    // fp32/tf32 -> bf16 downgrade below modifies it.
    cfg::read<Access::TensixCfgUnit, ft::R2.Out_data_format, Sec::S0>(hal::gpr<p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_3>());

    // BH fast-tilize forces a 16-bit DEST view in the math thread (MOVA2D cannot
    // safely write Dst32b in this flow - see _llk_math_fast_tilize_init_). That makes
    // TF32/Float32 SrcA incompatible: MOVA2D(TF32) + Dst16b is ISA UB, and MOVA2D(bf16)
    // + Dst32b is also UB. The only consistent combination is bf16 SrcA + Dst16b, so
    // downgrade SrcA output to Float16_b here when the caller requested fp32/tf32.
    // The unpacker performs the fp32 -> bf16 conversion on the L1 -> SrcA path. This
    // matches the precision Metal previously consumed from the fp32 fast-tilize path.
    const std::uint32_t effective_dst_format =
        (unpack_dst_format == static_cast<std::uint32_t>(DataFormat::Float32) || unpack_dst_format == static_cast<std::uint32_t>(DataFormat::Tf32))
            ? static_cast<std::uint32_t>(DataFormat::Float16_b)
            : unpack_dst_format;
    if (effective_dst_format != unpack_dst_format)
    {
        sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle>();
        cfg::write<Access::TensixCfgUnit, ft::R2.Out_data_format, Sec::S0>(effective_dst_format);
    }

    cfg::write<Access::TensixCfgUnit, ft::R2.Haloize_mode, Sec::S0, 0>();

    // Tile_x_dim = 32, Tile_y_dim = ct_dim, Tile_z_dim = 16
    hal::scalar::set<(TILE_C_DIM << 16) | TILE_C_DIM>(hal::gpr<p_gpr_unpack::TMP0>());
    cfg::write<Access::TensixCfgUnit, ft::R5.Tile_x_dim_cntx0, Sec::S0, GprTransferSize::Bits32, WrcfgCompletion::Deferred>(hal::gpr<p_gpr_unpack::TMP0>());

    hal::scalar::set_low(hal::gpr<p_gpr_unpack::TMP0>(), ct_dim);
    hal::scalar::set_high<FACE_R_DIM>(hal::gpr<p_gpr_unpack::TMP0>());
    cfg::write<Access::TensixCfgUnit, ft::TileDsc.YDim, Sec::S0, GprTransferSize::Bits32, WrcfgCompletion::Deferred>(hal::gpr<p_gpr_unpack::TMP0>());

    // BH HW bug: TileDescriptor words 1-3 (YDim, ZDim) are not tracked as unpacker
    // resources. Explicit stall ensures the write completes.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();

    // X counter end = first chunk's tile width. CH1_Z stride stays at 4-wide
    // (8 SrcA rows) regardless of unit_dim - natural gaps in SrcA for unit_dim < 4.
    // Matching init_unit_dim to the first chunk eliminates one reinit_xdim per row.
    // TT_SETADCXX(UNP_A, init_unit_dim * TILE_C_DIM - 1, 0x0)
    adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = init_unit_dim * TILE_C_DIM - 1}}.for_clients<adc::Client::Unpacker0>());

    // CH1 Z stride: controls SrcA dest address gap between reads.
    // Uses effective_dst_format because Float32/Tf32 are downgraded to bf16 above.
    const std::uint32_t ch1_x_stride =
        (effective_dst_format == static_cast<std::uint32_t>(DataFormat::Float32) || effective_dst_format == static_cast<std::uint32_t>(DataFormat::Int32) ||
         effective_dst_format == static_cast<std::uint32_t>(DataFormat::Tf32))
            ? 4
            : 2;
    // stride = 4 * 32 * 2 = 256 bytes = 8 contiguous SrcA rows per read
    cfg::write<Access::TensixCfgUnit, ft::Unp0Ch1.Zstride, Sec::S0>(4 * TILE_C_DIM * ch1_x_stride);

    _llk_unpack_fast_tilize_mop_config_();
}

// Reconfigure X counter for a different unit_dim without full reinit.
// CH1_Z stride and MOP stay unchanged - only the read width changes.
inline void _llk_unpack_fast_tilize_reinit_xdim_(const std::uint32_t unit_dim)
{
    namespace adc = hal::counters::adc;
    // TT_SETADCXX(UNP_A, unit_dim * TILE_C_DIM - 1, 0x0)
    adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = unit_dim * TILE_C_DIM - 1}}.for_clients<adc::Client::Unpacker0>());
}

// One call = one row-chunk (one unit_dim, one MOP run).
// Block height is always 1; multiple rows and chunks are loops in the caller.
inline void _llk_unpack_fast_tilize_block_(
    const std::uint32_t base_address,
    [[maybe_unused]] const std::uint32_t tile_index,
    [[maybe_unused]] const std::uint32_t unpack_src_format,
    [[maybe_unused]] const std::uint32_t unit_dim,
    [[maybe_unused]] const std::uint32_t num_faces = 4)
{
    namespace sync = hal::sync;
    namespace adc  = hal::counters::adc;

    // Standard BH unpacker context dance (same as the other BH unpack passes).
    // Programs REG3 Base_address for the current cfg context via cfg[] write,
    // then synchronises Trisc<->T6 so the MOP runs with that base in place.
    volatile std::uint32_t tt_reg_ptr* cfg = hal::cfg::detail::state_cfg_bank();
    hal_port::wait_for_next_context(2);
    _llk_unpack_configure_single_address_(base_address, cfg);
    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

    // L1 addressing via CH0 counters. The caller folds any tile/column offset
    // into base_address, so Y starts at zero for this chunk and Z selects tensor
    // row (0..31).
    adc::set<adc::AddrCounter {.channel0 = {.x = 0, .y = 0}}.for_clients<adc::Client::Unpacker0>()>(); // reset CH0_X=0, CH0_Y=0 (SETADCXY 0b0011)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0>()>(); // reset all Z,W = 0

    // Hoist zmask high 16 bits - they persist in mop_zmask_hi16 until changed.
    // MOP_CFG(ZMASK >> 16) + MOP(MASK_LOOP, 32 - 1, ZMASK & 0xFFFF)
    constexpr std::uint32_t ZMASK = 0x80808080;
    hal::mop::Runner<hal::mop::MopTemplate::Template0>::run<32, ZMASK>();

    // Release the unpacker context acquired above and advance the software
    // tracker so the next call targets the other cfg context slot.
    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();
    hal_port::switch_config_context(unp_cfg_context);
}

template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_fast_tilize_uninit_()
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    namespace adc  = hal::counters::adc;
    namespace ft   = hal_port::fast_tilize;
    using cfg::Access;
    using cfg::GprTransferSize;
    using cfg::Sec;
    using cfg::WrcfgCompletion;

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();

    cfg::write<Access::TensixCfgUnit, ft::Unp0Ch1.Zstride, Sec::S0, GprTransferSize::Bits32, WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_0>());
    cfg::write<Access::TensixCfgUnit, ft::R5.Tile_x_dim_cntx0, Sec::S0, GprTransferSize::Bits32, WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_1>());
    cfg::write<Access::TensixCfgUnit, ft::TileDsc.YDim, Sec::S0, GprTransferSize::Bits32, WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_2>());
    // Restore Out_data_format (init may have downgraded it from fp32/tf32 to bf16).
    cfg::write<Access::TensixCfgUnit, ft::R2.Out_data_format, Sec::S0, GprTransferSize::Bits32, WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_unpack::SR_UNPACK_UNTILIZER_STATE_3>());

    adc::set<adc::AddrCounter {.channel0 = {.y = 0}, .channel1 = {.y = 0}}.for_clients<adc::Client::Unpacker0>()>();                 // SETADCXY mask 0b1010
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0>()>(); // SETADCZW mask 0b1111
}
