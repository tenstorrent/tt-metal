// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_AB_compressed_custom_mm.h. Same instruction sequence
// and access kinds as production, expressed through the HAL, except the HAL-GAPs marked below.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/nop.h"
#include "hal/replay.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_assert.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// The custom_mm UNPACR encodings and the counter reset, shared with llk_unpack_AB_custom_mm.h (identical block there); the guard
// keeps a translation unit that includes both to one definition.
#ifndef HAL_PORT_CUSTOM_MM_UNPACK_OPS_DEFINED_
#define HAL_PORT_CUSTOM_MM_UNPACK_OPS_DEFINED_

namespace hal_port
{

// TT_OP_UNPACR_COMMON(SrcA, 0b00000000, 1): a full weight tile into SrcA, no counter increments, set dvalid.
inline constexpr std::uint32_t custom_mm_unpack_a = hal::unpack::get_operation(hal::unpack::DataTransfer {
    .engine  = hal::unpack::Engine::Unpacker0,
    .context = hal::unpack::ContextSelection::explicit_context(0, 0),
    .handoff = hal::unpack::SourceHandoff::FlipAndSetDataValid,
});
static_assert(custom_mm_unpack_a == TT_OP_UNPACR_COMMON(SrcA, 0b00000000, 1), "UNPACR SrcA encoding");

// TT_OP_UNPACR_COMMON(SrcB, 0b00010001, 0): first SrcB face, CH0 Z += 1 and CH1 Z += 1, keep the bank.
inline constexpr std::uint32_t custom_mm_unpack_b_face0 = hal::unpack::get_operation(hal::unpack::DataTransfer {
    .engine     = hal::unpack::Engine::Unpacker1,
    .increments = {.channel0 = {.z = 1}, .channel1 = {.z = 1}},
    .context    = hal::unpack::ContextSelection::explicit_context(0, 0),
    .handoff    = hal::unpack::SourceHandoff::Keep,
});
static_assert(custom_mm_unpack_b_face0 == TT_OP_UNPACR_COMMON(SrcB, 0b00010001, 0), "UNPACR SrcB face 0 encoding");

// TT_OP_UNPACR_COMMON(SrcB, 0b00110100, 1): second SrcB face, CH0 Y += 1 and CH1 Z += 3 (wraps back to row 0), set
// dvalid.
inline constexpr std::uint32_t custom_mm_unpack_b_face1 = hal::unpack::get_operation(hal::unpack::DataTransfer {
    .engine     = hal::unpack::Engine::Unpacker1,
    .increments = {.channel0 = {.y = 1}, .channel1 = {.z = 3}},
    .context    = hal::unpack::ContextSelection::explicit_context(0, 0),
    .handoff    = hal::unpack::SourceHandoff::FlipAndSetDataValid,
});
static_assert(custom_mm_unpack_b_face1 == TT_OP_UNPACR_COMMON(SrcB, 0b00110100, 1), "UNPACR SrcB face 1 encoding");

// TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111) then TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010): Z and W of both channels, then
// Y of both channels, for both unpackers.
inline __attribute__((always_inline)) void custom_mm_reset_unpack_counters()
{
    namespace adc = hal::counters::adc;
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
    adc::set<adc::AddrCounter {.channel0 = {.y = 0}, .channel1 = {.y = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();
}

// TTI_STALLWAIT(STALL_UNPACK, UNPACK), TTI_UNPACR_NOP(SrcB, ..., CLR_SRC), TTI_STALLWAIT(STALL_UNPACK, UNPACK1): zero
// both SrcB banks once both are free. It must run alone: a both-bank SrcB clear drops SrcA writes that unpacker 0
// makes while it runs.
inline __attribute__((always_inline)) void custom_mm_clear_srcb_both_banks()
{
    namespace sync = hal::sync;
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();
    // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the clear-source flavor (CLR_SRC) has no HAL
    // descriptor, so the production encoding is kept.
    TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::Unpacker1Idle>();
}

} // namespace hal_port
#endif // HAL_PORT_CUSTOM_MM_UNPACK_OPS_DEFINED_

namespace hal_port
{

// TTI_WRCFG(gpr, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32): Deferred, the recorded TTI_NOP follows.
template <std::uint32_t Gpr>
inline __attribute__((always_inline)) void compressed_custom_mm_write_tile_descriptor()
{
    namespace cfg          = hal::cfg;
    constexpr auto TileDsc = cfg::Thcon[cfg::Reg0].TileDescriptor;
    static_assert(TileDsc.InDataFormat.addr32(cfg::Sec::S0) == THCON_SEC0_REG0_TileDescriptor_ADDR32, "tile descriptor word");
    cfg::write<cfg::Access::TensixCfgUnit, TileDsc.InDataFormat, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(hal::gpr<Gpr>());
}

// HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the clear-source flavor (CLR_SRC with Stall_Clr_Cntrl) has
// no HAL descriptor, so the production encoding is kept.
inline __attribute__((always_inline)) void compressed_custom_mm_clr_srca_stall()
{
    TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
}

// One format's recorded sequence: CLR_SRC stall, tile descriptor, NOP, UNPACR SrcA, CFGSHIFTMASK (scratch_sel), NOP,
// the two SrcB faces.
template <std::uint32_t Gpr, std::uint32_t ScratchSel>
inline __attribute__((always_inline)) void compressed_custom_mm_record_format()
{
    compressed_custom_mm_clr_srca_stall();
    compressed_custom_mm_write_tile_descriptor<Gpr>();
    hal::nop::thread();
    TTI_INSN(custom_mm_unpack_a);
    // HAL-GAP: CFGSHIFTMASK (Base_address += SCRATCH_SEC[ScratchSel]_val) has no HAL API; production encoding kept.
    TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, ScratchSel, THCON_SEC0_REG3_Base_address_ADDR32);
    hal::nop::thread();
    TTI_INSN(custom_mm_unpack_b_face0);
    TTI_INSN(custom_mm_unpack_b_face1);
}

} // namespace hal_port

// CUSTOM_MM
// Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
// in0 tile shape: [{1, 2, 4, 8}, 32]
// in1 tile shape: [32, 32]
// rt_dim: 1
// ct_dim: any integer from 1 to 64, as long as the ct_dim output tiles fit in the DEST sync region
//         (custom_mm_tile_rows of the CustomMmDestLayout each); nothing else bounds it
// kt_dim: even number from 2 to 254 (inclusive); one SrcB face walk covers at most 255 k-tiles
// fidelity: LoFi only
// throttle: not supported

inline void _llk_unpack_AB_compressed_custom_mm_mop_config_()
{
    hal::replay::record<hal::replay::BufferRange {0 /*start*/, 25 /*count*/}>(
        []
        {
            // Bfp8
            hal_port::compressed_custom_mm_record_format<p_gpr_unpack::PERF_UNPACK_NUM_TILES_3, 0>();
            // Bfp4
            hal_port::compressed_custom_mm_record_format<p_gpr_unpack::PERF_UNPACK_NUM_TILES_2, 1>();
            // Bfp2
            hal_port::compressed_custom_mm_record_format<p_gpr_unpack::PERF_UNPACK_NUM_TILES_1, 2>();
            hal_port::compressed_custom_mm_clr_srca_stall();
        });
}

/**
 * @brief Configure the unpack thread for a compressed custom_mm block matmul, whose weight tiles each carry their own
 *        BFP format.
 *
 * Records the per-format unpack sequences into the replay buffer, sets the unpacker X ends, and loads the per-format
 * SrcA tile-size increments (SCRATCH_SEC0-2) and tile descriptors (GPRs PERF_UNPACK_NUM_TILES_1-3) the execute uses.
 *
 * @tparam transpose: Transpose the SrcA read, values = <true/false>
 * @tparam clear_src: Zero both SrcB banks once here, values = <true/false>. Only unpB_face_r_dim rows of each
 *                    SrcB face are unpacked, so zeroing the rest saves FPU power.
 * @param unpB_face_r_dim: Activation rows per face, 1, 2, 4 or 8. Sets unpacker 1's X end.
 * @note Call this before @ref _llk_unpack_AB_compressed_custom_mm_, and again after any other op has run, in
 *       particular one that writes SrcB, SCRATCH_SEC0-2 or those GPRs (a custom_mm execute rewrites SCRATCH_SEC0/1):
 *       the execute relies on all of them and sets none.
 * @note On the math thread, pair with @ref _llk_math_compressed_custom_mm_init_.
 */
template <bool transpose = false, bool clear_src = true>
inline void _llk_unpack_AB_compressed_custom_mm_init_(const std::uint32_t unpB_face_r_dim)
{
    namespace hcfg = hal::cfg;
    namespace adc  = hal::counters::adc;

    volatile std::uint32_t* cfg = hcfg::detail::state_cfg_bank(); // get_cfg_pointer()

    // cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose ? 1 : 0): one RMWCIB, compile-time value
    hcfg::write<hcfg::Access::TensixCfgUnit, hcfg::Thcon[hcfg::Reg2].Haloize_mode, hcfg::Sec::S0, (transpose ? 1u : 0u)>();

    constexpr std::uint32_t unpA_x_end = TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1;
    const std::uint32_t unpB_x_end     = unpB_face_r_dim * FACE_C_DIM - 1;
    // TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0)
    adc::set<adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = unpA_x_end}}.for_clients<adc::Client::Unpacker0>()>();
    // TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0)
    adc::set(adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = unpB_x_end}}.for_clients<adc::Client::Unpacker1>());

    _llk_unpack_AB_compressed_custom_mm_mop_config_();

    if constexpr (clear_src)
    {
        // Clear SrcB as we only unpack into 1/8 FPU rows so zeroing them gives power savings
        // This particular instruction clears both banks after waiting for both of them to be free
        // It must run alone: a both-bank SrcB clear drops SrcA writes that unpacker 0 makes while it runs
        hal_port::custom_mm_clear_srcb_both_banks();
    }

    hal_port::custom_mm_reset_unpack_counters();

    // These are RISC-V stores, so they can overtake unpacks still queued from the previous op (e.g. a
    // tilize whose CFGSHIFTMASK reads SCRATCH_SEC0); wait for its contexts to retire first.
    hal_port::wait_for_next_context(1);

    static_assert(hcfg::Scratch::val.addr32(hcfg::Sec::S0) == SCRATCH_SEC0_val_ADDR32, "scratch word");
    static_assert(hcfg::Scratch::val.addr32(hcfg::Sec::S2) == SCRATCH_SEC2_val_ADDR32, "scratch word");
    cfg[hcfg::Scratch::val.addr32(hcfg::Sec::S0)] = 68;
    cfg[hcfg::Scratch::val.addr32(hcfg::Sec::S1)] = 36;
    cfg[hcfg::Scratch::val.addr32(hcfg::Sec::S2)] = 20;
    // HAL-GAP: no HAL API for RISC-V regfile (GPR) stores; production lines kept.
    regfile[p_gpr_unpack::PERF_UNPACK_NUM_TILES_3] = 0x10 | static_cast<std::uint32_t>(DataFormat::Bfp8_b);
    regfile[p_gpr_unpack::PERF_UNPACK_NUM_TILES_2] = 0x10 | static_cast<std::uint32_t>(DataFormat::Bfp4_b);
    regfile[p_gpr_unpack::PERF_UNPACK_NUM_TILES_1] = 0x10 | static_cast<std::uint32_t>(DataFormat::Bfp2_b);
}

constexpr std::uint32_t get_replay_insn_for_combo(const std::uint8_t combo)
{
    std::uint8_t prev = combo & 0b11;
    std::uint8_t curr = (combo >> 3) & 0b11;

    bool use_b         = (combo >> 2) & 0b1;
    bool need_reconfig = prev != curr;
    bool need_stall    = need_reconfig && (prev == 1 || curr == 1);

    std::uint32_t start_idx    = curr == 3 ? 0 : curr == 2 ? 8 : 16;
    std::uint32_t start_offset = 3;
    std::uint32_t replay_len   = 3;

    if (use_b)
    {
        replay_len += 2;
    }
    if (need_reconfig)
    {
        start_offset -= 2;
        replay_len += 2;
    }
    if (need_stall)
    {
        start_offset -= 1;
        replay_len += 1;
    }

    if (curr == 0)
    {
        if (use_b)
        {
            return hal::replay::get_operation(hal::replay::BufferRange {22, need_stall ? 3u : 2u});
        }
        else
        {
            return need_stall ? hal::replay::get_operation(hal::replay::BufferRange {24, 1}) : hal::nop::thread_operation();
        }
    }

    return hal::replay::get_operation(hal::replay::BufferRange {start_idx + start_offset, replay_len});
}

/**
 * @brief Unpack a kt_dim x ct_dim block of compressed weight tiles into SrcA and the matching activation tiles into
 *        SrcB, switching the SrcA format per tile as the metadata says.
 *
 * @param base_address_a: L1 address of the compressed weight stream (SrcA), in the 16-byte-word encoding of
 *                        L1_ADDRESS(). Tiles are packed back to back at their own format's size; zero tiles take no
 *                        space.
 * @param base_address_b: L1 address of the activations (SrcB), in the same encoding.
 * @param base_address_meta: Byte address of the per-tile metadata, read by the RISC-V (so not L1_ADDRESS()). Each
 *                           word holds 10 tiles: the previous tile's format in bits 1:0, then per tile a use_b bit
 *                           and a 2-bit format (0 zero, 1 bfp2, 2 bfp4, 3 bfp8).
 * @param kt_dim: Inner dimension in tiles, an even number from 2 to 254.
 * @param ct_dim: Output width in tiles, 1 to 64; DEST capacity bounds it, see @ref _llk_math_compressed_custom_mm_.
 * @note Call @ref _llk_unpack_AB_compressed_custom_mm_init_ first.
 * @note Every call ends on the to/from-bfp2 format stall, so the next call's metadata can start from the
 *       previous-format sentinel (0) without encoding one.
 * @note On the math thread, pair with @ref _llk_math_compressed_custom_mm_.
 */
inline void _llk_unpack_AB_compressed_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t base_address_meta,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1)
{
    static constexpr std::uint32_t FMTABLE[32] = {
        get_replay_insn_for_combo(0b000'00), // 0b000'00 zero to zero
        get_replay_insn_for_combo(0b000'01), // 0b000'01 bfp2 to zero
        get_replay_insn_for_combo(0b000'10), // 0b000'10 bfp4 to zero
        get_replay_insn_for_combo(0b000'11), // 0b000'11 bfp8 to zero

        get_replay_insn_for_combo(0b001'00), // 0b001'00 zero to zero with b
        get_replay_insn_for_combo(0b001'01), // 0b001'01 bfp2 to zero with b
        get_replay_insn_for_combo(0b001'10), // 0b001'10 bfp4 to zero with b
        get_replay_insn_for_combo(0b001'11), // 0b001'11 bfp8 to zero with b

        get_replay_insn_for_combo(0b010'00), // 0b010'00 zero to bfp2
        get_replay_insn_for_combo(0b010'01), // 0b010'01 bfp2 to bfp2
        get_replay_insn_for_combo(0b010'10), // 0b010'10 bfp4 to bfp2
        get_replay_insn_for_combo(0b010'11), // 0b010'11 bfp8 to bfp2

        get_replay_insn_for_combo(0b011'00), // 0b011'00 zero to bfp2 with b
        get_replay_insn_for_combo(0b011'01), // 0b011'01 bfp2 to bfp2 with b
        get_replay_insn_for_combo(0b011'10), // 0b011'10 bfp4 to bfp2 with b
        get_replay_insn_for_combo(0b011'11), // 0b011'11 bfp8 to bfp2 with b

        get_replay_insn_for_combo(0b100'00), // 0b100'00 zero to bfp4
        get_replay_insn_for_combo(0b100'01), // 0b100'01 bfp2 to bfp4
        get_replay_insn_for_combo(0b100'10), // 0b100'10 bfp4 to bfp4
        get_replay_insn_for_combo(0b100'11), // 0b100'11 bfp8 to bfp4

        get_replay_insn_for_combo(0b101'00), // 0b101'00 zero to bfp4 with b
        get_replay_insn_for_combo(0b101'01), // 0b101'01 bfp2 to bfp4 with b
        get_replay_insn_for_combo(0b101'10), // 0b101'10 bfp4 to bfp4 with b
        get_replay_insn_for_combo(0b101'11), // 0b101'11 bfp8 to bfp4 with b

        get_replay_insn_for_combo(0b110'00), // 0b110'00 zero to bfp8
        get_replay_insn_for_combo(0b110'01), // 0b110'01 bfp2 to bfp8
        get_replay_insn_for_combo(0b110'10), // 0b110'10 bfp4 to bfp8
        get_replay_insn_for_combo(0b110'11), // 0b110'11 bfp8 to bfp8

        get_replay_insn_for_combo(0b111'00), // 0b111'00 zero to bfp8 with b
        get_replay_insn_for_combo(0b111'01), // 0b111'01 bfp2 to bfp8 with b
        get_replay_insn_for_combo(0b111'10), // 0b111'10 bfp4 to bfp8 with b
        get_replay_insn_for_combo(0b111'11), // 0b111'11 bfp8 to bfp8 with b
    };

    LLK_ASSERT(kt_dim <= 254, "compressed custom_mm: kt_dim above 254 needs the SrcB walk split");

    namespace hcfg = hal::cfg;
    namespace sync = hal::sync;

    volatile std::uint32_t* cfg = hcfg::detail::state_cfg_bank(); // get_cfg_pointer()

    const std::uint32_t address_a  = base_address_a;
    const std::uint32_t address_b  = base_address_b;
    const std::uint32_t full_iters = (kt_dim * ct_dim) / 10;
    const std::uint32_t rem_iters  = (kt_dim * ct_dim) % 10;
    const std::uint32_t* meta_ptr  = reinterpret_cast<const std::uint32_t*>(base_address_meta);

    hal_port::wait_for_next_context(1);
    hal_port::reset_config_context();

    constexpr std::uint32_t base_a_addr = hcfg::Unpacker[0].Cntx[0].Base.addr32(hcfg::Sec::S0);
    constexpr std::uint32_t base_b_addr = hcfg::Unpacker[1].Cntx[0].Base.addr32(hcfg::Sec::S0);
    static_assert(base_a_addr == THCON_SEC0_REG3_Base_address_ADDR32 && base_b_addr == THCON_SEC1_REG3_Base_address_ADDR32, "unpacker base words");
    cfg[base_a_addr] = address_a;
    cfg[base_b_addr] = address_b;

    // semaphore_post(semaphore::UNPACK_SYNC): MMIO
    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>();

    // TTI_STALLWAIT(STALL_UNPACK, TRISC_CFG)
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

    // HAL-GAP: no public HAL call issues a runtime-selected, pre-encoded instruction word (the FMTABLE replay
    // handles); TT_INSN kept as production.

    for (std::uint32_t i = 0; i < full_iters; ++i)
    {
        std::uint32_t meta = meta_ptr[i];

        std::uint32_t idx0 = (meta >> 0) & 0b11111;
        std::uint32_t idx1 = (meta >> 3) & 0b11111;
        std::uint32_t idx2 = (meta >> 6) & 0b11111;
        std::uint32_t idx3 = (meta >> 9) & 0b11111;
        std::uint32_t idx4 = (meta >> 12) & 0b11111;
        std::uint32_t idx5 = (meta >> 15) & 0b11111;
        std::uint32_t idx6 = (meta >> 18) & 0b11111;
        std::uint32_t idx7 = (meta >> 21) & 0b11111;
        std::uint32_t idx8 = (meta >> 24) & 0b11111;
        std::uint32_t idx9 = (meta >> 27) & 0b11111;

        TT_INSN(FMTABLE[idx0]);
        TT_INSN(FMTABLE[idx1]);
        TT_INSN(FMTABLE[idx2]);
        TT_INSN(FMTABLE[idx3]);
        TT_INSN(FMTABLE[idx4]);
        TT_INSN(FMTABLE[idx5]);
        TT_INSN(FMTABLE[idx6]);
        TT_INSN(FMTABLE[idx7]);
        TT_INSN(FMTABLE[idx8]);
        TT_INSN(FMTABLE[idx9]);
    }
    std::uint32_t meta = meta_ptr[full_iters];
    for (std::uint32_t i = 0; i < rem_iters; ++i)
    {
        std::uint32_t idx0 = (meta >> 0) & 0b11111;
        TT_INSN(FMTABLE[idx0]);
        meta >>= 3;
    }

    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>();

    // The to/from bfp2 stall is encoded per tile only within a call; the first tile's prev format is a
    // metadata sentinel, not the last format of the previous call, so always end on the stall. It only has
    // to precede the next call's unpacks, so it goes after the SEMGET to keep it out of the context handshake.
    hal_port::compressed_custom_mm_clr_srca_stall();

    hal_port::wait_for_next_context(1);
    hal_port::reset_config_context();

    hal_port::custom_mm_reset_unpack_counters();
}
