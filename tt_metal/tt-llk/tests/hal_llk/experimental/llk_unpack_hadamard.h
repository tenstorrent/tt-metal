// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LLK unpack primitives for the H128 (1x128) Hadamard transform.
//
// Tile geometry (matches hadamard.h):
//   h16   ([16, 16], 1 face): H_16 fills face (0, 0). -> srcB face 0.
//   input: the first 128 datums at the input address, in L1 order (e.g.
//          face 0 rows 0..7 of a [32, 32] tile, or 4x[1, 32] pages).
//          -> srcA face 0 rows 0..7; rows 8..15 zeroed.
//
// Single-face unpack: each phase issues exactly one UNPACR per src
// register, reading one 16x16 face per operand.
//
// Both phases are issued back-to-back on the unpack thread inside
// _llk_unpack_hadamard_h128_ (MM1 runs concurrently on the math thread):
//
//   Phase 1 (context 0, before MM1): one UNPACR per src register —
//     h16 -> srcB face 0, X_pad -> srcA bank 0 face 0.
//
//   Phase 2 (context 1, overlaps MM1): one UNPACR streams H_16 into
//     srcA's *other* bank. Setting dvalid hands that bank to the FPU, so
//     after MM1's CLR_A the FPU reads H_16 from srcA bank 1 for MM2.
//     srcB's dvalid is never cleared during the two passes, so the
//     FPU keeps H_16 in srcB until the math thread's MOVD2B overwrites
//     rows 0..7 with the MM1 result.
//
// Context scheme (avoids a per-tile CFG write + TRISC_CFG stall for the
// H_16 stream): the srcA unpacker base lives in two hardware config
// contexts. Phase 1 runs on context 0 (configuring context 0's srcA/srcB
// bases per tile) and ends with switch_config_context, so phase 2 runs on
// context 1. The init preprograms context 1's srcA base once to the H_16
// tile; since phase 1 only ever touches context 0, every phase-2 UNPACR
// reads H_16 with no per-tile reconfiguration. Phase 2 ends by resetting
// the context back to 0 for the next tile.
//
// Requirement: H_16 must sit at a fixed L1 address for the program's
// lifetime (single-slot, persistent tile, h16_tile_index == 0). Phase 1
// recomputes its srcB address each tile so it tolerates CB rotation, but
// the preprogrammed context-1 srcA base does not.

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_unpack_hadamard.h. Same instruction sequence and access
// kinds as production, expressed through the HAL.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "hal/cfg.h"
#include "hal/sync.h"
#include "hal/tensix_counters.h"
#include "hal/unpack.h"
#include "llk_unpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// Datums the input unpack reads per call: the top 8 rows of srcA face 0.
constexpr std::uint32_t HADAMARD_H128_INPUT_DATUMS = 8 * FACE_C_DIM;

// One-shot unpack init for the H128 transform. Call once at init, after
// llk_unpack_hw_configure.
//
//   1. Single-face (16x16) config: read exactly one face per operand
//      (transpose off, no partial face).
//   2. Preprogram context 1's srcA (SEC0) base with the H_16 tile's fixed
//      L1 address so phase-2 needs no per-tile CFG write (see "Context
//      scheme" in the file header). H_16 must stay resident there.
//   3. Reset to context 0 so the first tile's phase-1 unpack runs there.
inline void _llk_unpack_hadamard_h128_init_(const std::uint32_t h16_address)
{
    // Two-context handshake op: an earlier op's init may have left Auto TTSync on, and tt-metal calls no uninit.
    hal_port::unpack_sync_mode<hal_port::UnpackSyncMode::Handshake>();
    LLK_ASSERT(is_valid_L1_address(h16_address), "H_16 L1 address must be in valid L1 memory region");

    // ── Single-face config ────────────────────────────────────────────
    // Full 16x16-face X span. srcA rests here; the per-tile path narrows it
    // to 8 rows for the input read and restores it (see _llk_unpack_hadamard_h128_).
    constexpr std::uint32_t kFaceXEnd = FACE_R_DIM * FACE_C_DIM - 1;

    namespace hcfg = hal::cfg;
    namespace adc  = hal::counters::adc;

    // Hadamard never transposes faces; force the face-transpose (haloize)
    // mode off in case a prior datacopy left it on.
    hcfg::write<hcfg::Access::TensixCfgUnit, hcfg::Thcon[hcfg::Reg2].Haloize_mode, hcfg::Sec::S0, 0>();

    // Zero the z/w address counters for both unpackers: TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}, .channel1 = {.z = 0, .w = 0}}.for_clients<adc::Client::Unpacker0, adc::Client::Unpacker1>()>();

    // Full single-face span for srcA (input/h16) and srcB (h16): SETADCXX(UNP_A / UNP_B, kFaceXEnd, 0x0).
    // Compile-time span: production's TT_ form with a constexpr value.
    constexpr adc::AddrCounter face_span {.channel0 = {.x = 0}, .channel1 = {.x = kFaceXEnd}};
    adc::set<face_span.for_clients<adc::Client::Unpacker0>()>();
    adc::set<face_span.for_clients<adc::Client::Unpacker1>()>();

    // ── Context-1 H_16 preprogram ─────────────────────────────────────
    // Whole-word MMIO store, as production.
    constexpr std::uint32_t base_cntx1 = hcfg::Unpacker[0].Cntx[1].Base.addr32(hcfg::Sec::S0);
    static_assert(base_cntx1 == THCON_SEC0_REG3_Base_cntx1_address_ADDR32, "unpacker A context-1 base word");
    volatile std::uint32_t tt_reg_ptr *cfg = hcfg::detail::state_cfg_bank();
    cfg[base_cntx1]                        = h16_address;

    // Start on context 0 so the first tile's phase-1 unpack runs there.
    hal_port::reset_config_context();
}

// One tile's worth of unpack: phase 1 (h16 -> srcB, input -> srcA bank 0)
// then phase 2 (H_16 -> srcA bank 1). See the file header for the phase and
// context scheme. The 1x128 input carries only the 128 real values (no
// bottom-8 zero pad): phase 1 zeros srcA bank 0 at the full span, narrows to
// 8 rows for the input UNPACR, then restores the full span (needed for the
// phase-2 H_16 stream and the next tile's full-span zero-src). Operands:
//   base_address_a / tile_index_a / tile_size_a -> h16  (srcB)
//   base_address_b / tile_index_b / tile_size_b -> input (srcA)
inline void _llk_unpack_hadamard_h128_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b)
{
    namespace sync   = hal::sync;
    namespace adc    = hal::counters::adc;
    namespace unpack = hal::unpack;

    volatile std::uint32_t *cfg = hal::cfg::detail::state_cfg_bank();

    const std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a; // h16
    const std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b; // input

    constexpr std::uint32_t kFaceXEnd  = FACE_R_DIM * FACE_C_DIM - 1; // full 16x16 face
    constexpr std::uint32_t kInputXEnd = HADAMARD_H128_INPUT_DATUMS - 1;

    // UNPACR(Src, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Set Dvalid*/, RAREFYB_DISABLE, 0, 0 /*ContextIdInc*/, 0, 0, 1)
    constexpr unpack::DataTransfer unpack_srcb {.engine = unpack::Engine::Unpacker1, .handoff = unpack::SourceHandoff::FlipAndSetDataValid};
    constexpr unpack::DataTransfer unpack_srca {.engine = unpack::Engine::Unpacker0, .handoff = unpack::SourceHandoff::FlipAndSetDataValid};
    // SETADCXX(UNP_A, x_end, 0x0)
    constexpr auto srca_span = [](const std::uint32_t x_end) { return adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = x_end}}; };

    // ── Phase 1 (context 0): h16 -> srcB, input -> srcA bank 0 ──────────
    // Context 1 permanently stores the base address of the 16x16
    // Hadamard matrix, while context 0 is reprogrammed for every input tile.
    // Wait until context 0 is free so tile N+1 cannot overwrite tile N's base
    // address configuration before tile N finishes using it.
    hal_port::wait_for_next_context(1);

    // SEC0 (srcA) <- input, SEC1 (srcB) <- h16.
    _llk_unpack_configure_addresses_(address_b, address_a, cfg);

    sync::semaphore::post<sync::Access::MMIO, hal_port::UNPACK_SYNC_SEM>(); // Trisc::SEMPOST for context acquire

    // Stall unpacker until pending CFG writes from Trisc have completed.
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::RiscvAccessProcessed>();

    // h16 -> srcB (full face, Set Dvalid).
    unpack::run<unpack_srcb>();

    // Zero srcA bank 0 at the FULL span (ZEROSRC honors the X span, so this
    // must precede the narrow) -> rows 8..15 zero for MM1's 16-row reduction.
    // No Set Dvalid: the input UNPACR below sets it and hands the bank to FPU.
    // HAL-GAP: hal::nop encodes only the pure-delay UNPACR_NOP; the zero-source flavor (UNP_ZEROSRC) has no HAL
    // descriptor, so the production encoding is kept.
    TTI_UNPACR_NOP(SrcA, 0, 0, 0 /*no Set Dvalid*/, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);

    // Narrow srcA to the top 8 rows (128 datums) for the input read.
    adc::set<srca_span(kInputXEnd).for_clients<adc::Client::Unpacker0>()>();

    // input -> srcA bank 0 (top 8 rows, Set Dvalid).
    unpack::run<unpack_srca>();

    // Restore full face span: phase 2 streams the entire 16x16 H_16 into
    // srcA bank 1, and the next tile's zero-src needs the full span.
    adc::set<srca_span(kFaceXEnd).for_clients<adc::Client::Unpacker0>()>();

    sync::semaphore::get<hal_port::UNPACK_SYNC_SEM>(); // T6::SEMGET for context release
    hal_port::switch_config_context(unp_cfg_context);  // -> context 1

    // ── Phase 2 (context 1): stream H_16 into srcA bank 1 (overlaps MM1) ─
    // SEC0 context-1 srcA base was preprogrammed to H_16 by _init_, so no
    // per-tile CFG write is needed. Set Dvalid hands srcA bank 1 to the FPU
    // for MM2.
    unpack::run<unpack_srca>();
    sync::wait::stall<sync::StallTarget::Unpack, sync::StallCondition::Unpacker0Idle | sync::StallCondition::Unpacker1Idle>();

    // Re-pin phase 1 to context 0 for the next tile (one SETC16, no stall).
    hal_port::reset_config_context();
}
