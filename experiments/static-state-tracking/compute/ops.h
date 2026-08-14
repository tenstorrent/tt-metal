// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The compute ops.
//
// Each configuring op is a pure function  Tag<S_in> -> Tag<S_out> with an
// explicit THREE-PART footprint over State fields (the audit-table contract):
//   * requires — fields (and values) the op depends on. Unsatisfiable
//     requirements are static_asserts (compile error, never silent corruption);
//     satisfiable ones become `if constexpr` reconfigure guards: when the state
//     proves the hardware already matches, the *_cfg call is not compiled at
//     all — zero bytes, zero cycles.
//   * writes — fields the op sets to a known new value (reflected in *Next).
//   * clobbers — fields whose hardware the op damages without establishing the
//     tracked value. Where the new contents are still describable we prefer a
//     tracked write of the new value (see PackStrideProgram); a true clobber
//     resets the field to unknown so the next dependent op must reconfigure.
//
// The same compile-time `State` threads identically through all three TRISC
// builds (UNPACK / MATH / PACK); the UNPACK()/MATH()/PACK() gates decide which
// physical instructions each TRISC emits.

#ifndef SST_COMPUTE_OPS_H
#define SST_COMPUTE_OPS_H

#include <cstdint>

#include "defs.h"
#include "hw_math.h"
#include "hw_pack.h"
#include "hw_unpack.h"
#include "experiments/static-state-tracking/inc/state.h"
#include "experiments/static-state-tracking/tensor/all.h"

namespace sst {
namespace compute {

using namespace tensor;

// Build-define-derived traits. Used ONLY to (a) establish the initial tracked
// DestCfg at hw_startup and (b) drive the tile_regs_* dest-sync protocol
// helpers, which are per-call protocol, not tracked configuration. Every
// configuring op derives its traits FROM the tracked state instead (TraitsFrom)
// — fp32-dest-acc is runtime-mutable state (audit: _llk_*_set_fp32_dest_acc_),
// so an op keyed on the build define instead of the state would go stale the
// moment a kernel toggles it.
using ActiveKernelTraits = KernelTraits<
    (DST_ACCUM_MODE != 0),
    (DST_SYNC_MODE == DstSync::SyncFull) ? DstSyncMode::SyncFull : DstSyncMode::SyncHalf>;

// The hw-layer traits carrier instantiated from the TRACKED dest config.
// Callers must have already required `S.global.dest.known`.
template <const State& S>
using TraitsFrom = KernelTraits<S.global.dest.value.fp32_acc, S.global.dest.value.sync>;

// ---------------------------------------------------------------------------
// Next-state builders (pure constexpr) + ODR-safe named results.
// ---------------------------------------------------------------------------

// copy_tile writes: unpack program (mode+geometry), math FPU program
// (mode+geometry). Descriptors and global cfg are requires-only (unchanged).
constexpr State with_copy(State s, const TileConfig& tile_config) {
    s.unpack.mode = Tracked<UnpackMode>{UnpackMode::DataCopy};
    s.unpack.tile_config = Tracked<TileConfig>{tile_config};
    s.math.mode = Tracked<MathMode>{MathMode::DataCopy};
    s.math.tile_config = Tracked<TileConfig>{tile_config};
    return s;
}

// untilize_block writes: pack program (mode+geometry) — and the packer output
// STRIDES, which are descriptor-owned registers (PCK0_ADDR_CTRL_ZW_REG_0 +
// SCRATCH_SEC2; audit: `_llk_pack_untilize_init_` aliases the
// `_llk_pack_hw_configure_` stride group). That aliasing is recorded as a
// tracked write of `stride_prog = Untilize` rather than left as silent state
// drift: a later Default-mode pack op requires Default and re-emits strides.
constexpr State with_untilize(
    State s, const TileConfig& tile_config, uint16_t tiles_per_block, uint16_t tiles_per_row) {
    s.pack.mode = Tracked<PackMode>{PackMode::Untilize};
    s.pack.tile_config = Tracked<TileConfig>{tile_config};
    s.pack.tiles_per_block = Tracked<uint16_t>{tiles_per_block};
    s.pack.tiles_per_row = Tracked<uint16_t>{tiles_per_row};
    s.pack.stride_prog = Tracked<PackStrideProgram>{PackStrideProgram::Untilize};
    return s;
}

template <const State& S, typename TileT>
struct CopyNext {
    static constexpr State value = with_copy(S, Resolver<TileT>::tile_config());
};
template <const State& S, uint16_t TilesPerBlock, uint16_t TilesPerRow, typename TileT>
struct UntilizeNext {
    static constexpr State value = with_untilize(S, Resolver<TileT>::tile_config(), TilesPerBlock, TilesPerRow);
};

// hw_startup writes: both unpack operand descriptors, the pack descriptor
// (with its relu / l1_acc sub-fields and Default strides — configure_pack
// programs all of them), and the core-global ALU-format + DEST configs.
// It CLOBBERS every op program: starting from kInitial leaves unpack/math/pack
// program fields unknown, which is the explicit clobber — a previous kernel's
// MOP/ADDR_MOD contents are dead, so the first op of each engine must program
// its own (the loop combinator hoists that to exactly once).
constexpr State with_startup(
    State s,
    const TileConfig& in_a,
    const TileConfig& in_b,
    const TileConfig& output,
    bool fp32,
    DstSyncMode sync,
    bool remap) {
    s.unpack.src_a_desc = Tracked<TileConfig>{in_a};
    s.unpack.src_b_desc = Tracked<TileConfig>{in_b};

    s.pack.desc = Tracked<TileConfig>{output};
    s.pack.relu = Tracked<uint32_t>{0};      // configure_pack programs relu_config = 0
    s.pack.l1_acc = Tracked<bool>{false};    // and leaves L1-acc disabled
    s.pack.stride_prog = Tracked<PackStrideProgram>{PackStrideProgram::Default};

    s.global.alu_fmt = Tracked<AluFmtCfg>{AluFmtCfg{in_a.data_format, in_b.data_format, /*int8_math=*/false}};
    s.global.dest = Tracked<DestCfg>{DestCfg{fp32, sync, remap}};
    return s;
}
template <typename TileInA, typename TileInB, typename TileOut, bool Remap>
struct StartupNext {
    static constexpr State value = with_startup(
        kInitial,
        Resolver<TileInA>::tile_config(),
        Resolver<TileInB>::tile_config(),
        Resolver<TileOut>::tile_config(),
        ActiveKernelTraits::fp32_dest_acc,
        ActiveKernelTraits::dst_sync,
        Remap);
};

// ---------------------------------------------------------------------------
// hw_startup: the ONE explicit setup. Base HW configure on every TRISC, plus
// an explicit MATH remap write. Returns known descriptor + global domains;
// per-engine op programs remain unknown until the first copy / untilize,
// which the loop combinator hoists so they run exactly once.
// ---------------------------------------------------------------------------
// `Remap` selects the DST production layout the whole kernel uses:
//   Remap=true  (default) — untilize pipelines: MATH writes DST in the
//                stride-16 remapped layout so a DST-draining untilize packer
//                consumes it directly. Must be enabled BEFORE any DST write.
//   Remap=false — tiled-output pipelines (matmul + default pack): DST stays in
//                the natural tiled layout the default packer expects.
template <typename TileInA, typename TileInB, typename TileOut, bool Remap = true>
ALWI auto hw_startup() {
    // Each is consumed inside per-TRISC macro-gated calls below (the two input
    // configs on UNPACK/MATH, the output on PACK), so on the other TRISC builds it
    // is unused — [[maybe_unused]] silences that per-build warning without
    // duplicating the (constexpr) resolve at each call site.
    [[maybe_unused]] constexpr TileConfig tile_config_in_a = Resolver<TileInA>::tile_config();
    [[maybe_unused]] constexpr TileConfig tile_config_in_b = Resolver<TileInB>::tile_config();
    [[maybe_unused]] constexpr TileConfig tile_config_out = Resolver<TileOut>::tile_config();

    // Base HW configure only — the stuff every kernel needs regardless of which
    // ops it runs (formats, tile sizes, strides, DST-sync, dest-offset regs). We
    // deliberately do NOT build any op MOP here: the pack MOP is op-specific
    // (Default vs Untilize vs Tilize) and every pack op programs its own, so a
    // MOP built at startup is always thrown away. The first pack op builds it
    // exactly once (straight-line: the only time; in a loop: hoisted by `loop`).
    UNPACK((hw::unpack_hw_cfg<ActiveKernelTraits>(tile_config_in_a, tile_config_in_b)));
    MATH((hw::math_pack_sync_cfg<ActiveKernelTraits>()));
    MATH((hw::math_hw_cfg<ActiveKernelTraits>()));
    PACK((hw::pack_hw_cfg<ActiveKernelTraits>(tile_config_out)));
    PACK((hw::pack_dest_cfg<ActiveKernelTraits>()));

    // Establish a known DEST layout in hardware for both modes. Writing false
    // matters when a prior kernel left remap/swizzle enabled.
    MATH((hw::math_remap_cfg(Remap)));

    return Tag<StartupNext<TileInA, TileInB, TileOut, Remap>::value>{};
}

// ---------------------------------------------------------------------------
// tile_regs_* — DST ownership handshake between MATH and PACK. State-transparent
// (they change no HW configuration), so they do not thread state. They key on
// the build defines (ActiveKernelTraits) because they are dest-sync PROTOCOL —
// per-call class-A behavior, deliberately outside the tracked State.
// ---------------------------------------------------------------------------
ALWI void tile_regs_acquire() { MATH((hw::math_wait_for_dest_available())); }
ALWI void tile_regs_commit() { MATH((hw::math_dest_section_done<ActiveKernelTraits>())); }
ALWI void tile_regs_wait() { PACK((hw::packer_wait_for_math_done())); }
ALWI void tile_regs_release() { PACK((hw::pack_dest_section_done<ActiveKernelTraits>())); }

// ---------------------------------------------------------------------------
// copy_tile: datacopy one tile A -> DST[dst_idx].
//
// requires: global.dest known (fp32 selects the MOV vs ELWADD datacopy MOP)
//           global.alu_fmt.srca_format == TileT's format (ALU consumes SrcA)
//           unpack.src_a_desc == TileT's config (descriptor must match the
//             tile being unpacked; granular descriptor reconfig is not yet
//             ported, so a mismatch is a compile error, not a reconfigure)
//           unpack/math programs == datacopy@geometry (reconfigured on mismatch)
// writes:   unpack.{mode,tile_config}, math.{mode,tile_config}
// clobbers: — (its counters/dvalid traffic is class-A protocol, balanced per call)
// ---------------------------------------------------------------------------
template <const State& S, typename TileT, Backend B>
ALWI auto copy_tile(Tag<S>, const Tensor<TileT, B>& in, uint32_t in_idx, uint32_t dst_idx) {
    constexpr TileConfig tile_config = Resolver<TileT>::tile_config();

    static_assert(S.global.dest.known, "copy_tile requires an established DEST config — call hw_startup first");
    static_assert(
        S.unpack.src_a_desc.matches(tile_config),
        "copy_tile requires the SrcA descriptor configured for this tile type (established by hw_startup; "
        "granular descriptor reconfiguration is not yet ported to SST)");
    static_assert(
        S.global.alu_fmt.known && S.global.alu_fmt.value.srca_format == tile_config.data_format,
        "copy_tile requires the ALU SrcA format to match this tile type");

    // MATH: single datacopy-MOP sub-step (depends on mode + geometry + fp32).
    constexpr bool m_all =
        !S.math.mode.matches(MathMode::DataCopy) || !S.math.tile_config.matches(tile_config);

    // UNPACK: format sub-step keys on geometry (tile_config); MOP sub-step keys on op
    // mode AND geometry — its outer loop count is tile_config.num_faces, so a same-mode
    // geometry change must still re-emit the MOP.
    if constexpr (!S.unpack.tile_config.matches(tile_config)) {
        UNPACK((hw::unpack_datacopy_face_cfg(tile_config)));
    }
    if constexpr (!S.unpack.mode.matches(UnpackMode::DataCopy) || !S.unpack.tile_config.matches(tile_config)) {
        UNPACK((hw::unpack_datacopy_mop_cfg(tile_config)));
    }
    if constexpr (m_all) {
        MATH((hw::math_a2d_cfg<TraitsFrom<S>>(tile_config)));
    }

    UNPACK((hw::unpack_a(in.tile_addr_16B(in_idx))));
    MATH((hw::math_a2d(dst_idx)));
    (void)in;
    (void)in_idx;
    (void)dst_idx;
    return Tag<CopyNext<S, TileT>::value>{};
}

// ---------------------------------------------------------------------------
// untilize_block: pack a block of DST tiles -> L1 in untilized layout.
//
// requires: global.dest known, with remap enabled (the strided untilize packer
//             consumes the stride-16 remapped DST layout)
//           pack.desc == TileT's config (output formats programmed by startup)
//           pack program == untilize@{block,row,geometry} (reconfigured on
//             mismatch, split into MOP / stride sub-steps)
// writes:   pack.{mode,tile_config,tiles_per_block,tiles_per_row},
//           pack.stride_prog = Untilize  — the tracked form of this op's
//             aliasing write into descriptor-owned stride registers (§ audit:
//             _llk_pack_untilize_init_ writes PCK0_ADDR_CTRL_ZW_REG_0_Zstride)
// clobbers: — (the L1 dest address it moves per call is class-A protocol)
//
// TilesPerBlock/TilesPerRow lead the template list so callers write
// `untilize_block<tiles_per_block, tiles_per_row>(state, out, block_index)`.
// ---------------------------------------------------------------------------
template <uint16_t TilesPerBlock, uint16_t TilesPerRow, const State& S, typename TileT, Backend B>
ALWI auto untilize_block(Tag<S>, const Tensor<TileT, B>& out, uint32_t col_tile_offset) {
    constexpr TileConfig tile_config = Resolver<TileT>::tile_config();

    static_assert(S.global.dest.known, "untilize_block requires an established DEST config — call hw_startup first");
    static_assert(
        S.global.dest.value.remap,
        "untilize_block consumes the stride-16 remapped DST layout — hw_startup<..., Remap=true> required");
    static_assert(
        S.pack.desc.matches(tile_config),
        "untilize_block requires the PACK descriptor configured for this tile type (established by hw_startup; "
        "granular descriptor reconfiguration is not yet ported to SST)");

    // The untilize PACR MOP depends on mode + block width (TilesPerBlock) + geometry —
    // its outer loop count is tile_config.face_r_dim, so a same-mode/same-block geometry
    // change must still re-emit it.
    if constexpr (
        !S.pack.mode.matches(PackMode::Untilize) || !S.pack.tiles_per_block.matches(TilesPerBlock) ||
        !S.pack.tile_config.matches(tile_config)) {
        PACK((hw::pack_untilize_mop_cfg<TilesPerBlock>(tile_config)));
    }
    // The per-row strides / output offset depend on geometry (tile_config) + full row
    // width (TilesPerRow) — and on WHO programmed the stride registers last: coming out
    // of startup they hold the Default-pack values (stride_prog == Default), so the
    // first untilize must re-emit them even when geometry fields happen to match.
    if constexpr (
        !S.pack.tile_config.matches(tile_config) || !S.pack.tiles_per_row.matches(TilesPerRow) ||
        !S.pack.stride_prog.matches(PackStrideProgram::Untilize)) {
        PACK((hw::pack_untilize_row_cfg<TilesPerRow>(tile_config)));
    }

    PACK((hw::pack_untilize<TilesPerBlock>(out.l1_addr_16B, col_tile_offset, tile_config, /*dst_tile_index=*/0)));
    (void)out;
    (void)col_tile_offset;
    return Tag<UntilizeNext<S, TilesPerBlock, TilesPerRow, TileT>::value>{};
}

}  // namespace compute
}  // namespace sst

#endif  // SST_COMPUTE_OPS_H
