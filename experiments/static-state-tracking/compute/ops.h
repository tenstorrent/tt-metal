// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The compute ops.
//
// Each configuring op is a pure function  Tag<S_in> -> Tag<S_out>:
//   1. Inspect the incoming compile-time state `S`.
//   2. Emit the engine reconfigure ONLY if `S` does not already satisfy the op,
//      via `if constexpr`. When the state proves the hardware is already
//      configured, the *_cfg call is not compiled at all — zero bytes, zero
//      cycles. This is the manual-tracking payoff: the `*_init` calls fold into
//      the op and vanish once redundant.
//   3. Emit the actual per-tile / per-block engine work, which always runs.
//   4. Return the new state.
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

// Reconfigure guards below are written as compile-time `if constexpr (cond)`:
// when the tracked state already proves a reconfigure redundant the condition is
// false and the code inside is never generated.

// Every op reconfigures at SUB-STEP granularity: a config sub-step
// is emitted only if the specific tracked field it depends on changed.

namespace sst {
namespace compute {

using namespace tensor;

using ActiveKernelTraits = KernelTraits<
    (DST_ACCUM_MODE != 0),
    (DST_SYNC_MODE == DstSync::SyncFull) ? DstSyncMode::SyncFull : DstSyncMode::SyncHalf>;

// ---------------------------------------------------------------------------
// Next-state builders (pure constexpr) + ODR-safe named results.
// ---------------------------------------------------------------------------
constexpr State with_copy(State s, const TileConfig& tile_config) {
    s.operations.unpack.mode = Tracked<UnpackMode>{UnpackMode::DataCopy};
    s.operations.unpack.tile_config = Tracked<TileConfig>{tile_config};
    s.operations.math.mode = Tracked<MathMode>{MathMode::DataCopy};
    s.operations.math.tile_config = Tracked<TileConfig>{tile_config};
    return s;
}
constexpr State with_untilize(
    State s, const TileConfig& tile_config, uint16_t tiles_per_block, uint16_t tiles_per_row) {
    s.operations.pack.mode = Tracked<PackMode>{PackMode::Untilize};
    s.operations.pack.tile_config = Tracked<TileConfig>{tile_config};
    s.operations.pack.tiles_per_block = Tracked<uint16_t>{tiles_per_block};
    s.operations.pack.tiles_per_row = Tracked<uint16_t>{tiles_per_row};
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

// hw_startup establishes independent operand descriptors and the DEST layout.
// Immutable FP32/DST-sync configuration lives in KernelTraits, outside State.
// Operation programs remain unknown until the first op init.
constexpr State with_startup(
    State s, const TileConfig& in_a, const TileConfig& in_b, const TileConfig& output, bool remap) {
    s.operands.src_a = Tracked<TileConfig>{in_a};
    s.operands.src_b = Tracked<TileConfig>{in_b};
    s.operands.pack = Tracked<TileConfig>{output};

    s.layout.remap = Tracked<bool>{remap};
    return s;
}
template <typename TileInA, typename TileInB, typename TileOut, bool Remap>
struct StartupNext {
    static constexpr State value = with_startup(
        kInitial,
        Resolver<TileInA>::tile_config(),
        Resolver<TileInB>::tile_config(),
        Resolver<TileOut>::tile_config(),
        Remap);
};

// ---------------------------------------------------------------------------
// hw_startup: the ONE explicit setup. Base HW configure on every TRISC, plus
// an explicit MATH remap write. Returns known operand/layout domains; per-engine
// operation programs remain unknown until the first copy / untilize,
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
// (they change no HW configuration), so they do not thread state.
// ---------------------------------------------------------------------------
ALWI void tile_regs_acquire() { MATH((hw::math_wait_for_dest_available())); }
ALWI void tile_regs_commit() { MATH((hw::math_dest_section_done<ActiveKernelTraits>())); }
ALWI void tile_regs_wait() { PACK((hw::packer_wait_for_math_done())); }
ALWI void tile_regs_release() { PACK((hw::pack_dest_section_done<ActiveKernelTraits>())); }

// ---------------------------------------------------------------------------
// copy_tile: datacopy one tile A -> DST[dst_idx]. Configures UNPACK + MATH for
// datacopy only when the incoming state does not already say so.
// ---------------------------------------------------------------------------
template <const State& S, typename TileT, Backend B>
ALWI auto copy_tile(Tag<S>, const Tensor<TileT, B>& in, uint32_t in_idx, uint32_t dst_idx) {
    constexpr TileConfig tile_config = Resolver<TileT>::tile_config();

    // MATH: single datacopy-MOP sub-step (depends on mode + geometry).
    constexpr bool m_all =
        !S.operations.math.mode.matches(MathMode::DataCopy) || !S.operations.math.tile_config.matches(tile_config);

    // UNPACK: format sub-step keys on geometry (tile_config); MOP sub-step keys on op
    // mode AND geometry — its outer loop count is tile_config.num_faces, so a same-mode
    // geometry change must still re-emit the MOP.
    if constexpr (!S.operations.unpack.tile_config.matches(tile_config)) {
        UNPACK((hw::unpack_datacopy_face_cfg(tile_config)));
    }
    if constexpr (
        !S.operations.unpack.mode.matches(UnpackMode::DataCopy) ||
        !S.operations.unpack.tile_config.matches(tile_config)) {
        UNPACK((hw::unpack_datacopy_mop_cfg(tile_config)));
    }
    if constexpr (m_all) {
        MATH((hw::math_a2d_cfg<ActiveKernelTraits>(tile_config)));
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
// Configures PACK for untilize (+ geometry) only when the incoming state does
// not already match.  TilesPerBlock/TilesPerRow lead the template list so callers write
// `untilize_block<tiles_per_block, tiles_per_row>(state, out, block_index)`.
// ---------------------------------------------------------------------------
template <uint16_t TilesPerBlock, uint16_t TilesPerRow, const State& S, typename TileT, Backend B>
ALWI auto untilize_block(Tag<S>, const Tensor<TileT, B>& out, uint32_t col_tile_offset) {
    constexpr TileConfig tile_config = Resolver<TileT>::tile_config();

    // The untilize PACR MOP depends on mode + block width (TilesPerBlock) + geometry —
    // its outer loop count is tile_config.face_r_dim, so a same-mode/same-block geometry
    // change must still re-emit it. The per-row strides / output offset depend on
    // geometry (tile_config) + full row width (TilesPerRow).
    if constexpr (
        !S.operations.pack.mode.matches(PackMode::Untilize) ||
        !S.operations.pack.tiles_per_block.matches(TilesPerBlock) ||
        !S.operations.pack.tile_config.matches(tile_config)) {
        PACK((hw::pack_untilize_mop_cfg<TilesPerBlock>(tile_config)));
    }
    if constexpr (
        !S.operations.pack.tile_config.matches(tile_config) || !S.operations.pack.tiles_per_row.matches(TilesPerRow)) {
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
