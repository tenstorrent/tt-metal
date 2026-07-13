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

// ---------------------------------------------------------------------------
// Next-state builders (pure constexpr) + ODR-safe named results.
// ---------------------------------------------------------------------------
constexpr State with_copy(State s, const TileConfig& tc) {
    s.u.mode = Tracked<UnpackMode>{UnpackMode::DataCopy};
    s.u.tc = Tracked<TileConfig>{tc};
    s.m.mode = Tracked<MathMode>{MathMode::DataCopy};
    s.m.tc = Tracked<TileConfig>{tc};
    return s;
}
constexpr State with_untilize(State s, const TileConfig& tc, uint16_t block_ct, uint16_t full_ct) {
    s.p.mode = Tracked<PackMode>{PackMode::Untilize};
    s.p.tc = Tracked<TileConfig>{tc};
    s.p.block_ct = Tracked<uint16_t>{block_ct};
    s.p.full_ct = Tracked<uint16_t>{full_ct};
    return s;
}

template <const State& S, typename TileT>
struct CopyNext {
    static constexpr State value = with_copy(S, Resolver<TileT>::tile_config());
};
template <const State& S, uint16_t Bct, uint16_t Fct, typename TileT>
struct UntilizeNext {
    static constexpr State value = with_untilize(S, Resolver<TileT>::tile_config(), Bct, Fct);
};

// hw_startup establishes the DST-produce layout for the whole kernel (it emits
// the remap toggle up front when Remap=true.
constexpr State with_startup(State s, bool remap) {
    s.m.remap = Tracked<bool>{remap};
    return s;
}
template <bool Remap>
struct StartupNext {
    static constexpr State value = with_startup(kInitial, Remap);
};

// ---------------------------------------------------------------------------
// hw_startup: the ONE explicit setup. Base HW configure on every TRISC, plus
// the untilize MATH remap (output is row-major; the remap is constant for the
// whole kernel so it is not tracked). Returns the all-unknown state — the
// per-engine op modes are still established by the first copy / untilize, which
// the loop combinator hoists so they run exactly once.
// ---------------------------------------------------------------------------
// `Remap` selects the DST production layout the whole kernel uses:
//   Remap=true  (default) — untilize pipelines: MATH writes DST in the
//                stride-16 remapped layout so a DST-draining untilize packer
//                consumes it directly. Must be enabled BEFORE any DST write.
//   Remap=false — tiled-output pipelines (matmul + default pack): DST stays in
//                the natural tiled layout the default packer expects.
template <typename TileIn, typename TileOut, bool Remap = true>
ALWI auto hw_startup() {
    // Each is consumed inside per-TRISC macro-gated calls below (tc_in on
    // UNPACK/MATH, tc_out on PACK), so on the other TRISC builds it is unused —
    // [[maybe_unused]] silences that per-build warning without duplicating the
    // (constexpr) resolve at each call site.
    [[maybe_unused]] constexpr TileConfig tc_in = Resolver<TileIn>::tile_config();
    [[maybe_unused]] constexpr TileConfig tc_out = Resolver<TileOut>::tile_config();

    // Base HW configure only — the stuff every kernel needs regardless of which
    // ops it runs (formats, tile sizes, strides, DST-sync, dest-offset regs). We
    // deliberately do NOT build any op MOP here: the pack MOP is op-specific
    // (Default vs Untilize vs Tilize) and every pack op programs its own, so a
    // MOP built at startup is always thrown away. The first pack op builds it
    // exactly once (straight-line: the only time; in a loop: hoisted by `loop`).
    UNPACK((hw::unpack_hw_cfg(tc_in)));
    MATH((hw::math_pack_sync_cfg()));
    MATH((hw::math_hw_cfg(tc_in)));
    PACK((hw::pack_hw_cfg(tc_out)));
    PACK((hw::pack_dest_cfg()));

    if constexpr (Remap) {
        // Row-major output => enable the DEST stride-16 remap up front, before any
        // producer writes DST.
        MATH((hw::math_remap_cfg(true)));
    }

    return Tag<StartupNext<Remap>::value>{};
}

// ---------------------------------------------------------------------------
// tile_regs_* — DST ownership handshake between MATH and PACK. State-transparent
// (they change no HW configuration), so they do not thread state.
// ---------------------------------------------------------------------------
ALWI void tile_regs_acquire() { MATH((hw::math_wait_for_dest_available())); }
ALWI void tile_regs_commit() { MATH((hw::math_dest_section_done())); }
ALWI void tile_regs_wait() { PACK((hw::packer_wait_for_math_done())); }
ALWI void tile_regs_release() { PACK((hw::pack_dest_section_done())); }

// ---------------------------------------------------------------------------
// copy_tile: datacopy one tile A -> DST[dst_idx]. Configures UNPACK + MATH for
// datacopy only when the incoming state does not already say so.
// ---------------------------------------------------------------------------
template <const State& S, typename TileT, Backend B>
ALWI auto copy_tile(Tag<S>, const Tensor<TileT, B>& in, uint32_t in_idx, uint32_t dst_idx) {
    constexpr TileConfig tc = Resolver<TileT>::tile_config();

    // MATH: single datacopy-MOP sub-step (depends on mode + geometry).
    constexpr bool m_all = !S.m.mode.matches(MathMode::DataCopy) || !S.m.tc.matches(tc);

    // UNPACK: format sub-step keys on geometry (tc); MOP sub-step keys on op mode.
    if constexpr (!S.u.tc.matches(tc)) {
        UNPACK((hw::unpack_datacopy_face_cfg(tc)));
    }
    if constexpr (!S.u.mode.matches(UnpackMode::DataCopy)) {
        UNPACK((hw::unpack_datacopy_mop_cfg(tc)));
    }
    if constexpr (m_all) {
        MATH((hw::math_a2d_cfg(tc)));
    }

    UNPACK((hw::unpack_a(in.tile_addr_16B(in_idx))));
    MATH((hw::math_a2d(dst_idx)));
    (void)in;
    (void)in_idx;
    (void)dst_idx;
    return Tag<CopyNext<S, TileT>::value>{};
}

// ---------------------------------------------------------------------------
// pack_untilize_dest: pack a block of DST tiles -> L1 in untilized layout.
// Configures PACK for untilize (+ geometry) only when the incoming state does
// not already match.  Bct/Fct lead the template list so callers write
// `pack_untilize_dest<block_ct, full_ct>(state, out, block_index)`.
// ---------------------------------------------------------------------------
template <uint16_t Bct, uint16_t Fct, const State& S, typename TileT, Backend B>
ALWI auto pack_untilize_dest(Tag<S>, const Tensor<TileT, B>& out, uint32_t col_tile_offset) {
    constexpr TileConfig tc = Resolver<TileT>::tile_config();

    // The untilize PACR MOP depends on mode + block width (Bct); the per-row
    // strides / output offset depend on geometry (tc) + full row width (Fct).
    // Changing only the block width thus reprograms the MOP but keeps the strides;
    // changing only the row width keeps the MOP and redoes the strides.
    if constexpr (!S.p.mode.matches(PackMode::Untilize) || !S.p.block_ct.matches(Bct)) {
        PACK((hw::pack_untilize_mop_cfg<Bct>(tc)));
    }
    if constexpr (!S.p.tc.matches(tc) || !S.p.full_ct.matches(Fct)) {
        PACK((hw::pack_untilize_row_cfg<Fct>(tc)));
    }

    PACK((hw::pack_untilize<Bct>(out.l1_addr_16B, col_tile_offset, tc, /*dst_tile_index=*/0)));
    (void)out;
    (void)col_tile_offset;
    return Tag<UntilizeNext<S, Bct, Fct, TileT>::value>{};
}

}  // namespace compute
}  // namespace sst

#endif  // SST_COMPUTE_OPS_H
