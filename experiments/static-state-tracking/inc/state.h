// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The state model.
//
// `State` is a compile-time description of what the three Tensix engines
// (UNPACK / MATH / PACK) are currently configured to do. It is NOT stored at
// runtime: it lives entirely in the type system and is threaded from op to op.
// Every op is a pure function  Tag<S_in> -> Tag<S_out>.
//
// C++17 note (the whole reason this file looks the way it does):
//   * C++17 has NO class-type non-type template parameters. You cannot write
//     `template <State S>`.
//   * You CAN write `template <const State& S>` — a reference to an object with
//     static storage duration AND linkage is a legal C++17 NTTP.
//   * Therefore every `State` used as a template argument must be anchored in a
//     namespace-scope `inline constexpr` (see `kInitial`) or a `static
//     constexpr` data member of a template (see `PhiMerge` / the *Next structs
//     in ops.h). We NEVER use a function-local `static constexpr State` as an
//     NTTP — that has no linkage and fails to link without LTO. Getting this
//     wrong is the single biggest footgun of the approach.

#ifndef SST_STATE_H
#define SST_STATE_H

#include <cstdint>

#include "tracked.h"

#include "../tensor/resolver.h"  // sst::tensor::TileConfig — the tile-geometry currency

namespace sst {

// TileConfig — the tile geometry / format currency.
// Re-exported here so `sst::TileConfig` keeps resolving for the state model.
using tensor::TileConfig;

// ---------------------------------------------------------------------------
// Per-engine op modes. The dominant reconfigure in real kernels is a MODE
// switch (datacopy <-> untilize <-> matmul), so mode is a first-class field.
// ---------------------------------------------------------------------------
enum class UnpackMode : uint8_t {
    None = 0,
    DataCopy = 1,
    MatmulAB = 2,
    TilizeA = 3,
    EltwiseAB = 4,
    ReduceAB = 5,
    EltwiseColB = 6  // SrcA + col-broadcast SrcB (the COL-bcast unpack MOP)
};
enum class MathMode : uint8_t { None = 0, DataCopy = 1, Matmul = 2, Reduce = 3, Tilize = 4, Eltwise = 5 };
enum class PackMode : uint8_t { None = 0, Default = 1, Untilize = 2, Tilize = 3, Reduce = 4 };

// The binary eltwise sub-kind. It selects a different MATH addrmod/MOP, so it is
// tracked in the MATH state: switching add<->sub<->mul must reprogram MATH even
// when the mode is already Eltwise for the same tile geometry.
enum class EltwiseOp : uint8_t { None = 0, Add = 1, Sub = 2, Mul = 3, MulBcastCol = 4, SubBcastCol = 5 };

// The reduce pool sub-kind. MAX uses a GMPOOL, SUM/AVG a GAPOOL (they share the
// GAPOOL path; SUM vs AVG differ only in the host-provided scaler). It selects a
// different MATH pool instruction, so it is tracked in the MATH state.
enum class ReduceOp : uint8_t { None = 0, Max = 1, Sum = 2 };

// The SFPU unary op the SFPU config registers (LREG constants + macro sequences +
// addrmods) are currently programmed for. An SFPU op runs in place on DST and does
// NOT touch the unpack/FPU datapath, so it is tracked as its own MATH field: it
// only governs whether the (expensive) SFPU init must be re-emitted.
enum class SfpuOp : uint8_t { None = 0, Exp = 1, Recip = 2, Max = 3 };

// ---------------------------------------------------------------------------
// The three engine sub-states. Each field is Tracked: known or widened-away.
// ---------------------------------------------------------------------------
struct UnpackCfg {
    Tracked<UnpackMode> mode{};
    Tracked<TileConfig> tc{};
    Tracked<uint16_t> full_ct{};  // tilize geometry: the flat block-row stride, in tiles

    constexpr bool operator==(const UnpackCfg& o) const { return mode == o.mode && tc == o.tc && full_ct == o.full_ct; }
    constexpr bool operator!=(const UnpackCfg& o) const { return !(*this == o); }
    static constexpr UnpackCfg merge(const UnpackCfg& a, const UnpackCfg& b) {
        return UnpackCfg{sst::merge(a.mode, b.mode), sst::merge(a.tc, b.tc), sst::merge(a.full_ct, b.full_ct)};
    }
};

struct MathCfg {
    Tracked<MathMode> mode{};
    Tracked<TileConfig> tc{};  // format/geometry the datacopy MOP + ALU depend on
    // The DST-produce layout: true = the stride-16 remapped (row-major) layout a
    // DST-draining untilize packer consumes; false = the natural tiled layout the
    // default/tilize packer expects. It is a DEST_ACCESS_CFG toggle that must be
    // set BEFORE MATH writes DST, so it rides in the MATH state and any op that
    // writes DST (matmul, tilize fill, copy) declares the layout it needs. A
    // matmul->untilize pipeline flips it on; a tilize flips it back off.
    Tracked<bool> remap{};
    // Which binary eltwise op the MATH addrmod/MOP is programmed for. Only
    // meaningful when mode==Eltwise; the datacopy/matmul/tilize paths leave it
    // None. Tracked so add<->sub<->mul re-emits the MATH configure even when the
    // mode is unchanged.
    Tracked<EltwiseOp> ew{};
    // Which reduce pool the MATH addrmod/pool instruction is programmed for. Only
    // meaningful when mode==Reduce; other paths leave it None. Tracked so
    // max<->sum re-emits the MATH configure even when the mode is unchanged.
    Tracked<ReduceOp> rp{};
    // Which SFPU unary op the SFPU config registers hold. An SFPU op runs in place
    // on DST, so it is orthogonal to `mode` (the FPU datapath that produced DST
    // stays configured): it is tracked separately and only gates the SFPU init.
    // None means the SFPU has not been programmed for any of our unary ops.
    Tracked<SfpuOp> sfpu{};

    constexpr bool operator==(const MathCfg& o) const {
        return mode == o.mode && tc == o.tc && remap == o.remap && ew == o.ew && rp == o.rp && sfpu == o.sfpu;
    }
    constexpr bool operator!=(const MathCfg& o) const { return !(*this == o); }
    static constexpr MathCfg merge(const MathCfg& a, const MathCfg& b) {
        return MathCfg{
            sst::merge(a.mode, b.mode),
            sst::merge(a.tc, b.tc),
            sst::merge(a.remap, b.remap),
            sst::merge(a.ew, b.ew),
            sst::merge(a.rp, b.rp),
            sst::merge(a.sfpu, b.sfpu)};
    }
};

struct PackCfg {
    Tracked<PackMode> mode{};
    Tracked<TileConfig> tc{};
    Tracked<uint16_t> block_ct{};  // untilize geometry: tiles per inner block
    Tracked<uint16_t> full_ct{};   // untilize geometry: tiles per full row

    constexpr bool operator==(const PackCfg& o) const {
        return mode == o.mode && tc == o.tc && block_ct == o.block_ct && full_ct == o.full_ct;
    }
    constexpr bool operator!=(const PackCfg& o) const { return !(*this == o); }
    static constexpr PackCfg merge(const PackCfg& a, const PackCfg& b) {
        return PackCfg{
            sst::merge(a.mode, b.mode),
            sst::merge(a.tc, b.tc),
            sst::merge(a.block_ct, b.block_ct),
            sst::merge(a.full_ct, b.full_ct)};
    }
};

struct State {
    UnpackCfg u{};
    MathCfg m{};
    PackCfg p{};

    constexpr bool operator==(const State& o) const { return u == o.u && m == o.m && p == o.p; }
    constexpr bool operator!=(const State& o) const { return !(*this == o); }
    static constexpr State merge(const State& a, const State& b) {
        return State{UnpackCfg::merge(a.u, b.u), MathCfg::merge(a.m, b.m), PackCfg::merge(a.p, b.p)};
    }
};

// The one namespace-scope anchor: the all-unknown starting state.
inline constexpr State kInitial{};

// ---------------------------------------------------------------------------
// Tag<S>: lifts a `const State&` into a compile-time value we can thread and
// compare purely in the type system.
// ---------------------------------------------------------------------------
template <const State& S>
struct Tag {
    static constexpr const State& state = S;
};

constexpr Tag<kInitial> initial() { return {}; }

template <const State& A, const State& B>
constexpr bool operator==(Tag<A>, Tag<B>) {
    return A == B;
}
template <const State& A, const State& B>
constexpr bool operator!=(Tag<A>, Tag<B>) {
    return !(A == B);
}

// ---------------------------------------------------------------------------
// Phi (control-flow join). `PhiMerge<A, B>::value` is a `static constexpr
// State` data member — it HAS linkage, so `Tag<PhiMerge<A, B>::value>` is a
// legal NTTP. This is the ODR-safe way to name a merged state.
// ---------------------------------------------------------------------------
template <const State& A, const State& B>
struct PhiMerge {
    static constexpr State value = State::merge(A, B);
};

template <const State& A, const State& B>
constexpr auto phi(Tag<A>, Tag<B>) {
    return Tag<PhiMerge<A, B>::value>{};
}
template <const State& A, const State& B, typename... Rest>
constexpr auto phi(Tag<A>, Tag<B>, Rest... rest) {
    return phi(Tag<PhiMerge<A, B>::value>{}, rest...);
}

}  // namespace sst

#endif  // SST_STATE_H
