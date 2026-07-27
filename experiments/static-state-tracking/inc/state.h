// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The hardware-configuration state model.
//
// `State` is a compile-time description of independently reconfigurable Tensix
// hardware domains. It intentionally separates:
//   * immutable kernel traits (kept outside the mergeable State),
//   * SrcA / SrcB / PACK operand descriptors,
//   * UNPACK / MATH / PACK operation programs, and
//   * DEST layout.
//
// C++17 note (the whole reason this file looks the way it does):
//   * C++17 has NO class-type non-type template parameters. You cannot write
//     `template <State S>`.
//   * You CAN write `template <const State& S>` — a reference to an object with
//     static storage duration AND linkage is a legal C++17 NTTP.
//   * Therefore every `State` used as a template argument must be anchored in a
//     namespace-scope `inline constexpr` (see `kInitial`) or a `static
//     constexpr` data member of a template (see `PhiMerge` / the *Next structs
//     in ops.h). We NEVER use a function- local `static constexpr State` as an
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
enum class UnpackMode : uint8_t { None = 0, DataCopy = 1, Matmul = 2, Tilize = 3, Eltwise = 4, Reduce = 5 };
enum class MathMode : uint8_t { None = 0, DataCopy = 1, Matmul = 2, Reduce = 3, Tilize = 4, Eltwise = 5 };
enum class PackMode : uint8_t { None = 0, Default = 1, Untilize = 2, Tilize = 3, Reduce = 4 };
enum class DstSyncMode : uint8_t { SyncHalf = 0, SyncFull = 1 };

// Immutable JIT/compile-time configuration. Traits are available to operation
// recipes but are not threaded through control flow or widened at joins.
template <bool Fp32DestAcc, DstSyncMode DstSync>
struct KernelTraits {
    static constexpr bool fp32_dest_acc = Fp32DestAcc;
    static constexpr DstSyncMode dst_sync = DstSync;
};

// The binary eltwise sub-kind. It selects a different MATH addrmod/MOP, so it is
// tracked in the MATH state: switching add<->sub<->mul must reprogram MATH even
// when the mode is already Eltwise for the same tile geometry.
enum class EltwiseOp : uint8_t { None = 0, Add = 1, Sub = 2, Mul = 3 };

// The SrcB broadcast dimension the eltwise datapath is programmed for.
enum class BroadcastDim : uint8_t { None = 0, Col = 1, Row = 2, Scalar = 3 };

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
// The physical source/packer descriptors are independent. Matmul in particular
// may have asymmetric SrcA and SrcB formats/geometries, while pack reconfiguration
// changes neither source.
struct OperandCfg {
    Tracked<TileConfig> src_a{};
    Tracked<TileConfig> src_b{};
    Tracked<TileConfig> pack{};

    constexpr bool operator==(const OperandCfg& o) const {
        return src_a == o.src_a && src_b == o.src_b && pack == o.pack;
    }
    constexpr bool operator!=(const OperandCfg& o) const { return !(*this == o); }
    static constexpr OperandCfg merge(const OperandCfg& a, const OperandCfg& b) {
        return OperandCfg{sst::merge(a.src_a, b.src_a), sst::merge(a.src_b, b.src_b), sst::merge(a.pack, b.pack)};
    }
};

// Operation-program domains. These describe persistent MOP / ADDR_MOD and
// operation-specific geometry, not operand descriptor state.
struct UnpackCfg {
    Tracked<UnpackMode> mode{};
    Tracked<TileConfig> tile_config{};
    Tracked<uint16_t> tiles_per_row{};  // tilize geometry: the flat block-row stride, in tiles

    constexpr bool operator==(const UnpackCfg& o) const {
        return mode == o.mode && tile_config == o.tile_config && tiles_per_row == o.tiles_per_row;
    }
    constexpr bool operator!=(const UnpackCfg& o) const { return !(*this == o); }
    static constexpr UnpackCfg merge(const UnpackCfg& a, const UnpackCfg& b) {
        return UnpackCfg{
            sst::merge(a.mode, b.mode),
            sst::merge(a.tile_config, b.tile_config),
            sst::merge(a.tiles_per_row, b.tiles_per_row)};
    }
};

struct MathCfg {
    Tracked<MathMode> mode{};
    Tracked<TileConfig> tile_config{};  // format/geometry the datacopy MOP + ALU depend on
    // Which binary eltwise op the MATH addrmod/MOP is programmed for. Only
    // meaningful when mode==Eltwise; the datacopy/matmul/tilize paths leave it
    // None. Tracked so add<->sub<->mul re-emits the MATH configure even when the
    // mode is unchanged.
    Tracked<EltwiseOp> eltwise{};
    // Which SrcB broadcast the eltwise addressing is programmed for. Orthogonal
    // to `eltwise`: it feeds both the SrcB unpack MOP and the MATH addrmod. Only
    // meaningful when mode==Eltwise; other paths leave it None. Tracked so
    // toggling the broadcast re-emits only the addressing configure.
    Tracked<BroadcastDim> broadcast{};
    // Which reduce pool the MATH addrmod/pool instruction is programmed for. Only
    // meaningful when mode==Reduce; other paths leave it None. Tracked so
    // max<->sum re-emits the MATH configure even when the mode is unchanged.
    Tracked<ReduceOp> reduce{};
    // Which SFPU unary op the SFPU config registers hold. An SFPU op runs in place
    // on DST, so it is orthogonal to `mode` (the FPU datapath that produced DST
    // stays configured): it is tracked separately and only gates the SFPU init.
    // None means the SFPU has not been programmed for any of our unary ops.
    Tracked<SfpuOp> sfpu{};

    constexpr bool operator==(const MathCfg& o) const {
        return mode == o.mode && tile_config == o.tile_config && eltwise == o.eltwise && broadcast == o.broadcast &&
               reduce == o.reduce && sfpu == o.sfpu;
    }
    constexpr bool operator!=(const MathCfg& o) const { return !(*this == o); }
    static constexpr MathCfg merge(const MathCfg& a, const MathCfg& b) {
        return MathCfg{
            sst::merge(a.mode, b.mode),
            sst::merge(a.tile_config, b.tile_config),
            sst::merge(a.eltwise, b.eltwise),
            sst::merge(a.broadcast, b.broadcast),
            sst::merge(a.reduce, b.reduce),
            sst::merge(a.sfpu, b.sfpu)};
    }
};

struct PackCfg {
    Tracked<PackMode> mode{};
    Tracked<TileConfig> tile_config{};
    Tracked<uint16_t> tiles_per_block{};  // untilize geometry: tiles per inner block
    Tracked<uint16_t> tiles_per_row{};    // untilize geometry: tiles per full row

    constexpr bool operator==(const PackCfg& o) const {
        return mode == o.mode && tile_config == o.tile_config && tiles_per_block == o.tiles_per_block &&
               tiles_per_row == o.tiles_per_row;
    }
    constexpr bool operator!=(const PackCfg& o) const { return !(*this == o); }
    static constexpr PackCfg merge(const PackCfg& a, const PackCfg& b) {
        return PackCfg{
            sst::merge(a.mode, b.mode),
            sst::merge(a.tile_config, b.tile_config),
            sst::merge(a.tiles_per_block, b.tiles_per_block),
            sst::merge(a.tiles_per_row, b.tiles_per_row)};
    }
};

struct OperationCfg {
    UnpackCfg unpack{};
    MathCfg math{};
    PackCfg pack{};

    constexpr bool operator==(const OperationCfg& o) const {
        return unpack == o.unpack && math == o.math && pack == o.pack;
    }
    constexpr bool operator!=(const OperationCfg& o) const { return !(*this == o); }
    static constexpr OperationCfg merge(const OperationCfg& a, const OperationCfg& b) {
        return OperationCfg{
            UnpackCfg::merge(a.unpack, b.unpack), MathCfg::merge(a.math, b.math), PackCfg::merge(a.pack, b.pack)};
    }
};

struct DestLayoutCfg {
    Tracked<bool> remap{};

    constexpr bool operator==(const DestLayoutCfg& o) const { return remap == o.remap; }
    constexpr bool operator!=(const DestLayoutCfg& o) const { return !(*this == o); }
    static constexpr DestLayoutCfg merge(const DestLayoutCfg& a, const DestLayoutCfg& b) {
        return DestLayoutCfg{sst::merge(a.remap, b.remap)};
    }
};

struct State {
    OperandCfg operands{};
    OperationCfg operations{};
    DestLayoutCfg layout{};

    constexpr bool operator==(const State& o) const {
        return operands == o.operands && operations == o.operations && layout == o.layout;
    }
    constexpr bool operator!=(const State& o) const { return !(*this == o); }
    static constexpr State merge(const State& a, const State& b) {
        return State{
            OperandCfg::merge(a.operands, b.operands),
            OperationCfg::merge(a.operations, b.operations),
            DestLayoutCfg::merge(a.layout, b.layout)};
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
