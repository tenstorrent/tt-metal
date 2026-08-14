// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The hardware-configuration state model.
//
// `State` is a compile-time description of independently reconfigurable Tensix
// hardware, structured after the LLK state-audit evidence (llk_state_map.csv,
// writer-set clustering): a field is the largest group of persistent resources
// that always change together under every LLK writer. Fields are scoped to a
// domain:
//   * thread-local — UNPACK (SrcA/SrcB descriptors + op program), MATH (FPU +
//     SFPU programs), PACK (descriptor + relu/l1-acc sub-fields + op program);
//   * core-global — ALU format config and DEST config, whose registers the
//     audit shows written from more than one TRISC (ALU_ACC_CTRL zero-flag from
//     T0+T1, fp32-dest-acc split across T1 ALU_ACC_CTRL_Fp32 / T2
//     PCK_DEST_RD_CTRL, dest_offset_id shadow shared T1+T2).
// Persistent hardware whose retention is per-call (counters, L1 base/dest
// addresses, semaphores, GPRs) is deliberately NOT in the State — it is
// governed by invariants at op boundaries, not tracked fields.
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

// KernelTraits is the hw-layer VALUE CARRIER for (fp32-dest-acc, dst-sync): the
// hw::* configure functions template on it. It is no longer the source of
// truth for the kernel — that is `State::global.dest` (see DestCfg below).
// ops.h derives a KernelTraits instantiation FROM the tracked state at each
// consuming op, and uses the build-define-derived instantiation only to
// ESTABLISH the initial DestCfg at hw_startup and in the tile_regs_* dest
// protocol helpers (protocol, not configuration).
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

// Which op family last programmed the packer output strides
// (PCK0_ADDR_CTRL_*_REG_0 + SCRATCH_SEC2). Audit evidence for tracking this
// separately from the descriptor: `_llk_pack_untilize_init_` writes
// PCK0_ADDR_CTRL_ZW_REG_0_Zstride + SCRATCH_SEC2 — stride registers that
// `_llk_pack_hw_configure_` (the descriptor writer) also owns. An untilize op
// therefore ALIASES descriptor-owned registers: after it runs, the strides no
// longer hold the default-pack values even though the descriptor formats are
// intact. Tracking the last stride programmer turns that silent clobber into a
// tracked write: a later Default-mode pack op requires `Default` and re-emits
// the strides; a repeated untilize proves `Untilize` still holds and elides.
enum class PackStrideProgram : uint8_t { None = 0, Default = 1, Untilize = 2 };

// ---------------------------------------------------------------------------
// Core-global fields. These registers live in the shared CFG space and the
// audit shows multiple TRISCs writing them, so they cannot be thread-local.
// The same compile-time State threads through all three TRISC builds, so the
// three replicas agree by construction; the ops that WRITE these fields are
// the single-writer points (rule: one owning op per transition).
// ---------------------------------------------------------------------------

// ALU format configuration (audit: ALU_FORMAT_SPEC_REG0/2, ALU_ACC_CTRL_INT8,
// Zero_Flag_disabled_src — the zero-flag bit is written from both T0 and T1).
struct AluFmtCfg {
    uint32_t srca_format = 0;
    uint32_t srcb_format = 0;
    bool int8_math = false;

    constexpr bool operator==(const AluFmtCfg& o) const {
        return srca_format == o.srca_format && srcb_format == o.srcb_format && int8_math == o.int8_math;
    }
    constexpr bool operator!=(const AluFmtCfg& o) const { return !(*this == o); }
};

// DEST configuration (audit: ALU_ACC_CTRL_Fp32/SFPU_Fp32 on T1 +
// PCK_DEST_RD_CTRL_Read_32b on T2 + DEST_ACCESS_CFG remap/swizzle + the
// SyncHalf/SyncFull bank scheme whose dest_offset_id shadow is shared T1+T2).
// fp32_acc lives HERE, not in an immutable KernelTraits: the dedicated
// `_llk_math_set_fp32_dest_acc_` / `_llk_pack_set_fp32_dest_acc_` entry points
// (used mid-kernel by transpose_dest / fast-tilize) prove it is runtime-mutable
// state that a kernel may toggle and the tracker must follow.
struct DestCfg {
    bool fp32_acc = false;
    DstSyncMode sync = DstSyncMode::SyncHalf;
    bool remap = false;  // stride-16 remapped DST layout (DEST_ACCESS_CFG remap+swizzle)

    constexpr bool operator==(const DestCfg& o) const {
        return fp32_acc == o.fp32_acc && sync == o.sync && remap == o.remap;
    }
    constexpr bool operator!=(const DestCfg& o) const { return !(*this == o); }
};

struct GlobalCfg {
    Tracked<AluFmtCfg> alu_fmt{};
    Tracked<DestCfg> dest{};

    constexpr bool operator==(const GlobalCfg& o) const { return alu_fmt == o.alu_fmt && dest == o.dest; }
    constexpr bool operator!=(const GlobalCfg& o) const { return !(*this == o); }
    static constexpr GlobalCfg merge(const GlobalCfg& a, const GlobalCfg& b) {
        return GlobalCfg{sst::merge(a.alu_fmt, b.alu_fmt), sst::merge(a.dest, b.dest)};
    }
};

// ---------------------------------------------------------------------------
// Thread-local domains. Descriptors live WITH their owning engine (the audit
// maps THCON_SEC0/SEC1 to the unpack thread and the PCK stride/format group to
// the pack thread); each engine additionally has one op-program field whose
// value is the PARAMETER TUPLE that generated it (mode + geometry + sub-op),
// never the identity of the init that ran — so identical re-inits from
// different call sites compare equal and unify.
// ---------------------------------------------------------------------------

struct UnpackState {
    // Operand descriptors. Independent per operand: the audit's
    // reconfig_data_format_srca/srcb writer sets are disjoint (SEC0 vs SEC1).
    Tracked<TileConfig> src_a_desc{};
    Tracked<TileConfig> src_b_desc{};
    // Op program (MOP + replay + haloize/x-end op cfg).
    Tracked<UnpackMode> mode{};
    Tracked<TileConfig> tile_config{};
    Tracked<uint16_t> tiles_per_row{};  // tilize geometry: the flat block-row stride, in tiles

    constexpr bool operator==(const UnpackState& o) const {
        return src_a_desc == o.src_a_desc && src_b_desc == o.src_b_desc && mode == o.mode &&
               tile_config == o.tile_config && tiles_per_row == o.tiles_per_row;
    }
    constexpr bool operator!=(const UnpackState& o) const { return !(*this == o); }
    static constexpr UnpackState merge(const UnpackState& a, const UnpackState& b) {
        return UnpackState{
            sst::merge(a.src_a_desc, b.src_a_desc),
            sst::merge(a.src_b_desc, b.src_b_desc),
            sst::merge(a.mode, b.mode),
            sst::merge(a.tile_config, b.tile_config),
            sst::merge(a.tiles_per_row, b.tiles_per_row)};
    }
};

struct MathState {
    // FPU program (MOP + ADDR_MODs + CLR_DVALID cfg): mode + sub-op + geometry.
    Tracked<MathMode> mode{};
    Tracked<TileConfig> tile_config{};  // format/geometry the datacopy MOP + ALU depend on
    // Which binary eltwise op the MATH addrmod/MOP is programmed for. Only
    // meaningful when mode==Eltwise; the datacopy/matmul/tilize paths leave it
    // None. Tracked so add<->sub<->mul re-emits the MATH configure even when the
    // mode is unchanged.
    Tracked<EltwiseOp> eltwise{};
    // Which SrcB broadcast the eltwise addressing is programmed for. Orthogonal
    // to `eltwise`: it feeds both the SrcB unpack MOP and the MATH addrmod.
    Tracked<BroadcastDim> broadcast{};
    // Which reduce pool the MATH addrmod/pool instruction is programmed for.
    Tracked<ReduceOp> reduce{};
    // SFPU program — its own field, orthogonal to the FPU program: an SFPU op
    // runs in place on DST and gates only the (expensive) SFPU init.
    Tracked<SfpuOp> sfpu{};

    constexpr bool operator==(const MathState& o) const {
        return mode == o.mode && tile_config == o.tile_config && eltwise == o.eltwise && broadcast == o.broadcast &&
               reduce == o.reduce && sfpu == o.sfpu;
    }
    constexpr bool operator!=(const MathState& o) const { return !(*this == o); }
    static constexpr MathState merge(const MathState& a, const MathState& b) {
        return MathState{
            sst::merge(a.mode, b.mode),
            sst::merge(a.tile_config, b.tile_config),
            sst::merge(a.eltwise, b.eltwise),
            sst::merge(a.broadcast, b.broadcast),
            sst::merge(a.reduce, b.reduce),
            sst::merge(a.sfpu, b.sfpu)};
    }
};

struct PackState {
    // Pack descriptor: formats/geometry (audit: THCON pack format group,
    // Exp_section/Row_start, programmed by pack hw-configure + reconfig).
    Tracked<TileConfig> desc{};
    // Descriptor SUB-FIELDS with their own dedicated reconfig entry points in
    // the audit (`_llk_pack_relu_config_`, `_llk_pack_reconfig_l1_acc_`): they
    // toggle independently and at high frequency in fused kernels, so folding
    // them into `desc` would force full descriptor re-emits on every toggle.
    Tracked<uint32_t> relu{};   // packed STACC_RELU config value (0 = disabled)
    Tracked<bool> l1_acc{};     // THCON Pack_L1_Acc
    // Who programmed the packer output strides last — see PackStrideProgram.
    Tracked<PackStrideProgram> stride_prog{};
    // Op program (MOP + replay + addr-mods + row-set mapping).
    Tracked<PackMode> mode{};
    Tracked<TileConfig> tile_config{};
    Tracked<uint16_t> tiles_per_block{};  // untilize geometry: tiles per inner block
    Tracked<uint16_t> tiles_per_row{};    // untilize geometry: tiles per full row

    constexpr bool operator==(const PackState& o) const {
        return desc == o.desc && relu == o.relu && l1_acc == o.l1_acc && stride_prog == o.stride_prog &&
               mode == o.mode && tile_config == o.tile_config && tiles_per_block == o.tiles_per_block &&
               tiles_per_row == o.tiles_per_row;
    }
    constexpr bool operator!=(const PackState& o) const { return !(*this == o); }
    static constexpr PackState merge(const PackState& a, const PackState& b) {
        return PackState{
            sst::merge(a.desc, b.desc),
            sst::merge(a.relu, b.relu),
            sst::merge(a.l1_acc, b.l1_acc),
            sst::merge(a.stride_prog, b.stride_prog),
            sst::merge(a.mode, b.mode),
            sst::merge(a.tile_config, b.tile_config),
            sst::merge(a.tiles_per_block, b.tiles_per_block),
            sst::merge(a.tiles_per_row, b.tiles_per_row)};
    }
};

struct State {
    UnpackState unpack{};
    MathState math{};
    PackState pack{};
    GlobalCfg global{};

    constexpr bool operator==(const State& o) const {
        return unpack == o.unpack && math == o.math && pack == o.pack && global == o.global;
    }
    constexpr bool operator!=(const State& o) const { return !(*this == o); }
    static constexpr State merge(const State& a, const State& b) {
        return State{
            UnpackState::merge(a.unpack, b.unpack),
            MathState::merge(a.math, b.math),
            PackState::merge(a.pack, b.pack),
            GlobalCfg::merge(a.global, b.global)};
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
