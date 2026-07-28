# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off for rms_norm PHASE 4 — `AddUnary(eps)` then `Rsqrt` on a REDUCE_ROW result.

WHAT IS ISOLATED
----------------
Phase 4 of rms_norm consumes one tile per tile-row out of `cb_rms_sum` (a REDUCE_ROW result: only
COLUMN 0 of each tile is meaningful, because the consumer is a `BroadcastDim::Col` FPU multiply that
replicates column 0 across the row) and publishes 1/rms into `cb_rms_recip`.

This bench reconstructs ONLY that: `ht` resident tiles in, `ht` tiles out, nothing else on the core.
Both CBs are backed directly on L1-sharded tensors (`cb_descriptor_from_sharded_tensor`), so there is
NO reader, NO writer and ZERO NoC traffic — the whole device kernel duration is phase-4 compute. The
loop structure mirrors the op: `n_groups` sequential groups of `ht` tiles (the op's row-block loop),
so the focus geometry is reproduced exactly as 4 groups x 8 tiles = 32 tiles per core on 64 cores.

THE THREE LEVERS UNDER TEST (independent compile-time axes)
----------------------------------------------------------
1. WINDOW  (`blocked`)  — the op's chain runs `EltwiseShape::tiles(ht)` whose `block_size` defaults
   to 1, and `eltwise_chain` additionally CLAMPS block_size to 1 for any chain with a per-tile
   (`InputLifecycle::Streaming`) CB reader (see the block_size doc in eltwise_chain.hpp: "Streaming
   CB-reader chains consume one tile per iter, so block_size is clamped to 1 for them", and
   `detail::policy_supports_block`). So the op pays a whole tile_regs_acquire/commit/wait/release
   round plus a pack phase PER TILE. `blocked` puts all `min(ht, DEST_AUTO_LIMIT)` tiles in ONE
   dst-sync window.
2. LANES   (`scope`)    — the SFPU walks a tile as 32 vector ops (4 faces x 4 row-groups x 2 column
   parities). A col-0 result only lives in the EVEN-parity vectors of faces 0 and 2, i.e. 8 of the
   32 vectors. `scope=c` keeps faces 0+2 (16 vectors, `VectorMode::C`); `scope=cskip` additionally
   strides the DEST address by 2 to drop the odd-parity vectors (8 vectors).
3. PASSES  (`fused`)    — `+eps` and `rsqrt` are two independent SFPU passes over the tile, each
   paying its own DEST-address setup + STALLWAIT and its own full vector walk. `fused` computes
   `rsqrt(x + eps)` in ONE pass by adding eps inside the sqrt body's input.

PRECISION CONTRACT (fixed, never a lever)
-----------------------------------------
Every variant runs under the SAME `ComputeConfigDescriptor`: `math_fidelity=HiFi2`,
`fp32_dest_acc_en=False`, `math_approx_mode=False`, `dst_full_sync_en=False` — the focus case's
user config. The fused body calls the IDENTICAL accurate rsqrt kernel the stock `rsqrt_tile` calls
(`_calculate_sqrt_body_<APPROX, RECIPROCAL=true, FAST_APPROX=false>`, i.e. the SQRT_23-bits
algorithm) with the same `!fp32_dest_acc_en` round-to-nearest store. It is the same function at the
same precision, differing only in that the intermediate `x + eps` no longer round-trips through the
bf16 DEST (which makes `fused` slightly MORE accurate than the baseline, never less).

The FORMAT axis (`in_fmt`/`out_fmt`) is a separate, explicitly-numeric option: the op's CBs are
Float32 (4 KB) tiles feeding a bf16 DEST, so their extra 2 KB per tile carries no information that
survives the DEST. Narrowing them is measured and its PCC reported.
"""

import ttnn

TILE = 32

CB_IN = 0  # the REDUCE_ROW result (op: cb_rms_sum)
CB_OUT = 16  # 1/rms                (op: cb_rms_recip)
CB_ONES = 1  # bcast probe only: an all-ones srcA block

ZONE_NAME = "PH4"

# ---- variant taxonomy -------------------------------------------------------
# impl: 0 = kernel_lib eltwise_chain (what the op does today), 1 = raw LLK loop.
# blocked: 0 = one dst-sync window per tile (the op's clamped block_size=1), 1 = one window per
#          min(ht, DEST_AUTO_LIMIT) tiles.
# scope: 0 = VectorMode::RC (all 32 vectors), 1 = VectorMode::C (16), 2 = C + even-parity stride (8).
# fused: 0 = two SFPU passes (add_unary then rsqrt), 1 = one pass computing rsqrt(x + eps).
_IMPL_CHAIN, _IMPL_RAW = 0, 1
_RC, _C, _CSKIP, _NONE = 0, 1, 2, 3

VARIANTS = {
    # name                  (impl,        blocked, scope,  fused)
    "baseline": (_IMPL_CHAIN, 0, _RC, 0),  # the op's current phase 4
    "chain_blocked": (_IMPL_CHAIN, 1, _RC, 0),  # lever 1 through the helper surface
    # GRADUATION-READY: the op's own `eltwise_chain` call (so the chain keeps owning the CB
    # lifecycle, the dtype reconfig and the dst-sync window — the parts the op actually needs from
    # the helper) with the two stock SFPU elements replaced by ONE custom `UnaryOp` element that
    # computes rsqrt(x + eps) over the column-0 vectors only.
    "chain_fused_cskip": (_IMPL_CHAIN, 0, _CSKIP, 1),
    "raw_rc": (_IMPL_RAW, 0, _RC, 0),  # bridge: raw == baseline?
    "raw_rc_blk": (_IMPL_RAW, 1, _RC, 0),  # lever 1 alone
    "raw_c": (_IMPL_RAW, 0, _C, 0),  # lever 2 (coarse) alone
    "raw_cskip": (_IMPL_RAW, 0, _CSKIP, 0),  # lever 2 (parity) alone
    "raw_fused_rc": (_IMPL_RAW, 0, _RC, 1),  # lever 3 alone
    "raw_fused_c": (_IMPL_RAW, 0, _C, 1),  # levers 2(coarse)+3
    "raw_cskip_blk": (_IMPL_RAW, 1, _CSKIP, 0),  # levers 1+2
    "raw_fused_cskip": (_IMPL_RAW, 0, _CSKIP, 1),  # levers 2+3
    "raw_fused_cskip_blk": (_IMPL_RAW, 1, _CSKIP, 1),  # levers 1+2+3
    # ---- ABLATIONS (wrong output by construction: identity copy, no SFPU). Not candidates; they
    # measure the per-tile copy+pack+dst-window FLOOR, i.e. the irreducible cost of phase 4 existing
    # as its own compute pass at all — which is exactly the upside still on the table for folding
    # `+eps`/`rsqrt` into the reduce's `post_reduce_op` slot so no separate pass is needed.
    "abl_copy_pack": (_IMPL_RAW, 0, _NONE, 0),
    "abl_copy_pack_blk": (_IMPL_RAW, 1, _NONE, 0),
}
BASELINE = "baseline"
ABLATIONS = ("abl_copy_pack", "abl_copy_pack_blk")

# Number of 32-lane SFPU vector ops each variant runs per tile (the lane-cost model).
VECTORS_PER_TILE = {
    "baseline": 64,
    "chain_blocked": 64,
    "chain_fused_cskip": 8,
    "raw_rc": 64,
    "raw_rc_blk": 64,
    "raw_c": 32,
    "raw_cskip": 16,
    "raw_fused_rc": 32,
    "raw_fused_c": 16,
    "raw_cskip_blk": 16,
    "raw_fused_cskip": 8,
    "raw_fused_cskip_blk": 8,
    "abl_copy_pack": 0,
    "abl_copy_pack_blk": 0,
}

LABEL = {
    "baseline": "eltwise_chain, block=1 (clamped), RC x2 passes",
    "chain_blocked": "eltwise_chain, Chunked policies + block=DEST_AUTO_LIMIT, RC x2",
    "chain_fused_cskip": "eltwise_chain + ONE custom UnaryOp element (fused, col-0 vectors)",
    "raw_rc": "raw loop, window=1 tile, RC x2 passes",
    "raw_rc_blk": "raw loop, window=DEST tiles, RC x2 passes",
    "raw_c": "raw loop, window=1 tile, VectorMode::C x2 passes",
    "raw_cskip": "raw loop, window=1 tile, C + even-parity stride x2 passes",
    "raw_fused_rc": "raw loop, window=1 tile, RC, ONE fused pass rsqrt(x+eps)",
    "raw_fused_c": "raw loop, window=1 tile, VectorMode::C, ONE fused pass",
    "raw_cskip_blk": "raw loop, window=DEST tiles, C + parity stride x2 passes",
    "raw_fused_cskip": "raw loop, window=1 tile, C + parity stride, ONE fused pass",
    "raw_fused_cskip_blk": "raw loop, window=DEST tiles, C + parity stride, ONE fused pass",
    "abl_copy_pack": "ABLATION: copy+pack only, NO SFPU (per-tile floor)",
    "abl_copy_pack_blk": "ABLATION: copy+pack only, NO SFPU, window=DEST tiles",
}

# Which output columns each variant leaves correct (the op only ever reads column 0).
VALID_COLS = {
    "baseline": 32,
    "chain_blocked": 32,
    "chain_fused_cskip": 1,
    "raw_rc": 32,
    "raw_rc_blk": 32,
    "raw_c": 16,
    "raw_cskip": 1,
    "raw_fused_rc": 32,
    "raw_fused_c": 16,
    "raw_cskip_blk": 1,
    "raw_fused_cskip": 1,
    "raw_fused_cskip_blk": 1,
    "abl_copy_pack": 0,  # ablation: identity copy, no valid output at all
    "abl_copy_pack_blk": 0,
}

# NOTE: resolved lazily. ttnn/ttnn/operations/__init__.py exec_module()s every .py under
# ttnn/ttnn/operations/ during `import ttnn`, so module-level `ttnn.<attr>` access here would run
# against a half-initialized ttnn.
_DTYPE_NAMES = {"fp32": "float32", "bf16": "bfloat16"}


# =============================================================================
# Compute kernel — phase 4 only.
#
# CT args: [impl, blocked, scope, fused]
# RT args: [ht, n_groups, eps_bits]
# =============================================================================
_KERNEL = r"""
// Isolated rms_norm phase 4: AddUnary(eps) -> Rsqrt over `ht` REDUCE_ROW-result tiles.
//
// RAW-LLK JUSTIFICATION (kernel-head note required by the bake-off rules):
//   The `raw_*` variants bypass `compute_kernel_lib::eltwise_chain` + `eltwise_scalar.hpp`'s
//   `AddUnary` + `eltwise_math.hpp`'s `Rsqrt`. Two mechanisms are NOT reachable through those
//   helpers today:
//     (a) SFPU WORK SCOPE. `rsqrt_tile` hardcodes `VectorMode::RC` and `ITERATIONS = 8`, and
//         `add_unary_tile` likewise; neither the compute-API wrapper nor the chain element exposes
//         a vector-mode / iteration / address-stride knob. A REDUCE_ROW result is column-0-only, so
//         28 of the 32 vector ops per tile per pass are computed and never read.
//     (b) PASS FUSION. `AddUnary` and `Rsqrt` are separate chain elements, hence separate SFPU
//         passes with separate DEST-address setup + STALLWAIT and separate full vector walks. The
//         chain has no "unary op with a pre-added scalar" element.
//   Everything else (CB lifecycle, dst-sync window, pack) is reproduced faithfully so the measured
//   delta is attributable to (a)/(b)/the window size alone; `raw_rc` is the bridge variant that
//   proves the raw reproduction costs the same as the helper chain at RC x 2 passes.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_binop_with_unary.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

namespace ckl = compute_kernel_lib;
using ckernel::VectorMode;

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_out = 16;

constexpr uint32_t IMPL_CHAIN = 0;
constexpr uint32_t SCOPE_RC = 0, SCOPE_C = 1, SCOPE_CSKIP = 2, SCOPE_NONE = 3;

// ---------------------------------------------------------------------------
// Scoped stock SFPU calls — same functors the compute API uses, with the vector mode threaded
// through instead of hardcoded to RC.
// ---------------------------------------------------------------------------
ALWI void add_unary_vm(uint32_t idst, uint32_t param, VectorMode vm) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
        (APPROX, ckernel::ADD_UNARY, 8 /* ITERATIONS */), idst, vm, param));
}
ALWI void rsqrt_vm(uint32_t idst, VectorMode vm) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt,
        (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE, false /*FAST_APPROX*/, false /*legacy_compat*/), idst, vm));
}

// ---------------------------------------------------------------------------
// Even-parity ("cskip") bodies. The SFPU walks a face as
// [rg0-even, rg0-odd, rg1-even, rg1-odd, ...]; column 0 lives only in the EVEN-parity vectors, so we
// visit offsets 0,2,4,6 and skip the odd ones. Net dst_reg advance is +8 == the stock ITERATIONS=8,
// so `VectorMode::C`'s face-0 -> face-2 stepping composes unchanged: column 0 for all 32 rows in 4
// vector ops per face instead of 8.
//
// The fused body is bit-for-bit the stock accurate rsqrt kernel
// (`_calculate_sqrt_body_<APPROX, RECIPROCAL=true, FAST_APPROX=false>` + the `!fp32_dest_acc_en`
// round-to-nearest store, exactly what `calculate_rsqrt<..., legacy_compat=false>` runs) with `+eps`
// folded into its argument, so `+eps` costs no pass of its own.
// ---------------------------------------------------------------------------
#ifdef TRISC_MATH
template <int NVEC, int STRIDE>
sfpi_inline void rsqrt_body_strided() {
    for (int d = 0; d < NVEC; d++) {
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(
            sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) { t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest); }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}
template <int NVEC, int STRIDE>
sfpi_inline void add_body_strided(uint32_t param) {
    const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(param);
    for (int d = 0; d < NVEC; d++) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] + eps;
        sfpi::dst_reg += STRIDE;
    }
}
template <int NVEC, int STRIDE>
sfpi_inline void fused_body_strided(uint32_t param) {
    const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(param);
    for (int d = 0; d < NVEC; d++) {
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(
            sfpi::dst_reg[0] + eps);
        if constexpr (!DST_ACCUM_MODE) { t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest); }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}
#endif

// One tile's worth of phase-4 SFPU work at DEST slot `idst`.
template <uint32_t scope, uint32_t fused>
ALWI void phase4_sfpu(uint32_t idst, uint32_t eps_bits) {
    if constexpr (scope == SCOPE_NONE) {
        // ABLATION: keep the copy/pack/dst-window scaffolding, drop the payload.
        (void)idst;
        (void)eps_bits;
    } else if constexpr (fused) {
        if constexpr (scope == SCOPE_CSKIP) {
            // 4 even-parity vectors per face, faces 0 and 2 => 8 vector ops, ONE pass.
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                [eps_bits]() { fused_body_strided<4, 2>(eps_bits); }, idst, VectorMode::C)));
        } else if constexpr (scope == SCOPE_C) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                [eps_bits]() { fused_body_strided<8, 1>(eps_bits); }, idst, VectorMode::C)));
        } else {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                [eps_bits]() { fused_body_strided<8, 1>(eps_bits); }, idst, VectorMode::RC)));
        }
    } else if constexpr (scope == SCOPE_CSKIP) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            [eps_bits]() { add_body_strided<4, 2>(eps_bits); }, idst, VectorMode::C)));
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            []() { rsqrt_body_strided<4, 2>(); }, idst, VectorMode::C)));
    } else {
        constexpr VectorMode vm = (scope == SCOPE_C) ? VectorMode::C : VectorMode::RC;
        add_unary_vm(idst, eps_bits, vm);
        rsqrt_vm(idst, vm);
    }
}

// ---------------------------------------------------------------------------
// GRADUATION-READY chain element. Derives from the same `compute_kernel_lib::UnaryOp` CRTP base every
// op in eltwise_math.hpp / eltwise_scalar.hpp derives from, so `eltwise_chain` keeps owning the CB
// lifecycle, the dtype reconfig and the dst-sync window — only the SFPU body changes. Runtime eps is
// an instance field and `exec` is overridden to consume it, exactly like `AddUnary` / `Power`.
//
// Its natural library home is eltwise_math.hpp (an `SfpuScope`-parameterised `Rsqrt`, plus a fused
// `RsqrtAddUnary`); it is written locally here only so the bake-off can measure it without touching
// kernel_lib.
namespace compute_kernel_lib {
template <Dst Slot = Dst::D0>
struct RsqrtAddUnaryColZero : UnaryOp<RsqrtAddUnaryColZero<Slot>, Slot> {
    uint32_t eps_bits;
    constexpr explicit RsqrtAddUnaryColZero(uint32_t e) noexcept : eps_bits(e) {}
    static ALWI void init() { rsqrt_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        const uint32_t idst = to_u32(Slot) + slot_offset;
        const uint32_t eps = eps_bits;
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            [eps]() { fused_body_strided<4, 2>(eps); }, idst, VectorMode::C)));
    }
};
}  // namespace compute_kernel_lib

template <uint32_t scope, uint32_t fused>
ALWI void phase4_sfpu_init() {
    if constexpr (scope == SCOPE_NONE) {
        return;
    }
    if constexpr (!fused) {
        binop_with_scalar_tile_init();
    }
    rsqrt_tile_init();  // programs the sqrt body's vConst*Prgm constants; shared by every rsqrt path
}

void kernel_main() {
    constexpr uint32_t IMPL = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCKED = get_compile_time_arg_val(1);
    constexpr uint32_t SCOPE = get_compile_time_arg_val(2);
    constexpr uint32_t FUSED = get_compile_time_arg_val(3);

    static_assert(
        IMPL != IMPL_CHAIN || (SCOPE == SCOPE_RC && FUSED == 0) || (SCOPE == SCOPE_CSKIP && FUSED == 1),
        "chain variants: stock RC x2, or the custom fused col-0 element");

    const uint32_t ht = get_arg_val<uint32_t>(0);
    const uint32_t n_groups = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_in, cb_in, cb_out);

    // The CBs are backed on the resident L1 shards; publish the input once so the chain's
    // cb_wait_front sees it (mirrors the zero-copy sharded-input contract).
    cb_reserve_back(cb_in, ht * n_groups);
    cb_push_back(cb_in, ht * n_groups);

    constexpr uint32_t DEST_BLOCK = ckl::DEST_AUTO_LIMIT;

    for (uint32_t g = 0; g < n_groups; ++g) {
        DeviceZoneScopedN("PH4");
        if constexpr (IMPL == IMPL_CHAIN && SCOPE == SCOPE_CSKIP) {
            // ---- graduation shape: the op's chain, one custom fused col-0 SFPU element ----
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(ht),
                ckl::CopyTile<cb_in, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
                ckl::RsqrtAddUnaryColZero<ckl::Dst::D0>{eps_bits},
                ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming>{});
        } else if constexpr (IMPL == IMPL_CHAIN) {
            if constexpr (BLOCKED == 0) {
                // ---- EXACTLY the op's current phase 4 (rms_norm_compute.cpp, zone cmp_rsqrt) ----
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(ht),
                    ckl::CopyTile<cb_in, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
                    ckl::AddUnary<ckl::Dst::D0>{eps_bits},
                    ckl::Rsqrt<>{},
                    ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming>{});
            } else {
                // Chunked in/out are the policies that actually permit block_size > 1
                // (`Streaming` is clamped to 1, and its per-tile reserve/push is what gave PCC 0.0
                // when Refinement 5 tried block_size > 1 with Streaming).
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(ht, DEST_BLOCK),
                    ckl::CopyTile<
                        cb_in, ckl::Dst::D0, ckl::InputLifecycle::Chunked, ckl::CopyTileReconfig::Input,
                        ckl::OperandKind::Block>{},
                    ckl::AddUnary<ckl::Dst::D0>{eps_bits},
                    ckl::Rsqrt<>{},
                    ckl::PackTile<cb_out, ckl::OutputLifecycle::Chunked>{});
            }
        } else {
            // ---- raw reproduction of the same CB lifecycle + dst-sync window ----
            const uint32_t blk = (BLOCKED == 0) ? 1u : (ht < DEST_BLOCK ? ht : DEST_BLOCK);
            phase4_sfpu_init<SCOPE, FUSED>();
            for (uint32_t base = 0; base < ht; base += blk) {
                const uint32_t n = (ht - base < blk) ? (ht - base) : blk;
                cb_wait_front(cb_in, n);
                cb_reserve_back(cb_out, n);
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_in);
                for (uint32_t j = 0; j < n; ++j) {
                    copy_tile(cb_in, j, j);
                }
                for (uint32_t j = 0; j < n; ++j) {
                    phase4_sfpu<SCOPE, FUSED>(j, eps_bits);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < n; ++j) {
                    pack_tile(j, cb_out, j);
                }
                tile_regs_release();
                cb_push_back(cb_out, n);
                cb_pop_front(cb_in, n);
            }
        }
    }
}
"""


# =============================================================================
# BCAST PROBE — the load-bearing safety check for the lane-scoping levers.
#
# `raw_cskip*` leaves columns 1..31 of the output holding whatever was copied in. The real op's
# consumer is `BinaryFpu<cb_input_tiles, cb_rms_recip, Mul, BroadcastDim::Col>`; this probe runs that
# exact primitive (`mul_tiles_bcast<BroadcastType::COL>`, srcB = the 1/rms tile) against a POISONED
# tile whose column 0 is valid and whose columns 1..31 hold garbage, and packs the product. If the
# product is correct everywhere, the broadcast provably reads column 0 ONLY, and scoping phase 4 to
# column 0 cannot change the op's output.
# =============================================================================
_PROBE_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_ones = 1;   // srcA: an all-ones full block (stands in for x)
    constexpr uint32_t cb_recip = 0;  // srcB: the col-0-valid / poisoned 1/rms tile
    constexpr uint32_t cb_out = 16;
    const uint32_t n = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_ones, cb_recip, cb_out);
    cb_reserve_back(cb_ones, n);  cb_push_back(cb_ones, n);
    cb_reserve_back(cb_recip, n); cb_push_back(cb_recip, n);
    cb_wait_front(cb_ones, n);
    cb_wait_front(cb_recip, n);
    cb_reserve_back(cb_out, n);
    init_bcast<ckernel::EltwiseBinaryType::ELWMUL, ckernel::BroadcastType::COL>(cb_ones, cb_recip, cb_out);
    for (uint32_t i = 0; i < n; ++i) {
        tile_regs_acquire();
        mul_tiles_bcast<ckernel::BroadcastType::COL>(cb_ones, cb_recip, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, i);
        tile_regs_release();
    }
    cb_push_back(cb_out, n);
}
"""


# =============================================================================
# Host side
# =============================================================================
def core_range_set(grid_x, grid_y):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])


def sharded_memory_config(tiles_per_core, grid_x, grid_y):
    """`tiles_per_core` 32x32 tiles stacked vertically on each core of a grid_x by grid_y grid."""
    return ttnn.create_sharded_memory_config(
        shape=(tiles_per_core * TILE, TILE),
        core_grid=core_range_set(grid_x, grid_y),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def tensor_height(tiles_per_core, grid_x, grid_y):
    return tiles_per_core * TILE * grid_x * grid_y


def compute_config(fp32_dest_acc_en=False):
    """The focus case's user precision contract. IDENTICAL for every variant — never a lever.

    `fp32_dest_acc_en` is exposed ONLY so the bench can prove the lane-scoped element is also
    CORRECT under the op's other supported DEST mode (the op's SUPPORTED matrix allows both). The
    focus case pins it False and every perf number is taken there; it is never varied to gain speed.
    """
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=False,
        dst_full_sync_en=False,
    )


def _uniform_runtime_args(grid_x, grid_y, args):
    rt = ttnn.RuntimeArgs()
    for x in range(grid_x):
        for y in range(grid_y):
            rt[x][y] = list(args)
    return rt


def create_program_descriptor(
    input_tensor, output_tensor, *, variant, ht, n_groups, eps_bits, grid_x, grid_y, fp32_dest_acc_en=False
):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    impl, blocked, scope, fused = VARIANTS[variant]
    cores = core_range_set(grid_x, grid_y)
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[impl, blocked, scope, fused],
        runtime_args=_uniform_runtime_args(grid_x, grid_y, [ht, n_groups, eps_bits]),
        config=compute_config(fp32_dest_acc_en),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run_op(input_tensor, output_tensor, *, variant, ht, n_groups, eps_bits, grid_x, grid_y, fp32_dest_acc_en=False):
    descriptor = create_program_descriptor(
        input_tensor,
        output_tensor,
        variant=variant,
        ht=ht,
        n_groups=n_groups,
        eps_bits=eps_bits,
        grid_x=grid_x,
        grid_y=grid_y,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    return ttnn.generic_op([input_tensor, output_tensor], descriptor)


def run_bcast_probe(recip_tensor, ones_tensor, output_tensor, *, tiles_per_core, grid_x, grid_y):
    cores = core_range_set(grid_x, grid_y)
    compute = ttnn.KernelDescriptor(
        kernel_source=_PROBE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[],
        runtime_args=_uniform_runtime_args(grid_x, grid_y, [tiles_per_core]),
        config=compute_config(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, recip_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_ONES, ones_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
    ]
    descriptor = ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)
    return ttnn.generic_op([recip_tensor, ones_tensor, output_tensor], descriptor)


def dtype_of(fmt):
    return getattr(ttnn, _DTYPE_NAMES[fmt])
