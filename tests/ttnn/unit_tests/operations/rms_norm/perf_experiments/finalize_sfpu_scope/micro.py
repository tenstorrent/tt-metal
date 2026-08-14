# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off (part P4): SCOPE + FUSE rms_norm's FINALIZE onto the axis that carries data.

WHAT IS ISOLATED
    rms_norm finalizes its statistic inside the reduce helper's `post_reduce_op`
    extension point:

        binop_with_scalar_tile_init();
        mul_unary_tile(dst, 1/W);      // mean = Sum(x^2) * (1/W)
        add_unary_tile(dst, eps);      // + epsilon
        rsqrt_tile_init();
        rsqrt_tile(dst);               // 1/rms

    Each of the three SFPU calls hardcodes `VectorMode::RC` + `ITERATIONS = 8`, i.e. a
    walk over the WHOLE 32x32 DEST tile = 32 32-lane vector ops each, 96 in total.
    But the tile is a REDUCE_ROW result: the only meaningful data is COLUMN 0 (one
    value per row).  Its single consumer is a `BroadcastDim::Col` operand that reads
    column 0 and replicates it across the row, so ~31/32 of every vector op computes
    lanes nobody reads.

    This bench isolates ONLY that SFPU cost.  One core, one bf16 tile resident in L1.
    The tile is copied into DEST ONCE and packed out ONCE, both OUTSIDE a
    DeviceZoneScopedN; inside the zone a MATH-thread-only loop applies the finalize
    `reps` times.  The test reads TRISC_1 (math), so the number is pure SFPU cycles:
    no unpack, no pack, no CB handshake, no reduce, no NoC.

TWO INDEPENDENT LEVERS, measured separately and together
  * SCOPE — how many vector ops the walk visits.  A col-0 result needs every
    row-group but only the EVEN column parity (column 0 is even, and parity is the
    INNER walk axis), so `ITERATIONS` cannot isolate it: it takes a hand-addressed
    sfpi body that strides DEST by 2 under `VectorMode::C`.  32 -> 16 -> 8 vectors.
  * FUSION — how many walks there are.  mul, add and rsqrt are three SEPARATE walks
    over the same tile today.  One sfpi body that does `rsqrt(v*inv_w + eps)` per
    vector collapses three walks into one.  This is the bigger lever and it is
    invisible from the helper API.

    Together: 96 vector ops -> 8.  Plus one SFPU init per finalize instead of two
    (`rsqrt_tile_init()` is self-contained: `llk_math_eltwise_unary_sfpu_init<rsqrt>`
    already runs the config-reg + ADDR_MOD invariant that `binop_with_scalar_tile_init()`
    exists to run).

PRECISION IS A CONTRACT, NOT A LEVER
    Every variant runs under the SAME user config: math_fidelity=HiFi2,
    fp32_dest_acc_en=False, math_approx_mode=False, bf16 in.  `APPROX` is never
    flipped and no approximate rsqrt is used anywhere: `fused*` calls the identical
    non-approximate `_calculate_sqrt_body_<APPROX, RECIPROCAL=true>` the stock
    `rsqrt_tile` calls.  The ONE numerical difference the fusion makes is that the
    two intermediates (mean, mean+eps) stay in the SFPU's fp32 lane registers instead
    of round-tripping through the 16-bit DEST — i.e. MORE precision, not less.
    `fused_cskip_bf16` re-inserts that DEST truncation explicitly so the two can be
    told apart; both are measured and both report their error.

VARIANTS (`stock` is the shipped op's approach — the honest baseline)
    none                empty reps-loop, no SFPU.  Math-loop overhead floor (~0).
    stock               BASELINE.  2 inits + mul/add/rsqrt, each VectorMode::RC.  [96]
    stock_noinit        the same 3 calls with the 2 inits hoisted out of the loop. [96]
    vmode_c             the same 3 calls threaded to VectorMode::C.                [48]
    cskip_each          3 SEPARATE even-parity sfpi bodies (scope only, no fusion). [24]
    fused_rc            ONE fused body, VectorMode::RC (fusion only, no scope).     [32]
    fused_c             ONE fused body, VectorMode::C.                              [16]
    fused_cskip         ONE fused body + even-parity stride.  CANDIDATE.            [8]
    fused_cskip_noinit  the candidate with its single init hoisted.                 [8]
    fused_cskip_bf16    the candidate with the baseline's DEST truncation restored. [8]

Correctness is checked at reps=1 on each variant's valid region against an fp64
torch golden; the rest of the tile keeps whatever the scope did not touch (and is
never read by the op — proven end-to-end in test_whole_op.py).  Perf is measured,
never asserted.

MEASURED — Blackhole p150b @1350 MHz, one core, ONE fresh launch per variant,
reps=2000, W=1024, eps=1e-6, HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False.
MATH (TRISC_1) ns for ONE finalize; copy+pack outside the zone (isolation check:
max unpack 0.011 ns, max pack 0.007 ns inside the zone).

    variant              inits  vec  math ns   vs stock  ns/vec   col-0 PCC
    none                     0    0      0.0        —        —    1.0000000
    stock  (THE OP TODAY)    2   96    989.7    1.00x     10.3    0.9999709
    stock_noinit             0   96    976.3    1.01x     10.2    0.9999709
    vmode_c                  2   48    514.8    1.92x     10.7    0.9999709
    cskip_each               2   24    283.7    3.49x     11.8    0.9999709
    fused_rc                 1   32    897.9    1.10x     28.1    0.9999709
    fused_c                  1   16    459.4    2.15x     28.7    0.9999709
    fused_cskip              1    8    240.1    4.12x     30.0    0.9999709   <- CANDIDATE
    fused_cskip_noinit       0    8    230.5    4.29x     28.8    0.9999709
    fused_cskip_bf16         1    8    263.9    3.75x     33.0    0.9999709
    fused_cskip_nonneg       1    8    222.4    4.45x     27.8    0.9999709
    cskip_rsqrt_only         1    8    204.5    4.84x     25.6    0.9999709   (floor)

Every variant is BIT-IDENTICAL on column 0 (same max rel err 0.00313, same PCC to
7 digits): at W=1024 the 1/W multiply is exact and eps is below bf16 resolution,
so neither the scope nor the fp32 intermediate moves a value.  `fused_cskip_bf16`
(the truncation restored) confirms it — same numbers, 24 ns more.

Reading of the result:
  * SCOPE is the lever, not fusion.  96 -> 24 vectors alone is 3.49x; fusion
    alone (fused_rc, 96 -> 32) is only 1.10x, because the fused body cannot be
    unrolled (SFPU register pressure) and so costs ~2.8x more per vector.  On top
    of the scope, fusion is still worth 283.7 -> 240.1 (1.18x): it removes one
    whole address walk.
  * `cskip_rsqrt_only` (204.5 ns) is the FLOOR — the rsqrt any col-0 finalize must
    pay.  The candidate is 18% above it; the stock chain is 4.8x above it.
  * The two SFPU inits cost only 13 ns of the 989.7; the init reduction is real
    but it is not where the win lives.
"""

import ttnn

TILE = 32

CB_IN = 0  # one bf16 input tile, sharded L1 (resident)
CB_OUT = 16  # one fp32 output tile, sharded L1

ZONE_NAME = "FIN_SFPU"

VARIANTS = (
    "none",
    "stock",
    "stock_noinit",
    "vmode_c",
    "cskip_each",
    "fused_rc",
    "fused_c",
    "fused_cskip",
    "fused_cskip_noinit",
    "fused_cskip_bf16",
    "fused_cskip_nonneg",
    "cskip_rsqrt_only",
)
BASELINE = "stock"
ABLATION = "none"
CANDIDATE = "fused_cskip"

_VARIANT_ID = {name: i for i, name in enumerate(VARIANTS)}

# 32-lane SFPU vector ops each variant runs (the MATH cost is ~flat per vector op).
_VECTORS = {
    "none": 0,
    "stock": 96,
    "stock_noinit": 96,
    "vmode_c": 48,
    "cskip_each": 24,
    "fused_rc": 32,
    "fused_c": 16,
    "fused_cskip": 8,
    "fused_cskip_noinit": 8,
    "fused_cskip_bf16": 8,
    "fused_cskip_nonneg": 8,
    "cskip_rsqrt_only": 8,
}

# SFPU init calls inside the timed loop (per finalize).
_INITS = {
    "none": 0,
    "stock": 2,
    "stock_noinit": 0,
    "vmode_c": 2,
    "cskip_each": 2,
    "fused_rc": 1,
    "fused_c": 1,
    "fused_cskip": 1,
    "fused_cskip_noinit": 0,
    "fused_cskip_bf16": 1,
    "fused_cskip_nonneg": 1,
    "cskip_rsqrt_only": 1,
}

LABEL = {
    "none": "empty loop (no SFPU)",
    "stock": "3 calls, VectorMode::RC + 2 inits  (THE OP TODAY)",
    "stock_noinit": "3 calls, VectorMode::RC, inits hoisted",
    "vmode_c": "3 calls, VectorMode::C",
    "cskip_each": "3 even-parity sfpi bodies (scope, no fusion)",
    "fused_rc": "1 fused body, VectorMode::RC (fusion, no scope)",
    "fused_c": "1 fused body, VectorMode::C",
    "fused_cskip": "1 fused body, even-parity stride (dst_reg+=2)",
    "fused_cskip_noinit": "fused_cskip, init hoisted",
    "fused_cskip_bf16": "fused_cskip + DEST bf16 truncation of intermediates",
    "fused_cskip_nonneg": "fused_cskip, negative-input NaN guard dropped",
    "cskip_rsqrt_only": "DIAGNOSTIC FLOOR: rsqrt alone on column 0",
}

# (r0, r1, c0, c1) half-open region each variant leaves correct.
VALID_REGION = {
    "none": (0, TILE, 0, TILE),  # identity copy
    "stock": (0, TILE, 0, TILE),
    "stock_noinit": (0, TILE, 0, TILE),
    "vmode_c": (0, TILE, 0, 16),
    "cskip_each": (0, TILE, 0, 1),
    "fused_rc": (0, TILE, 0, TILE),
    "fused_c": (0, TILE, 0, 16),
    "fused_cskip": (0, TILE, 0, 1),
    "fused_cskip_noinit": (0, TILE, 0, 1),
    "fused_cskip_bf16": (0, TILE, 0, 1),
    "fused_cskip_nonneg": (0, TILE, 0, 1),
    "cskip_rsqrt_only": (0, TILE, 0, 1),
}


def vectors(variant):
    return _VECTORS[variant]


def inits(variant):
    return _INITS[variant]


# =============================================================================
# Compute kernel.
#
# RAW LLK, deliberately (this bench exists to price exactly that): the three
# convenience wrappers `mul_unary_tile` / `add_unary_tile` / `rsqrt_tile` all
# HARDCODE `VectorMode::RC` and `ITERATIONS = 8` at their call site
# (binop_with_scalar.h:57-70, rsqrt.h:36-45), so neither the scope nor the fusion
# is reachable through them.  Both go through the same `SFPU_UNARY_CALL` /
# `_llk_math_eltwise_unary_sfpu_params_` entry point the wrappers themselves use;
# the only thing bypassed is the fixed VectorMode/ITERATIONS the wrapper bakes in.
#
# CT args: [variant_id]      RT args: [reps, inv_w_bits, eps_bits]
# =============================================================================
_KERNEL = r"""
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "ckernel_sfpu_binop_with_unary.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

using ckernel::VectorMode;

// ---------------------------------------------------------------------------
// (a) the three stock calls, with VectorMode threaded through instead of hardcoded
// ---------------------------------------------------------------------------
ALWI void mul_scoped(uint32_t idst, uint32_t p, VectorMode vm) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
        (APPROX, 2 /*MUL*/, 8 /*ITERATIONS*/), idst, vm, p));
}
ALWI void add_scoped(uint32_t idst, uint32_t p, VectorMode vm) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
        (APPROX, 0 /*ADD*/, 8 /*ITERATIONS*/), idst, vm, p));
}
ALWI void rsqrt_scoped(uint32_t idst, VectorMode vm) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt,
        (APPROX, 8 /*ITERATIONS*/, DST_ACCUM_MODE, false /*FAST_APPROX*/, false /*legacy_compat*/), idst, vm));
}

#ifdef TRISC_MATH
// ---------------------------------------------------------------------------
// (b) even-parity sfpi bodies.  The SFPU walks a face as
//     [rg0-even, rg0-odd, rg1-even, ...] -- column parity is the INNER axis and
//     column 0 is EVEN, so visiting offsets 0,2,4,6 (dst_reg += 2) keeps every
//     row-group of the face at half the vectors.  Net dst_reg advance is +8 ==
//     the stock ITERATIONS=8, so VectorMode::C's face0 -> face2 stepping composes
//     unchanged: column 0 for all 32 rows in 8 vector ops instead of 16 (or 32).
// ---------------------------------------------------------------------------
template <int STRIDE, int NITER>
sfpi_inline void body_mul(uint32_t p) {
    const sfpi::vFloat s = ckernel::sfpu::Converter::as_float(p);
    for (int i = 0; i < NITER; i++) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * s;
        sfpi::dst_reg += STRIDE;
    }
}
template <int STRIDE, int NITER>
sfpi_inline void body_add(uint32_t p) {
    const sfpi::vFloat s = ckernel::sfpu::Converter::as_float(p);
    for (int i = 0; i < NITER; i++) {
        sfpi::dst_reg[0] = sfpi::dst_reg[0] + s;
        sfpi::dst_reg += STRIDE;
    }
}
template <int STRIDE, int NITER>
sfpi_inline void body_rsqrt() {
    for (int i = 0; i < NITER; i++) {
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) { t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest); }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}

// ---------------------------------------------------------------------------
// (c) THE FUSED FINALIZE: rsqrt(v * inv_w + eps), one walk instead of three.
//     Same non-approximate sqrt body the stock rsqrt_tile calls -- APPROX and
//     FAST_APPROX are untouched, so this is the identical function at the
//     identical precision, just computed in one pass.
//     TRUNC=true restores the baseline's intermediate round trip through the
//     16-bit DEST (bf16 truncate after the mul and after the add), so the two
//     numerical stories can be told apart.
// ---------------------------------------------------------------------------
sfpi_inline sfpi::vFloat bf16_trunc(sfpi::vFloat x) {
    return sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vUInt>(x) & 0xFFFF0000u);
}
// NOT unrolled, deliberately: the sqrt body alone nearly fills the 8 SFPU LREGs,
// and holding the two scalar constants live across an UNROLLED copy of it spills
// ("internal compiler error: cannot store sfpu register" — measured).  `unroll 1`
// keeps one body live at a time, which is the same discipline the sfpu_tile_scope
// example records for its c_skip bodies.
//     NONNEG=true passes FAST_APPROX to the sqrt body.  On the NON-approximate
//     branch (which is what runs here, APPROX being fixed false by the user's
//     math_approx_mode) that flag does exactly ONE thing: it elides the trailing
//     `v_if(x < 0) y = NaN` guard.  It does NOT change the Newton iteration or any
//     computed value for x >= 0 -- and the finalize's input is Sum(x^2)*(1/W) + eps,
//     a sum of squares plus a positive epsilon, so x < 0 is unreachable.  Same
//     numbers, three fewer instructions per vector.
template <int STRIDE, int NITER, bool TRUNC, bool NONNEG = false>
sfpi_inline void body_fused(uint32_t inv_w_bits, uint32_t eps_bits) {
#pragma GCC unroll 1
    for (int i = 0; i < NITER; i++) {
        const sfpi::vFloat inv_w = ckernel::sfpu::Converter::as_float(inv_w_bits);
        const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
        sfpi::vFloat v = sfpi::dst_reg[0] * inv_w;
        if constexpr (TRUNC) { v = bf16_trunc(v); }
        v = v + eps;
        if constexpr (TRUNC) { v = bf16_trunc(v); }
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, NONNEG>(v);
        if constexpr (!DST_ACCUM_MODE) { t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest); }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STRIDE;
    }
}
#endif

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_out = 16;
    constexpr uint32_t variant = get_compile_time_arg_val(0);
    const uint32_t reps = get_arg_val<uint32_t>(0);   // runtime -> the loop is real, not unrolled away
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits = get_arg_val<uint32_t>(2);

    compute_kernel_hw_startup(cb_in, cb_in, cb_out);
    copy_tile_init(cb_in);

    // Hoisted inits for the *_noinit variants (the others init inside the loop,
    // exactly where the op's `finalize` lambda does).
    if constexpr (variant == 2) {           // stock_noinit
        binop_with_scalar_tile_init();
        rsqrt_tile_init();
    } else if constexpr (variant == 8) {    // fused_cskip_noinit
        rsqrt_tile_init();
    }

    cb_reserve_back(cb_in, 1);
    cb_push_back(cb_in, 1);   // sharded input already resident -- mark available once
    cb_wait_front(cb_in, 1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);   // seed DEST[0] once -- OUTSIDE the timed zone
    {
        DeviceZoneScopedN("FIN_SFPU");   // records per-TRISC; the test reads TRISC_1 (math)
        for (uint32_t r = 0; r < reps; ++r) {
            if constexpr (variant == 0) {
                // none: math-loop overhead floor
            } else if constexpr (variant == 1) {          // stock -- the op today
                binop_with_scalar_tile_init();
                mul_unary_tile(0, inv_w_bits);
                add_unary_tile(0, eps_bits);
                rsqrt_tile_init();
                rsqrt_tile(0);
            } else if constexpr (variant == 2) {          // stock_noinit
                mul_unary_tile(0, inv_w_bits);
                add_unary_tile(0, eps_bits);
                rsqrt_tile(0);
            } else if constexpr (variant == 3) {          // vmode_c
                binop_with_scalar_tile_init();
                mul_scoped(0, inv_w_bits, VectorMode::C);
                add_scoped(0, eps_bits, VectorMode::C);
                rsqrt_tile_init();
                rsqrt_scoped(0, VectorMode::C);
            } else if constexpr (variant == 4) {          // cskip_each
                binop_with_scalar_tile_init();
                MATH((_llk_math_eltwise_unary_sfpu_params_(body_mul<2, 4>, 0, VectorMode::C, inv_w_bits)));
                MATH((_llk_math_eltwise_unary_sfpu_params_(body_add<2, 4>, 0, VectorMode::C, eps_bits)));
                rsqrt_tile_init();
                MATH((_llk_math_eltwise_unary_sfpu_params_(body_rsqrt<2, 4>, 0, VectorMode::C)));
            } else if constexpr (variant == 5) {          // fused_rc
                rsqrt_tile_init();
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    body_fused<1, 8, false>, 0, VectorMode::RC, inv_w_bits, eps_bits)));
            } else if constexpr (variant == 6) {          // fused_c
                rsqrt_tile_init();
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    body_fused<1, 8, false>, 0, VectorMode::C, inv_w_bits, eps_bits)));
            } else if constexpr (variant == 7) {          // fused_cskip  (CANDIDATE)
                rsqrt_tile_init();
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    body_fused<2, 4, false>, 0, VectorMode::C, inv_w_bits, eps_bits)));
            } else if constexpr (variant == 8) {          // fused_cskip_noinit
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    body_fused<2, 4, false>, 0, VectorMode::C, inv_w_bits, eps_bits)));
            } else if constexpr (variant == 9) {          // fused_cskip_bf16
                rsqrt_tile_init();
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    body_fused<2, 4, true>, 0, VectorMode::C, inv_w_bits, eps_bits)));
            } else if constexpr (variant == 10) {         // fused_cskip_nonneg
                rsqrt_tile_init();
                MATH((_llk_math_eltwise_unary_sfpu_params_(
                    body_fused<2, 4, false, true>, 0, VectorMode::C, inv_w_bits, eps_bits)));
            } else {                                      // cskip_rsqrt_only (diagnostic floor)
                rsqrt_tile_init();
                MATH((_llk_math_eltwise_unary_sfpu_params_(body_rsqrt<2, 4>, 0, VectorMode::C)));
            }
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out, 0);   // pack once (post-zone) so the host can verify the scoped region
    cb_push_back(cb_out, 1);
    tile_regs_release();
    cb_pop_front(cb_in, 1);
}
"""


# =============================================================================
# Host side
# =============================================================================
def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def create_sharded_memory_config():
    """One 32x32 tile, height-sharded onto a single core."""
    return ttnn.create_sharded_memory_config(
        shape=(TILE, TILE),
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def target_compute_config():
    """The focus case's user config -- FIXED for every variant, never a lever."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )


def create_program_descriptor(input_tensor, output_tensor, *, variant, inv_w_bits, eps_bits, reps=1):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    compute = ttnn.KernelDescriptor(
        kernel_source=_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[_VARIANT_ID[variant]],
        runtime_args=[(ttnn.CoreCoord(0, 0), [reps, inv_w_bits, eps_bits])],
        config=target_compute_config(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, input_tensor),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor),
    ]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run_op(input_tensor, *, variant, inv_w_bits, eps_bits, reps=1):
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILE]),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        input_tensor.device(),
        create_sharded_memory_config(),
    )
    descriptor = create_program_descriptor(
        input_tensor, output, variant=variant, inv_w_bits=inv_w_bits, eps_bits=eps_bits, reps=reps
    )
    return ttnn.generic_op([input_tensor, output], descriptor)
