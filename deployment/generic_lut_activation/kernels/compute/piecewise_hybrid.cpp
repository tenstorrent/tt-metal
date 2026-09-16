// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// =============================================================================
// PIECEWISE HYBRID — generic RVV/SFPU dual-stream variant of the embedded-LUT
// activation kernel (magic 0xC0FFEE20).
// =============================================================================
// This file is NOT compiled standalone: the run_csv.sh codegen (env
// TT_ACT_HYBRID_KERNEL=1) emits the usual adhoc[N].cpp defines block
// (EMBEDDED_LUT, POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE, LUT_DATA, INPUT_MIN/MAX,
// BASIS_*, optional SEGMENT_DEGREES / POLY_PARITY_*) and then includes THIS
// file instead of piecewise_generic.cpp.
//
// Streams (tile parity = LOCAL tile index i on this core, shared bit-for-bit
// with the committed hybrid reader/writer dataflow kernels and the host's
// HYBRID_RVV_SHARE gcd reduction):
//     is_rvv(i) = (i % rvv_den) < rvv_num
//   SFPU stream: c_0 -> unpack -> DEST -> math -> pack -> c_16. The math body
//     is the UNMODIFIED production dispatch
//     sfpi::piecewise_generic_lut_dispatch<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>
//     pulled in below by including piecewise_generic.cpp verbatim (with its
//     kernel_main renamed away), so USE_DUAL_EVAL / basis / adaptive-degree
//     routing is byte-identical to a share=0 production run.
//   RVV stream:  c_1 -> TRISC2 Zve32f generic evaluator -> c_17, raw
//     stream-MMIO-register protocol (same as hybrid_demo.cpp: we are the sole
//     acker of c_1 and sole producer of c_17; c_0/c_16 stay 100% llk-managed).
//
// -----------------------------------------------------------------------------
// GENERIC RVV EVALUATOR (pack thread only)
// -----------------------------------------------------------------------------
// Works for any embedded polynomial-cascade LUT with NUM_SEGMENTS <= 64 and
// POLY_DEGREE <= 16, fp32 only. Two init-time tables (scalar code, untimed):
//   (a) 256-entry uint32 cell table over the uniform grid
//       [LUT_DATA[0], LUT_DATA[NUM_SEGMENTS]] (abs-space when
//       BASIS_INPUT_ABS_X): cell k -> segment index of the cell's LEFT edge.
//   (b) AoS coefficient records: one per segment, (POLY_DEGREE+1) fp32 padded
//       with +0.0f to the next power-of-two bytes, so record offset = seg <<
//       HYB_REC_SHIFT. Zero-padding is EXACT for Horner (acc=0 -> acc*x+c == c
//       bit-exactly), so per-segment adaptive degrees (SEGMENT_DEGREES) come
//       out bit-identical to the production eval_polynomial<effective_degree>.
// Hot loop @ e32m2 (vl=8), 2-way chunk interleave, per element:
//   cell = clamp(vfcvt_rtz((x_eval - lo) * inv_step), 0, 255)   [.vf scalars]
//   seg  = cell_table[cell]                                     [vluxei32]
//   seg += (x_eval >= bnd_hi[seg]); seg -= (x_eval < bnd_lo[seg])
//        (bidirectional one-step fix-up against the REAL breakpoints; the
//         sentinels make edge segments sticky: bnd_hi[last] = NaN — the >=
//         compare never fires, even for x = +inf — and bnd_lo[0] = -inf)
//   (POLY_DEGREE+1) coefficient vluxei32 loads, then straight full-degree
//   Horner HIGH-TO-LOW with fused vfmadd, matching the LUT contract
//   p(x) = sum c_j x^j:  acc = c_D; for j = D-1..0: acc = fma(acc, x_eval, c_j)
//   — the same operation ORDER as the production eval_polynomial<D> sfpu_mad
//   chain (piecewise_generic.cpp:2142+). NOTE the two engines still round
//   independently (SFPMAD vs Zve32f vfmadd), so RVV-vs-SFPU byte identity is
//   NOT claimed; each stream's accuracy class is validated by the harness ULP
//   report, and each stream is byte-STABLE across shares (the self-consistency
//   gates). Both engines are DAZ+FTZ, so subnormal coefficients (e.g. gelu
//   seg0 c0 = 5.55e-42) behave as +0.0 identically on both.
//   For POLY_PARITY_ODD/EVEN LUTs the production SFPU path evaluates an
//   x^2-Horner over the nonzero-parity coefficients while this evaluator runs
//   the full Horner over the same (zero-interleaved) coefficients: the same
//   polynomial in a different (still-Horner) association, so roundings differ
//   by a few ULP but the fit's accuracy class holds (harness-verified).
// Post-polynomial basis reconstruction mirrors piecewise_generic.cpp:3292-3329
//   IN ORDER: (1) BASIS_MUL_ABS_X_BEFORE_POST: acc *= |x|;
//   (2) BASIS_AFFINE_EVEN: acc = fma(EVEN_SCALE*|x| (rounded mul), acc,
//       fma(SCALE, x_orig, BIAS));  (3) BASIS_CLAMP_MAX: acc = min(acc, CM);
//   (4) BASIS_POST_SIGN_X: acc = copysign(acc, x_orig);
//   (5) BASIS_LEFT_TAIL_ZERO: x_orig < T -> 0;  (6) BASIS_RIGHT_TAIL_IDENTITY:
//   x_orig > T -> x_orig.  Tails are unconditional overrides AFTER the affine
//   mad, exactly like production.
// Out-of-range x: the CELL INDEX is clamped to [0,255] but the polynomial is
//   ALWAYS evaluated at the RAW x_eval (never a clamped x); combined with the
//   sticky-edge fix-up sentinels this reproduces the production cascade's
//   natural edge-segment selection for x outside [boundaries[0], top].
//   Boundary ownership uses >= (a breakpoint belongs to its RIGHT segment),
//   identical to the production v_if(x >= lut[seg]) cascade.
//
// LOUD-FAILURE CONTRACT: init computes the resolution metric
//   max over k of (cell_table[min(k+2,255)] - cell_table[k]), also folding in
//   (NUM_SEGMENTS-1) - cell_table[254] for the clamped top cells. The 2-cell
//   window absorbs the <=1-cell fuzz between the vector rtz((x-lo)*inv_step)
//   cell and the scalar left-edge cell table. metric <= 1 guarantees the
//   bidirectional one-step fix-up lands the exact production segment for every
//   x. metric > 1 (a segment narrower than ~2 grid cells) writes error word
//   0xE0000001 + the metric to scratch and the run must be FAILED by the
//   runner; the kernel still streams tiles (best effort) so the pipeline never
//   deadlocks — truthfulness lives in the error word, not in a hang.
//
// -----------------------------------------------------------------------------
// L1 SCRATCH MAP — uint32 words at 0x160000 (core-local; dump core (0,0) via
// RVV_SCRATCH_CSV):
//   hdr[0]  = 0xC0FFEE20 magic (written LAST)
//   hdr[1]  = t_first lo   -- wall clock on the pack thread AFTER table init,
//   hdr[2]  = t_first hi      BEFORE tile 0 (init is deliberately untimed)
//   hdr[3]  = t_last lo    -- wall clock AFTER the last push of EITHER stream
//   hdr[4]  = t_last hi       was ISSUED (SFPU packs are Tensix-queued)
//   hdr[5]  = sfpu_tiles_done      (per-tile, progress-visible)
//   hdr[6]  = rvv_tiles_done       (per-tile, progress-visible)
//   hdr[7]  = last_rvv_local_idx   (0xFFFFFFFF if no RVV tile ran)
//   hdr[8]  = n_tiles   hdr[9] = rvv_num   hdr[10] = rvv_den
//   hdr[11] = breadcrumb stage: 0x10 pack entry, 0x20 tables built,
//             0x30 in loop, 0x40 loop done, 0x50 header complete
//   hdr[12] = breadcrumb tile (local index in flight)
//   hdr[13] = elapsed cycles (t_last - t_first), low 32 bits
//   hdr[14] = ERROR word: 0 = OK; 0xE0000001 = index-LUT resolution violation
//   hdr[15] = resolution metric (max segments spanned by any 2-cell window)
//   hdr[16] = NUM_SEGMENTS   hdr[17] = POLY_DEGREE   hdr[18] = HYB_REC_SHIFT
//   hdr[19] = fp32 bits of grid lo   hdr[20] = bits of grid hi
//   hdr[21] = bits of inv_step
//   [+0x100] bnd_hi  float[64]  (word  64): boundaries[s+1]; NaN for last seg
//   [+0x200] bnd_lo  float[64]  (word 128): boundaries[s];  -inf for seg 0
//   [+0x300] cell    uint32[256](word 192): segment of each grid cell's left edge
//   [+0x700] coeff records      (word 448): NUM_SEGMENTS << HYB_REC_SHIFT bytes
// =============================================================================

// ---------------------------------------------------------------------------
// Pull in the ENTIRE production kernel (sfpi evaluators, dispatch, helpers)
// with its kernel_main renamed out of the way. The production math is thereby
// reused VERBATIM — no copied evaluator code to drift.
// ---------------------------------------------------------------------------
#define kernel_main hyb_production_kernel_main_unused
#include "piecewise_generic.cpp"
#undef kernel_main

// ---------------------------------------------------------------------------
// Support matrix — refuse at COMPILE TIME anything whose SFPU epilogue the RVV
// evaluator does not reproduce. Silent wrong math is never an option.
// ---------------------------------------------------------------------------
#ifdef USE_BF16
#error "piecewise_hybrid: fp32 only (the host HYBRID_RVV_SHARE guard also rejects bf16)"
#endif
#ifndef EMBEDDED_LUT
#error "piecewise_hybrid: requires an embedded LUT (generated adhoc defines)"
#endif
#ifdef FUSE_GRAD_MUL
#error "piecewise_hybrid: fused-grad (_bw) runs are rejected (host guard agrees)"
#endif
#if EVAL_METHOD_IS_STANDALONE || !defined(EVAL_METHOD_POLY_CASCADE)
#error "piecewise_hybrid: only the piecewise polynomial cascade eval method is supported"
#endif
#if defined(RANGE_REDUCTION_EXP) || defined(RANGE_REDUCTION_TRIG) || defined(RANGE_REDUCTION_TAN) || \
    defined(RANGE_REDUCTION_LOG) || defined(RANGE_REDUCTION_CBRT) || defined(EVAL_METHOD_REDUCED_POLY)
#error "piecewise_hybrid: range-reduced LUTs are not supported by the RVV evaluator"
#endif
#if defined(ASYMPTOTIC_FACTOR_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || \
    defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_X)
#error "piecewise_hybrid: asymptotic-factored LUTs are not supported by the RVV evaluator"
#endif
#if defined(AFFINE_IDENTITY) || defined(AFFINE_COLLAPSE) || defined(CLAMPED_AFFINE_COLLAPSE) || \
    (defined(TT_ACT_EVAL_KIND) && (TT_ACT_EVAL_KIND != 0) && (TT_ACT_EVAL_KIND != TT_ACT_EVAL_POLY_CASCADE))
#error "piecewise_hybrid: algebraic whole-function collapses bypass the cascade dispatch"
#endif
#if defined(HAS_CRITICAL_POINT) || defined(POSTCOMPOSE_AFFINE_Y)
#error "piecewise_hybrid: critical-point / postcompose epilogues are not implemented on the RVV stream"
#endif
#if defined(BASIS_MUL_SQRT_1_MINUS_ABS) || defined(BASIS_POST_REFLECT_PI) || defined(BASIS_RIGHT_TAIL_ABS_AFFINE)
#error "piecewise_hybrid: sqrt_factored / reflect-pi / abs-affine-tail bases are not implemented on the RVV stream"
#endif
#if defined(DST_COEFF_STORE) || defined(DST_COEFF_PROBE)
#error "piecewise_hybrid: DST coefficient store must be disabled (set DST_COEFF_DISABLE=1)"
#endif

static_assert(NUM_SEGMENTS >= 1 && NUM_SEGMENTS <= 64, "piecewise_hybrid: NUM_SEGMENTS must be 1..64");
static_assert(POLY_DEGREE <= 16, "piecewise_hybrid: POLY_DEGREE must be <= 16");
static_assert(
    LUT_SIZE == (NUM_SEGMENTS + 1) + NUM_SEGMENTS * (POLY_DEGREE + 1), "piecewise_hybrid: unexpected LUT layout");

// ---------------------------------------------------------------------------
// Compile-time layout constants (all three TUs see these; only pack uses them).
// ---------------------------------------------------------------------------
constexpr uint32_t hyb_rec_bytes_pow2() {
    uint32_t need = (POLY_DEGREE + 1u) * 4u;
    uint32_t p = 4u;
    while (p < need) {
        p <<= 1;
    }
    return p;
}
constexpr uint32_t HYB_REC_BYTES = hyb_rec_bytes_pow2();
constexpr uint32_t hyb_log2u(uint32_t v) {
    uint32_t s = 0;
    while ((1u << s) < v) {
        s++;
    }
    return s;
}
constexpr uint32_t HYB_REC_SHIFT = hyb_log2u(HYB_REC_BYTES);
static_assert(HYB_REC_BYTES <= 128, "record must fit the 0x700..0x26FF coefficient window");

// Uniform 256-cell index grid over the segment-lookup domain. With
// BASIS_INPUT_ABS_X the lookup runs in |x|-space, so the grid covers the
// BOUNDARY domain [LUT_DATA[0], LUT_DATA[NUM_SEGMENTS]] — NOT the signed
// input range. Division happens at COMPILE TIME (no scalar fdiv exists on
// TRISC2).
constexpr float HYB_GRID_LO = LUT_DATA[0];
constexpr float HYB_GRID_HI = LUT_DATA[NUM_SEGMENTS];
constexpr float HYB_GRID_STEP = (HYB_GRID_HI - HYB_GRID_LO) / 256.0f;
constexpr float HYB_GRID_INV_STEP = 256.0f / (HYB_GRID_HI - HYB_GRID_LO);
static_assert(HYB_GRID_HI > HYB_GRID_LO, "degenerate boundary domain");

// Deterministic stream parity — identical on all three TRISCs, in the hybrid
// reader/writer dataflow kernels, and in the host's gcd reduction.
static inline bool hybrid_is_rvv(uint32_t i, uint32_t rvv_num, uint32_t rvv_den) { return (i % rvv_den) < rvv_num; }

// ============================================================================
// RVV stream (pack TRISC only — stock tt-metal enables Zve32f on TRISC2 via
// ComputeConfig::enable_trisc2_rvv, so every RVV type/intrinsic/include stays
// strictly inside this guard).
// ============================================================================
#ifdef TRISC_PACK
#include <riscv_vector.h>
#include "internal/tt-1xx/risc_common.h"  // get_timestamp(), invalidate_l1_cache()
#include "domain_actions_rvv.h"

static constexpr uint32_t HYB_SCRATCH = 0x160000;
static constexpr uint32_t HYB_BND_HI_OFF = 0x100;
static constexpr uint32_t HYB_BND_LO_OFF = 0x200;
static constexpr uint32_t HYB_CELL_OFF = 0x300;
static constexpr uint32_t HYB_COEFF_OFF = 0x700;
static constexpr uint32_t HYB_ERR_INDEX_RESOLUTION = 0xE0000001u;

// ---- raw dual-stream CB protocol (verified in hybrid_demo.cpp) -------------
// All four counters are per-stream NOC-overlay scratch registers (stream N =
// CB N), MMIO-readable/writable from any RISC, NOT behind the BH L1 cache.
// 16-bit wrap math throughout, matching dataflow_api.h / llk_io_pack.h.

static inline uint16_t hyb_in_tiles_received(uint32_t cb) {
    return (uint16_t)reg_read((uint32_t)(uintptr_t)get_cb_tiles_received_ptr((int)cb));
}

static inline void hyb_in_wait(uint32_t cb, uint16_t my_acked, uint16_t want) {
    while ((uint16_t)(hyb_in_tiles_received(cb) - my_acked) < want) {
    }
}

// pop n tiles of input CB `cb` — we are the SOLE acker, so a plain MMIO store
// is the complete pop. `my_acked` is the local mirror (reg zeroed at launch).
static inline void hyb_in_pop(uint32_t cb, uint16_t& my_acked, uint16_t n) {
    my_acked += n;
    get_cb_tiles_acked_ptr((int)cb)[0] = my_acked;
}

// byte address of the t-th RVV-stream tile in CB `cb`. Pack-side CB init sets
// write=true so fifo_wr_ptr == base (16-byte units); we never llk-push these
// CBs, so it stays at base.
static inline uint32_t hyb_tile_addr(uint32_t cb, uint32_t t) {
    LocalCBInterface& i = get_local_cb_interface(cb);
    uint32_t base16B = i.fifo_wr_ptr;
    uint32_t slot = t % i.fifo_num_pages;
    return (base16B + slot * i.fifo_page_size) << 4;  // CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT
}

// wait for n free pages in output CB `cb` — RISC poll against our OWN
// received mirror (mirrors llk_wait_for_free_tiles).
static inline void hyb_out_reserve(uint32_t cb, uint16_t my_received, uint16_t n) {
    LocalCBInterface& i = get_local_cb_interface(cb);
    while ((uint16_t)((uint16_t)i.fifo_num_pages -
                      (uint16_t)(my_received -
                                 (uint16_t)reg_read((uint32_t)(uintptr_t)get_cb_tiles_acked_ptr((int)cb)))) < n) {
    }
}

// publish n tiles of output CB `cb` — we are the SOLE producer. Fence + last-
// word read-back order the L1 data stores before the MMIO count store.
static inline void hyb_out_push(uint32_t cb, uint16_t& my_received, uint32_t last_tile_addr, uint16_t n) {
    asm volatile("fence" ::: "memory");
    (void)*(volatile uint32_t*)last_tile_addr;
    my_received += n;
    get_cb_tiles_received_ptr((int)cb)[0] = my_received;
}

// ---- init: build the scratch tables (scalar, untimed) ----------------------
// No C float->int casts anywhere on this thread (scalar fcvt is RNE-locked):
// the cell table is built by WALKING cells with a running segment cursor —
// integer index math + scalar float compares/mults only (all RNE-safe).
// Returns the resolution metric (max segments spanned by any 2-cell window).
static inline uint32_t hyb_build_tables() {
    float* bnd_hi = (float*)(HYB_SCRATCH + HYB_BND_HI_OFF);
    float* bnd_lo = (float*)(HYB_SCRATCH + HYB_BND_LO_OFF);
    uint32_t* cell_tab = (uint32_t*)(HYB_SCRATCH + HYB_CELL_OFF);

    // (a) AoS coefficient records, zero-padded to the power-of-two record.
    for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
        float* rec = (float*)(HYB_SCRATCH + HYB_COEFF_OFF + (s << HYB_REC_SHIFT));
        for (uint32_t j = 0; j <= POLY_DEGREE; j++) {
            rec[j] = LUT_DATA[(NUM_SEGMENTS + 1) + s * (POLY_DEGREE + 1) + j];
        }
        for (uint32_t j = POLY_DEGREE + 1; j < (HYB_REC_BYTES / 4); j++) {
            rec[j] = 0.0f;
        }
    }

    // (b) fix-up boundary tables with sticky-edge sentinels.
    //     bnd_hi[last] = quiet NaN: `x >= NaN` is false for EVERY x including
    //     +inf, so the last segment never increments past the table.
    //     bnd_lo[0] = -inf: `x < -inf` is false for every x, so segment 0
    //     never decrements. Written as bit patterns (no float literals of
    //     inf/NaN needed).
    for (uint32_t s = 0; s < NUM_SEGMENTS; s++) {
        if (s + 1 < NUM_SEGMENTS) {
            bnd_hi[s] = LUT_DATA[s + 1];
        } else {
            ((uint32_t*)bnd_hi)[s] = 0x7FC00000u;  // qNaN
        }
        if (s > 0) {
            bnd_lo[s] = LUT_DATA[s];
        } else {
            ((uint32_t*)bnd_lo)[s] = 0xFF800000u;  // -inf
        }
    }
    for (uint32_t s = NUM_SEGMENTS; s < 64; s++) {  // defensive fill for OOB seg reads
        ((uint32_t*)bnd_hi)[s] = 0x7FC00000u;
        ((uint32_t*)bnd_lo)[s] = 0xFF800000u;
    }

    // (c) cell table: segment of each cell's LEFT edge (>=: a breakpoint
    //     belongs to its RIGHT segment, same as the production cascade).
    uint32_t seg = 0;
    for (uint32_t k = 0; k < 256; k++) {
        float left = HYB_GRID_LO + (float)(int)k * HYB_GRID_STEP;  // int->float is exact for k<=255
        while (seg + 1 < NUM_SEGMENTS && left >= LUT_DATA[seg + 1]) {
            seg++;
        }
        cell_tab[k] = seg;
    }

    // (d) resolution metric over 2-cell windows (absorbs the <=1-cell rounding
    //     fuzz of the vector rtz((x-lo)*inv_step) index vs these left edges),
    //     folding in the clamped top cells which must reach the last segment
    //     within one fix-up step.
    uint32_t metric = 0;
    for (uint32_t k = 0; k + 2 < 256; k++) {
        uint32_t d = cell_tab[k + 2] - cell_tab[k];
        if (d > metric) {
            metric = d;
        }
    }
    uint32_t top = (NUM_SEGMENTS - 1) - cell_tab[254];
    if (top > metric) {
        metric = top;
    }
    return metric;
}

// ---- generic RVV tile evaluator ---------------------------------------------
// 1024 fp32 elements, e32m2 (vl=8) = 128 chunks, 2-way interleave (A/B chains
// hide the vluxei latency; named registers only — no arrays of sizeless types,
// no tuple intrinsics). See the file header for the exact numeric contract.
static inline void hyb_rvv_eval_tile(const float* x, float* yout) {
    const float* ctab = (const float*)(HYB_SCRATCH + HYB_COEFF_OFF);
    const uint32_t* cell_tab = (const uint32_t*)(HYB_SCRATCH + HYB_CELL_OFF);
    const float* bnd_hi = (const float*)(HYB_SCRATCH + HYB_BND_HI_OFF);
    const float* bnd_lo = (const float*)(HYB_SCRATCH + HYB_BND_LO_OFF);

    size_t vl = __riscv_vsetvl_e32m2(8);
    vuint32m2_t vzero = __riscv_vmv_v_x_u32m2(0, vl);
#if defined(BASIS_AFFINE_EVEN)
    vfloat32m2_t vaffbias = __riscv_vfmv_v_f_f32m2(BASIS_AFFINE_BIAS, vl);
#endif

    for (int c = 0; c < 128; c += 2) {
        vfloat32m2_t xoA = __riscv_vle32_v_f32m2(x + c * 8, vl);
        vfloat32m2_t xoB = __riscv_vle32_v_f32m2(x + c * 8 + 8, vl);
#if defined(BASIS_INPUT_ABS_X)
        vfloat32m2_t xeA = __riscv_vfsgnjx_vv_f32m2(xoA, xoA, vl);  // |x|
        vfloat32m2_t xeB = __riscv_vfsgnjx_vv_f32m2(xoB, xoB, vl);
#else
        vfloat32m2_t xeA = xoA;
        vfloat32m2_t xeB = xoB;
#endif
        // cell = clamp(rtz((xe - lo) * inv_step), 0, 255)  — vector rtz only;
        // the raw xe (NOT a clamped copy) feeds the polynomial below.
        vint32m2_t iA = __riscv_vfcvt_rtz_x_f_v_i32m2(
            __riscv_vfmul_vf_f32m2(__riscv_vfsub_vf_f32m2(xeA, HYB_GRID_LO, vl), HYB_GRID_INV_STEP, vl), vl);
        vint32m2_t iB = __riscv_vfcvt_rtz_x_f_v_i32m2(
            __riscv_vfmul_vf_f32m2(__riscv_vfsub_vf_f32m2(xeB, HYB_GRID_LO, vl), HYB_GRID_INV_STEP, vl), vl);
        iA = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(iA, 0, vl), 255, vl);
        iB = __riscv_vmin_vx_i32m2(__riscv_vmax_vx_i32m2(iB, 0, vl), 255, vl);
        vuint32m2_t cbA = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_i32m2_u32m2(iA), 2, vl);
        vuint32m2_t cbB = __riscv_vsll_vx_u32m2(__riscv_vreinterpret_v_i32m2_u32m2(iB), 2, vl);
        vuint32m2_t segA = __riscv_vluxei32_v_u32m2(cell_tab, cbA, vl);
        vuint32m2_t segB = __riscv_vluxei32_v_u32m2(cell_tab, cbB, vl);

        // bidirectional one-step fix-up against the REAL breakpoints
        vuint32m2_t sbA = __riscv_vsll_vx_u32m2(segA, 2, vl);
        vuint32m2_t sbB = __riscv_vsll_vx_u32m2(segB, 2, vl);
        vfloat32m2_t bhA = __riscv_vluxei32_v_f32m2(bnd_hi, sbA, vl);
        vfloat32m2_t bhB = __riscv_vluxei32_v_f32m2(bnd_hi, sbB, vl);
        vfloat32m2_t blA = __riscv_vluxei32_v_f32m2(bnd_lo, sbA, vl);
        vfloat32m2_t blB = __riscv_vluxei32_v_f32m2(bnd_lo, sbB, vl);
        vbool16_t upA = __riscv_vmfge_vv_f32m2_b16(xeA, bhA, vl);
        vbool16_t upB = __riscv_vmfge_vv_f32m2_b16(xeB, bhB, vl);
        vbool16_t dnA = __riscv_vmflt_vv_f32m2_b16(xeA, blA, vl);
        vbool16_t dnB = __riscv_vmflt_vv_f32m2_b16(xeB, blB, vl);
        segA = __riscv_vadd_vv_u32m2(segA, __riscv_vmerge_vxm_u32m2(vzero, 1, upA, vl), vl);
        segB = __riscv_vadd_vv_u32m2(segB, __riscv_vmerge_vxm_u32m2(vzero, 1, upB, vl), vl);
        segA = __riscv_vsub_vv_u32m2(segA, __riscv_vmerge_vxm_u32m2(vzero, 1, dnA, vl), vl);
        segB = __riscv_vsub_vv_u32m2(segB, __riscv_vmerge_vxm_u32m2(vzero, 1, dnB, vl), vl);
        vuint32m2_t offA = __riscv_vsll_vx_u32m2(segA, HYB_REC_SHIFT, vl);
        vuint32m2_t offB = __riscv_vsll_vx_u32m2(segB, HYB_REC_SHIFT, vl);

        // straight full-degree Horner, high-to-low, fused vfmadd
        vfloat32m2_t accA = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offA, vl);
        vfloat32m2_t accB = __riscv_vluxei32_v_f32m2(ctab + POLY_DEGREE, offB, vl);
#pragma GCC unroll 17
        for (int j = (int)POLY_DEGREE - 1; j >= 0; j--) {
            vfloat32m2_t cjA = __riscv_vluxei32_v_f32m2(ctab + j, offA, vl);
            vfloat32m2_t cjB = __riscv_vluxei32_v_f32m2(ctab + j, offB, vl);
            accA = __riscv_vfmadd_vv_f32m2(accA, xeA, cjA, vl);
            accB = __riscv_vfmadd_vv_f32m2(accB, xeB, cjB, vl);
        }

        // basis reconstruction / tails, in production epilogue order
#if defined(BASIS_MUL_ABS_X_BEFORE_POST)
        accA = __riscv_vfmul_vv_f32m2(accA, xeA, vl);
        accB = __riscv_vfmul_vv_f32m2(accB, xeB, vl);
#endif
#if defined(BASIS_AFFINE_EVEN)
        {
            // inner = fma(SCALE, x_orig, BIAS); t1 = EVEN_SCALE*|x| (rounded
            // mul, like the production sfpu_mad's A operand); acc = fma(acc, t1, inner)
            vfloat32m2_t innerA = __riscv_vfmacc_vf_f32m2(vaffbias, BASIS_AFFINE_SCALE, xoA, vl);
            vfloat32m2_t innerB = __riscv_vfmacc_vf_f32m2(vaffbias, BASIS_AFFINE_SCALE, xoB, vl);
            vfloat32m2_t t1A = __riscv_vfmul_vf_f32m2(xeA, BASIS_AFFINE_EVEN_SCALE, vl);
            vfloat32m2_t t1B = __riscv_vfmul_vf_f32m2(xeB, BASIS_AFFINE_EVEN_SCALE, vl);
            accA = __riscv_vfmadd_vv_f32m2(accA, t1A, innerA, vl);
            accB = __riscv_vfmadd_vv_f32m2(accB, t1B, innerB, vl);
        }
#endif
#if defined(BASIS_CLAMP_MAX)
        accA = __riscv_vfmin_vf_f32m2(accA, BASIS_CLAMP_MAX_VALUE, vl);
        accB = __riscv_vfmin_vf_f32m2(accB, BASIS_CLAMP_MAX_VALUE, vl);
#endif
#if defined(BASIS_POST_SIGN_X)
        accA = __riscv_vfsgnj_vv_f32m2(accA, xoA, vl);
        accB = __riscv_vfsgnj_vv_f32m2(accB, xoB, vl);
#endif
#if defined(BASIS_LEFT_TAIL_ZERO)
        {
            vbool16_t mzA = __riscv_vmflt_vf_f32m2_b16(xoA, BASIS_LEFT_TAIL_ZERO_THRESHOLD, vl);
            vbool16_t mzB = __riscv_vmflt_vf_f32m2_b16(xoB, BASIS_LEFT_TAIL_ZERO_THRESHOLD, vl);
            accA = __riscv_vfmerge_vfm_f32m2(accA, 0.0f, mzA, vl);
            accB = __riscv_vfmerge_vfm_f32m2(accB, 0.0f, mzB, vl);
        }
#endif
#if defined(BASIS_RIGHT_TAIL_IDENTITY)
        {
            vbool16_t miA = __riscv_vmfgt_vf_f32m2_b16(xoA, BASIS_RIGHT_TAIL_IDENTITY_THRESHOLD, vl);
            vbool16_t miB = __riscv_vmfgt_vf_f32m2_b16(xoB, BASIS_RIGHT_TAIL_IDENTITY_THRESHOLD, vl);
            accA = __riscv_vmerge_vvm_f32m2(accA, xoA, miA, vl);
            accB = __riscv_vmerge_vvm_f32m2(accB, xoB, miB, vl);
        }
#endif
        accA = tt_rvv_finalize_domain_actions(xoA, accA, vl);
        accB = tt_rvv_finalize_domain_actions(xoB, accB, vl);
        __riscv_vse32_v_f32m2(yout + c * 8, accA, vl);
        __riscv_vse32_v_f32m2(yout + c * 8 + 8, accB, vl);
    }
}
#endif  // TRISC_PACK

// ============================================================================
// Kernel entry. Unpack/math threads run the UNMODIFIED production per-tile
// body over their subset of tiles; the pack thread runs the global interleave
// loop (production llk pack sequence for SFPU tiles, raw-protocol RVV
// evaluation for RVV tiles).
// ============================================================================
void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t rvv_num = get_arg_val<uint32_t>(1);
    uint32_t rvv_den = get_arg_val<uint32_t>(2);
    if (rvv_den == 0) {  // defensive: never divide by zero, degrade to SFPU-only
        rvv_den = 1;
        rvv_num = 0;
    }

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_rvv_in = tt::CBIndex::c_1;
    constexpr auto cb_rvv_out = tt::CBIndex::c_17;
    (void)cb_rvv_in;
    (void)cb_rvv_out;

    const auto& lut_ref = LUT_DATA;
    auto p_lut = &lut_ref;
    (void)p_lut;

    init_sfpu(cb_in, cb_out);

#ifndef TRISC_PACK
    // ------------------------------------------------------------------
    // UNPACK + MATH threads: production per-tile body over SFPU tiles only.
    // The math body is the production dispatch — identical routing (dual /
    // single / blend / adaptive degree) to a share=0 run.
    // ------------------------------------------------------------------
    for (uint32_t i = 0; i < n_tiles; i++) {
        if (hybrid_is_rvv(i, rvv_num, rvv_den)) {
            continue;  // RVV tile: never unpacked, never touches DEST
        }
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        copy_tile(cb_in, 0, 0);
#ifdef TRISC_MATH
        sfpi::piecewise_generic_lut_dispatch<POLY_DEGREE, NUM_SEGMENTS, LUT_SIZE>(*p_lut);
#endif
        tile_regs_commit();
        cb_pop_front(cb_in, 1);
    }
#else
    // ------------------------------------------------------------------
    // PACK thread: global interleave loop.
    //   SFPU tile: production llk pack sequence (fire-and-forget into the
    //              Tensix FIFO) for the next SFPU tile.
    //   RVV tile:  raw-protocol wait/reserve, generic RVV evaluate
    //              c_1 -> c_17, raw pop/push — runs while queued SFPU pack
    //              sequences drain.
    // ------------------------------------------------------------------
    volatile uint32_t* hdr = (volatile uint32_t*)HYB_SCRATCH;
    hdr[0] = 0;  // clear magic first: a partial header is never mistaken for done
    hdr[5] = 0;
    hdr[6] = 0;
    hdr[7] = 0xFFFFFFFFu;
    hdr[8] = n_tiles;
    hdr[9] = rvv_num;
    hdr[10] = rvv_den;
    hdr[11] = 0x10;  // breadcrumb: pack entry
    hdr[12] = 0xFFFFFFFFu;
    hdr[14] = 0;
    hdr[15] = 0;
    hdr[16] = NUM_SEGMENTS;
    hdr[17] = POLY_DEGREE;
    hdr[18] = HYB_REC_SHIFT;
    *(volatile float*)(HYB_SCRATCH + 19 * 4) = HYB_GRID_LO;
    *(volatile float*)(HYB_SCRATCH + 20 * 4) = HYB_GRID_HI;
    *(volatile float*)(HYB_SCRATCH + 21 * 4) = HYB_GRID_INV_STEP;

    uint32_t metric = hyb_build_tables();
    hdr[15] = metric;
    if (metric > 1) {
        // Index-LUT resolution violation: a segment narrower than ~2 grid
        // cells. LOUD failure — the runner must abort on this word. We still
        // stream tiles below so the reader/writer never deadlock.
        hdr[14] = HYB_ERR_INDEX_RESOLUTION;
    }
    hdr[11] = 0x20;  // breadcrumb: tables built

    uint16_t my_acked_c1 = 0;       // local mirror of c_1 acked (reg zeroed at launch)
    uint16_t rvv_out_received = 0;  // local mirror of c_17 received (ditto)
    uint32_t sfpu_done = 0;
    uint32_t rvv_done = 0;

    uint64_t t_first = get_timestamp();
    hdr[1] = (uint32_t)t_first;
    hdr[2] = (uint32_t)(t_first >> 32);
    hdr[11] = 0x30;  // breadcrumb: in loop

    for (uint32_t i = 0; i < n_tiles; i++) {
        hdr[12] = i;
        if (!hybrid_is_rvv(i, rvv_num, rvv_den)) {
            // ---- SFPU tile: production pack-side body, verbatim order ----
            tile_regs_wait();            // queued SEMWAIT (math-done gate)
            cb_reserve_back(cb_out, 1);  // RISC poll on c_16 acked
            pack_tile(0, cb_out);        // queued MOP sequence
            cb_push_back(cb_out, 1);     // queued SETDMAREG/STALLWAIT(PACK)/STOREREG
            tile_regs_release();         // queued dest-section done
            sfpu_done++;
            hdr[5] = sfpu_done;
        } else {
            // ---- RVV tile, ordinal r = rvv_done ----
            uint32_t r = rvv_done;
            hyb_in_wait(cb_rvv_in, my_acked_c1, 1);
            invalidate_l1_cache();  // BH: L1 data reads after MMIO count poll
            hyb_out_reserve(cb_rvv_out, rvv_out_received, 1);
            uint32_t src = hyb_tile_addr(cb_rvv_in, r);
            uint32_t dst = hyb_tile_addr(cb_rvv_out, r);
            hyb_rvv_eval_tile((const float*)src, (float*)dst);
            hyb_in_pop(cb_rvv_in, my_acked_c1, 1);  // after the tile's last RVV load
            hyb_out_push(cb_rvv_out, rvv_out_received, dst, 1);
            rvv_done++;
            hdr[6] = rvv_done;
            hdr[7] = i;
        }
    }
    hdr[11] = 0x40;  // breadcrumb: loop done

    uint64_t t_last = get_timestamp();
    hdr[3] = (uint32_t)t_last;
    hdr[4] = (uint32_t)(t_last >> 32);
    hdr[13] = (uint32_t)(t_last - t_first);
    hdr[11] = 0x50;        // breadcrumb: header complete
    hdr[0] = 0xC0FFEE20u;  // magic LAST
#endif  // TRISC_PACK
}
