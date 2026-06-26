// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

/**
 * ============================================================================
 * Canonical eval_method taxonomy for the generic-LUT activation kernels.
 * ============================================================================
 *
 * This header is the SINGLE source of truth for "which evaluator owns the
 * approximation". The codegen (both tools/generate_adhoc_kernel.py and
 * run_csv.sh) emits exactly ONE EVAL_METHOD_* selector; the kernel dispatcher
 * reads that one tag and routes. This replaces the historically overloaded
 * RANGE_REDUCTION_* namespace, where reduce-then-poly, standalone
 * hardware-exponent-ALU evaluators, and the Newton-root evaluator (which fits
 * no polynomial at all) all shared one prefix and had to be disambiguated by
 * scattered #if chains.
 *
 *   EVAL_METHOD_POLY_CASCADE     - piecewise polynomial cascade (the default).
 *                                  Parity / dual-eval / adaptive-degree / blend
 *                                  are ORTHOGONAL modifiers selected inside the
 *                                  cascade family, not separate methods.
 *   EVAL_METHOD_RATIONAL_CASCADE - piecewise P(x)/Q(x) cascade (rational base
 *                                  kernel). Implied by including
 *                                  piecewise_rational.cpp.
 *   EVAL_METHOD_BLEND            - coefficient-blend variant of the cascade,
 *                                  AUTO-SELECTED by the dispatcher from a
 *                                  POLY_CASCADE config (never emitted directly).
 *   EVAL_METHOD_REDUCED_POLY     - Cody-Waite / mantissa reduce-then-poly
 *                                  (exp / trig / tan / log / cbrt). Range
 *                                  reduction now means ONLY this: reduce the
 *                                  argument, run the SAME poly cascade, expand.
 *                                  The specific reduction is carried by the
 *                                  REDUCE_* sub-tag.
 *   EVAL_METHOD_EXPONENT_ALU     - bit-decompose (exman/exexp/setexp) then a
 *                                  single reduced-domain Horner. STANDALONE:
 *                                  bypasses the segment cascade. The kind is
 *                                  carried by the EXPONENT_ALU_* sub-tag
 *                                  (EXP2 / LOG2 / POW).
 *   EVAL_METHOD_NEWTON_ROOT      - magic-seed + Newton recurrence, NO polynomial
 *                                  fit. FIRST-CLASS method (formerly nested as an
 *                                  "exponent_alu kind"), STANDALONE: bypasses the
 *                                  cascade and ignores the LUT entirely.
 *   EVAL_METHOD_AFFINE_COLLAPSE  - y = c0 + c1*x (one SFPMAD), or identity y = x.
 *   EVAL_METHOD_CLAMPED_AFFINE_COLLAPSE
 *                                - y = min(max(c0 + c1*x, low), high). Exact
 *                                  whole-function algebraic collapse detected
 *                                  from CSV coefficients; bypasses segment
 *                                  selection and Horner.
 *
 * Selection policy:
 *   1. Prefer whole-function collapses when the coefficient algebra proves the
 *      entire fit is identity, affine, or clamped-affine.
 *   2. Use standalone methods when the CSV metadata declares a non-cascade
 *      evaluator (exponent-ALU or Newton-root).
 *   3. Use reduced-poly when the metadata declares real range reduction.
 *   4. Otherwise use the poly/rational cascade; future per-segment lowering
 *      belongs inside that family, not as activation-name branches.
 *
 * Data-detail macros (coefficient arrays, magic constants, scales, degrees)
 * keep their established names (EXP_HW_COEFFS, NEWTON_ROOT_MAGIC, LOG_HW_SCALE,
 * ...) — those are payload, not routing, and renaming them buys nothing.
 *
 * BEHAVIOR-PRESERVING SHIM
 * ------------------------
 * The deep kernel bodies in piecewise_generic.cpp / piecewise_rational.cpp were
 * written against the legacy RANGE_REDUCTION_* feature macros. To guarantee
 * byte-identical numerics, this header DERIVES those legacy macros from the
 * clean selectors. New code (the dispatcher, the standalone-evaluator gate)
 * reads EVAL_METHOD_*; the legacy bodies keep compiling unchanged. There is no
 * numeric difference between the old and new tag — it is a pure rename + shim.
 */

#pragma once

// ----------------------------------------------------------------------------
// EXPONENT_ALU: clean kind sub-tags -> legacy standalone-evaluator feature macros
// ----------------------------------------------------------------------------
#if defined(EVAL_METHOD_EXPONENT_ALU)
#if defined(EXPONENT_ALU_EXP2)
#define RANGE_REDUCTION_EXP_HW
#elif defined(EXPONENT_ALU_LOG2)
#define RANGE_REDUCTION_LOG_HW
#elif defined(EXPONENT_ALU_POW)
#define RANGE_REDUCTION_POW_HW
#else
#error "EVAL_METHOD_EXPONENT_ALU requires one of EXPONENT_ALU_EXP2/LOG2/POW"
#endif
#endif

// ----------------------------------------------------------------------------
// NEWTON_ROOT: first-class standalone method -> legacy feature macro
// ----------------------------------------------------------------------------
#if defined(EVAL_METHOD_NEWTON_ROOT)
#define RANGE_REDUCTION_NEWTON_ROOT
#endif

// ----------------------------------------------------------------------------
// REDUCED_POLY: Cody-Waite / mantissa reduce-then-poly. The REDUCE_* sub-tag
// names the reduction; map to the legacy RANGE_REDUCTION_* the cascade uses.
// (These five genuinely ARE range reduction, so the legacy name is apt — we
// keep the umbrella selector EVAL_METHOD_REDUCED_POLY for explicit routing.)
// ----------------------------------------------------------------------------
#if defined(EVAL_METHOD_REDUCED_POLY)
#if defined(REDUCE_EXP)
#define RANGE_REDUCTION_EXP
#elif defined(REDUCE_TRIG)
#define RANGE_REDUCTION_TRIG
#elif defined(REDUCE_TAN)
#define RANGE_REDUCTION_TAN
#elif defined(REDUCE_LOG)
#define RANGE_REDUCTION_LOG
#elif defined(REDUCE_CBRT)
#define RANGE_REDUCTION_CBRT
#else
#error "EVAL_METHOD_REDUCED_POLY requires one of REDUCE_EXP/TRIG/TAN/LOG/CBRT"
#endif
#endif

// ----------------------------------------------------------------------------
// Convenience predicates for the dispatcher and the reduction-helper gate.
// ----------------------------------------------------------------------------

// Standalone evaluators own the whole approximation and bypass the cascade.
#if defined(EVAL_METHOD_EXPONENT_ALU) || defined(EVAL_METHOD_NEWTON_ROOT)
#define EVAL_METHOD_IS_STANDALONE 1
#else
#define EVAL_METHOD_IS_STANDALONE 0
#endif

// Any method that needs the range-reduction / exponent-decompose helpers
// (reduce_*, exp_hw_eval, log_hw_eval, pow_hw_eval, newton_root_eval) pulled in.
#if defined(EVAL_METHOD_REDUCED_POLY) || defined(EVAL_METHOD_EXPONENT_ALU) || defined(EVAL_METHOD_NEWTON_ROOT)
#define EVAL_METHOD_NEEDS_REDUCTION_HELPERS 1
#else
#define EVAL_METHOD_NEEDS_REDUCTION_HELPERS 0
#endif
