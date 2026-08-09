# SFPU Optimization Story — Design Spec

**Status:** approved design (2026-06-20)
**Location:** `tt_metal/programming_examples/generic_lut_activation_embedded/tutorials/sfpu_optimization/`

## Purpose

A reusable, runnable, explainable artifact that teaches **how to optimize a Tensix SFPU
vector kernel from scratch and quantify each win**. It presents the evolution of a
piecewise-polynomial (and piecewise-rational) activation evaluator as a *ladder* of
minimal kernels, each adding exactly one optimization, with measured device timing and
static analysis showing the cumulative benefit of every step.

It is **pedagogical**, not production: the kernels are deliberately minimal and each
isolates one idea. The production kernels (`piecewise_generic_specialized.cpp` etc.)
combine all of these; this artifact unbundles them so each can be understood and measured
on its own.

## Audience

TT-Metal kernel engineers learning SFPU optimization. Assumes familiarity with the SFPU
model (vFloat/vInt 32-lane SIMD, `v_if`/`v_endif` predication, `dst_reg[]`, Horner
evaluation) at the level of the curriculum's hardware model.

## The benchmark problem (fixed across all rungs)

To make device-µs directly comparable, **every rung computes the identical function on the
identical input.** The benchmark is *synthetic but structured* so that every optimization
is both valid and exercised:

- **Act I (polynomial):** 16 segments, max degree 8, **odd parity** (even-index coeffs are
  exactly zero), with **mixed per-segment effective degrees** (several segments have
  trailing-zero high-order coeffs). Coefficients are drawn from a **fixed-seed RNG** so the
  artifact is deterministic and reproducible.
- **Act II (rational):** single-region (or few-region) n8d8, **odd numerator / even
  denominator** parity, fixed-seed random coeffs.
- **Shape:** 256 tiles (512×512), fp32, identical input tensor for all rungs.

Rationale for "structured random": parity is only *correct* on a genuinely odd/even
polynomial, and adaptive-degree only *wins* when some segments are reducible. A purely
random dense polynomial would make those rungs either wrong or no-ops. The construction is
documented in the paper as a deliberate choice — it exercises the full optimization set
rather than mimicking one real activation. (Real-activation numbers already live in the
parent example's sweeps; this artifact isolates the *mechanism*.)

## The rungs

### Act I — Polynomial

| rung | file | optimization | primary lever |
|------|------|--------------|---------------|
| P0 | `p0_naive.cpp` | Naive Horner: runtime loop over coeffs, runtime segment search | baseline |
| P1 | `p1_vectorized.cpp` | `vFloat` SIMD Horner (32 lanes), runtime coeff loop | 32-wide throughput |
| P2 | `p2_unrolled.cpp` | compile-time template/`constexpr` unrolled Horner + segment cascade | remove loop overhead, const-fold coeffs, avoid constprop spills |
| P3 | `p3_dual.cpp` | dual-eval: 2 `dst_reg` rows per iteration, shared coeff loads | ILP — hide SFPU pipeline latency |
| P4 | `p4_parity.cpp` | parity x²-Horner (stride-2 coefficient access) | ~½ the FMAs |
| P5 | `p5_adaptive.cpp` | adaptive per-segment degree (skip zero high-order coeffs) | fewer FMAs on reducible segments |

### Act II — Rational

| rung | file | optimization |
|------|------|--------------|
| R0 | `r0_naive.cpp` | separate numerator/denominator Horner + reciprocal inside each segment |
| R1 | `r1_unrolled.cpp` | compile-time unroll |
| R2 | `r2_interleaved.cpp` | interleaved num/den Horner (lockstep FMA chains, ILP) |
| R3 | `r3_parity.cpp` | parity x²-Horner (odd num / even den) |
| R4 | `r4_deferred.cpp` | deferred reciprocal — one `sfpu_reciprocal` outside all `v_if`s |

Each rung's source is a standalone compute kernel readable in isolation; the diff between
consecutive rungs is the single optimization being taught.

## Metrics (per rung)

**Measured (device):**
- Tracy `DEVICE` compute µs at the fixed shape, 3 runs → min.
- Per-step speedup (vs previous rung) and cumulative speedup (vs P0 / R0).

**Static analysis:**
- Analytical FMA count and reciprocal count, derived from degree/segments/parity (rigorous,
  always available).
- **Best-effort** real SFPU instruction count via `riscv-tt-elf-objdump` on the JIT-compiled
  TRISC `.o` (counts `sfp*` ops). Marked best-effort because `-flto` can complicate
  attribution; analytical FMA counts are the rigorous backbone if objdump is noisy.
- Qualitative live-register note per rung (why pressure rises/falls).

**Correctness gate (light):**
- Each rung's output must match P0 (resp. R0) within an fp tolerance. This validates the
  central claim — "same function, strictly faster" — and catches a rung that optimizes
  itself into being wrong.

## Architecture & reuse

The driver reuses existing infrastructure rather than reinventing it:
- **Host:** the existing adhoc host (`generic_lut_activation.cpp` + `KERNEL_VARIANT`
  define) runs any compute kernel placed in the adhoc slot.
- **Build/profile loop:** same pattern as `run_csv.sh` — swap the rung kernel into
  `kernels/compute/adhoc/adhoc.cpp`, `ninja` the adhoc target, run with
  `TT_METAL_DEVICE_PROFILER=1`, extract µs via
  `profiler_helpers.sh::extract_profiler_compute_time`.

Per-rung driver step:
1. `cp kernels/compute/<rung>.cpp` → adhoc slot (with the generated bench coeff header).
2. `ninja -C build_Release …_adhoc`.
3. Run at fixed shape under device profiler → µs (3×, min).
4. Dump output CSV → correctness check vs P0.
5. `objdump` the TRISC `.o` → SFPU instruction count (best-effort).
6. Append a row to `results.csv`.

New code is only: the rung kernels, `gen_bench.py`, `run_tutorial.sh`, and the paper.

## File layout

```
tutorials/sfpu_optimization/
  DESIGN.md                      # this spec
  SFPU_OPTIMIZATION_STORY.md     # the paper (numbers injected from results.csv)
  gen_bench.py                   # fixed-seed coeffs + reference values + bench header
  run_tutorial.sh                # driver (build + profile + static analysis + assemble)
  kernels/compute/
    p0_naive.cpp … p5_adaptive.cpp
    r0_naive.cpp … r4_deferred.cpp
  results.csv                    # generated: per-rung µs, speedup, FMA, instr count
  results.md                     # generated: rendered table
```

## The paper (`SFPU_OPTIMIZATION_STORY.md`)

Narrative structure:
1. **The problem** — evaluating piecewise activations on the SFPU; what makes it slow.
2. **SFPU primer** — 32-lane SIMD, predicated `v_if`, Horner, the cost model (FMAs × lanes,
   predication = all branches execute).
3. **Act I: the polynomial ladder** — one section per rung: the idea, the code delta, the
   measured win, the static-analysis "why."
4. **Act II: the rational ladder** — same treatment.
5. **Cumulative results table** — the headline (per-step and cumulative speedup), injected
   from `results.csv`.
6. **When each optimization applies** — a decision guide (parity needs odd/even; dual-eval
   trades registers; adaptive needs reducible degrees; deferred reciprocal needs rational).
7. **Reproduce it** — `./run_tutorial.sh`.

## Reproducibility

- `gen_bench.py` uses a hard-coded seed → identical coeffs every run.
- `run_tutorial.sh` regenerates `results.csv` and re-injects the paper's numbers, so the
  artifact is self-updating on new silicon/toolchain.

## Out of scope

- Range reduction (Cody-Waite exp/trig/log) — orthogonal to the eval-perf story; would
  muddy the ladder. May be a future Act III.
- Real-activation accuracy tuning — covered by the parent example's sweeps.
- Multi-core scaling — the production host already uses `split_work_to_cores`; the tutorial
  fixes the shape and focuses on per-element compute.

## Risks

- **objdump under -flto:** instruction attribution may be imperfect → metric marked
  best-effort; analytical FMA counts carry the rigor.
- **Rung isolation vs duplication:** rung kernels will share boilerplate (CB setup, segment
  search). Accept minor duplication in favor of each rung being independently readable — the
  whole point is per-rung clarity.
- **Device-serial driver:** all rungs profile on the exclusive device; the driver runs them
  sequentially (one process), like the existing sweeps.
