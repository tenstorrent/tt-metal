# SFPU Vector Kernel Optimization — an Applicability Map

The usual kernel-optimization writeup is a ladder: "do A, then B, then C, get N× faster."
That hides the question an engineer actually has: **which optimization applies to *my* kernel,
and where does it break?** Every SFPU optimization here has a **bound** — almost always the
register file — and beyond that bound it doesn't just stop helping, it **crashes the compiler.**

So this artifact is a **measured phase diagram**: we sweep the parameter space (optimization ×
polynomial degree × segment count × parity) on real Blackhole silicon and record, for every
cell, whether it **compiles** (a register-spill ICE = the frontier), is **correct**, and how
**fast** it is. The result is a map you can read off: *for degree D, N segments, parity p — use
this; above here it spills; below here it's not worth it.*

> Everything is runnable and deterministic:
> `./run_tutorial.sh all` (the per-optimization rungs) and `./bounds_sweep.sh` +
> `python3 phase_diagram.py` (the phase diagrams). Fixed-seed benchmark (`gen_bench.py` /
> `gen_sweep.py`).

---

## 1. The SFPU cost model (why the bounds exist)

Three hardware facts drive everything:

- **32-lane SIMD.** One `vFloat` is 32 elements; one FMA does 32 multiply-adds. There's no
  scalar mode — a `vFloat` Horner step is already vectorized.
- **`v_if` is predication, not branching.** *All lanes execute both sides*, masked on write. So
  a segment cascade `for s: v_if(x>=b[s]) {…}` evaluates **every segment's polynomial for every
  element**. Cost ∝ `Σ(segments × degree)`, independent of which segment an element lands in.
- **The register file is small.** Hold too many live `vFloat`s and GCC's allocator spills — at
  `-O3 -flto` it doesn't spill gracefully, it **aborts with a reload ICE**. This is *the* bound
  on almost every optimization below.

Headline static metric: **live coefficient registers**. That number, not the FMA count,
decides whether a kernel compiles.

## 2. The optimizations (what each does)

| name | idea | live-reg cost |
|---|---|---|
| **cascade** | per-segment Horner inside each `v_if` (the baseline) | ~1 (acc) |
| **unroll** | compile-time template Horner (no loop overhead/spills) | same |
| **dual** | 2 DST rows/iter, independent chains → ILP hides latency | ~2× data regs |
| **parity** | odd/even function → x²-Horner, half the Horner length | **½ the coeffs** |
| **adaptive** | per-segment compile-time degree → skip zero coeffs | same |
| **blend** | cascade *selects* coeffs (moves); ONE Horner after → kills the `×segments` term | **degree+1 coeffs** |
| **deferred recip** (rational) | one reciprocal after the cascade, not per-segment | — |

`blend` is the structural win — it's the only one that attacks the dominant `O(segments×degree)`
term — but it's also the one that pays in **live coefficient registers**, so it's the one the
register file bounds hardest.

## 3. The phase diagram (measured, 256 tiles, fp32, Blackhole)

Cells are device µs; **`ICE`** = the register-spill compiler crash (the frontier).

**cascade — parity off / on** (the baseline; cheap on registers, fits everywhere, but pays the full `seg×deg`):

| deg\seg | 4 | 16 | 64 | | 4 | 16 | 64 |
|---|---|---|---|---|---|---|---|
| **2** | 4.2 | 12.7 | 47.1 | | 3.5 | 8.3 | 29.5 |
| **4** | 5.5 | 18.0 | 70.9 | | 4.1 | 11.8 | 43.6 |
| **8** | 8.2 | 28.6 | 134.1 | | 5.3 | 17.1 | 64.9 |
| **16** | 13.5 | 49.9 | 240.1 | | 7.9 | 27.8 | 125.0 |

**blend — parity off / on** (the structural win — and its register cliff):

| deg\seg | 4 | 16 | 64 | | 4 | 16 | 64 |
|---|---|---|---|---|---|---|---|
| **2** | 3.8 | 10.9 | 40.1 | | 3.3 | 7.2 | 25.7 |
| **4** | 4.8 | 14.6 | 54.4 | | 3.6 | 9.2 | 33.0 |
| **6** | **ICE** | **ICE** | **ICE** | | 4.0 | 11.2 | 40.3 |
| **8** | **ICE** | **ICE** | **ICE** | | 4.5 | 12.9 | 47.4 |
| **12** | **ICE** | **ICE** | **ICE** | | 5.5 | 16.6 | 61.8 |
| **16** | **ICE** | **ICE** | **ICE** | | **ICE** | **ICE** | **ICE** |

## 4. The discovered bounds

**Register-ICE frontier** (the smallest degree that spills the file):

| variant | parity off | parity on |
|---|---|---|
| cascade | fits to 16 | fits to 16 |
| dual | fits to 16 | fits to 16 |
| **blend** | **ICE at degree ≥ 6** (safe ≤ 4) | **ICE at degree ≥ 16** (safe ≤ 12) |
| blend + dual | ICE at degree ≥ 6 | ICE at degree ≥ 16 |

The mechanism is exactly the live-coefficient count: blend holds `degree+1` coeff registers;
parity halves that (odd terms only), so it **doubles the degree headroom** (4 → 12). Stacking
dual on blend doesn't change the frontier (the coeff registers dominate). *This is the bound an
engineer hits in practice — and why P5/parity blend fit while the non-parity / rational blend
crashed.*

**Blend break-even** (blend µs vs cascade µs): blend wins for **N ≥ 4 at every fitting degree**,
and the win **grows with both segments and degree** — from ~1.1× up to **1.55×** (parity, D12,
N64). The bigger the cascade (`seg×deg`), the more blend's select-then-evaluate saves.

### The decision rule (read straight off the map)
```
if function has parity (odd/even):
    if degree ≤ 12 and segments ≥ 4:  use BLEND  (best; up to 1.55× over cascade)
    else:                              use CASCADE+parity
else (no parity):
    if degree ≤ 4 and segments ≥ 4:   use BLEND
    else:                              use CASCADE   (blend ICEs at degree ≥ 6)
dual / blend+dual: not worth the register risk here — ILP gain < the spill risk.
```

## 5. One path through the map: the fixed-benchmark ladder

Walking a single representative cell (16 seg, deg 8, odd parity) optimization-by-optimization —
the classic "ladder" view — is just one column of the diagram:

| step | µs | cumulative |
|---|---|---|
| naive (runtime loops) | 85.1 | 1.0× |
| + unroll | 22.4 | 3.8× |
| + dual | 16.9 | 5.0× |
| + parity | 16.2 | 5.3× |
| + adaptive | 15.8 | 5.4× |
| **+ blend** | **12.4** | **6.8×** |

Lessons the ladder alone would mislead you on, but the map makes obvious:
- **Unroll is the giant** (3.8×): a naive SFPU kernel is overhead/spill-bound, not arithmetic-bound.
- **Parity/adaptive barely move the clock** (~1.03× each) — the kernel isn't FMA-bound; halving
  FMAs is the wrong lever. (Their real value, the map shows, is **register headroom for blend**.)
- **Blend is the structural win** — and only because parity kept it under the ICE frontier.

Rational is the analogous story (45.2 → 8.8µs, 5.2×): unroll → interleaved num/den → parity →
deferred reciprocal. Rational *blend* ICEs (numerator-odd + denominator-even = 9 live coeffs).

## 6. What does NOT help (measured negatives)

- **FMA-shaving** (Estrin, immediate-Horner, bf16-compute): dead. The map shows the kernel is
  cascade/register-bound, not arithmetic-bound — parity already proved halving FMAs buys ~3%.
- **Multi-tile DST batching**: predicted 1.3–1.8×, measured **~1%** — at 256 tiles the kernel is
  not dispatch-bound; the handshake is already hidden behind the eval.
- **SFPLUTFP32 hardware LUT**: a 6-band, degree-1, fp16, hardware-fixed-boundary instruction —
  err ~2000 on a degree-8 function. A genuine fast path for LUT-moldable activations
  (sigmoid/tanh), **not** a general piecewise-poly replacement.

## 7. Relationship to the production kernel

These rungs are a **teaching model**, not the shipping kernel. Production
(`piecewise_generic_specialized.cpp`, `piecewise_rational_specialized.cpp`) combines all the
*winning* techniques plus range reduction, asymptotic factoring, and dtype paths — and it
already carries the same register-frontier guard the map quantifies (e.g. parity not stacked on
dual at high degree). The map tells you **which technique to switch on for a given fit, and where
it will spill** — directly useful for tuning the production kernel per activation.

## 8. Reproduce

```bash
cd $TT_METAL_HOME/tt_metal/programming_examples/generic_lut_activation_embedded/tutorials/sfpu_optimization
./run_tutorial.sh all          # the per-optimization rungs on the fixed benchmark
./bounds_sweep.sh              # the full phase-diagram grid (device-serial)
python3 phase_diagram.py       # render bounds_phase_diagram.md
```
- `gen_bench.py` / `gen_sweep.py` — deterministic benchmark + per-cell kernel generation.
- `kernels/compute/p*.cpp`, `r*.cpp` — the optimization rungs (one idea each).
- `lib/score.py`, `lib/static_analysis.py` — correctness gate + FMA/register accounting.

Build prerequisites: the example's standard build (parent `README.md` → "Setup & Build").
