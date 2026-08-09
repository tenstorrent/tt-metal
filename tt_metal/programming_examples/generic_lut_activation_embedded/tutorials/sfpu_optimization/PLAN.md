# SFPU Optimization Story — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, explainable tutorial artifact that teaches SFPU vector-kernel optimization as a ladder of minimal kernels (naive Horner → adaptive degree, then a rational Act II), quantifying each step's win with device-µs + static analysis.

**Architecture:** Each "rung" is a self-contained compute kernel that computes one fixed benchmark function. A bash driver swaps each rung into the existing adhoc kernel slot, builds it, profiles it under Tracy, checks its output against the P0/R0 baseline, and records µs + instruction counts to `results.csv`. A markdown paper narrates the ladder with the generated numbers injected.

**Tech Stack:** Tenstorrent SFPU C++ kernels (sfpi: `vFloat`/`vInt`, `v_if`, `dst_reg[]`), the existing adhoc host (`generic_lut_activation.cpp` + `KERNEL_VARIANT`), `run_csv.sh`-style Tracy profiling (`extract_profiler_compute_time`), Python (numpy) for bench generation, bash for the driver.

**Spec:** `tutorials/sfpu_optimization/DESIGN.md`

**Working dir for all paths:** `tt_metal/programming_examples/generic_lut_activation_embedded/` (call it `$EX`). Run device commands from repo root with `TT_POLY_FIT_DIR` unset (this artifact is self-contained — it does not need tt-polynomial-fitter).

---

## File Structure

```
$EX/tutorials/sfpu_optimization/
  DESIGN.md                    # exists
  PLAN.md                      # this file
  gen_bench.py                 # T1 — deterministic coeffs + reference + C++ header
  kernels/common/bench_lut.h   # T1 — generated; constexpr LUT + reference metadata
  kernels/compute/p0_naive.cpp … p5_adaptive.cpp   # T3–T8
  kernels/compute/r0_naive.cpp … r4_deferred.cpp   # T9–T13
  run_tutorial.sh              # T2 — driver
  lib/static_analysis.py       # T2 — analytical FMA counts + objdump parsing
  results.csv                  # generated
  SFPU_OPTIMIZATION_STORY.md   # T14 — the paper
```

**Reuse, do not modify:** the adhoc target `programming_examples_generic_lut_activation_embedded_adhoc`, its host `generic_lut_activation.cpp`, and `profiler_helpers.sh`. The driver places a rung's kernel at `$EX/kernels/compute/adhoc/adhoc.cpp` (the slot the adhoc target compiles), exactly as `run_csv.sh` does.

**Interface contract (every rung kernel must satisfy):** a rung kernel is a drop-in replacement for `adhoc/adhoc.cpp`. Study the current generated `adhoc/adhoc.cpp` (produced by any `run_csv.sh` run) for the exact required shape: the `#include`s, the `namespace sfpi`, the `void <name>()` SFPU entry expected by `KERNEL_VARIANT="adhoc/adhoc"`, and how the LUT `std::array<float, N>` is declared and indexed. Rung kernels differ ONLY in the eval body; the host/CB/LLK scaffolding is identical and copied verbatim from a known-good generated `adhoc.cpp`.

---

## Task 1: Deterministic benchmark generator

**Files:**
- Create: `$EX/tutorials/sfpu_optimization/gen_bench.py`
- Create (generated): `$EX/tutorials/sfpu_optimization/kernels/common/bench_lut.h`
- Create (generated): `$EX/tutorials/sfpu_optimization/bench_reference.csv`

- [ ] **Step 1: Write `gen_bench.py`**

Deterministic, no device. Produces the Act I polynomial benchmark and a reference. Key requirements:
- `numpy` with a hard-coded seed (`np.random.default_rng(20260620)`).
- Act I poly: `NUM_SEG=16`, `MAX_DEG=8`, **odd parity** (even-index coeffs forced to 0.0), **mixed effective degree** (for ~half the segments, zero out coeffs above a per-segment cap drawn from {3,5,7}). Uniform segmentation over `[-8, 8]`.
- Emit `bench_lut.h` containing, OUTSIDE any namespace:
  - `constexpr uint32_t BENCH_NUM_SEGMENTS = 16;`
  - `constexpr uint32_t BENCH_MAX_DEGREE = 8;`
  - `constexpr float BENCH_RANGE_MIN = -8.0f;` / `BENCH_RANGE_MAX = 8.0f;`
  - `constexpr uint32_t BENCH_SEGMENT_DEGREES[16] = {…};` (per-segment effective degree)
  - `constexpr std::array<float, LUT_SIZE> BENCH_LUT = {…};` in the SAME layout the production adhoc LUT uses (boundaries block then per-segment coeff blocks of `MAX_DEGREE+1`). Confirm the exact layout by reading a generated `adhoc/adhoc.cpp`.
- Emit `bench_reference.csv` with `input,output` for a dense sweep over `[-8,8]` (262144 fp32 points), where `output` is the exact piecewise polynomial evaluated in fp64 (the function the kernels approximate — for the tutorial the "reference" IS the polynomial, so all rungs must match it and each other).

```python
#!/usr/bin/env python3
"""Deterministic SFPU-tutorial benchmark: a fixed odd-parity, mixed-degree
piecewise polynomial. No device, no external deps beyond numpy."""
import numpy as np, argparse, os
SEED=20260620; NUM_SEG=16; MAX_DEG=8; LO,HI=-8.0,8.0
def build():
    rng=np.random.default_rng(SEED)
    bounds=np.linspace(LO,HI,NUM_SEG+1)
    seg_deg=rng.choice([3,5,7,8],size=NUM_SEG)            # mixed effective degree
    coeffs=np.zeros((NUM_SEG,MAX_DEG+1))
    for s in range(NUM_SEG):
        for d in range(seg_deg[s]+1):
            if d%2==0: continue                            # odd parity: even coeffs = 0
            coeffs[s,d]=rng.uniform(-0.5,0.5)/(d+1)
    return bounds,seg_deg,coeffs
def lut_flat(bounds,coeffs):
    # layout: [NUM_SEG+1 boundaries][seg0 c0..cMAX][seg1 ...]  (match adhoc.cpp)
    flat=list(bounds)
    for s in range(NUM_SEG): flat+=list(coeffs[s])
    return flat
def emit_header(path,bounds,seg_deg,coeffs):
    flat=lut_flat(bounds,coeffs); n=len(flat)
    with open(path,"w") as f:
        f.write("// GENERATED by gen_bench.py — do not edit\n#pragma once\n#include <array>\n#include <cstdint>\n")
        f.write(f"constexpr uint32_t BENCH_NUM_SEGMENTS={NUM_SEG};\n")
        f.write(f"constexpr uint32_t BENCH_MAX_DEGREE={MAX_DEG};\n")
        f.write(f"constexpr float BENCH_RANGE_MIN={LO}f;\nconstexpr float BENCH_RANGE_MAX={HI}f;\n")
        f.write("constexpr uint32_t BENCH_SEGMENT_DEGREES[%d]={%s};\n"%(NUM_SEG,",".join(map(str,seg_deg))))
        f.write("constexpr std::array<float,%d> BENCH_LUT={{%s}};\n"%(n,",".join(f"{v:.9e}f" for v in flat)))
def emit_reference(path,bounds,coeffs):
    xs=np.linspace(LO,HI,262144)
    out=np.empty_like(xs)
    for i,x in enumerate(xs):
        s=min(np.searchsorted(bounds,x,side="right")-1,NUM_SEG-1); s=max(s,0)
        out[i]=sum(coeffs[s,d]*x**d for d in range(MAX_DEG+1))
    np.savetxt(path,np.c_[xs,out],delimiter=",",header="input,output",comments="",fmt="%.9e")
if __name__=="__main__":
    here=os.path.dirname(os.path.abspath(__file__))
    b,sd,c=build()
    os.makedirs(os.path.join(here,"kernels/common"),exist_ok=True)
    emit_header(os.path.join(here,"kernels/common/bench_lut.h"),b,sd,c)
    emit_reference(os.path.join(here,"bench_reference.csv"),b,c)
    print("wrote bench_lut.h + bench_reference.csv")
```

- [ ] **Step 2: Run it**

Run: `cd $EX/tutorials/sfpu_optimization && /usr/bin/python3 gen_bench.py`
Expected: `wrote bench_lut.h + bench_reference.csv`; `bench_lut.h` exists with a 16-entry `BENCH_SEGMENT_DEGREES` and a `BENCH_LUT` array.

- [ ] **Step 3: Validate the LUT layout against a real adhoc.cpp**

Run any `run_csv.sh ... --tiles 1` once, open the generated `$EX/kernels/compute/adhoc/adhoc.cpp`, and confirm `bench_lut.h`'s boundary/coeff ordering matches the production layout (boundaries first, then `MAX_DEGREE+1` coeffs per segment). Fix `lut_flat()` if the production layout differs (e.g. coeff-major). Re-run Step 2.

- [ ] **Step 4: Commit**

```bash
git add $EX/tutorials/sfpu_optimization/gen_bench.py $EX/tutorials/sfpu_optimization/kernels/common/bench_lut.h $EX/tutorials/sfpu_optimization/bench_reference.csv
git commit -m "sfpu-tutorial: deterministic benchmark generator + LUT/reference"
```

---

## Task 2: Driver + static-analysis helper

**Files:**
- Create: `$EX/tutorials/sfpu_optimization/lib/static_analysis.py`
- Create: `$EX/tutorials/sfpu_optimization/run_tutorial.sh`

- [ ] **Step 1: Write `lib/static_analysis.py`**

Two functions, no device:
- `fma_count(num_seg, seg_degrees, parity, dual)` → analytical FMA count per element for the piecewise eval (Horner: `degree` FMAs/segment; parity halves it via `ceil(degree/2)`; dual doesn't change per-element FMAs but is noted). Returns an int.
- `count_sfpu_insns(obj_path)` → best-effort: run `riscv-tt-elf-objdump -d <obj>` (compiler at `runtime/sfpi/compiler/bin/`), count mnemonics starting `sfp`. Return int or `None` if objdump/obj missing.

```python
import subprocess, os, math, re
SFPI="/localdev/nkapre/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
def fma_count(num_seg, seg_degrees, parity=False, dual=False):
    per_seg = (lambda d: math.ceil(d/2) if parity else d)
    return sum(per_seg(d) for d in seg_degrees)   # predicated cascade: all segments evaluated
def count_sfpu_insns(obj_path):
    if not (os.path.exists(SFPI) and os.path.exists(obj_path)): return None
    try:
        out=subprocess.run([SFPI,"-d",obj_path],capture_output=True,text=True,timeout=60).stdout
    except Exception: return None
    return len(re.findall(r"\b(sfp[a-z0-9_]+)\b", out))
```

- [ ] **Step 2: Write `run_tutorial.sh`**

Driver. For a list of rungs (arg `--act poly|rational|all`), for each rung kernel: copy it (and ensure it `#include`s `bench_lut.h`) into the adhoc slot, build, profile at the fixed shape, dump output, check correctness vs `bench_reference.csv`, compute static analysis, append to `results.csv`.

```bash
#!/bin/bash
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
EX="$REPO/tt_metal/programming_examples/generic_lut_activation_embedded"
ADHOC_SLOT="$EX/kernels/compute/adhoc/adhoc.cpp"
BIN="$REPO/build_Release/programming_examples/programming_examples_generic_lut_activation_embedded_adhoc"
source "$EX/profiler_helpers.sh"
SYSPY=/usr/bin/python3
SHAPE_TILES=256
OUT="$HERE/results.csv"
ACT="${1:-all}"; [[ "$1" == "--act" ]] && ACT="$2"

/usr/bin/python3 "$HERE/gen_bench.py"
echo "rung,act,us,fma,sfpu_insns,max_abs_err,status" > "$OUT"

POLY="p0_naive p1_vectorized p2_unrolled p3_dual p4_parity p5_adaptive"
RAT="r0_naive r1_unrolled r2_interleaved r3_parity r4_deferred"
case "$ACT" in poly) RUNGS="$POLY";; rational) RUNGS="$RAT";; *) RUNGS="$POLY $RAT";; esac

for rung in $RUNGS; do
  src="$HERE/kernels/compute/${rung}.cpp"
  [[ ! -f "$src" ]] && { echo "$rung,,,,,,MISSING_KERNEL" >> "$OUT"; continue; }
  cp "$src" "$ADHOC_SLOT"
  ninja -C "$REPO/build_Release" programming_examples_generic_lut_activation_embedded_adhoc >/dev/null 2>&1 \
    || { echo "$rung,,,,,,BUILD_FAIL" >> "$OUT"; continue; }
  best=999999; dump="/tmp/sfputut_${rung}.csv"
  for run in 1 2 3; do
    pd="/tmp/sfputut_prof_${rung}_${run}"; mkdir -p "$pd"
    [[ $run -eq 1 ]] && DUMP="$dump" || DUMP=/dev/null
    DUMP_OUTPUT_CSV="$DUMP" TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR="$pd" \
      "$BIN" --activation tutorial --precision fp32 \
      --range-min -8 --range-max 8 --tiles $SHAPE_TILES >/dev/null 2>&1
    t=$(extract_profiler_compute_time "$pd/.logs/profile_log_device.csv" "$EX" 2>/dev/null)
    [[ -n "$t" && "$t" != "0" ]] && (( $(echo "$t < $best"|bc -l) )) && best="$t"
  done
  obj=$(ls -t "$REPO"/build_Release/../**/kernels/**/trisc1/*.o 2>/dev/null | head -1)  # adjust glob to JIT cache
  read fma insns err <<<"$("$SYSPY" "$HERE/lib/score.py" "$rung" "$dump" "$obj")"
  st="OK"; (( $(echo "$err > 1e-2"|bc -l 2>/dev/null||echo 0) )) && st="ACCURACY_FAIL"
  echo "$rung,$ACT,$best,$fma,$insns,$err,$st" >> "$OUT"
done
echo "results -> $OUT"; column -t -s, "$OUT"
```

- [ ] **Step 3: Write `lib/score.py`** (correctness vs reference + static analysis glue)

```python
import sys, csv, math
sys.path.insert(0, __file__.rsplit("/",1)[0])
from static_analysis import fma_count, count_sfpu_insns
rung, dump, obj = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv)>3 else "")
# reference
ref={}
for a,b in list(csv.reader(open(__file__.rsplit("/",2)[0]+"/bench_reference.csv")))[1:]:
    ref[round(float(a),5)]=float(b)
err=0.0
try:
    for a,b in list(csv.reader(open(dump)))[1:]:
        x=round(float(a),5)
        if x in ref: err=max(err, abs(float(b)-ref[x]))
except Exception: err=9.99
parity = "parity" in rung or "interleaved" in rung or rung[0]=="r"
# seg_degrees: import from gen_bench's choice — read bench_lut.h
import re
hdr=open(__file__.rsplit("/",2)[0]+"/kernels/common/bench_lut.h").read()
sd=[int(x) for x in re.search(r"BENCH_SEGMENT_DEGREES\[\d+\]=\{([^}]*)\}",hdr).group(1).split(",")]
adaptive = "adaptive" in rung
degs = sd if adaptive else [max(sd)]*len(sd)
fma = fma_count(len(degs), degs, parity="parity" in rung)
insns = count_sfpu_insns(obj) if obj else None
print(fma, insns if insns is not None else "NA", f"{err:.3e}")
```

- [ ] **Step 4: Make executable + dry-run the non-device parts**

```bash
chmod +x $EX/tutorials/sfpu_optimization/run_tutorial.sh
/usr/bin/python3 -c "import sys; sys.path.insert(0,'$EX/tutorials/sfpu_optimization/lib'); import static_analysis as s; print(s.fma_count(16,[8]*16))"
```
Expected: prints `128` (16 segments × 8 FMAs).

- [ ] **Step 5: Resolve the JIT `.o` glob**

The `obj=$(ls -t ...)` line is best-effort and MUST be fixed to the real JIT cache path. After T3 runs P0 once, locate the compiled TRISC object under `~/.cache/tt-metal-cache/.../adhoc/.../trisc1/*.o` and update the glob in `run_tutorial.sh`. If it can't be pinned reliably, set `insns=NA` and rely on analytical FMA counts (acceptable per spec).

- [ ] **Step 6: Commit**

```bash
git add $EX/tutorials/sfpu_optimization/run_tutorial.sh $EX/tutorials/sfpu_optimization/lib/
git commit -m "sfpu-tutorial: driver + static-analysis/correctness scoring"
```

---

## Task 3: Rung P0 — naive Horner (baseline + wiring proof)

**Files:**
- Create: `$EX/tutorials/sfpu_optimization/kernels/compute/p0_naive.cpp`

- [ ] **Step 1: Obtain the known-good kernel scaffold**

Run `run_csv.sh` once on any coeff CSV with `--tiles 1`, then copy the generated `$EX/kernels/compute/adhoc/adhoc.cpp` to `kernels/compute/p0_naive.cpp` as the scaffold. Replace its embedded LUT with `#include "../common/bench_lut.h"` and its eval body with the naive implementation below. Keep ALL CB/LLK/host-facing scaffolding identical (this is the contract from the File Structure section).

- [ ] **Step 2: Write the naive eval body**

Inside the SFPU entry, over `for (int d=0; d<32; d++)` DST rows, naive = runtime loops, no unroll, no parity:

```cpp
// p0_naive: runtime segment search + runtime Horner loop (the baseline)
for (int i = 0; i < 32; i++) {
    vFloat x = dst_reg[i];
    // runtime linear segment search (predicated)
    vFloat result = 0.0f;
    for (uint32_t s = 0; s < BENCH_NUM_SEGMENTS; s++) {
        v_if (x >= BENCH_LUT[s]) {
            // runtime Horner over MAX_DEGREE
            const uint32_t base = (BENCH_NUM_SEGMENTS + 1) + s * (BENCH_MAX_DEGREE + 1);
            vFloat acc = BENCH_LUT[base + BENCH_MAX_DEGREE];
            for (int d = (int)BENCH_MAX_DEGREE - 1; d >= 0; d--)
                acc = acc * x + BENCH_LUT[base + d];
            result = acc;
        }
        v_endif;
    }
    dst_reg[i] = result;
}
```

- [ ] **Step 3: Run P0 via the driver (device)**

Run: `cd $REPO && $EX/tutorials/sfpu_optimization/run_tutorial.sh --act poly` (it will attempt all poly rungs; only P0 exists, others log MISSING_KERNEL).
Expected: `results.csv` has a `p0_naive` row with a real `us`, `fma=128`, `max_abs_err < 1e-2`, `status=OK`. If `ACCURACY_FAIL`, debug the eval/LUT-layout until P0 matches `bench_reference.csv` (this is the correctness anchor for all later rungs).

- [ ] **Step 4: Commit**

```bash
git add $EX/tutorials/sfpu_optimization/kernels/compute/p0_naive.cpp
git commit -m "sfpu-tutorial: P0 naive Horner baseline (+ verified vs reference)"
```

---

## Tasks 4–8: Polynomial rungs P1–P5

Each task: **copy P0's scaffold verbatim**, replace ONLY the eval body with the version below, run the driver, confirm `status=OK` (matches reference) and record the µs, commit. The optimization delta is exactly the eval body shown.

### Task 4: P1 — vectorized Horner
**File:** `kernels/compute/p1_vectorized.cpp`
- [ ] **Step 1: Eval body** — identical structure to P0 but make explicit that the Horner accumulator is a 32-lane `vFloat` (P0 already is, on SFPU). The teaching delta vs a hypothetical scalar baseline: one `vFloat` Horner step processes 32 elements. (If P0 is already vectorized on SFPU, P1's role is to *name and measure* the SIMD width; keep the body identical to P0 and document that the SFPU is inherently 32-wide — the scalar baseline is conceptual. Record P1 = P0 µs and explain in the paper.) Confirm with the implementer whether a genuinely scalar baseline is feasible; if not, fold P0+P1 into one rung and renumber. **Decision point flagged for execution.**
- [ ] **Step 2:** Run driver, confirm OK. **Step 3:** Commit.

### Task 5: P2 — compile-time unrolled
**File:** `kernels/compute/p2_unrolled.cpp`
- [ ] **Step 1: Eval body** — replace runtime loops with `constexpr`/template recursion so the compiler emits a flat FMA chain and folds coefficient indices. Use a recursive `template<uint32_t D> eval_horner` and `template<uint32_t S> unroll_seg`, each `__attribute__((always_inline))`:

```cpp
template <int D> static inline vFloat horner(const float* c, vFloat x) {
    if constexpr (D < 0) return vFloat(0.0f);
    else return horner<D-1>(c, x) * x + c[D];   // note: build hi->lo; adjust to match P0 order
}
template <uint32_t S> static inline void seg(vFloat x, vFloat& r) {
    if constexpr (S < BENCH_NUM_SEGMENTS) {
        constexpr uint32_t base=(BENCH_NUM_SEGMENTS+1)+S*(BENCH_MAX_DEGREE+1);
        v_if (x >= BENCH_LUT[S]) { r = horner<BENCH_MAX_DEGREE>(&BENCH_LUT[base], x); } v_endif;
        seg<S+1>(x, r);
    }
}
// in the d-loop: vFloat r=0.0f; seg<0>(x,r); dst_reg[i]=r;
```
- [ ] **Step 2:** Run driver, confirm OK + record µs (expect win vs P0 from no loop overhead / no constprop spills). **Step 3:** Commit.

### Task 6: P3 — dual-eval (ILP)
**File:** `kernels/compute/p3_dual.cpp`
- [ ] **Step 1: Eval body** — process two DST rows per iteration sharing each coefficient load, two independent Horner chains interleaved. Loop `for (int i=0;i<32;i+=2)`, evaluate `x0=dst_reg[i]`, `x1=dst_reg[i+1]` with a dual template that loads `c[D]` once and applies to both accumulators:

```cpp
template <int D> static inline void horner2(const float* c, vFloat x0, vFloat x1, vFloat& a0, vFloat& a1) {
    if constexpr (D < 0) { a0=vFloat(0.0f); a1=vFloat(0.0f); }
    else { horner2<D-1>(c,x0,x1,a0,a1); vFloat cd=c[D]; a0=a0*x0+cd; a1=a1*x1+cd; }
}
```
- [ ] **Step 2:** Run driver, confirm OK + record (expect ILP win). **Step 3:** Commit.

### Task 7: P4 — parity x²-Horner
**File:** `kernels/compute/p4_parity.cpp`
- [ ] **Step 1: Eval body** — the benchmark is odd, so evaluate in x² basis: `P(x) = x*(c1 + c3*x² + c5*x⁴ + …)`. Precompute `x2=x*x` once per DST row, Horner over odd coeffs with stride 2, multiply by `x` at the end. Build on P3's dual structure (parity + dual together):

```cpp
// odd parity: coeffs c1,c3,...; eval Q(x2) then result = x*Q(x2)
template <int K> static inline vFloat horner_odd(const float* c, vFloat x2) { // K = top odd index
    if constexpr (K < 1) return vFloat(0.0f);
    else return horner_odd<K-2>(c, x2) * x2 + c[K];
}
// result = x * horner_odd<MAXODD>(base, x2);
```
- [ ] **Step 2:** Run driver, confirm OK (parity output must still match the odd reference) + record (~½ FMAs → `fma` column drops, µs drops). **Step 3:** Commit.

### Task 8: P5 — adaptive per-segment degree
**File:** `kernels/compute/p5_adaptive.cpp`
- [ ] **Step 1: Eval body** — use `BENCH_SEGMENT_DEGREES[S]` (constexpr) as the per-segment template degree so reducible segments skip zero high-order coeffs. Build on P4:

```cpp
template <uint32_t S> static inline void seg_adaptive(vFloat x, vFloat x2, vFloat& r) {
    if constexpr (S < BENCH_NUM_SEGMENTS) {
        constexpr uint32_t deg = BENCH_SEGMENT_DEGREES[S];     // constexpr -> valid template arg
        constexpr uint32_t base=(BENCH_NUM_SEGMENTS+1)+S*(BENCH_MAX_DEGREE+1);
        v_if (x >= BENCH_LUT[S]) { r = x * horner_odd<deg|1 ? deg : deg-1>(&BENCH_LUT[base], x2); } v_endif;
        seg_adaptive<S+1>(x, x2, r);
    }
}
```
(Adjust the top-odd-index expression to the correct odd degree ≤ `deg`.)
- [ ] **Step 2:** Run driver, confirm OK + record (fewer FMAs on reducible segments → further µs drop). **Step 3:** Commit.

---

## Task 9: Extend bench + Rung R0 — naive rational

**Files:**
- Modify: `gen_bench.py` (add `build_rational()` → append `BENCH_R_*` constants + rational reference to a second header `bench_rational_lut.h` / `bench_rational_reference.csv`)
- Create: `kernels/compute/r0_naive.cpp`

- [ ] **Step 1:** Extend `gen_bench.py` with an n8d8 odd-num/even-den rational (fixed seed, few segments), emit `kernels/common/bench_rational_lut.h` (num coeffs, den coeffs, boundaries) + `bench_rational_reference.csv` (fp64 P(x)/Q(x)). Run it.
- [ ] **Step 2:** Write `r0_naive.cpp` (scaffold from P0): per segment, runtime Horner for numerator and denominator separately, then `sfpu_reciprocal_iter<3>(den)` and multiply — reciprocal **inside** each `v_if`.
- [ ] **Step 3:** Run `run_tutorial.sh --act rational`; confirm `r0_naive` OK vs rational reference, record baseline µs. **Step 4:** Commit.

---

## Tasks 10–13: Rational rungs R1–R4

Each: copy R0 scaffold, replace eval body, run `--act rational`, confirm OK vs rational reference, record, commit.

- **Task 10 — R1 unrolled** (`r1_unrolled.cpp`): template-unroll the num/den Horners and segment cascade (same pattern as P2). Commit.
- **Task 11 — R2 interleaved** (`r2_interleaved.cpp`): evaluate numerator and denominator Horner chains **in lockstep** (alternate FMAs on the two independent chains) to hide SFPU latency — analogous to P3's dual but the two chains are num & den. Commit.
- **Task 12 — R3 parity** (`r3_parity.cpp`): odd numerator → x·(x² Horner); even denominator → x² Horner; stride-2 access. Commit.
- **Task 13 — R4 deferred reciprocal** (`r4_deferred.cpp`): return `P(x)` and `Q(x)` per segment, do **one** `sfpu_reciprocal_iter<3>` **outside** all `v_if`s, then `result = P * recip`. Record (saves N-1 reciprocals). Commit.

---

## Task 14: The paper

**Files:**
- Create: `$EX/tutorials/sfpu_optimization/SFPU_OPTIMIZATION_STORY.md`
- Create: `$EX/tutorials/sfpu_optimization/lib/inject_results.py`

- [ ] **Step 1:** Write `SFPU_OPTIMIZATION_STORY.md` following the spec's narrative: (1) the problem, (2) SFPU primer + cost model, (3) Act I ladder — one subsection per rung with the code delta, measured win, and the static-analysis "why", (4) Act II ladder, (5) cumulative results table (placeholder block `<!-- RESULTS:poly --> … <!-- /RESULTS -->`), (6) when-each-applies decision guide, (7) reproduce (`./run_tutorial.sh`).
- [ ] **Step 2:** Write `lib/inject_results.py` — read `results.csv`, render markdown tables (per-step + cumulative speedup), replace the `<!-- RESULTS:* -->` blocks in the paper. Run it.
- [ ] **Step 3:** Commit paper + injector.

---

## Task 15: Full run + finalize

- [ ] **Step 1:** `./run_tutorial.sh all` end-to-end on device → full `results.csv` (11 rungs, all `OK`).
- [ ] **Step 2:** `/usr/bin/python3 lib/inject_results.py` → numbers injected into the paper.
- [ ] **Step 3:** Sanity: cumulative speedup is monotonic-ish and every rung `status=OK` (correctness preserved). Note any rung that doesn't win and explain in the paper (e.g. P1==P0 if SFPU already 32-wide).
- [ ] **Step 4:** Commit `results.csv`, `results.md`, finalized paper. Optionally add a one-line pointer from the example's main `README.md` to `tutorials/sfpu_optimization/SFPU_OPTIMIZATION_STORY.md`.

```bash
git add $EX/tutorials/sfpu_optimization/
git commit -m "sfpu-tutorial: full results + finalized optimization-story paper"
```

---

## Notes for the implementer
- **Device is exclusive & serial** — the driver profiles rungs one at a time; never run two device processes.
- **JIT cache** — kernel edits are picked up by JIT; if a rung's results look stale, `rm -rf ~/.cache/tt-metal-cache`.
- **Correctness is the test** — there is no pytest here; each rung's gate is "max_abs_err vs the reference < 1e-2 AND output matches P0/R0 shape". A rung that gets faster but fails this is a bug, not a win.
- **Flagged decision (Task 4):** if a genuinely scalar (non-SIMD) SFPU baseline isn't practical, merge P0/P1 into a single "vectorized Horner" baseline and renumber the ladder; update the paper accordingly.
- **objdump (`insns`)** is best-effort; if it can't be pinned to the right `.o`, leave `NA` and rely on analytical FMA counts — the spec accepts this.
