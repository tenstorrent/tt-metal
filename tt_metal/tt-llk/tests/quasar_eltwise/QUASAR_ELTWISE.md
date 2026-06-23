<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Quasar Eltwise Generic-LUT Flow

This document is the reference for the **Quasar eltwise generic-LUT activation
flow**: the end-to-end pipeline that builds a per-eval-method SFPU LUT kernel,
compiles it with the qsr32 SFPI toolchain, runs it on the **craq-sim Quasar
`ttsim`**, and PCC/ULP-compares the on-sim output against the
tt-polynomial-fitter ground-truth golden.

It covers the existing methods (`polynomial`, `rational`) and the three eval
methods ported into the Quasar tree (`exponent_alu`, `newton_root`,
`parity`/`adaptive`), with per-method compile/run/gap status, the
Quasar-specific SFPU adaptation notes, and the **pinned, validated sim build**.

## Pinned, validated simulator (load-bearing)

All status in this document was validated against this exact sim build. Pin it.

| Item                | Value                                                                              |
|---------------------|------------------------------------------------------------------------------------|
| craq-sim repo       | `/localdev/nkapre/craq-sim-quasar`                                                  |
| SHA                 | `3ed587aa25499116dfb1a88eb0c04686cabd029b` (short `3ed587aa`)                       |
| Branch              | `quasar` (tracks `remotes/origin/quasar`)                                           |
| Pinned sim `.so`    | `/localdev/nkapre/craq-sim-quasar/src/_out/release_qsr/libttsim.so`                 |
| SOC descriptor      | `tt_metal/soc_descriptors/quasar_32_arch.yaml`                                      |

This is **the validated sim build** for the flow. The notes below (esp. the
`SFPEXMAN` / `SFPADDEXP` / `SFPADDI` gaps) are properties of *this* `ttsim`
build; a different SHA may behave differently.

## The flow (4 stages)

The `tt-llk` pytest harness owns the actual build + on-sim execution
(`tt_metal/tt-llk/tests/python_tests/helpers/test_config.py` + `conftest.py`).
The `tt_metal/tt-llk/tests/quasar_eltwise/` driver scripts are a thin, documented
layer that wires the pinned sim and the per-method env contract on top of that
harness.

```
1. build SFPI         qsr32 SFPI toolchain (runtime/sfpi), per-TRISC compile
                      of the selected Quasar test source (3 #ifdef sections:
                      LLK_TRISC_UNPACK / _MATH / _PACK; only _MATH evaluates
                      the LUT and lands params.h + the GENERIC_LUT_DATA header).
        |
2. compile per method  map eval_method -> Quasar test source; bake the fitter
                      coefficients / metadata into build.h via the
                      GENERIC_LUT_DATA TemplateParameter.
        |
3. run under craq-sim  pytest --run-simulator with the pinned libttsim.so,
                      TT_METAL_SLOW_DISPATCH_MODE=1, CHIP_ARCH=quasar. Models
                      the eltwise_unary_sfpu recipe:
                      UNPACK -> SrcA -> FPU datacopy (MOVA2D) -> Dest ->
                      SFPU(embedded LUT) -> PACK. The SFPU evaluator processes
                      2 Dest rows per iteration (SFP_ROWS / _incr_counters_),
                      not dst_reg[0..31].
        |
4. golden compare     golden = tt-polynomial-fitter ground_truth over the FULL
                      original domain (faithful reduce+reconstruct when range
                      reduction is enabled). Metric: calculate_pcc + binade ULP.
                      Gate: PCC >= 0.99 in ground_truth mode.
```

### Single entry point: `run_quasar.sh`

```
./run_quasar.sh [-m EVAL_METHOD] [-a ACTIVATION] [-c CSV] [--compile-only] [--no-compile] [-t PCC]
```

For one `(activation, eval_method)` it: builds the config (maps the method to
its Quasar test source + per-method env contract), runs the SFPI compile-only
sanity gate (`compile_llk_quasar.sh`), runs the test under the pinned sim, and
parses/gates the PCC/ULP the test prints. See `README.md` for the full option
table; `compile_llk_quasar.sh [eval_method] [activation]` is the standalone
compile-only gate (no device/sim).

```
./run_quasar.sh -m polynomial  -a gelu
./run_quasar.sh -m rational    -a atanh -c <atanh_n8d8_s1_..._rational.csv>
./run_quasar.sh -m newton_root -a sqrt
./run_quasar.sh -m parity      -a tanh  --compile-only
```

### Pinned environment

```
TT_METAL_HOME=/localdev/nkapre/tt-metal
TT_METAL_SIMULATOR=/localdev/nkapre/craq-sim-quasar/src/_out/release_qsr/libttsim.so
TT_METAL_SLOW_DISPATCH_MODE=1   CHIP_ARCH=quasar     # forced by run_quasar.sh
```

## Eval-method coverage

| eval_method     | Quasar test source                          | CSV env var      | compiles | runs on sim | status |
|-----------------|---------------------------------------------|------------------|----------|-------------|--------|
| `polynomial`    | `generic_lut_activation_quasar_test.cpp`    | `QUASAR_LUT_CSV` | yes      | yes         | existing — piecewise-polynomial (degree N / segments S), in-cascade reduce+reconstruct RR methods 1-8 (exp/log/cbrt/trig/tan + expalu_exp2/log2/pow) |
| `rational`      | `generic_lut_rational_quasar_test.cpp`      | `QUASAR_LUT_CSV` | yes      | yes         | existing — P(x)/Q(x), iterative approx_recip + 2 Newton rounds, deferred reciprocal |
| `exponent_alu`  | `generic_lut_expalu_quasar_test.cpp`        | `QUASAR_LUT_CSV` | yes      | yes         | **ported** — standalone bit-decompose + single reduced-domain Horner that BYPASSES the segment cascade (exp/log/pow + sqrt/rsqrt/cbrt) |
| `newton_root`   | `generic_lut_newton_root_quasar_test.cpp`   | `QUASAR_NR_CSV`  | yes      | yes (sqrt, rsqrt) | **ported** — magic-seed + Newton/Householder. cbrt (N=3) branch written & compiles, not sim-validated (no newton_root cbrt CSV in fitter) |
| `parity`        | `generic_lut_parity_quasar_test.cpp`        | `QUASAR_LUT_CSV` | yes      | yes         | **ported** — polynomial parity x²-Horner (POLY_PARITY_ODD/EVEN) + adaptive per-segment degree (SEGMENT_DEGREES[]); orthogonal modifiers of POLY_CASCADE |

Sources: `tt_metal/tt-llk/tests/sources/quasar/`. Python goldens:
`tt_metal/tt-llk/tests/python_tests/quasar/`.

### Ported-method validation results (pinned sim)

The three `run_quasar.sh` default configs were re-validated from the relocated
`tt_metal/tt-llk/tests/quasar_eltwise/` driver against the pinned `3ed587aa` sim;
each runs **2 parametrized cases (fp32 + bf16), 0 skipped**:

| `run_quasar.sh` invocation       | default CSV                                 | fp32 PCC            | bf16 PCC           |
|----------------------------------|---------------------------------------------|---------------------|--------------------|
| `-m expalu` (`-a sigmoid`)       | `sigmoid_p5_s1_uniform_fpminimax_ulp.csv`   | 0.9999999962793374  | 0.9999957726870178 |
| `-m newton_root -a sqrt`         | `sqrt_p0_s1_uniform_fpminimax_ulp.csv`      | 0.9999999458124683  | 0.9999557093489042 |
| `-m parity -a tanh`              | algorithmic odd-parity LUT (deg [3,5,5,3])  | 0.9999999866691556  | 0.9999968783334164 |

`exponent_alu` (`generic_lut_expalu_quasar_test.cpp`) — all variants PASS vs
fitter ground_truth (fp32 / bf16 PCC):

| variant  | mode                              | fp32 PCC    | bf16 PCC    |
|----------|-----------------------------------|-------------|-------------|
| exp      | mode 1 (exp_p4_s1)                | 0.99999717  | 0.99999486  |
| sigmoid  | mode 1 + sigmoid compose          | 0.99999999  | 0.99999577  |
| log2     | mode 2 (log2_p4_s1)               | 0.99999996  | 0.99998683  |
| sqrt     | mode 3 root_n=2 (sqrt_p2_s1)      | 0.99999911  | 0.99995707  |
| cbrt     | mode 3 root_n=3 odd-root sign     | 0.99999999  | 0.99999527  |
| rsqrt    | mode 3 reciprocal path            | 0.99999999  | 0.99999996  |

`newton_root` (`generic_lut_newton_root_quasar_test.cpp`):

| variant | magic        | N | recip | iters | fp32 PCC      | bf16 PCC      | bf16 ULP |
|---------|--------------|---|-------|-------|---------------|---------------|----------|
| sqrt    | `0x5f1110a0` | 2 | no    | 2     | 0.99999996    | 0.99996       | max 1    |
| rsqrt   | `0x5f3759df` | 2 | yes   | 2     | 0.9999999991  | 0.9999999616  | max 1    |

`parity`/`adaptive` (`generic_lut_parity_quasar_test.cpp`) — 5/5 PASS:

| variant                                        | PCC                |
|------------------------------------------------|--------------------|
| default odd-parity LUT, fp32, deg [3,5,5,3]    | 0.9999999866691556 |
| default odd-parity LUT, bf16, deg [3,5,5,3]    | 0.9999968783334164 |
| EVEN-parity single-seg cos CSV, deg [4]        | 0.9999993731679687 |
| ODD-parity single-seg sin CSV, deg [5]         | 0.9999994620387233 |
| real fitter sin_p3_s8 (parity=none fallback)   | 0.9999999809522045 |

> **Metric note.** PCC ~1.0 is the load-bearing correctness signal. Raw fp32
> ULP explodes near roots / for tiny-magnitude outputs even when PCC ~1.0
> (project convention: bit-distance ULP near roots is a metric artifact, not a
> kernel error). bf16 max ULP (1 on the Newton roots, 54-55% bit-exact) is the
> meaningful tightness figure.

## Quasar SFPU adaptation notes

The qsr32 SFPU on this `ttsim` build differs from Blackhole; the ports required
these substitutions (all numerically exact or numerically identical, documented
inline in each source):

- **`SFPADDI` removed** (ttsim aborts `tensix_execute_sfpaddi`). The `-O3`
  instruction-combine pass folds `acc*x + c` (a separate SFPMUL/SFPMAD + an
  SFPADD of a round constant's SFPLOADI) into a single `SFPADDI` immediate.
  Every Horner step MUST be emitted as one **fused SFPMAD** via the
  `fma_const()` helper (`__builtin_rvtt_sfpmad` with the constant as the
  3rd/addend operand) so the backend never produces a standalone SFPADD to
  fold. Used in all ported sources (poly/rational already followed this). Build
  with `-mno-tt-tensix-optimize-combine` for the reciprocal composes.

- **`SFPADDEXP` / `SFPDIVP2` missing** (newton_root). The Blackhole reference
  uses `addexp(x, -1)` (exponent decrement = `x * 0.5`). On Quasar this op is
  absent, so it is replaced by the multiply `x * 0.5f`. `0.5` is an exact fp32
  value, so the result is **bit-identical** to the exponent decrement for all
  finite normals.

- **`SFPEXMAN` missing** (exponent_alu) — *contradicts the original task premise
  that SFPEXMAN was present.* The Blackhole `exp_hw_eval` uses
  `exman(ImplicitOne)` + `shft(Logical)` for the branch-free float→int
  decompose; ttsim aborts `MissingSpecification: tensix_execute_sfpexman`. The
  port re-derives the identical decompose with the sim-supported floor/round
  path (`SFP_STOCH_RND` + `setexp`-based ldexp) — the same toolkit the rational
  kernel uses (which already warns "Do NOT use exman"). Numerically identical;
  on real Quasar silicon the `exman` path could be restored for speed.
  `log_hw_eval` / `pow_hw_eval` never used exman.

- **`exman` (FractionOnly) avoided generally** — use `setexp(in, 127)` like the
  BH log kernel for the mantissa-in-[1,2) decompose.

- **int→float path** — the only Quasar int→float route is `vSMag`
  (`convert<vFloat>(convert<vSMag>(...))`); used for the cbrt N=3 int math.

All other intrinsics (`reinterpret`, `vUInt >> 1`, integer MAGIC-i subtract,
`setexp` / `exexp` / `setsgn`, `vConst1`, `exexp` decompose) exist on Quasar
sfpi and are used verbatim.

## Known gaps

- **newton_root cbrt (N=3)**: kernel branch fully written & compiles, but **not
  sim-validated** — no newton_root cbrt CSV exists in the fitter (all
  `cbrt_*.csv` use poly/rational range reduction, not newton_root metadata).
  Validate when such a CSV is produced.
- **parity x²-Horner triggers only for parity-constrained fits**
  (single-segment, origin-centered). Multi-segment fitter CSVs (e.g.
  `sin_p3_s8`) correctly detect `parity=none` and use the natural-basis Horner
  fallback (validated). This matches the embedded-kernel behavior.
- **Environment / pin drift (test runner only, not the kernels)**: the harness
  needs `tt-exalens==0.3.20` (`ParsedElfFile`) + `pytest-xdist==3.8.0`; the
  shared `python_env` has 0.3.21 (renamed to `ElfFile`), which blocks conftest
  import. The **canonical, portable** fix is the tt-llk test venv:

  ```bash
  bash tt_metal/tt-llk/tests/setup_external_testing_env.sh   # builds tests/.venv from requirements.txt (tt-exalens 0.3.20)
  ```

  Both `run_quasar.sh` and `quasar_sweep.sh` **auto-detect** `tests/.venv` first
  (then repo `python_env`, then `python3`) — no machine-specific or `/tmp` paths.
  Override with `VENV_PY=<interpreter>` if needed. The sim path defaults to the
  pinned craq-sim Quasar build (override `TT_METAL_SIMULATOR` per machine).

## Full corpus sweep (deployment validator)

`quasar_sweep.sh` is a **deployment validator**: by default (`--approximation best`) it
reads the fitter's TRUE deployed pick per activation from `best.csv`
(`best_<metric>_{fitting,degree,num_segments}`, metric=ulp) and runs the ONE shipping
config — rational if the deployed fitting is rational (or degree is `n/m`), else polynomial.
This is exactly what ships; no per-category guessing, no degeneracy/oversized guards.
The coeff CSV is found by a **tolerant glob** (the fitter's recorded config does not always
have an exactly-named CSV — e.g. `log` records `p8_s1` but only `log_p8_s2_*` exists, so we
match same-degree, fewest-segments-first). No env setup beyond the venv above:

```bash
cd tt_metal/tt-llk/tests/python_tests/quasar
./quasar_sweep.sh --activations all                 # DEFAULT = deployment sweep (best.csv), 60 activations
./quasar_sweep.sh --activations gelu,exp,tanh       # subset, one deployed config each
```

`--approximation polynomial|rational|both` is an **opt-in comparison** mode: it reads the
per-CATEGORY pick from `best_polynomial.csv` / `best_rational.csv` (same tolerant glob, no
guards) and reports each category's real PCC — honest, not patched. Use it to compare a
forced poly vs. rational fit for an activation against its deployed `best` pick.

```bash
./quasar_sweep.sh --activations all --approximation both   # opt-in: poly + rational per-category comparison
```

Results (PCC/ULP per activation × mode × precision, threshold PCC≥0.99) → `/tmp/quasar_sweep_results.txt`. The specialized eval-methods (newton_root, expalu, parity) are validated per-op via `run_quasar.sh -m <method>` (the sweep covers the poly/rational fits that exist for every activation).

## File map

| File                                                                                       | Role |
|--------------------------------------------------------------------------------------------|------|
| `tt_metal/tt-llk/tests/quasar_eltwise/run_quasar.sh`                                        | single entry point (config → compile gate → sim run → golden compare) |
| `tt_metal/tt-llk/tests/quasar_eltwise/compile_llk_quasar.sh`                                | per-eval-method SFPI compile-only gate (no device/sim) |
| `tt_metal/tt-llk/tests/quasar_eltwise/README.md`                                            | usage / option tables / legacy-script deprecation |
| `tt_metal/tt-llk/tests/quasar_eltwise/QUASAR_ELTWISE.md`                                    | this document |
| `tt_metal/tt-llk/tests/sources/quasar/generic_lut_activation_quasar_test.cpp`              | polynomial cascade + RR methods 1-8 (existing) |
| `tt_metal/tt-llk/tests/sources/quasar/generic_lut_rational_quasar_test.cpp`                | rational P(x)/Q(x) (existing) |
| `tt_metal/tt-llk/tests/sources/quasar/generic_lut_expalu_quasar_test.cpp`                  | exponent-ALU standalone evaluator (ported) |
| `tt_metal/tt-llk/tests/sources/quasar/generic_lut_newton_root_quasar_test.cpp`             | Newton-root magic-seed (ported) |
| `tt_metal/tt-llk/tests/sources/quasar/generic_lut_parity_quasar_test.cpp`                  | parity x²-Horner + adaptive degree (ported) |
| `tt_metal/tt-llk/tests/python_tests/quasar/test_generic_lut_*_quasar.py`                   | per-method python goldens (fitter ground_truth, PCC/ULP) |
| `tt_metal/tt-llk/tests/python_tests/quasar/quasar_sweep.sh`                                 | deployment validator: runs the fitter's deployed pick (best.csv) per activation; `--approximation both` for opt-in poly/rational comparison |

## Dataflow Buffer (DFB) status — why slow-dispatch / tt-llk, and the path to DFB

**Why this flow uses the tt-llk standalone path (not the tt-metal host dispatch):**
Quasar replaces circular buffers with **Dataflow Buffers (DFBs)** — the Metal-2.0 mechanism
(`tt_metal/api/.../experimental/metal2_host_api/dataflow_buffer_spec.hpp`; cf.
`circular_buffer_constants.h`: *"TEMPORARY ... will be replaced by Dataflow Buffers (DFBs)"*).
Quasar **drops legacy CB support**: a host program using the CB-based `DataMovementKernel`
is rejected — *"DataMovementKernel is not supported on Quasar. Use `QuasarDataMovementKernel`"*
(`kernel.hpp:340`). The `generic_lut_activation` host program is CB-based, so the full
tt-metal host-dispatch path is a dead end on Quasar without a Metal-2.0 rewrite.

The **tt-llk flow has no host-dispatch layer at all** — `compile_llk_quasar.sh` compiles the
3 TRISC kernels directly with SFPI and the emulation runner executes them on the sim
(slow-dispatch). It never touches CBs *or* DFBs, so it sidesteps the CB→DFB transition and
**works today**. That is why all validation here runs `TT_METAL_SLOW_DISPATCH_MODE=1`.

**The DFB attempt that exists (and its blocker):**
A Metal-2.0 ProgramSpec/DFB eltwise path *was* prototyped — `git 71c9425e14f`
*"Milestone 1: Quasar eltwise binary ADD proof"* on branch **`origin/dchen/binary_quasar`**
(uses `experimental::quasar::QuasarDataMovementKernel` + the DFB/ProgramSpec API). Per its
commit message it **compiles, links, and runs end-to-end on the craq-sim Quasar — but the
output reads back as zero** (a functional bug in the single-tile Quasar DFB add path). It was
later **cleaned up on that branch and never merged to `origin/main`**.

So it is **not** "DFB doesn't run on craq-sim" — the sim *does* have DFB emulation
(`tt_metal/impl/emulation/emulated_program_runner.cpp` handles `QuasarDataMovementKernel`,
one thread per DM processor 0..7). The blocker is the **zero-output functional bug** in the
single-tile DFB path. (Root-cause investigation in progress; this note will be updated.)

**Path to a DFB-backed flow:**
1. Root-cause + fix the single-tile DFB zero-output bug (trace QuasarDataMovementKernel → DFB
   alloc/bind → compute → readback).
2. Port the eltwise-LUT host driver onto the Metal-2.0 DFB API (`program_spec` +
   `dataflow_buffer_spec` + `QuasarDataMovementKernel`).
3. Re-run the same `quasar_sweep.sh --activations all` on the DFB path and confirm parity
   with the tt-llk slow-dispatch numbers (60/60).
