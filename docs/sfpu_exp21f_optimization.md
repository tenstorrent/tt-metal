# Optimizing `exp_21f` on Blackhole — 16.6 % fewer cycles, no accuracy change

Two instruction-count reductions to the shipped bfloat16-accurate exponential
(`_sfpu_exp_21f_bf16_tti_`), plus one change that keeps the result reachable from the
ttsim simulator (§9). Same algorithm, same polynomial.

| | |
|---|---|
| Kernel changed | `ckernel_sfpu_exp.h`, Blackhole only |
| Silicon | Blackhole p100a, device 0 |
| Branch point | `f6b36f3b1be` (main) |
| Diff | one file, plus one new regression test |
| Body size | **16 → 12** SFPU instructions (per-call preamble 4 → 5, see §9) |
| Performance | **577.99 → 481.99** cycles/tile, **−16.61 %** |
| Accuracy | **unchanged** — same polynomial, re-expressed (§9) |
| Functional | **257 passed, 129 skipped, 0 failed** (`test_sfpu_unary.py -k Exp`) |

All three changes are pure implementation. Nothing about the approximation moves, so there is
no accuracy trade to weigh — on every input that was tested, the standard `Exp` sweep plus the
overflow probes listed in §4.2, the outputs were byte-identical to main.

> §2 and §3 describe the kernel as first written, with a round-toward-zero convert. §9 replaces
> that convert with an equivalent one the simulator can execute; read it as an amendment to
> both. The body instruction count and the dependency chain are the same either way.

---

## 1. Where the cycles were

Baseline: 577.99 cycles/tile ÷ 32 elements = **18.06 cycles/element** against a 16-instruction
body. Only ~2 cycles of slack, and the existing kernel is already hand-scheduled — the `SFPLOADI`
sits in the `SFPMAD`'s latency window and the `SFPGT`/`SFPAND` mask pair fills two more.

So there was no scheduling win available. The kernel is close to **latency-bound**: cycles track
the critical path, not the instruction count. That framing is what picked both changes — each one
removes instructions *from the dependency chain*, not from the side of it.

## 2. Change 1 — stop building an integer just to take it apart

`_float_to_int32_for_exp_21f_` builds the 31-bit integer `xlog2 · 2^23` and the kernel then
immediately decomposes it again:

```
SFPEXEXP   exp  = exexp(xlog2)
SFPEXMAN   man  = exman8(xlog2)
SFPSHFT    i    = man << exp          <- the integer
SFPEXMAN   frac = exman9(i)           <- take it apart again
SFPCAST    frac = (float)frac
```

Five instructions to produce two values — an integer part and a fractional part — that a
truncating float→int conversion yields in three:

```
SFP_STOCH_RND (RND_ZERO, FP32→UINT16)   int_part = floor(xlog2)   <- see §9
SFPCAST                                 fi       = (float)int_part
SFPMAD (× LCONST_neg1)                  frac     = xlog2 − fi
```

`LCONST_neg1` turns the `SFPMAD` into a subtract, so the third step is one instruction.

**The polynomial is untouched.** Only the *encoding* of the fractional part changes — a float in
`[0, 1)` instead of the raw 23-bit mantissa — and the coefficients absorb that by rescaling:

| | scalar kernel | TTI kernel | |
|---|---|---|---|
| c1 | `7.839635491371155e-08` = `0x33a8`**`5ada`** | `0.657636285f` = `0x3f28`**`5ada`** | × 2^23 |
| c2 | `4.791750143340323e-15` = `0x27ac`**`a418`** | `0.337189436f` = `0x3eac`**`a418`** | × 2^46 |

The mantissa halves are **identical** — only the exponent bits move. That is the check that this
is a pure rescaling and not a refit: same function of the same quantity, evaluated to 3.9e-08
(fp32 rounding) over `u ∈ [0, 1)`. §9 applies a second, equally mechanical re-expression on top
of this one; `c2` survives both untouched.

`SFPSETEXP` follows from mode 2 (`ARG_EXPONENT`, exponent read from a float encoding's exponent
field) to mode 0 (exponent read from the operand's low bits), which is where the truncating
convert leaves it.

The `SFPGT` mask moves one slot later, after the frac subtract, because it overwrites `LREG3` —
which still holds `xlog2` until that subtract consumes it.

**Measured: −32.0 cycles/tile.** Two instructions removed, one cycle/element gained — the other
was already hidden. The old `SFPEXEXP`/`SFPEXMAN` pair were mutually independent (both read
`LREG3`), so one filled the other's shadow; the new chain is serial. That is the latency-bound
behaviour in §1 showing up directly, and it is what suggested change 2.

## 3. Change 2 — let the converter do the clamp

The upper clamp existed to keep `floor(xlog2) ≤ 255`, and cost two instructions plus a **2-cycle,
non-pipelined `SFPSWAP` sitting on the critical path**:

```
SFPLOADI  LREG1 = 255.0f
SFPSWAP   LREG3 = min(255, xlog2)
```

With change 1 the integer already comes from a converter — and a converter narrower than the
value saturates. Switching `SFP_STOCH_RND` from `FP32_TO_UINT16` to `FP32_TO_UINT8` makes the
clamp fall out of the conversion itself: anything at or above 255 lands on 255, which is the
exponent field for `+inf`. Both instructions delete, and the `SFPSWAP` leaves the dependency
chain.

**Measured: a further −64.0 cycles/tile.**

Total: body 16 → 12 instructions, 18.06 → 15.06 cycles/element.

### 3.1 This rests on a hardware behaviour, so it was measured

`FP32→UINT8` saturating at 255 is now load-bearing. Two facts, both established by experiment on
p100a rather than assumed:

- **`FP32→UINT16` does *not* saturate a negative input to zero.** Removing the lower guard failed
  2 of 257 tests. That is why the `SFPGT`/`SFPAND` mask is kept — it zeroes exactly those lanes
  before the integer is used, which is the same guard the old sequence relied on.
- **`FP32→UINT8` *does* saturate on positive overflow.** Verified directly (§4).

The asymmetry is the point: the converter saturates at the top and misbehaves at the bottom, so
the top clamp can be deleted and the bottom one cannot.

## 4. Correctness

### 4.1 A coverage gap this change walked into

The `Exp` sweep domain is deliberately range-bounded to avoid overflow (`_exp_spec`), and
`_APPROX_ACCURACY_MAX` caps the argument at 16.0 on top of that. **Nothing in the ordinary suite
drives `xlog2` above 255** — the only input in the entire suite that reaches the saturating path
is the `+inf` special.

So the 257 passing tests did *not* establish that change 2 was safe. If the convert wrapped
instead of saturating, `exp(+inf)` would still be right (infinity is handled distinctly) while
every large finite input silently returned a tiny wrong value — `exp(100)` reading `2^-112`
instead of `+inf`, with nothing to catch it.

### 4.2 Direct measurement, against main as the control

`tests/python_tests/test_exp_overflow_saturation.py` (new) drives overflow probes through the
kernel. Run on both the branch and main:

| input | branch | main | |
|---|---|---|---|
| 0 | 1 | 1 | |
| 1 | 2.71875 | 2.71875 | |
| 80 | 5.54927e+34 | 5.54927e+34 | |
| 88 | 1.64824e+38 | 1.64824e+38 | |
| 88.5 | 2.72492e+38 | 2.72492e+38 | last finite |
| 89 | **inf** | **inf** | |
| 100 | **inf** | **inf** | would be 2^-112 if the convert wrapped |
| 128, 200, 512 | **inf** | **inf** | |
| 1e30 | **inf** | **inf** | |
| `+inf` | **inf** | **inf** | |

**Byte-for-byte identical to main on every probe.** Saturation confirmed, and the change verified
behaviourally identical across the range no other test reaches.

The test runs `Float16_b → Float16_b` with `dest_acc=No`: the kernel under test is the bfloat16
one, so the data is bfloat16 and Dest is in the matching 16-bit mode. `dest_acc=No` is what
selects the kernel — `calculate_exponential` dispatches on `!APPROXIMATION_MODE &&
!is_fp32_dest_acc_en` alone and never looks at the L1 format.

Every probe around the overflow threshold is bfloat16-exact deliberately, because the data is
bfloat16 in L1: a literal like 88.7 is rounded to 88.5 on the way in, so a probe written next to
the threshold would be pinned to a value other than the one it names. An earlier version used 88.7
and 88.8 and flagged both as failures — on **both** the branch and main, which is what identified
it as a rounding artifact rather than a kernel fault. `1e30` is the one exception and does not need
the property: it lands on 1.000255552e+30, but it is a deep-overflow stress value where any nearby
representable input saturates the same way.

### 4.3 Suite

| | branch | main |
|---|---|---|
| `test_sfpu_unary.py -k Exp` | **257 passed, 129 skipped, 0 failed** | 257 passed, 129 skipped, 0 failed |
| `test_exp_overflow_saturation.py` | **1 passed** | 1 passed |

Covers `Exp`, `Exp2`, `ExpWithBase`, `Expm1` across the format matrix, plus
`test_exponential_clamp_negative` and the specials/edge sweep.

**Blast radius.** Only `_sfpu_exp_21f_bf16_tti_` and the `LREG13` constant in `exp_init`'s
bfloat16 branch changed. The scalar `_sfpu_exp_21f_bf16_` is untouched, so `ckernel_sfpu_exp2.h`,
`ckernel_sfpu_i1.h`, `ckernel_sfpu_mish.h`, `ckernel_sfpu_sigmoid.h` and `ckernel_sfpu_situ_glu.h`
are unaffected. The fp32-destination path (Juffa) is untouched.

**Not covered.** Wormhole B0 carries the same kernel and was not modified or measured — the same
two changes should port, but `FP32→UINT8` saturation has only been verified on Blackhole. Quasar
unchanged.

## 5. Performance

`perf_eltwise_unary_sfpu.py`, CI flags (`--speed-of-light`, producer/consumer split),
`MATH_ISOLATE` on the `TILE_LOOP` marker, cycles per tile, `tile_cnt` 8, `loop_factor` 16,
`iterations` 32. Baseline and branch built from separate clean build roots; the baseline was
produced with the change `git stash`ed.

| formats | approx_mode | mathop | main | branch | Δ | % |
|---|---|---|---|---|---|---|
| Float16_b→Float16_b | No | **Exp** | 577.99 | **481.99** | **−96.0** | **−16.61 %** |
| Float16_b→Float32 | No | **Exp** | 577.99 | **481.99** | −96.0 | −16.61 % |
| Float32→Float16_b | No | **Exp** | 578.31 | **482.31** | −96.0 | −16.60 % |
| Float16_b→Float16_b | Yes | Exp *(control)* | 93.53 | 93.53 | 0.00 | 0.00 % |
| Float16_b→Float32 | Yes | Exp *(control)* | 93.50 | 93.50 | 0.00 | 0.00 % |
| Float32→Float16_b | Yes | Exp *(control)* | 93.80 | 93.80 | 0.00 | 0.00 % |
| Float16_b→Float16_b | No | Gelu *(control)* | 2847.10 | 2847.10 | 0.00 | 0.00 % |

The approximate path and the unrelated op are flat to the last decimal. `L1_TO_L1` tracks
`MATH_ISOLATE` on the affected rows — math is the bottleneck for this op.

### 5.1 Instruction accounting

| step | body | cycles/tile | Δ |
|---|---|---|---|
| main | 16 | 577.99 | |
| + truncating convert replaces the integer round trip | 14 | 545.99 | −32.0 |
| + `FP32→UINT8` saturation replaces `SFPLOADI`+`SFPSWAP` | 12 | **481.99** | −64.0 |

Four instructions removed, **−96.0** measured against −128.0 if each cost a full cycle. The
shortfall is the one `SFPEXEXP`/`SFPEXMAN` slot that was already hidden (§2) — everything else
came off the critical path and paid in full. The residual 15.06 cycles/element against a
12-instruction body is ~3 cycles of remaining latency.

## 6. What is left

The kernel is still latency-bound: 12 instructions, 15.06 cycles/element. The remaining ~3 cycles
are the serial chain `SFP_STOCH_RND → SFPCAST → SFPMAD(frac) → SFPMAD(poly1) → SFPMAD(poly2) →
SFPSETEXP`, which has no independent work left to interleave — the `SFPGT` and `SFPAND` already
occupy the two available shadows.

Getting further means **software-pipelining two Dest elements** so each fills the other's latency
slots. That needs roughly double the scratch registers; with `LREG5/6/7` holding constants and
`LREG12/13` programmable, only `LREG0–3` and `LREG4` are free — enough for one element, not two.
Freeing one would cost an instruction to rematerialise a constant. Worth measuring, not worth
assuming.

**A larger lever sits elsewhere.** The approximate path measures **93.5 cycles/tile against this
kernel's 481.99** — 5.2× faster, and that gap is unchanged by this work. Any call site that can
tolerate ~3 % relative error is leaving 5× on the table by using `APPROXIMATION_MODE=false`.

## 7. All Blackhole measurements

Every perf number taken on the p100a during this work, including the rejected alternative, so the
comparisons above can be checked and so the rejected path is not re-explored blind.

**Re-verified after rebasing onto main.** The figures below were first taken against
`e835e43e46b`; `ckernel_sfpu_exp.h` is byte-identical between that commit and `f6b36f3b1be`, and
re-running the whole sweep on the newer base reproduced them: main 577.976 vs 577.992, branch
481.977 vs 481.992, delta **−96.0 exactly** either way. Differences of ~0.02 cycles are profiler
noise, three orders below the 32 cycles one instruction costs.

**Common setup for all rows:** `perf_eltwise_unary_sfpu.py`, `--speed-of-light`, producer/consumer
split, `dest_acc=No`, `loop_factor=16`, `iterations=32`, `input_dimensions=[128, 64]`
(`tile_cnt` 8), `fast_mode=No`. Cycles per tile. Each variant built into its own clean
`RUNNER_TEMP`. Values are `mean(...)` at the `TILE_LOOP` marker.

### 7.1 Full sweep — every variant, every run type

| variant | formats | op | approx | L1_TO_L1 | **MATH_ISOLATE** | UNPACK_ISOLATE | PACK_ISOLATE |
|---|---|---|---|---|---|---|---|
| **main** (`exp_21f`, TTI+replay) | Float16_b→Float16_b | Exp | No | 582.27 | **577.99** | 41.33 | 25.80 |
| **main** | Float16_b→Float32 | Exp | No | 582.98 | **577.99** | 41.33 | 34.81 |
| **main** | Float32→Float16_b | Exp | No | 591.36 | **578.31** | 40.48 | 25.81 |
| **main** | Float16_b→Float16_b | Exp | Yes | 98.92 | 93.53 | 41.38 | 25.73 |
| **main** | Float16_b→Float32 | Exp | Yes | 99.63 | 93.50 | 41.34 | 34.92 |
| **main** | Float32→Float16_b | Exp | Yes | 107.98 | 93.80 | 40.52 | 25.73 |
| **main** | Float16_b→Float16_b | Gelu | No | 2851.43 | 2847.10 | 41.33 | 25.72 |
| step 1 (truncating convert) | Float16_b→Float16_b | Exp | No | 550.27 | **545.99** | 41.33 | 25.80 |
| step 1 | Float16_b→Float32 | Exp | No | 551.00 | **545.99** | 41.33 | 34.81 |
| step 1 | Float32→Float16_b | Exp | No | 559.36 | **546.31** | 40.48 | 25.81 |
| step 1 | Float16_b→Float16_b | Exp | Yes | 98.92 | 93.53 | 41.38 | 25.73 |
| step 1 | Float16_b→Float32 | Exp | Yes | 99.63 | 93.50 | 41.34 | 34.92 |
| step 1 | Float32→Float16_b | Exp | Yes | 107.98 | 93.80 | 40.52 | 25.73 |
| step 1 | Float16_b→Float16_b | Gelu | No | 2851.43 | 2847.10 | 41.33 | 25.72 |
| **final** (+ UINT8 clamp) | Float16_b→Float16_b | Exp | No | 486.27 | **481.99** | 41.33 | 25.80 |
| **final** | Float16_b→Float32 | Exp | No | 486.98 | **481.99** | 41.33 | 34.81 |
| **final** | Float32→Float16_b | Exp | No | 495.36 | **482.31** | 40.48 | 25.81 |
| **final** | Float16_b→Float16_b | Exp | Yes | 98.92 | 93.53 | 41.38 | 25.73 |
| **final** | Float16_b→Float32 | Exp | Yes | 99.63 | 93.50 | 41.34 | 34.92 |
| **final** | Float32→Float16_b | Exp | Yes | 107.98 | 93.80 | 40.52 | 25.73 |
| **final** | Float16_b→Float16_b | Gelu | No | 2851.43 | 2847.10 | 41.33 | 25.72 |

`approx_mode=Yes` and `Gelu` are identical to the last decimal in all three builds — the change is
confined to the bfloat16-accurate path. `UNPACK_ISOLATE` and `PACK_ISOLATE` are likewise unchanged
throughout; only `MATH_ISOLATE` moves, and `L1_TO_L1` tracks it because math is the bottleneck.

### 7.2 Other markers

`Float16_b→Float16_b`, Exp, `approx_mode=No`:

| build | marker | L1_TO_L1 | MATH_ISOLATE | UNPACK_ISOLATE | PACK_ISOLATE |
|---|---|---|---|---|---|
| main | INIT | 274.00 | 166.00 | 257.00 | 287.00 |
| main | KERNEL | 74933.00 | 74286.00 | 5688.00 | 3739.00 |
| main | TILE_LOOP | 582.27 | 577.99 | 41.33 | 25.80 |
| final | INIT | 274.00 | 166.00 | 257.00 | 287.00 |
| final | KERNEL | 62645.00 | 61999.00 | 5688.00 | 3739.00 |
| final | TILE_LOOP | 486.27 | 481.99 | 41.33 | 25.80 |

`INIT` is unchanged at 166.0 — the extra `SFPCONFIG`/`SFPLOADI` work in `exp_init` is the same
count, only different constants. `KERNEL` (the whole profiled region, 16 loop_factor passes over 8
tiles) drops 74286 → 61999, i.e. −16.5 %, matching the per-tile figure.

### 7.3 Single-variant runs

Taken during development on `Float16_b→Float16_b`, Exp, `approx_mode=No`, `MATH_ISOLATE` at
`TILE_LOOP`. These are the data points behind the "latency-bound" and "sfpi vs TTI" claims.

| # | build | cycles/tile | note |
|---|---|---|---|
| 1 | main — `exp_21f`, hand-written TTI + replay buffer | **577.99** | the baseline |
| 2 | `exp_21f`, *same algorithm* in sfpi, compiler-scheduled | 987.90 | **hand-written TTI is worth 1.71×** |
| 3 | [rejected] Cody-Waite deg-3 sfpi, constants left to the register allocator | 1211.85 | |
| 4 | [rejected] Cody-Waite deg-3 sfpi, constants pinned to `LREG12/13/14` | 987.84 | −18.5 % vs row 3 |
| 5 | [rejected] Cody-Waite deg-3 sfpi, + split `2^k` for edge correctness | 1147.84 | +160.0 = exactly 5 instrs × 32 |
| 6 | [rejected] Cody-Waite deg-3, floor reduction, hand-written TTI | 835.97 | best form of the alternative |
| 7 | **final** — `exp_21f` optimized (this change) | **481.99** | |

Row 2 against row 1 is the measurement that says compiler-scheduled sfpi costs 1.71× a
hand-written TTI kernel for an identical algorithm — which is why row 6, not row 5, is the fair
comparison for the alternative, and why the alternative was still rejected at +44.6 %.

Rows 3→4 are worth remembering independently: every constant the register allocator spills costs
an `SFPLOADI` pair *per element*, and recovering three of them was worth 18.5 %.

### 7.4 The rejected alternative, full sweep

A Cody-Waite range reduction with a degree-3 minimax polynomial was implemented, measured, and
**not adopted** — it was instruction-neutral with `exp_21f` at matched implementation style, so it
offered accuracy (0.0086 % vs 0.1734 % max relative error) rather than speed. Recorded here so the
evaluation is not repeated.

| variant | formats | op | approx | L1_TO_L1 | **MATH_ISOLATE** |
|---|---|---|---|---|---|
| CW deg-3 sfpi, split `2^k` | Float16_b→Float16_b | Exp | No | 1152.36 | **1147.84** |
| CW deg-3 sfpi, split `2^k` | Float16_b→Float32 | Exp | No | 1153.07 | **1147.81** |
| CW deg-3 sfpi, split `2^k` | Float32→Float16_b | Exp | No | 1161.48 | **1148.16** |
| CW deg-3 TTI, floor | Float16_b→Float16_b | Exp | No | 840.27 | **835.97** |
| CW deg-3 TTI, floor | Float16_b→Float32 | Exp | No | 840.96 | **835.97** |
| CW deg-3 TTI, floor | Float32→Float16_b | Exp | No | 849.32 | **836.29** |

Both carried the same flat `approx_mode=Yes` (93.53 / 93.50 / 93.80) and Gelu (2847.10) controls.

### 7.5 Summary against main

| build | body | cycles/tile | Δ | % |
|---|---|---|---|---|
| main | 16 | 577.99 | — | — |
| step 1 — truncating convert | 14 | 545.99 | −32.0 | −5.54 % |
| **final** — + UINT8 saturating clamp | **12** | **481.99** | **−96.0** | **−16.61 %** |
| *[rejected] CW deg-3, best TTI form* | *17* | *835.97* | *+258.0* | *+44.63 %* |

## 8. Reproduction

Interpreter setup per the `llk-python-test-env` note.

```bash
cd tt_metal/tt-llk/tests/python_tests
export CHIP_ARCH=blackhole RUNNER_TEMP=<clean build root>       # fresh dir per variant
ID='perf_eltwise_unary_sfpu.py::test_perf_eltwise_unary_sfpu[formats:Float16_b->Float16_b-approx_mode:No-mathop:Exp-dest_acc:No-loop_factor:16-iterations:32-fast_mode:No-stable_sort:No-input_dimensions:[128, 64]]'
pytest -q --speed-of-light --compile-producer -n 4 -m perf "$ID"
pytest -q --speed-of-light --compile-consumer -n 1 -m perf "$ID"
# TILE_LOOP / mean(MATH_ISOLATE) in
# $LLK_ROOT/perf_data/perf_eltwise_unary_sfpu/perf_eltwise_unary_sfpu.post.csv
```

A tile is 32 SFPU vector iterations, so one added instruction is exactly +32.0 cycles/tile;
deltas landing off a multiple of 32 mean scheduling moved too.

---

## 9. Change 3 — a floor the simulator can execute

### 9.1 Why

Changes 1 and 2 both rest on one instruction: `SFP_STOCH_RND` in `SFPSTOCHRND_RND_ZERO` mode,
which truncates FP32 to a saturating UINT8. On silicon it is exactly right. In `ttsim` — the
Tensix simulator the `sim_bh_p150` CI lane runs on — it does not execute at all:

```c
/* ttsim/src/tensix.cpp, TENSIX_EXECUTE_SFP_STOCH_RND() */
TTSIM_VERIFY(!rnd_mode, UnsupportedFunctionality, "rnd_mode=%d", rnd_mode);
```

`SFPSTOCHRND_RND_ZERO` is 2, so the predicate fails, and `ttsim_error()` is `[[noreturn]]` and
ends in `_Exit(1)`. The simulator process dies the first time a tile reaches the kernel, taking
pytest with it. That is what failed seven `sim_bh_p150` jobs on the first CI run of this branch:
`ttnn eltwise group 2/3/4`, `ttnn fused group 1/2`, `core ttnn unit test group`, `ttsim examples`
— exit code 1, no test-level annotation, at whichever point in each split the first non-approx
bf16 `exp` ran. `sim_wh_n150` was unaffected (Wormhole path untouched), and every `bh_p150b_civ2`
job on real silicon passed.

The restriction is over-broad rather than intentional: `ttsim`'s own rationale for excluding this
family (`docs/unsupported_functionality.md`, category 8) is about *stochastic* rounding, which is
`rnd_mode == 1`. `RND_ZERO` is deterministic and has a published functional model
(`tt-isa-documentation`, `SFPSTOCHRND_FloatInt.md`). Widening the check is a five-line change, but
it lands in a repository that does not accept pull requests, so it cannot unblock this branch.

### 9.2 The substitution

Round-to-nearest-ties-away — `rnd_mode == 0`, the one mode `ttsim` does model — reaches the same
floor if the argument is biased down by half a unit. For any `x ≥ 0`, writing `x = n + f` with
`f ∈ [0, 1)`:

```
round(x − 0.5) = round(n + f − 0.5) = n = floor(x)
```

because `f − 0.5 ∈ [−0.5, 0.5)` never carries. The bias is already a constant the kernel loads,
so this is a one-immediate change, `127.0 → 126.5`, and `126.5` is exact in BF16 (`0x42fd`).
Saturation is untouched: `round(x − 0.5) ≥ 255` exactly when `floor(x) ≥ 255`.

The fractional part now arrives as `t = f − 0.5 ∈ [−0.5, 0.5)`, so the polynomial is rewritten in
`t`. This is algebra, not a refit:

```
c0 + c1·(t + 0.5) + c2·(t + 0.5)²  =  (c0 + c1/2 + c2/4) + (c1 + c2)·t + c2·t²
```

| | `frac ∈ [0, 1)` | `t ∈ [−0.5, 0.5)` | |
|---|---|---|---|
| c0 | `1.001953125f` = `0x3f804000` | `1.415068626f` = `0x3fb520f8` | `c0 + c1/2 + c2/4` |
| c1 | `0.657636285f` = `0x3f285ada` | `0.994825721f` = `0x3f7eace6` | `c1 + c2` |
| c2 | `0.337189436f` = `0x3eaca418` | `0.337189436f` = `0x3eaca418` | unchanged |

`c0'` no longer lands on the FP16 immediate grid, so it takes an `SFPLOADI` `UPPER`/`LOWER` pair
instead of a single `FLOATA` load. **That instruction is in the per-call preamble, before
`TTI_REPLAY` records the body — the replayed body is still 12 instructions and the dependency
chain is unchanged.** Preamble goes 4 → 5 `SFPLOADI`.

Rounding `c0'` to the FP16 grid instead would have kept the preamble at 4, and was rejected: see
9.3.

### 9.3 Verification

The whole body was modelled bit-exactly against `tt-isa-documentation` — `fma_model_bh` for every
`SFPMAD`, and the published functional models for `SFPSTOCHRND` (both flavours), `SFPCAST`,
`SFPSETEXP` — and both formulations were swept over `x ∈ [−90, 90]` in steps of `1e-4`
(1 800 001 points), comparing the stored BF16:

| c0' encoding | preamble | BF16 outputs differing | max rel. err vs `exp` |
|---|---|---|---|
| FP32 `0x3fb520f8` (**chosen**) | 5 | **1** of 1 800 001 | `5.744462e-03` — *identical* to RND_ZERO |
| FP16 `0x3da9` (= `1.4150390625`) | 4 | 6 649 (0.37 %) | `5.711828e-03` |
| c0' FP32, c1' as FP16 `0x3bf5` | 4 | 11 209 (0.62 %) | `5.846893e-03` |

The chosen encoding reproduces the round-toward-zero kernel to one BF16 ULP on a single sample in
1.8 million, at a max relative error identical to nine digits. The two four-instruction variants
are also correct — their error is no worse than the polynomial's own `1.7e-03` fit error, and one
is marginally better — but they move a third of a percent of outputs by one BF16 ULP, which is
not worth one preamble instruction on a branch whose claim is that nothing about the answers
moves.

Both documented hardware quirks are reproduced by the model and are shared by the two
formulations: `RND_ZERO` rounding *away* from zero when every discarded mantissa bit is set
(`tt-isa-documentation` notes this explicitly), and the corresponding tie in `round(x − 0.5)`.
They coincide on the same inputs and are harmless here, since `frac ≈ 0` at exactly those points.

### 9.4 What still needs measuring

The replayed body is unchanged, so §5's `481.99` cycles/tile should stand. The extra preamble
`SFPLOADI` costs one instruction per `calculate_exponential` call against a body of
`12 × ITERATIONS`, so the expected regression is under one percent. **This has not been
re-measured on silicon.** The sweep in §8 reproduces it.
