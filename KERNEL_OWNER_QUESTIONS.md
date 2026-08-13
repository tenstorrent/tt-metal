# Three questions for SFPU kernel owners — drafts, not yet filed

All three are divergences between a kernel and its golden that the ISA does not settle, found by
driving IEEE specials and domain boundaries through the tt-llk Python suite. Each is cheap for an owner
to adjudicate and expensive for a test to keep guessing about: until they are answered there is no way
to know whether the right outcome is a pass, an `xfail`, or a bug report, and a guess becomes a
permanent reason string that nobody re-derives.

**Between them they account for 33 of the 37 unary ops still outside cat-B coverage** — which is the
reason to ask rather than to keep working around them. Only 4 of the 37 need anything built:

| # | Question | Ops it decides |
|---|---|---|
| 1 | What should an approximation kernel do with an input outside its series' range? | **23** |
| 3 | Are SFPU comparisons defined for a `NaN` operand? | **9** |
| 2 | Why does `RsqrtCompat` saturate at the pole where `Rsqrt` does not? | **1** |

The first two were originally written up as one-op curiosities (`Log`, `signbit`). Driving the full
unary set showed both were single behaviours with wide blast radius. That is the argument for asking
now rather than after another tranche.

Measured on a Blackhole p300a, `ApproximationMode.No`, Float32 input — the only specials-carrying input
format reachable there. Reproduce with:

```bash
cd tt_metal/tt-llk
.claude/scripts/run_test.sh run --worktree $PWD --arch blackhole \
    --test test_sfpu_unary.py --k test_eltwise_unary_sfpu_edges
```

---

## 1. `Log` saturates a non-finite input, so no non-finite value survives it

**Question: is this intended, and should it be documented?**

| Probe | Golden (IEEE) | Hardware |
|---|---|---|
| `+inf` | `+inf` | **88.5** |
| `-inf` | `NaN` | **84.3** |
| `NaN` | `NaN` | **89.1** |
| `-0` | `-inf` | **-92.5** |

Every result is finite, and the three non-finite probes all land near `ln(FLT_MAX) = 88.7`. The kernel
appears to clamp its input to the format maximum and take the log of that, so a non-finite input cannot
produce a non-finite output.

**Why it matters to the suite.** `Log` is the last op of its tranche still outside `SPECIALS_READY_OPS`.
The other four (`Neg`, `Reciprocal`, `Sqrt`, `Rsqrt`) are enrolled, with genuine kernel divergences
recorded as non-strict xfails. `Log` cannot join until this is answered, because the three candidate
outcomes are indistinguishable from the test side:

- **Intended** — enrol with a documented xfail, reason "clamps its input to the format maximum".
- **Not intended** — file a kernel bug and enrol with an xfail pointing at it.
- **Intended but wrong for the API** — a kernel change, and the test follows it.

**What would settle it:** whether the input clamp is deliberate (e.g. to keep the `ln` polynomial in
range) and whether the API contract for `log` is meant to propagate `±inf` / `NaN`.

### This question is not about `Log`

Driving specials through the whole unary set shows the same shape in **22 further ops**, every one of
them a polynomial or LUT approximation being evaluated on an input its series was never meant to see.
The answers split two ways, and neither is IEEE:

**Saturates to the asymptote or to a magic constant**

| Op | Probe | Golden | Hardware |
|---|---|---|---|
| `LogWithBase` | `+inf` / `-inf` / `NaN` | `+inf` / `NaN` / `NaN` | `127.9` / `121.6` / `128.5` |
| `Digamma` | `NaN` | `NaN` | `89.09` |
| `Digamma` | `±0` | `∓inf` | `≈ -337920` |
| `I1` | `±inf`, `NaN` | `±inf`, `NaN` | `±1.1547668e37` |
| `Erf` | `NaN` | `NaN` | `1.0` |
| `Erfc` | `NaN` | `NaN` | `2.94e-12` |
| `Tanh` | `NaN` | `NaN` | `1.0` |
| `Sigmoid`, `TanhDerivative`, `Rdiv`, `Polygamma` | `NaN` | `NaN` | `0.0` |
| `Gelu`, `GeluDerivative` | `-inf` / `+inf` / `NaN` | `NaN` | `0.0` / `1.0` / `1.0` |
| `Lgamma` | `±0` | `+inf` | `-0.00051` |
| `UnaryPower`, `Rpow`, `CastFp32ToFp16a` | `NaN` (or `-inf`) | `NaN` / `0.0` | `+inf` |

**Returns `NaN` where a value is defined** — the same failure from the other side

| Op | Probe | Golden | Hardware |
|---|---|---|---|
| `Frac` | `±inf` | `±inf` | `NaN` |
| `SigmoidAppx` | `±inf` | `0.0` / `1.0` | `NaN` |
| `TanhDerivativeLut` | `±inf`, `NaN` | `0.0` | `NaN` |
| `Expm1Cw` | `+inf` | `+inf` | `NaN` |
| `Lgamma` | `±inf` | `+inf` | `NaN` |
| `SqrtCustom` | `+inf` / `-inf` | `+inf` / `NaN` | `NaN` / `+inf` |
| `Erfinv` | `±1` | `±inf` | `NaN` |

`LogWithBase` is worth singling out as evidence the cause is shared rather than per-op: its results are
`Log`'s multiplied by the dispatch scale `1/ln(2) ≈ 1.4427` (`88.7 × 1.4427 = 128.0`). It is the same
clamp, seen through a scale factor.

**So the real question is a contract question, and it is worth one answer rather than 23.** What should
a polynomial/LUT approximation kernel do with an input outside the range its series covers — propagate
per IEEE, saturate to the asymptote, or is it explicitly undefined and the caller's job to range-check?
Whatever the answer, it converts 23 held-out ops into either enrolments, a shared xfail, or one bug
report.

---

## 2. `RsqrtCompat(0)` saturates where plain `Rsqrt(0)` does not

**Question: which of the two is the intended answer at the shared pole?**

`RsqrtCompat` at `+0` returns `1.7014118e38` (`0x7F000000`) instead of `+inf`, on **all 8**
format/`dest_acc` combinations. Plain `Rsqrt` over the same probe does **not** diverge — it returns
`+inf` and agrees with the golden.

Two implementations of the same mathematical function disagree at their shared pole, and nothing in the
ISA prescribes either answer. `0x7F000000` is a suspiciously round bit pattern — it looks like a
deliberate saturation constant rather than an accident of the approximation, which is part of why this
needs an owner rather than a test-side decision.

**Why it matters to the suite.** `RsqrtCompat`'s 8 divergent cells are currently xfailed with the
reason "not prescribed by the ISA either way". That reason is honest but it is a placeholder: if the
saturation is deliberate the xfail should say so and cite the rationale, and if it is not, this is a
correctness bug in a function whose whole purpose is compatibility with `Rsqrt`.

**What would settle it:** whether `RsqrtCompat` is meant to saturate at the pole (and if so, why the
two implementations differ), or whether it should return `+inf` like `Rsqrt`.

---

## 3. SFPU comparisons rank `NaN` above every finite value

**Question: is the comparison result for a `NaN` operand defined, and if so, as what?**

IEEE 754 makes every ordered comparison involving `NaN` false — `x < y`, `x <= y`, `x > y` and `x >= y`
all return false, and only `!=` returns true. The SFPU instead behaves as though `NaN` were larger than
everything, which is what an unsigned magnitude comparison would do: a `NaN` has an all-ones exponent
and a set mantissa, so its bit pattern outranks any finite value.

**This is derived, not guessed.** The six unary comparison ops split exactly along the predicted line —
the three that ask "is x below the threshold" agree with the golden, and the three that ask "is x above
it" do not:

| Op | Expression | Golden (IEEE) | Hardware | Predicted by "NaN is greatest"? |
|---|---|---|---|---|
| `UnaryLt` | `x < 0.5` | `0.0` | `0.0` | ✅ false either way — **passes** |
| `UnaryLe` | `x <= 0.5` | `0.0` | `0.0` | ✅ false either way — **passes** |
| `UnaryMax` | `max(x, 0.0)` | `NaN` | `NaN` | ✅ returns the NaN — **passes** |
| `UnaryGt` | `x > 0.5` | `0.0` | **`1.0`** | ✅ true only if NaN > 0.5 |
| `UnaryGe` | `x >= 0.5` | `0.0` | **`1.0`** | ✅ same |
| `UnaryMin` | `min(x, 0.0)` | `NaN` | **`0.0`** | ✅ returns the *other* operand |

A test cannot produce that pattern by accident. The same rule then explains six further ops, each
returning its upper bound where IEEE would give `NaN`:

| Op | Hardware at `NaN` | Which constant that is |
|---|---|---|
| `Clamp` | `1.0` | `CLAMP_MAX` |
| `Hardtanh` | `1.0` | `CLAMP_MAX` |
| `Hardsigmoid` | `1.0` | the clamp ceiling |
| `ReluMax` | `5.0` | `RELU_MAX_THRESHOLD` |
| `Sign` | `1.0` | the `+1` branch — "not `<0`, not `==0`, therefore positive" |
| `Heaviside` | `1.0` | the `x > 0` branch |

**Why it matters to the suite.** Nine ops are held out of `SPECIALS_READY_OPS` by this one behaviour.
If it is defined, they can all be enrolled together with a single shared xfail reason — or, better, the
goldens can model it and the ops enrol with no xfail at all. If it is not defined, that is one bug
report covering nine ops rather than nine.

**Precedent for it being unspecified:** the ISA text already cited in this suite for `Sign` says
`SFPSETCC` is specified only for inputs that are not negative zero
(`tt-isa-documentation WormholeB0/.../VectorUnit.md`). `NaN` looks like the same kind of gap.

**What would settle it:** whether `SFPSETCC`'s ordered comparisons are specified for a `NaN` operand,
and whether the intended contract is IEEE unordered semantics or the magnitude ordering observed here.

---

## One question that was withdrawn — do not re-file it

An earlier revision listed a third question about `signbit(-0.0)`. It was read as a kernel-contract bug.
The delivery measurement since showed that the `-0.0` probe **never arrives** on the six combinations
where `Signbit` diverges: outside the unpack-to-dest path the datum goes through SrcA and the datacopy,
and the LREG holds `+0.0`. There is no kernel contract to question — it was a stimulus limitation, and
the suite now gates the probe out of those pipelines (`negative_zero_delivered()`).
