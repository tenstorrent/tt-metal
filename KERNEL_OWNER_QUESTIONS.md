# Questions for SFPU kernel owners — one answered by the ISA, two still open

Each is a divergence between a kernel and its golden, found by driving IEEE specials and domain
boundaries through the tt-llk Python suite. Until they are settled there is no way to know whether the
right outcome is a pass, an `xfail`, or a bug report, and a guess becomes a permanent reason string
that nobody re-derives.

> ## Read this first: Q3 is answered, and the answer reverses it
>
> [tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation) settles Q3 for Blackhole,
> and not in the direction the question assumed. `SFPGT`, `SFPLE` and `SFPSWAP` all specify a **total
> order** for FP32 in which `+NaN` is the largest value:
>
> > `-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN`
>
> So the hardware is behaving exactly as documented, and it is **the goldens that are wrong** — they
> model IEEE's unordered comparisons, which the SFPU does not implement. Those ops need a golden that
> models the total order, not an xfail against the kernel. See §3.
>
> That takes the count these questions decide from 33 ops down to **26**, and moves 7 of them from
> "blocked on an owner" to ordinary golden work.

**What is left for an owner**, and it is now mostly one question:

| # | Question | Ops it decides | Status |
|---|---|---|---|
| 1 | What should an approximation kernel do with an input outside its series' range? | **23** | **Open** — the ISA is silent by construction |
| 2 | Why does `RsqrtCompat` saturate at the pole where `Rsqrt` does not? | 1 | **Open**, but narrowed — see the ISA note in §2 |
| 3 | Are SFPU comparisons defined for a `NaN` operand? | 9 | **Answered** for Blackhole; 2 ops and the Wormhole gap remain |

Both remaining questions were originally written up as one-op curiosities (`Log`, `signbit`). Driving
the full unary set showed each was a single behaviour with wide blast radius.

**Sources.** Behaviour measured on a Blackhole p300a, `ApproximationMode.No`, Float32 input — the only
specials-carrying input format reachable there. ISA text quoted from
[tenstorrent/tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation), files under
`BlackholeA0/TensixTile/TensixCoprocessor/` and `WormholeB0/TensixTile/TensixCoprocessor/`. Reproduce
the measurements with:

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

### The ISA is silent here by construction, which is itself the answer to "why ask an owner"

Checked, so nobody re-checks: the ISA documents the *primitives* these kernels are built from, and
specifies each only within a stated range.

- **`SFPARECIP`** gives accuracy bounds that are explicitly range-limited — `0.9944 * 1/x <
  ApproxRecip(x) < 1/x * 1.0054` for reciprocal, and `0.9922 * e^x < ApproxExp(x) < e^x * 1.016` only
  for `0 ≤ x < 2` — and then recommends "software may wish to follow this instruction with other
  instructions to improve the accuracy (for example performing a few Newton-Raphson iterations)".
  Nothing about non-finite inputs.
- **`SFPLUTFP32`** selects coefficients by which range `Abs(LReg[3])` falls into and computes
  `a * b + c`, with a catch-all final range. It documents no special handling for `NaN` or `±inf`.

So `log`, `erf`, `tanh`, `gelu` and the rest are **software** built on primitives whose contract stops
at the edge of a stated range. The out-of-range behaviour of the composition is an LLK/API decision
that no ISA text can settle — which is precisely why this needs an owner rather than more measurement.

One inference worth putting to them: a LUT evaluated on `+inf` would land in the catch-all range and
compute `a * inf + c = ±inf`, yet the measured results are *finite* and clustered near
`ln(FLT_MAX)`. That is consistent with the kernels clamping their **input** before the LUT rather
than the LUT saturating, and it is the clamp whose intent is in question.

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

### The ISA narrows this: the saturation constant is not the hardware's

`SFPARECIP`'s functional model saturates its *own* way — an input below `2^-126` returns
**`0x7f800000`, i.e. `+inf`**, and an input at or above `2^126` returns `0`. `RsqrtCompat` returns
`0x7F000000` (`2^127`, `1.7014118e38`), which is **not** a value the instruction produces.

So the constant is a deliberate software clamp added above the primitive, and plain `Rsqrt` returning
`+inf` is the behaviour consistent with the underlying instruction. That does not decide which is
right for the API — a compat path may be clamping on purpose, for a caller that cannot take an `inf` —
but it does mean the question is "why was this clamp added", not "which one does the hardware do".

---

## 3. ~~SFPU comparisons rank `NaN` above every finite value~~ — answered by the ISA

**This is documented Blackhole behaviour, not a divergence.** The suite measured it, the ISA specifies
it, and the two agree exactly. What follows is kept because the measurement is what located the ISA
text, and because two ops and one architecture are still open.

### What the ISA says

`BlackholeA0/TensixTile/TensixCoprocessor/` specifies a **total order** for FP32 in three places:

| Instruction | Summary in `VectorUnit.md` |
|---|---|
| `SFPGT` | "Set per-lane flags based on `VD > VC`, where `-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN`" |
| `SFPLE` | "Set per-lane flags based on `VD <= VC`, where `-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN`" |
| `SFPSWAP` | `Min`/`Max`; "If C and D are instead FP32, this still determines whether C is less than D, using the total order where `-NaN < -Inf < ... < +Inf < +NaN`" |

All three route through `SignMagIsSmaller()`, which "treats C and D as sign-magnitude integers" — the
comparison is a bit-pattern compare remapped to two's complement, not an IEEE compare. A `+NaN` has an
all-ones exponent and a set mantissa, so it outranks every finite value **by design**.

### So the goldens are the wrong party

Every measured result follows from the total order, including the ops that *agree* — which is what
makes the explanation checkable rather than plausible:

| Op | Expression | Golden (IEEE) | Hardware | Under the documented total order |
|---|---|---|---|---|
| `UnaryLt` | `x < 0.5` | `0.0` | `0.0` | `+NaN` is not less — **false, agrees** |
| `UnaryLe` | `x <= 0.5` | `0.0` | `0.0` | same — **agrees** |
| `UnaryMax` | `max(x, 0.0)` | `NaN` | `NaN` | `+NaN` is the max — **agrees** |
| `UnaryGt` | `x > 0.5` | `0.0` | **`1.0`** | `+NaN` is greater — **hardware correct** |
| `UnaryGe` | `x >= 0.5` | `0.0` | **`1.0`** | same — **hardware correct** |
| `UnaryMin` | `min(x, 0.0)` | `NaN` | **`0.0`** | `+NaN` is the max, so min is the other operand — **hardware correct** |

`Clamp` → `1.0`, `Hardtanh` → `1.0`, `Hardsigmoid` → `1.0` and `ReluMax` → `5.0` are the same rule seen
through `SFPSWAP`: each returns its own upper-bound dispatch constant, which is what clamping a value
that ranks above everything must give.

**The action is therefore golden work, not an xfail.** Seven ops need goldens that model the total
order, after which they enrol as ordinary passes. Writing them off as kernel divergences would have
recorded a permanent, plausible-looking lie about documented hardware — the exact failure mode the
per-op gate exists to prevent.

### What is still open

1. **`Sign` and `Heaviside` are a different instruction.** They compare against zero, which is
   `SFPSETCC`, and its contract is explicitly conditioned: *"Provided that `VC` is neither negative
   zero nor any kind of NaN: Set per-lane flags based on `VC < 0` or `VC != 0` or `VC >= 0` or
   `VC == 0`"*. NaN is out of contract there, so their `1.0` at `NaN` is unspecified rather than
   documented — even though it is consistent with an `int32` test on a positive NaN's bit pattern.
   **Question: is `SFPSETCC` intended to be usable with a NaN operand, or must callers exclude it?**
2. **Wormhole has no total order, because it has no `SFPGT`/`SFPLE`.** Neither appears in
   `WormholeB0/TensixTile/TensixCoprocessor/VectorUnit.md`; `SFPSETCC` is the only comparison there,
   with the same "neither negative zero nor any kind of NaN" proviso. So the guarantee this section
   rests on is **Blackhole-only**, and the goldens may need arch-keying rather than one total-order
   model. **Question: what is the intended `NaN` comparison behaviour on Wormhole?** Nothing in this
   suite has been measured there.
3. **The op→instruction mapping is inferred from behaviour, not read from the kernels.** The pass/fail
   split matches the total order exactly, which is strong, but confirming which ops lower to
   `SFPGT`/`SFPLE`/`SFPSWAP` versus `SFPSETCC` needs a read of each kernel before the goldens are
   changed.

---

## One question that was withdrawn — do not re-file it

An earlier revision listed a third question about `signbit(-0.0)`. It was read as a kernel-contract bug.
The delivery measurement since showed that the `-0.0` probe **never arrives** on the six combinations
where `Signbit` diverges: outside the unpack-to-dest path the datum goes through SrcA and the datacopy,
and the LREG holds `+0.0`. There is no kernel contract to question — it was a stimulus limitation, and
the suite now gates the probe out of those pipelines (`negative_zero_delivered()`).
