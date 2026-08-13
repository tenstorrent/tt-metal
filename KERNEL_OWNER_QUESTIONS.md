# Questions for SFPU kernel owners — one answered by the ISA, two still open

Each is a divergence between a kernel and its golden, found by driving IEEE specials and domain
boundaries through the tt-llk Python suite. Until they are settled there is no way to know whether the
right outcome is a pass, an `xfail`, or a bug report, and a guess becomes a permanent reason string
that nobody re-derives.

> ## Read this first: Q3 was answered by the ISA, and is now fixed
>
> [tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation) settled Q3 for Blackhole,
> and not in the direction the question assumed. `SFPGT`, `SFPLE` and `SFPSWAP` all specify a **total
> order** for FP32 in which `+NaN` is the largest value:
>
> > `-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN`
>
> The hardware was behaving exactly as documented, and it was **the goldens that were wrong** — they
> modelled IEEE's unordered comparisons, which the SFPU does not implement. **Seven ops have since
> been fixed and enrolled** (`Clamp`, `Hardsigmoid`, `Hardtanh`, `ReluMax`, `UnaryGe`, `UnaryGt`,
> `UnaryMin`); they pass as ordinary tests, and the xfails this document was about to request were
> never written. See §3 for what is left of the question.
>
> **And not only for Blackhole.** `WormholeB0/…/SFPSWAP.md` carries the same `SignMagIsSmaller()` and the
> same total-order comment, so the order is specified there too — and all seven ops have since been
> measured green on a Wormhole n300. §3's item 2 used to say Wormhole had no total order; it was wrong.
>
> **That is the argument for this whole file.** Three questions were drafted against measured tables.
> One dissolved on contact with the ISA, and had it been filed as a kernel bug — or worse, silently
> xfailed — it would have left seven permanent, plausible-looking lies about documented hardware.
>
> It has since happened a second time, on Wormhole and again about the sign of a NaN — 49 failing
> variants across 10 ops that read as a kernel divergence and are documented behaviour. See the last
> section: **do not file it.**

**What is left for an owner**, and it is now mostly one question:

| # | Question | Ops it decides | Status |
|---|---|---|---|
| 1 | What should an approximation kernel do with an input outside its series' range? | **23** | **Open** — the ISA is silent by construction |
| 2 | Why does `RsqrtCompat` saturate at the pole where `Rsqrt` does not? | 1 | **Open**, but narrowed — see the ISA note in §2 |
| 3 | Are SFPU comparisons defined for a `NaN` operand? | 9 | **Answered and fixed** — 7 enrolled; the Wormhole gap is now measured and closed, leaving 2 ops |

Both remaining questions were originally written up as one-op curiosities (`Log`, `signbit`). Driving
the full unary set showed each was a single behaviour with wide blast radius.

**Sources.** The tables below are measured on a Blackhole p300a, `ApproximationMode.No`, Float32 input —
the only specials-carrying input format reachable there. §3's Wormhole items are measured on a Wormhole
n300; that record is [WORMHOLE_MEASUREMENT_RESULTS.md](WORMHOLE_MEASUREMENT_RESULTS.md). ISA text quoted
from [tenstorrent/tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation), files under
`BlackholeA0/TensixTile/TensixCoprocessor/` and `WormholeB0/TensixTile/TensixCoprocessor/`. Reproduce
the measurements with:

```bash
cd tt_metal/tt-llk
.claude/scripts/run_test.sh run --worktree $PWD --arch blackhole \
    --test test_sfpu_unary.py --k test_eltwise_unary_sfpu_edges
# and the same with --arch wormhole
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

### So the goldens were the wrong party — and have been fixed

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

**Done: seven ops now model the total order and are enrolled** as ordinary passes. The mapping was
confirmed against the kernels rather than inferred from behaviour — `_relu_max_body_` is
`v_if (result > threshold)`, a two-vector compare and therefore `SFPGT`, and `_calculate_clamp_` has
the same shape; `Hardsigmoid` turned out to *be* `_relu_max_body_(x/6 + 0.5, 1.0)`. Over 8000 finite
inputs the rewritten goldens are bit-identical to the ones they replace; only the NaN answers moved.

### What is still open

1. **`Sign` and `Heaviside` are a different instruction.** They compare against zero, which is
   `SFPSETCC`, and its contract is explicitly conditioned: *"Provided that `VC` is neither negative
   zero nor any kind of NaN: Set per-lane flags based on `VC < 0` or `VC != 0` or `VC >= 0` or
   `VC == 0`"*. NaN is out of contract there, so their `1.0` at `NaN` is unspecified rather than
   documented — even though it is consistent with an `int32` test on a positive NaN's bit pattern.
   **Question: is `SFPSETCC` intended to be usable with a NaN operand, or must callers exclude it?**
2. ~~**Wormhole has no total order, because it has no `SFPGT`/`SFPLE`.**~~ **Withdrawn — the premise was
   wrong.** Neither instruction appears in `WormholeB0/…/VectorUnit.md`, which is true, but the order
   does not depend on them: `WormholeB0/…/SFPSWAP.md` carries the same `SignMagIsSmaller()` model as
   Blackhole's, with the same comment word for word — *"using the total order where
   `-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN`"*. So the min/max/clamp half of the seven is
   **specified on Wormhole too**. For the four compare-shaped ops a two-vector `v_if (a > b)` must lower
   to something other than `SFPGT` there; the expansion is in the sfpi compiler backend rather than the
   headers, so which instruction it becomes is still unverified. **There is no question here for an
   owner** — only a follow-up for whoever wants a guarantee rather than an observation: disassemble a
   built `unary_gt` kernel and read the opcode.
3. ~~**The seven enrolled goldens are unverified on Wormhole.**~~ **Now verified: all seven pass 8/8 edge
   variants on a Wormhole n300, 0 xpassed**, and a direct probe over `+inf / -inf / NaN / ±0` reproduces
   the Blackhole table in this section value for value. No arch-keying needed.

   The same run also measured what item 1 above is about. **Wormhole answers identically to Blackhole:**
   `Sign(NaN) = 1.0` against a golden `0.0`, `Heaviside(NaN) = 1.0` against `0.5`, and the same `-0`
   divergence at `dest_acc=Yes`. So item 1 is one contract question about `SFPSETCC`, not two
   measurements — and it is the only part of this section still open.

---

## Two questions that were withdrawn — do not re-file either

**`signbit(-0.0)`.** An earlier revision listed this as a third question, read as a kernel-contract bug.
The delivery measurement since showed that the `-0.0` probe **never arrives** on the six combinations
where `Signbit` diverges: outside the unpack-to-dest path the datum goes through SrcA and the datacopy,
and the LREG holds `+0.0`. There is no kernel contract to question — it was a stimulus limitation, and
the suite now gates the probe out of those pipelines (`negative_zero_delivered()`).

**The generated-NaN sign on Wormhole.** The first Wormhole run of the edge sweep fails 49 variants across
10 ops (`Cos`, `Fmod`, `GeluAppx`, `Hardmish`, `Mish`, `Rsqrt`, `Silu`, `Sin`, `Softsign`, `Tan`), always
`golden=+inf` against `hw=-inf`, and it looks exactly like a kernel divergence worth asking about. **It is
not.** `SFPMAD.md` says the emitted NaN "is always the canonical NaN with bit pattern `0x7fc00000`" on
Blackhole and, on Wormhole, that "the sign bit might or might not be set". The conversion that makes the
sign visible is documented as well, and the ISA flags it itself — the packer's "NaN becomes infinity (this
is a potentially surprising behaviour)" and `SFPSTORE`'s "software is advised to avoid NaN inputs for this
conversion". So the kernels are behaving within spec and the goldens are asserting something the ISA
declines to promise: it is suite work, tracked as
[the expansion plan's §4](SFPU_EDGE_CASE_EXPANSION_PLAN.md). Filing it as a kernel bug would have been
the fourth question this file dissolved by reading the ISA first, and the second on the sign of a NaN.
