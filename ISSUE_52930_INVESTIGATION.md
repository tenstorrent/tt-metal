# Issue #52930 — reproduction, ISA cross-check, and root causes

**Issue:** [tenstorrent/tt-metal#52930](https://github.com/tenstorrent/tt-metal/issues/52930) — *[LLK] Problematic
SFPU edge cases*. Five findings from the edge sweep in
[PR #52416](https://github.com/tenstorrent/tt-metal/pull/52416), each described there as *not explained by the
ISA doc*.

**Question asked of this run:** can all five be reproduced in the existing setup on
`ldjurovic/sfpu_edge_cases_phase_3`; do we still get the recorded results; and for each, does
[tt-isa-documentation](https://github.com/tenstorrent/tt-isa-documentation) describe the behaviour — if not, what
is the root cause?

**Answer in one line:** all five reproduce. **Two of the five are documented hardware behaviour and are not
kernel bugs** (findings 1 and 2) — and the issue's stated diagnosis for both is wrong. **Three are genuine kernel
defects the ISA does not prescribe** (findings 3, 4, 5); all three are root-caused to a specific line below. Two
of the issue's recorded *values* are also wrong: erfinv returns NaN rather than a saturated finite, and
`RsqrtCompat(0)`'s underlying value is `0x7EFFFD9E`, not `0x7F000000`.

---

## 1. Environment

| | |
|---|---|
| Hardware | Wormhole n300 (UMD chip 0, board `010001461…`), silicon, not simulator |
| Tree | `tt-metal` @ `ldjurovic/sfpu_edge_cases_phase_3`, `26c61ff80e9` |
| SFPI | 7.69.0 (the pinned version; the earlier Wormhole run in `WORMHOLE_MEASUREMENT_RESULTS.md` used 7.68.0) |
| Runner | `tt_metal/tt-llk/.claude/scripts/run_test.sh` throughout; `pytest` was never invoked directly |
| Kernels under test | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/` — confirmed via `test_config.py:466-469`, which puts that directory on the include path |
| Date | 2026-08-17 |

**Can everything be run in the existing setup?** Yes, with no changes. `tests/.venv` and `tests/sfpi` are both
present and current; the two shipped suites that carry these findings collect and run as-is. Note the runner
script lives at `tt_metal/tt-llk/.claude/scripts/run_test.sh` — not at the tt-metal root, where an earlier note
placed it.

Commands used:

```bash
cd tt_metal/tt-llk
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_unary.py  --k "edges and (RsqrtCompat or Erfinv or Rsqrt)"
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_binary.py --k "binary_edges"
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_wh_issue52930_probe.py          # the instrument written for this run
```

### Why a new instrument was needed

All five findings are already recorded in the tree as **non-strict xfails** (`_EDGE_KNOWN_DIVERGENCES` /
`_EDGE_DIVERGENCE_REASON` in `test_sfpu_unary.py:600-688`, `_BINARY_EDGE_COMBINATIONS` /
`_BINARY_EDGE_REASON` in `test_sfpu_binary.py:1332-1405`). Rerunning them answers *"does it still diverge"* but
not *"what did the hardware return"*, and the ISA cross-check needs the bit pattern. So
`tests/python_tests/test_sfpu_wh_issue52930_probe.py` drives the same stimuli through the same drivers and prints
every `(input, golden, hardware)` triple with the fp32 bit pattern of each. It records rather than asserts.
**Not for merge.**

---

## 2. Rerun: does everything still reproduce?

### Shipped unary edge sweep — 24 variants

```
13 passed, 11 xfailed, 0 xpassed
```

| Op | Recorded | This run |
|---|---|---|
| `RsqrtCompat` | diverges on all 8 combinations | **8/8 XFAIL** ✅ |
| `Erfinv` | diverges on the 2 fp32-dest combinations only | **2/2 XFAIL** ✅ |
| `Rsqrt` (control) | does *not* diverge at the shared `+0` pole | **confirmed** — its 1 XFAIL is the separate `rsqrt(-0)` cat-B case ✅ |

### Shipped binary edge sweep — 160 variants

```
50 passed, 64 skipped, 30 xfailed, 16 xpassed
```

All 30 XFAIL are the `both_zero` and `nan_golden` classes of `div` / `xlogy` / `pow` / `fmod` / `remainder` —
findings 2 and 3, reproducing exactly. Nothing in those classes XPASSed.

**All 16 XPASS are the `negative_zero_golden` class** — a claim from the tree, not from the issue, and it is now
stale. See §5.1.

---

## 3. Measured values

Full record: `test_sfpu_wh_issue52930_probe.py` output (82 variant tables). The load-bearing rows:

### The controlled comparison that decides findings 2 and 5

`fmod(x, 0)`, whose kernel contains an explicit `b == 0 → quiet_NaN` guard:

| input → output | `dest_acc` | DEST width | pack | hardware |
|---|---|---|---|---|
| `Float16_b → Float16_b` | No | 16-bit | bf16 | `0x7F800000` (+inf) |
| `Float16_b → Float16_b` | Yes | fp32 | bf16 | `0x7F800000` (+inf) |
| `Float16_b → Float32` | No | 16-bit | fp32 | `0x7F800000` (+inf) |
| `Float16_b → Float32` | Yes | fp32 | fp32 | **`0x7FC00000` (NaN)** ✅ |
| `Float32 → Float16_b` | Yes | fp32 | bf16 | `0x7F800000` (+inf) |
| `Float32 → Float32` | Yes | fp32 | fp32 | **`0x7FC00000` (NaN)** ✅ |

The last two rows differ in **one variable only** — the pack output format — and `0x7FC00000` is exactly the
quiet NaN the kernel's own guard writes. So the kernel is right and the NaN is destroyed downstream, by the
narrowing to bf16.

### Findings 2 and 3 — the indeterminate forms

| Op | pair | golden | hw (bf16 anywhere) | hw (`Float32→Float32`, `dest_acc=Yes`) |
|---|---|---|---|---|
| `div` | `0 / 0` | NaN | `0x7F800000` +inf | **`0x7FC00000` NaN** ✅ |
| `xlogy` | `xlogy(0, 0)` | NaN | `0xFF800000` −inf | **`0xFFC00001` NaN** ✅ |
| `fmod` | `fmod(x, 0)` | NaN | `0x7F800000` +inf | **`0x7FC00000` NaN** ✅ |
| `remainder` | `remainder(x, 0)` | NaN | `0x7F800000` +inf | **`0x7FC00000` NaN** ✅ |
| `pow` | `0 ** 0` | `1.0` | `0x00000000` **+0** | `0x00000000` **+0** ❌ still wrong |

`div`, `xlogy`, `fmod` and `remainder` all produce a NaN wherever a NaN can survive. **`pow` is the only one
that stays wrong on the fully fp32 pipeline** — which is what separates finding 3 from finding 2.

Note `xlogy`'s `0xFFC00001`: mantissa LSB set, which is `SFPMAD.md`'s signature for an SFPU-emitted NaN (§4.2).

### Finding 4 — `RsqrtCompat(0)`

| input → output | `dest_acc` | hardware |
|---|---|---|
| `Float16_b → Float16_b` | No / Yes | `0x7F000000` = 1.7014118e38 |
| `Float32 → Float32` | Yes | **`0x7EFFFD9E` = 1.7013500e38** |

The issue records `0x7F000000` "on all 8 combinations". That is the **bf16-rounded view** of the real value:
`0x7EFFFD9E` rounds to bf16 `0x7F00`. The single computed value is `0x7EFFFD9E`, and §4.4 derives it exactly.

Also visible on that variant, unrelated to the pole: `rsqrt_compat(2.384e-07)` = `2044.9895` against a golden of
`2048.0` — ~0.15 % relative error, the legacy kernel's own accuracy.

### Finding 5 — `Erfinv(±1)` returns NaN, not a saturated value

| input → output | `dest_acc` | `erfinv(+1)` | `erfinv(-1)` |
|---|---|---|---|
| `Float16_b → Float16_b` | No / Yes | `0x7F800000` +inf ✅ | `0xFF800000` −inf ✅ |
| `Float16_b → Float32` | No | `0x7F800000` +inf ✅ | `0xFF800000` −inf ✅ |
| `Float16_b → Float32` | **Yes** | **`0x7FC00001` NaN** ❌ | **`0xFFC00001` NaN** ❌ |
| `Float32 → Float16_b` | No / Yes | `0x7F800000` +inf ✅ | `0xFF800000` −inf ✅ |
| `Float32 → Float32` | No | `0x7F800000` +inf ✅ | `0xFF800000` −inf ✅ |
| `Float32 → Float32` | **Yes** | **`0x7FC00001` NaN** ❌ | **`0xFFC00001` NaN** ❌ |

Two corrections to the issue text. The op does not *saturate* — it returns **NaN**, with the mantissa LSB set.
And the divergence is confined to the two combinations with an **fp32 DEST *and* an fp32 pack** because
everywhere else the same NaN is narrowed to ±infinity, which **coincidentally equals the golden ∓inf/±inf** and
so passes. The kernel is equally wrong on all eight; only two can see it.

### Finding 1 — `signbit(-0.0)`, measured directly for the first time

`test_sfpu_unary.py:578-584` flags this as *"NOT DIRECTLY MEASURED, and worth doing before anything is built on
it"*. It is now measured. Input `-0.0` (`0x80000000`):

| input → output | `dest_acc` | `unpack_to_dest` | `signbit(-0.0)` |
|---|---|---|---|
| `Float16_b → Float16_b` | No / Yes | false | `0.0` ❌ |
| `Float16_b → Float32` | No / Yes | false | `0.0` ❌ |
| `Float32 → Float16_b` | No | false | `0.0` ❌ |
| `Float32 → Float32` | No | false | `0.0` ❌ |
| `Float32 → Float16_b` | **Yes** | **true** | **`1.0` ✅** |
| `Float32 → Float32` | **Yes** | **true** | **`1.0` ✅** |

A clean 2/6 partition on `unpack_to_dest`, and `signbit(-1.0) = 1.0` on all eight. **The kernel's sign-bit read
is correct.** Where a real `-0.0` reaches the LREG it returns 1; on the six datacopy pipelines the datum arrives
as `+0.0`. This confirms the hypothesis the tree had recorded but not measured, and refutes the issue's reading.

---

## 4. ISA cross-check and root causes

### 4.1 Finding 1 — `signbit(-0.0)` → **documented hardware behaviour, not a kernel bug**

The issue calls this "a kernel-contract bug rather than a hardware one". It is the opposite.

On every pipeline except unpack-to-dest, the SFPU is fed through SrcB and a datacopy, which is `MOVB2D`
(`tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_datacopy.h:85-113`). Its functional model reads:

> ```c
> bool FlushDenormals = !ConfigState.ALU_ACC_CTRL_Zero_Flag_disabled_src;
> ...
> if (FlushDenormals && !(SrcBVal & 0xff)) SrcBVal = 0;
> ```
> — `WormholeB0/TensixTile/TensixCoprocessor/MOVB2D.md:56,81`

The assignment zeroes the **whole 19-bit datum, sign bit included**. `SrcASrcB.md` states the consequence
outright:

> "If using `MOVA2D` or `MOVB2D` with this type, `ALU_ACC_CTRL_Zero_Flag_disabled_src` should be set, as
> otherwise **the high 11 bits will be treated as zero when the low 8 bits are zero**."
> — `SrcASrcB.md:88`

For a `-0.0` the low 8 bits (exponent and low mantissa) are zero, so the sign bit is cleared and `+0.0` reaches
Dst. `FlushDenormals` is on unless `ALU_ACC_CTRL_Zero_Flag_disabled_src` is set, which the datacopy path does
not set.

**Verdict: the ISA describes this. Not a bug.** The `signbit` kernel reads the sign bit correctly
(`ckernel_sfpu_signbit.h`: `shft(as<vUInt>(in), -31)` then `int32_to_float`), which the `unpack_to_dest` column
proves. The tree already reflects this — `negative_zero_delivered()` keeps the probe off the pipelines that
cannot deliver it, and `_assert_signed_zero_partition_valid()` asserts `Signbit` has no divergence entries. **The
issue text is stale on this point and should be corrected**; there is a documented knob
(`ALU_ACC_CTRL_Zero_Flag_disabled_src`) if a `-0.0` ever has to survive the datacopy.

### 4.2 Finding 2 — `0/0` and `x%0` → inf — **documented hardware behaviour, not a kernel bug**

The issue attributes this to "the kernels' own reciprocal composition rather than the multiply". The measurement
in §3 refutes that: every one of these kernels produces a NaN, and `fmod`/`remainder` produce exactly the
`0x7FC00000` their explicit `v_if(b == 0.0f) { result = quiet_NaN(); }` guard writes
(`ckernel_sfpu_binary_fmod.h`). The NaN is then destroyed by the narrowing to bf16, and **the ISA documents that
narrowing in one sentence**:

> "**Rounding:** Denormals flushed to zero. Minus zero converted to positive zero. **If the exponent is 8 bits
> wide, NaN becomes infinity (this is a potentially surprising behavior).**"
> — `WormholeB0/TensixTile/TensixCoprocessor/Packers/FormatConversion.md:28`

fp32 → BF16 is exactly an 8-bit-exponent narrowing. The same page repeats it for the late conversion
("NaN can become infinity, if that is the outcome of truncating mantissa bits", line 81), and `SFPSTORE.md`
carries the DEST-side half:

> "If converting to BF16, the mantissa truncation can turn _some_ NaN values into infinity, so software is again
> advised to avoid NaN inputs for this conversion." — `SFPSTORE.md:38`

Which of the two stages fires depends on the variant — for `Float32→Float16_b` / `dest_acc=Yes` it can only be
the pack, since that variant differs from the passing `Float32→Float32` one in the pack format alone. Both
stages have a documented NaN→infinity conversion, so the observable is documented either way.

Two supporting details, both consistent:

* `xlogy`'s NaN is `0xFFC00001` and `Erfinv`'s is `0x7FC00001` — mantissa LSB set, which is precisely what the
  ISA guarantees for an arithmetic-emitted NaN: *"If a NaN is emitted, then the least significant bit of the
  mantissa is guaranteed to be set; other bits of the mantissa might or might not be set, and the sign bit might
  or might not be set."* (`SFPMAD.md:58`, and `FloatBitPatterns.md:25`). A NaN whose only set mantissa bit is
  low is *guaranteed* to become infinity when the low 16 bits are dropped.
* `Dst.md` notes in passing that *"Dst doesn't support NaNs or infinities"* for the 16-bit modes.

**Verdict: the ISA describes this. Not a kernel bug — but it is a real trap.** Any kernel that returns NaN
cannot deliver it on a bf16 pipeline. **The issue's diagnosis should be corrected**, and the xfail reasons in
`test_sfpu_binary.py:1376-1404` (which currently say "not explained by the ISA, and not expected to change on
Blackhole") should be rewritten to cite `Packers/FormatConversion.md:28` and re-scoped: they should assert NaN
on `Float32→Float32`/`dest_acc=Yes` and expect infinity elsewhere, rather than tolerating both.

### 4.3 Finding 3 — `0 ** 0` → 0 — **genuine kernel defect, ISA does not prescribe the result**

This is the only one of findings 2/3 that survives on the fully fp32 pipeline, so it is the kernel.

First, the tests do **not** use `calculate_sfpu_binary_pow` in `ckernel_sfpu_binary_pow.h`.
`MathOperation.SfpuElwpow` maps to `BinaryOp::POW` (`llk_params.py:241`), which dispatches to
`calculate_sfpu_binary_power` in **`ckernel_sfpu_binary.h:23-83`**. That function:

```c
// Base case when input is 0. ln(0) = -inf
v_if(base == 0.0f) { log_result = -std::numeric_limits<float>::infinity(); }
v_endif;

// Take exp(pow * log(base)) to produce base^pow
sfpi::vFloat val = pow * log_result;                    // <-- 0 * -inf  ==>  NaN
sfpi::vFloat result = _sfpu_exp_(sfpi::setsgn(val, 0));
v_if(val < 0) { result = sfpu_reciprocal_iter<2>(result); }   // <-- SFPSETCC on a NaN
v_endif;
```

`ln(0)` is handled — correctly — as `-inf`. Then for `pow == 0` the very next line forms the indeterminate
`0 × -inf`, which `SFPMAD.md:58` says is a NaN ("following the usual IEEE754 rules"). `exp(NaN)` then collapses
to `+0`, and the `v_if(val < 0)` guarding the negative-exponent reciprocal is evaluated on a NaN, which the
Wormhole ISA explicitly leaves **undefined**:

> `SFPSETCC` — "Provided that `VC` is neither negative zero nor any kind of NaN: Set per-lane flags based on
> `VC < 0` or …" — `VectorUnit.md:24`

A characterisation sweep confirms the defect is exactly this one cell, and that every neighbouring path is fine:

| pair | golden | hw | |
|---|---|---|---|
| `0 ** 0` | `1.0` | **`0.0`** | ❌ the only failure |
| `0 ** 1`, `0 ** 2` | `0.0` | `0.0` | ✅ `val = y·(−inf) = −inf` → reciprocal branch → `+0` |
| `1 ** 0`, `2 ** 0`, `4 ** 0`, `1e-30 ** 0` | `1.0` | `1.0` | ✅ `log_result` finite → `val = 0` → `exp(0) = 1` |
| `0 ** -1` | `inf` | `inf` | ✅ `val = +inf` → `exp(+inf)` |

So the issue's own diagnosis — "pow evaluates exp(b·ln a), so a composition artifact" — is **correct**, and the
indeterminate can now be named: `val = pow * log_result` at `(0, 0)`.

**Verdict: not prescribed by the ISA; a real kernel defect.** The mechanism (IEEE `0 × inf`, and an undefined
NaN comparison) is documented; the wrong answer is the kernel's. **Fix:** an explicit `v_if(pow == 0.0f)
{ result = 1.0f; }` after the composition. IEEE 754 defines `pow(x, 0) = 1` for every `x`, including `0` and
NaN, so the guard is unconditional and also removes the undefined `v_if(val < 0)` evaluation for this input.

### 4.4 Finding 4 — `RsqrtCompat(0)` → 1.7014e38 — **genuine kernel defect, ISA does not prescribe the result**

`rsqrt_compat` is a software composition (`tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_rsqrt_compat.h`), so
the ISA prescribes no answer — the issue is right about that. The value is fully derivable from the kernel:

`_calculate_rsqrt_compat_(0)` first takes `_sqrt_compat_(0)`, which has a `v_if (val != 0.0f)` guard and so
returns `0.0`. That `0.0` then goes to `_reciprocal_compat_`, which has **no zero guard at all**:

```c
sfpi::vFloat val = sfpi::setsgn(in, 1);   // -0.0
val = setexp(val, 126);                   // -0.5  <-- the input's magnitude is now gone
...                                       // Newton-Raphson converges to 1/0.5 = 2.0 (slightly under)
sfpi::vInt orig_exp = exexp(in);           // exexp(0.0)      = -127
sfpi::vInt new_exp  = exexp(result);       // exexp(1.99997…) =    0
new_exp -= orig_exp;                       // 0 - (-127)      =  127
new_exp += 126;                            //                 =  253
v_if (new_exp < 0) { result = 0.0F; new_exp = 0; }   // guards overflow only, not the zero input
v_endif;
return setexp(result, new_exp);
```

`exexp(0.0)` is `0 - 127 = -127` (`SFPEXEXP.md`: `Exp - Bias` on the raw field, which is 0 for a zero input), so
the exponent difference lands on **253** — an ordinary finite exponent, one short of the 255 that would mean
infinity. `setexp` writes the field and the surviving mantissa ≈ 1.99997 gives
`1.99997 × 2^(253-127) = 1.70135e38` = **`0x7EFFFD9E`**, matching the measurement to the bit. Rounded to bf16
that is `0x7F00` = `1.7014118e38`, the value in the issue.

So this is not saturation and there is no clamp involved: it is the exponent-difference arithmetic
`126 - exexp(in)` evaluated at `exexp(0.0) = -127`, in a function whose only guard covers the opposite overflow
direction. That also explains cleanly why **plain `Rsqrt` does not diverge at the same pole**: it uses
`sfpu_reciprocal_iter` (`ckernel_sfpu_recip.h`), which builds its scale as `~in.Exp` precisely so that, in its
own words, *"in.Exp == 0 results in ±inf, and in.Exp == 255 results in ±0"*. The modern kernel handles the pole
by construction; the legacy compat one never did.

**Verdict: not prescribed by the ISA; a real defect in a legacy kernel.** **Fix:** a zero/inf guard in
`_reciprocal_compat_`, or retire `_reciprocal_compat_`/`_calculate_rsqrt_compat_` in favour of
`sfpu_reciprocal_iter`, which is already correct here.

### 4.5 Finding 5 — `Erfinv(±1)` → NaN — **genuine kernel defect, root cause is `sqrt_custom(+inf)`**

`calculate_erfinv_body` (`ckernel_sfpu_erfinv.h`) evaluates `log(1 - x²)`, which at `x = ±1` is `log(0)`. The log
kernel handles that correctly — `ckernel_sfpu_log.h:84-105` uses `addexp(a, -1)` to wrap exponent 0 to 255 and
returns `-inf`. Propagating: `tmp = -4.33 + (-0.5)(-inf) = +inf`, and `calculated_value = +inf`. The next step is
`sfpu_sqrt_custom(+inf)`, and that is where it breaks:

```c
sfpi::vFloat approx = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));
sfpi::vFloat neg_half_val = val * -0.5f;
approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
```

For `val = +inf` the fast-inverse-sqrt seed is `as_float(0x5F370000 - 0x3FC00000) = as_float(0x1F770000)`
≈ `5.23e-20`. Squaring it gives ≈ `2.7e-39`, **below the fp32 minimum normal**, so it is a denormal — and
`SFPMAD.md:56,62` says *"Denormal inputs are treated as if they were zero"* and *"If the output (before rounding)
is denormal or negative zero, it'll be flushed to positive zero"*. So `approx * approx` is `+0`, and the next
multiply is `0 × (val * -0.5f)` = `0 × -inf` = **NaN** — again `SFPMAD.md:58`, and again with the mantissa LSB
set, which is exactly the `0x7FC00001` measured.

This chain predicts that `sqrt_custom(+inf)` is NaN *on its own*, independently of erfinv. Tested directly:

| op | `x = +inf` | `x = 4.0` | `x = 0.0` | `x = 1e-30` |
|---|---|---|---|---|
| `SqrtCustom` | **`0x7FC00001` NaN** ❌ | `1.99999` | `0.0` | `9.99999e-16` |
| `Sqrt` | `0x7F800000` **+inf** ✅ | `2.0` | `0.0` | `1.0e-15` |

Prediction confirmed, and the same NaN bit pattern. **The defect is in `sfpu_sqrt_custom`, not in `erfinv`** —
erfinv is one consumer of it. This is a finding beyond the issue's scope: `SqrtCustom(+inf) = NaN` is a bug in
its own right, and every other consumer of `sfpu_sqrt_custom` inherits it.

**Verdict: not prescribed by the ISA; a real kernel defect.** Both hardware behaviours in the chain (denormal
flush, IEEE `0 × inf`) are documented; the composition's failure to survive them is the kernel's. **Fix:** guard
`sfpu_sqrt_custom` for non-finite input (`v_if(val == inf) { out = val; }`, alongside the existing
`v_if(val != 0.0f)`), which repairs `erfinv(±1)` as a side effect. Note the issue's "saturates" wording should be
corrected to "returns NaN", and the "fp32-dest combinations only" scoping explained as visibility (§3), not as a
tolerance effect — the issue calls it "tolerance-shaped rather than semantic", which is wrong: it is semantic on
all eight combinations.

---

## 5. Findings this run turned up that the issue does not cover

### 5.1 The `negative_zero_golden` class no longer diverges on Wormhole — 16 XPASS

`test_sfpu_binary.py:1290-1304` records that a zero result's sign is lost on Wormhole (`div(0, -x)` → `+0.0`,
and likewise for `fmod`/`remainder`/`xlogy`), citing `SFPMAD.md`'s *"If the output (before rounding) is denormal
or negative zero, it'll be flushed to positive zero"*, and marks the class non-strict-xfail on Wormhole while
asserting it on Blackhole. **On this run all 16 of those cells XPASS** — `div`, `xlogy`, `fmod` and `remainder`,
at both `dest_acc` values.

The likely cause is that the kernels no longer rely on SFPMAD to carry the sign: `sfpu_reciprocal_iter`
(`ckernel_sfpu_recip.h`) ends with `y = sfpi::copysgn(y, in)`, and `div` is `in0 * sfpu_reciprocal_iter<2>(in1)`
(`ckernel_sfpu_binary.h:107`). `copysgn` is `SFPSETSGN`, not an SFPMAD, so the negative-zero flush does not apply
to it. That attribution is an inference from the kernel source; the 16 XPASS themselves are measured.

**Action:** `_WORMHOLE_ONLY_EDGE_CLASSES` and the comment block above it are now stale. The class should be
asserted on Wormhole too, so a regression fails instead of quietly returning to XFAIL. This is the same shape of
finding as items 3 and 4 in `WORMHOLE_MEASUREMENT_RESULTS.md`: a recorded Wormhole claim that no longer holds.

### 5.2 `SqrtCustom(+inf) = NaN`

See §4.5. Worth filing separately from erfinv, since it is the shared root cause.

### 5.3 Accuracy observations, not edge cases

* `rsqrt_compat(2.384e-07)` = `2044.99` vs `2048.0` — ~0.15 % relative error (`Float32→Float32`).
* `pow(4, 0.5)` = `1.99740` vs `2.0` — ~0.13 %, the `exp(b·ln a)` composition's own error.
* `SqrtCustom(4.0)` = `1.99999` vs `2.0`, against `Sqrt(4.0)` = exactly `2.0`.

---

## 6. Summary table

| # | Finding | Reproduces | ISA verdict | Root cause | Issue text accurate? |
|---|---|---|---|---|---|
| 1 | `signbit(-0.0)` → 0 | 6/8 (and 1 on the other 2) | **Documented** — `MOVB2D.md:56,81`, `SrcASrcB.md:88` | `MOVB2D` `FlushDenormals` zeroes the whole 19-bit datum, sign included, when the low 8 bits are 0 | ❌ calls it a kernel-contract bug; the kernel is correct |
| 2 | `0/0`, `x%0` → inf | Yes, on narrowing pipelines only | **Documented** — `Packers/FormatConversion.md:28`, `SFPSTORE.md:38` | fp32→BF16 conversion turns NaN into infinity; kernels do produce NaN | ❌ blames the kernels' reciprocal composition |
| 3 | `0**0` → 0 | Yes, all 6 incl. fp32 | **Not prescribed** | `ckernel_sfpu_binary.h:56` — `val = pow * log_result` = `0 × -inf` → NaN → `exp(NaN)` → +0 | ✅ diagnosis correct |
| 4 | `RsqrtCompat(0)` → 1.7014e38 | 8/8 | **Not prescribed** | `_reciprocal_compat_` exponent arithmetic `126 - exexp(0) = 253`; no zero guard | ⚠️ value is `0x7EFFFD9E`; `0x7F000000` is its bf16 rounding |
| 5 | `Erfinv(±1)` | 2/8 visible, wrong on all 8 | **Not prescribed** | `sqrt_custom(+inf)`: seed² underflows to +0, then `0 × -inf` → NaN | ❌ returns NaN, not a saturated value; semantic, not tolerance-shaped |

**Where the "we have a problem" cases stand:** findings 3, 4 and 5 are genuine kernel defects, each now located
to a line, each with a concrete fix, and none needing a hardware answer. Findings 1 and 2 need no kernel work —
they need the issue text and the corresponding xfail reasons corrected, and finding 2 in particular is worth
writing down as a standing constraint: **a kernel cannot return NaN on a bf16 pipeline.**

## 7. Files

| Path | Status |
|---|---|
| `tt_metal/tt-llk/tests/python_tests/test_sfpu_wh_issue52930_probe.py` | the instrument for this run — records, does not assert. **Not for merge** |
| `ISSUE_52930_INVESTIGATION.md` | this log |

One fix plan per kernel-level defect (findings 3, 4, 5). Findings 1 and 2 have no plan because they are
documented hardware behaviour — they need issue-text and xfail-reason corrections, not kernel work.

| Plan | Defect | Fix shape |
|---|---|---|
| [FIX_PLAN_52930_pow_zero_zero.md](FIX_PLAN_52930_pow_zero_zero.md) | finding 3 — `0**0` → 0 | add `pow == 0 → 1` guard in `calculate_sfpu_binary_power` |
| [FIX_PLAN_52930_reciprocal_compat_pole.md](FIX_PLAN_52930_reciprocal_compat_pole.md) | finding 4 — `RsqrtCompat(0)` → 1.7e38 | add the missing zero pole guard in `_reciprocal_compat_`; **scope first** — the public `recip()` defaults to this path |
| [FIX_PLAN_52930_sqrt_custom_infinity.md](FIX_PLAN_52930_sqrt_custom_infinity.md) | finding 5 + §5.2 — `Erfinv(±1)` → NaN | exclude non-finite input in `sfpu_sqrt_custom`; repairs `erfinv` as a side effect |
