# Fix plan — `sqrt_custom(+inf)` returns NaN, which is why `Erfinv(±1)` does too

**Issue:** [#52930](https://github.com/tenstorrent/tt-metal/issues/52930), finding 5 (and §5.2, which is the same
bug seen from its root).
**Evidence:** [ISSUE_52930_INVESTIGATION.md](ISSUE_52930_INVESTIGATION.md) §3, §4.5.
**Classification:** kernel defect. The two hardware behaviours in the chain are documented; the composition's
failure to survive them is the kernel's.

> **The issue reports this as an `erfinv` problem. It is not.** The defect is in `sfpu_sqrt_custom`, and `erfinv`
> is one of its consumers. Fixing `sqrt_custom` repairs `erfinv(±1)` as a side effect, and fixes every other
> consumer at the same time. Scope the work on the shared helper.

---

## 1. The defect

Two corrections to the issue text before anything else:

* **The op does not saturate — it returns NaN.** `Erfinv(±1)` gives `0x7FC00001` / `0xFFC00001`, not a large
  finite. The issue's "saturates rather than returning ±inf" is wrong.
* **It is not "tolerance-shaped rather than semantic".** It is semantic, and wrong on **all 8** format
  combinations. Only 2 can see it.

Measured on Wormhole n300:

| op | `x = +inf` | `x = 4.0` | `x = 0.0` | `x = 1e-30` |
|---|---|---|---|---|
| `SqrtCustom` | **`0x7FC00001` NaN** ❌ | `1.99999` | `0.0` | `9.99999e-16` |
| `Sqrt` | `0x7F800000` **+inf** ✅ | `2.0` | `0.0` | `1.0e-15` |

`Erfinv(±1)`, by combination:

| input → output | `dest_acc` | `erfinv(+1)` | why |
|---|---|---|---|
| `Float16_b → Float32` | **Yes** | **`0x7FC00001` NaN** ❌ | fp32 DEST *and* fp32 pack — the NaN is visible |
| `Float32 → Float32` | **Yes** | **`0x7FC00001` NaN** ❌ | same |
| all six others | — | `0x7F800000` +inf ✅ | **passes by accident** — see §3 |

## 2. Root cause

`calculate_erfinv_body` (`wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_erfinv.h:27`) evaluates
`log(1 - x²)`, which at `x = ±1` is `log(0)`. The log kernel gets that right — `ckernel_sfpu_log.h:84-105` uses
`addexp(a, -1)` to wrap exponent 0 to 255 and returns `-inf`. Propagating through erfinv:
`tmp = -4.33 + (-0.5)(-inf) = +inf`, so `calculated_value = +inf`, and line 39 calls
`sfpu_sqrt_custom<false, 2>(+inf)`.

`ckernel_sfpu_sqrt_custom.h:21-36`:

```c
24:      v_if(val != 0.0f) {                       // guards the 0 pole, not the inf pole
26:          sfpi::vFloat approx = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));
27:          sfpi::vFloat neg_half_val = val * -0.5f;
30:          approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
32:          out = approx * val;
     }
```

For `val = +inf` the fast-inverse-sqrt seed on line 26 is
`as_float(0x5F370000 - (0x7F800000 >> 1)) = as_float(0x1F770000)` ≈ `5.23e-20`. Line 30 then squares it:
`≈ 2.7e-39`, which is **below the fp32 minimum normal** (`1.18e-38`), so it is a denormal — and `SFPMAD.md:56,62`
says *"Denormal inputs are treated as if they were zero"* and *"If the output (before rounding) is denormal or
negative zero, it'll be flushed to positive zero"*. So `approx * approx` is `+0`.

`neg_half_val` is `-inf`. The next multiply is therefore `0 × -inf` = **NaN** (`SFPMAD.md:58`, IEEE rules), and
the same sentence explains the bit pattern: *"If a NaN is emitted, then the least significant bit of the mantissa
is guaranteed to be set"* — which is exactly the `...0001` measured. Everything downstream is NaN.

**This was confirmed independently rather than inferred:** the chain predicts `sqrt_custom(+inf)` is NaN on its
own, with no erfinv involved, and a direct probe of `SqrtCustom` shows exactly that (§1's first table). That is
what localises the defect to the helper.

## 3. Why 6 of 8 combinations pass, and why that is dangerous

The 6 passing combinations are **not** correct — they are the same NaN, narrowed. Wherever the pipeline drops
fp32 to BF16, `Packers/FormatConversion.md:28` turns the NaN into ±Infinity, and for `erfinv(±1)` the golden
*is* ∓inf/±inf — so the wrong answer is converted into the right one by a lossy format conversion.

Two consequences worth stating in the PR:

1. The scoping in the issue ("fp32-dest combinations only") is a statement about **visibility**, not about where
   the kernel is wrong.
2. This is the general trap from the investigation's §4.2 running in reverse: the bf16 narrowing usually *hides*
   a correct NaN, and here it *hides a wrong one*. Any future edge finding scoped to "fp32 combinations only"
   should be re-read with this in mind.

After the fix, all 8 return `±inf` — the 6 keep passing, now for the right reason, and the 2 stop failing.

## 4. Blast radius

`sfpu_sqrt_custom` is a shared helper in all three arches:

| Consumer | Argument it passes | Can it be non-finite? |
|---|---|---|
| `erfinv` (`ckernel_sfpu_erfinv.h:39,43`) | `tmp² - log·(1/a)`, then `tmp + sqrt(...)` | **Yes** — `+inf` at `x = ±1`. This is the reported bug |
| `asin` / `acos` range reduction (`ckernel_sfpu_trigonometry.h:503`) | `(1.0f - abs_v) * 0.5f` | For in-domain `|v| ≤ 1`, the argument is in `[0, 0.5]` — safe. For `v = ±inf` it is `-inf`, and for `|v| > 1` it is negative — both out of asin's domain, both unguarded in `sqrt_custom` |
| `SqrtCustom` op (`tests/helpers/include/sfpu_test_helpers.h:25`) | whatever the test drives | **Yes** — this is the direct probe |
| Blackhole | `blackhole/…/ckernel_sfpu_sqrt_custom.h` — same seed, same two Newton steps (unrolled rather than templated on `NEWTON_ITERATIONS`). **Same defect** | |
| Quasar | `quasar/…/ckernel_sfpu_sqrt_custom.h` exists and is consumed by its `trigonometry.h`; no `erfinv` seen there. Verify before assuming | |

These are metal-side ckernels, not tt-llk lib headers, so the `metal-integration.md` propagation checklist does
not apply — but the change must land on WH and BH together, since the suite runs both and the xfail table in §6
is not arch-keyed.

## 5. The fix — two options

### Option A (preferred): guard the non-finite input in `sqrt_custom`

`sqrt(+inf) = +inf` and `sqrt(NaN) = NaN` are both pass-throughs, so extending the existing guard covers them at
one instruction's cost. The current guard already establishes the pattern — `out` is pre-initialised to `val`
(line 23), and the `v_if` only overwrites it for inputs the iteration can handle:

```c
    sfpi::vFloat val = in;
    sfpi::vFloat out = val;   // already the correct answer for 0, +inf and NaN
    // The seed for +inf is ~5.2e-20; squaring it underflows to a denormal, which SFPMAD
    // flushes to +0, and the next multiply is then 0 * -inf = NaN. So exclude non-finite
    // inputs alongside zero -- for all three, passing `val` through is already right.
    v_if(val != 0.0f && sfpi::exexp(val, sfpi::ExponentMode::Biased) != 255) {
        ...unchanged...
    }
    v_endif;
```

Using the biased exponent field (`== 255` means inf or NaN) rather than a float comparison against infinity
matters: `VectorUnit.md:24` says `SFPSETCC` is only specified *"provided that `VC` is neither negative zero nor
any kind of NaN"*, so comparing a possibly-NaN `val` against `inf` with `<`/`==` is the undefined path this
codebase should stop walking. `SFPEXEXP` has no such caveat.

`-inf` and negative inputs: `sqrt(-inf)` and `sqrt(x < 0)` are NaN in IEEE. The guard above passes `-inf`
through unchanged, which returns `-inf` rather than NaN — better than the current garbage but still not IEEE. If
the negative side matters to a consumer, add `v_if(val < 0.0f) { out = NaN; }`; **note this is a behaviour change
for existing negative inputs and needs its own justification**, so keep it out of the minimal fix unless asin's
out-of-domain path turns out to depend on it.

### Option B: redirect to `_calculate_sqrt_body_`

`ckernel_sfpu_sqrt.h:25-60` handles both poles explicitly with an `infinity_bits` comparison, which is why plain
`Sqrt(+inf)` is correct. Retiring `sqrt_custom` in favour of it would delete the second implementation.

* **Pro:** one square root in the tree instead of two, and the surviving one is already correct and already
  tested.
* **Con:** different accuracy. `SqrtCustom(4.0)` = `1.99999` against `Sqrt(4.0)` = exactly `2.0`, so every
  consumer's numerics move — including `asin`/`acos`, whose polynomials were fitted around this helper's error.
  That needs an accuracy sweep of the trig ops, not just an edge check.

**Recommendation:** ship **Option A**. It is a one-line predicate change per arch that fixes the reported bug and
leaves all finite numerics bit-identical, so it does not need the trig-accuracy work. Option B is the better end
state and worth filing as a follow-up, but it is a different-sized job.

## 6. Test changes

### Remove the pins (same commit as the fix)

1. **`tt_metal/tt-llk/tests/python_tests/test_sfpu_unary.py:619-622`** — the `MathOperation.Erfinv` entry in
   `_EDGE_KNOWN_DIVERGENCES` (both combinations).
2. **`test_sfpu_unary.py:676`** — the `MathOperation.Erfinv` entry in `_EDGE_DIVERGENCE_REASON`.
3. **`test_sfpu_unary.py:598-599`** — the "STILL OPEN" comment block loses its `Erfinv at ±1` paragraph. Replace
   the "tolerance-shaped rather than semantic" claim rather than deleting it silently: it was wrong, and the file's
   convention is to record why a reading moved.

`_OP_SINGULARITIES` at `sfpu_domains.py:1540` already drives `±1` for `Erfinv`, so removal alone makes
`test_eltwise_unary_sfpu_edges[…-Erfinv-…]` **assert** `erfinv(±1) == ±inf` on all 8 combinations.

### Close the gap that let this hide

The real lesson is that **`SqrtCustom` is not in `SPECIALS_READY_OPS`** (`sfpu_domains.py:1992-2014` lists `Sqrt`
and `Rsqrt` but not `SqrtCustom`), so `+inf` was never driven at it — the defect had to be found through a
consumer. Enrol it:

* Add `MathOperation.SqrtCustom` to `SPECIALS_READY_OPS` with the IEEE contract, mirroring the existing
  `MathOperation.Sqrt` entry: `sqrt(+inf) = +inf`, `sqrt(-inf) = NaN`, `sqrt(NaN) = NaN`, `sqrt(±0) = ±0`.
* Confirm `UnarySFPUGolden`'s `SqrtCustom` golden defines those cases; if it routes to the same `_sqrt` as
  `Sqrt`, nothing to do.
* Expect the results to be observable **only** on the fp32-DEST/fp32-pack combinations (§3), which
  `specials_safe()` and the `_gate_unspecified_nan_sign()` gate should already scope correctly.
* If Option A's guard leaves `sqrt_custom(-inf) = -inf` rather than NaN, that row must be recorded as a known
  divergence with the reason above — **do not enrol the op and then paper over `-inf` silently.**

Same argument applies to `Erfinv`, also absent from `SPECIALS_READY_OPS`; enrolling it is optional and secondary.

## 7. Verification

```bash
cd tt_metal/tt-llk

# 1. Baseline both the root cause and the symptom.
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_wh_issue52930_probe.py --k "sqrt_custom_at_infinity or (unary_edge_values and Erfinv)"
#    expect: SqrtCustom(inf) -> 0x7FC00001, Erfinv(+/-1) -> 0x7FC00001/0xFFC00001 on the 2 fp32 combos

# 2. After the fix, the same probe must show:
#      SqrtCustom(inf) -> 0x7F800000
#      Erfinv(+1)      -> 0x7F800000 on ALL 8 (the 6 that passed by narrowing still pass)
#      SqrtCustom(4.0) -> 0x3FFFFFAC, 0.0 -> 0x00000000, 1e-30 -> unchanged   <-- the no-overreach check

# 3. Shipped sweeps with the tables removed, both arches.
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_unary.py --k "edges and (Erfinv or SqrtCustom or Sqrt)"
bash .claude/scripts/run_test.sh run --worktree $PWD --arch blackhole \
    --test test_sfpu_unary.py --k "edges and (Erfinv or SqrtCustom or Sqrt)"

# 4. The other consumers -- the accuracy check that Option A is supposed to make trivial.
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_unary.py --k "Asin or Acos or Erfinv or SqrtCustom"
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole --test test_sfpu_unary.py -m accuracy
```

Acceptance:
- `SqrtCustom(+inf) == +inf` (`0x7F800000`), both arches.
- `Erfinv(±1) == ±inf` on **all 8** combinations, both arches — not just the 2 that were failing.
- Every finite input bit-identical to baseline, `SqrtCustom(4.0) == 0x3FFFFFAC` included. Option A's whole
  justification is that this holds; if it does not, the guard is wrong.
- `asin` / `acos` accuracy unchanged.
- Erfinv's 2 xfail entries removed; `SqrtCustom` enrolled in `SPECIALS_READY_OPS` with any residual `-inf`
  divergence recorded explicitly rather than tolerated.
- No `perf_sfpu_erfinv_wh.py` regression from the added predicate.

## 8. Risks

| Risk | Mitigation |
|---|---|
| The guard is written as `val != INFINITY`, evaluating `SFPSETCC` on a possible NaN — the undefined path | §5 specifies the biased-exponent form (`exexp(val, Biased) != 255`), which has no NaN caveat |
| Widening the `v_if` predicate changes which lanes run the iteration and perturbs finite results | §7 step 2's finite rows are the check; they must be bit-identical |
| Extra predicate pushes the kernel past the SFPU register-allocator budget (this codebase has hit reload ICEs before — see the comment in `ckernel_sfpu_binary_pow.h`) | Compile both arches early; if it ICEs, hoist the exponent extraction or fold it into the existing comparison |
| `Erfinv`'s 6 accidentally-passing combinations regress to a *different* wrong answer | §7 acceptance requires all 8 to read `±inf`, so a regression there fails rather than hiding |
| Option B chosen under time pressure without the trig accuracy sweep | Option B is explicitly a follow-up; do not mix it into this PR |
| Quasar diverges and is missed | §4 requires checking quasar's `sqrt_custom` and its consumers before closing |

## 9. Effort

Option A: small — one predicate per arch (2, possibly 3 files), two table deletions, one `SPECIALS_READY_OPS`
entry, and the before/after probe comparison. The `SPECIALS_READY_OPS` enrolment is the part most likely to
surface further rows (`-inf`, negative inputs); budget for one extra measurement round there.

## 10. Related, file separately

* **`SqrtCustom(+inf) = NaN` in its own right** (investigation §5.2) — this plan fixes it, but it deserves its own
  issue so the fix is not filed as "an erfinv change" and so the other consumers are searchable from it.
* **`sqrt_custom` on negative input** — unguarded, and reachable from `asin`/`acos` when `|v| > 1`. Out of scope
  here (§5), but it is the same missing-domain-guard shape.
* **Two square-root implementations with different accuracy** — Option B. Worth an issue on its own.
