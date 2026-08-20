# Fix plan — `0 ** 0` returns 0 instead of 1

**Issue:** [#52930](https://github.com/tenstorrent/tt-metal/issues/52930), finding 3.
**Evidence:** [ISSUE_52930_INVESTIGATION.md](ISSUE_52930_INVESTIGATION.md) §3, §4.3.
**Classification:** kernel defect. The ISA prescribes no answer for `pow`; the mechanism it *does* document
(IEEE `0 × inf` → NaN, and `SFPSETCC` being undefined on a NaN) is what the kernel walks into.

---

## 1. The defect

`pow(0, 0)` returns `+0` (`0x00000000`) where C, torch, IEEE 754 and the suite's golden all give `1.0`.

Measured on Wormhole n300, and it is the **only** failing pair in its neighbourhood — which is what makes the fix
so tightly scoped:

| pair | golden | hw | |
|---|---|---|---|
| `0 ** 0` | `1.0` | **`0.0`** | ❌ |
| `0 ** 1`, `0 ** 2` | `0.0` | `0.0` | ✅ |
| `1 ** 0`, `2 ** 0`, `4 ** 0`, `1e-30 ** 0` | `1.0` | `1.0` | ✅ |
| `0 ** -1` | `inf` | `inf` | ✅ |

Unlike the `div`/`fmod`/`remainder`/`xlogy` findings in the same issue, this one **survives on
`Float32→Float32` with `dest_acc=Yes`** — the pipeline where nothing is narrowed — so it is not the documented
NaN→infinity conversion. It is the arithmetic.

## 2. Root cause

The op under test is `MathOperation.SfpuElwpow` → `BinaryOp::POW` (`llk_params.py:241`) →
`calculate_sfpu_binary_power` in **`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h:23-83`**.
Note this is *not* `calculate_sfpu_binary_pow` in `ckernel_sfpu_binary_pow.h`, which is a different, unused-here
implementation — an easy wrong turn when reading this code.

```c
51:  v_if(base == 0.0f) { log_result = -std::numeric_limits<float>::infinity(); }   // ln(0) = -inf, correct
52:  v_endif;
57:  sfpi::vFloat val = pow * log_result;                    // pow == 0  =>  0 * -inf  =>  NaN
60:  sfpi::vFloat result = _sfpu_exp_(sfpi::setsgn(val, 0)); // exp(NaN) collapses to +0
62:  v_if(val < 0) { result = sfpu_reciprocal_iter<2>(result); }   // SFPSETCC on a NaN: undefined on WH
```

Line 51 correctly special-cases `ln(0)`. Line 57 then forms the indeterminate `0 × -inf`, which `SFPMAD.md:58`
says is a NaN ("following the usual IEEE754 rules"). Everything after that is downstream of a NaN, including the
`v_if(val < 0)` on line 62, which `VectorUnit.md:24` explicitly leaves undefined:

> `SFPSETCC` — "**Provided that `VC` is neither negative zero nor any kind of NaN**: Set per-lane flags based on
> `VC < 0` or …"

So there are two problems on one line's worth of code: a wrong result, and an undefined-behaviour predicate
evaluation. Both are removed by the same guard.

## 3. The fix

IEEE 754 defines `pow(x, 0) = 1` for **every** `x` — including `0`, `±inf` and NaN. So the guard is
unconditional in `pow`, needs no interaction with the `base` sign handling below it, and can be written as a
final override:

```c
    // IEEE 754: pow(x, 0) == 1 for every x, including 0, +/-inf and NaN. Without this the
    // composition forms 0 * ln(0) = 0 * -inf = NaN at base == 0 (SFPMAD.md), exp(NaN)
    // collapses to +0, and the v_if(val < 0) below is evaluated on a NaN, which the ISA
    // leaves undefined (VectorUnit.md, SFPSETCC).
    v_if(pow == 0.0f) { result = 1.0f; }
    v_endif;
```

**Placement:** after the negative-base block ends (i.e. immediately before `return result;` at
`ckernel_sfpu_binary.h:82`). It must come last, because the negative-base branch would otherwise apply
`setsgn2`/NaN handling to the `1.0` — `(-2) ** 0` is `1`, not `-1`, and `pow_rounded == pow` holds for `pow == 0`
so that branch does execute.

**Why not guard at line 57 instead** (e.g. skip the multiply when `pow == 0`): `v_if` on the SFPU predicates
lanes rather than branching, so the NaN is still formed in the inactive lanes and the undefined `v_if(val < 0)`
still sees it. Overriding the result at the end is both simpler and complete.

**Cost:** one `SFPSETCC` + one `SFPLOADI`/`SFPMOV` pair per iteration, i.e. ~2-3 cycles on a kernel that already
runs a log polynomial, an exp and a reciprocal. Immaterial, but see §6 for the check.

### Also worth fixing while in here

`0 ** -1` returns `+inf`, which matches torch, but it gets there by accident: this kernel has **no**
`base == 0 && pow < 0 → NaN` guard at all, and the `+inf` falls out of `exp(+inf)`. The sibling
`_sfpu_binary_power_f32_` in `ckernel_sfpu_binary_pow.h` *does* carry such a guard and its docstring promises
NaN. The two implementations therefore disagree on `0 ** negative`. **Do not change behaviour here as part of
this fix** — torch and the golden both want `inf`, so the current answer is the right one — but file the
docstring/guard disagreement in `ckernel_sfpu_binary_pow.h` separately so the two do not drift further.

## 4. Blast radius

| | |
|---|---|
| Wormhole | `wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h:23-83` |
| Blackhole | **identical structure** — `blackhole/…/ckernel_sfpu_binary.h:51-57`, same `v_if(base == 0.0f)` and same `val = pow * log_result`. Same fix, same placement |
| Quasar | no `BinaryOp::POW` in its `ckernel_sfpu_binary.h` (semantic file naming); check `llk_sfpu/` for its pow path before assuming it is unaffected |
| Callers | every consumer of `BinaryOp::POW` — the binary `pow` compute API and TTNN's `pow` with a tensor exponent. Scalar/unary `power` goes through `calculate_unary_power` and is **not** on this path |

This is a metal-side ckernel, not a tt-llk lib header, so the `metal-integration.md` propagation checklist does
not apply — but the change must land on **both** WH and BH in the same PR, since the suite runs both arches and
the xfail table below is not arch-keyed.

## 5. Test changes

The current behaviour is *pinned* by non-strict xfails, so the fix will turn them into XPASS rather than a
failure. Both entries must be removed in the same commit:

1. **`tt_metal/tt-llk/tests/python_tests/test_sfpu_binary.py:1345-1352`** — delete the entire
   `MathOperation.SfpuElwpow` entry from `_BINARY_EDGE_COMBINATIONS` (all 6 combinations).
2. **`test_sfpu_binary.py:1389-1392`** — delete the `MathOperation.SfpuElwpow` entry from
   `_BINARY_EDGE_REASON`.

The assertion at `test_sfpu_binary.py:1409` (`set(_BINARY_EDGE_REASON) == set(_BINARY_EDGE_COMBINATIONS)`) keeps
these two honest — removing one without the other fails at collection, which is the intent.

Once both are gone, `test_sfpu_binary_edges[…-SfpuElwpow-…-both_zero]` **asserts** `0**0 == 1` on all 8
combinations. No new test is needed: `edge_pair_values` already generates the `(0, 0)` pair and
`_classify_edge_pair` already routes it to `both_zero`. That is the regression test.

**Optional hardening**, worth doing because the fix is broader than the one cell it repairs: `SfpuElwpow` is not
in `SPECIALS_READY_OPS`, so `pow(inf, 0)` and `pow(nan, 0)` — which the new guard also makes correct — are never
driven. Enrolling it needs a `BinarySFPUGolden` that defines the non-finite cases, and the results are only
observable on `Float32→Float32`/`dest_acc=Yes` per the NaN-narrowing rule. Treat as a follow-up, not a blocker.

## 6. Verification

```bash
cd tt_metal/tt-llk

# 1. Before the change: reproduce, and capture the baseline values.
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_wh_issue52930_probe.py --k "pow_zero_characterisation"
#    expect: 0**0 -> 0x00000000, every other pair ok

# 2. After the change: the same probe must show 0**0 -> 0x3F800000 and NOTHING else moved.
#    The 8 other pairs in _POW_PAIRS are the guard against a fix that overreaches.

# 3. The shipped sweep, both arches, with the two table entries removed.
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_binary.py --k "binary_edges and SfpuElwpow"
#    expect: all pass, 0 xfail, 0 xpass
bash .claude/scripts/run_test.sh run --worktree $PWD --arch blackhole \
    --test test_sfpu_binary.py --k "binary_edges and SfpuElwpow"

# 4. Regression: the op's ordinary accuracy sweep must be unchanged.
bash .claude/scripts/run_test.sh run --worktree $PWD --arch wormhole \
    --test test_sfpu_binary.py --k "SfpuElwpow"
```

Acceptance:
- `0 ** 0 == 1.0` exactly (`0x3F800000`), on all 8 format combinations, both arches.
- `x ** 0`, `0 ** y`, `0 ** -1` and the `4 ** 0.5` accuracy figure (`1.99740`, ~0.13 % — the composition's own
  error, not something this fix addresses) all **unchanged**.
- Both xfail table entries gone; collection-time assertion still passes.
- No new `perf_sfpu_*` regression on the pow path (§4's cost note; check `perf_eltwise_binary_sfpu.py` if the
  binary pow appears there).

## 7. Risks

| Risk | Mitigation |
|---|---|
| Guard placed before the negative-base block, breaking `(-2) ** 0` | Placement is specified as the last statement before `return`; `_POW_PAIRS` in the probe should gain `(-2.0, 0.0)` to pin it |
| `v_if(pow == 0.0f)` itself hits the SFPSETCC negative-zero caveat when `pow` is `-0.0` | `VectorUnit.md:47` excludes negative zero from `SFPSETCC`'s contract. `pow == -0.0` should also return 1; add `(0.0, -0.0)` and `(2.0, -0.0)` to the probe and confirm. If `-0.0` mis-predicates, compare on `setsgn(pow, 0)` instead |
| Register pressure — this kernel already carries a comment about reloading `base` for exactly that reason (line 51) | If the SFPU register allocator ICEs or spills, fold the guard into the existing negative-base `v_if` chain rather than adding a new live value |
| Fix lands on WH only, Blackhole then XPASSes with the tables removed | Land both arches in one PR; §6 step 3 runs both |

## 8. Effort

Small: two one-line kernel edits (WH + BH), two table deletions, one probe extension. The measurement
infrastructure already exists. The bulk of the work is §6's before/after comparison and the two-arch run.
