# Result — `sqrt_custom(+inf)` fixed; `Erfinv(±1)` repaired as a side effect

**Plan:** [FIX_PLAN_52930_sqrt_custom_infinity.md](FIX_PLAN_52930_sqrt_custom_infinity.md) (finding 5 and §5.2
of [ISSUE_52930_INVESTIGATION.md](ISSUE_52930_INVESTIGATION.md)). This was the last of the three kernel
defects from #52930 still unimplemented.

**Answer in one line:** the plan's Option A works exactly as specified — **every finite input is bit-identical,
`Erfinv(±1)` is now `±inf` on all 8 combinations, and `Asin`/`Acos` are unchanged to the bit** — at a cost of
**+3 cycles per SFPU iteration**, which is +9.7 % on `SqrtCustom` itself, +4.0 % on `Erfinv`, +4.0 % on the
`dest_acc=No` half of `Asin`/`Acos`, and **0.00 % on their `dest_acc=Yes` half**.

| | Wormhole n300, silicon | |
|---|---|---|
| Tree | `tt-metal` @ `ldjurovic/wrong_sfpu_edge_cases`, on top of `f20d29cb26d` | |
| Runner | `tt_metal/tt-llk/.claude/scripts/run_test.sh` throughout | |
| Date | 2026-08-17 | |

---

## 1. The fix

The plan's Option A verbatim, on **Wormhole and Blackhole**. Quasar was surveyed as the plan asked and
carries the same defect, but it is deliberately left untouched — Quasar is out of scope for this PR:

```c
    // Zero and the non-finite inputs all pass `val` straight through, which is already the
    // answer for +/-0 and +inf. Non-finite has to be excluded because the seed below is
    // ~5.2e-20 for +inf; squaring it underflows to a denormal, SFPMAD flushes that to +0,
    // and the next multiply is 0 * -inf = NaN -- so sqrt_custom(+inf) was NaN, and every
    // consumer inherited it (erfinv(+/-1)). Tested on the biased exponent field rather than
    // a compare against infinity because SFPSETCC is only specified for inputs that are not
    // NaN (VectorUnit.md), and this predicate has to be evaluated on a possible NaN.
    v_if(val != 0.0f && sfpi::exexp(val, sfpi::ExponentMode::Biased) != 255) {
```

No register-allocator ICE on either arch (the plan's §8 flagged that risk).

## 2. Precision — before vs after

Raw hardware bit patterns from `test_sfpu_wh_sqrtcustom_numerics.py`, `Float32→Float32 / dest_acc=Yes`
(the only pipeline where a NaN survives to be seen), against an exact fp64 `sqrt`:

| `x` | hw before | hw after | rel err before | rel err after |
|---|---|---|---|---|
| `0`, `-0` | `0x00000000` | `0x00000000` | — | — |
| `1` | `0x3F7FFFAC` | `0x3F7FFFAC` | `5.007e-06` | `5.007e-06` |
| `2` | `0x3FB504F3` | `0x3FB504F3` | `1.711e-08` | `1.711e-08` |
| `3` | `0x3FDDB3CC` | `0x3FDDB3CC` | `7.750e-07` | `7.750e-07` |
| `4` | `0x3FFFFFAC` | `0x3FFFFFAC` | `5.007e-06` | `5.007e-06` |
| `9` | `0x403FFFEE` | `0x403FFFEE` | `1.431e-06` | `1.431e-06` |
| `0.25`, `1.5`, `100`, `0.001`, `1e-10`, `1e+10`, `1e-30`, `1e+30`, `2.384e-07` | unchanged | unchanged | `1.0e-07 … 2.9e-06` | **identical** |
| `1.17549e-38`, `3.40282e+38` | `0x208AF000` / `0x600AF000` | unchanged | `1.171e+00` | `1.171e+00` |
| **`+inf`** | **`0x7FC00001` NaN** ❌ | **`0x7F800000` +inf** ✅ | — | — |
| **`NaN`** | `0x7FC00001` (kernel-made) | `0x7FC00000` (input preserved) ✅ | — | — |
| **`-inf`** | `0x7F800000` +inf | `0xFF800000` −inf | — | — |

**Finite normal-range max relative error: `5.007e-06` before, `5.007e-06` after — identical.** Option A's
entire justification is that this holds, and it does: across all 8 format combinations the only rows that moved
are the three non-finite ones.

`Asin` and `Acos`, the other two `sfpu_sqrt_custom` consumers, are **bit-identical on every probe value** —
the whole 16-value block diffs clean.

Two things this measurement surfaces that the fix does **not** change:

* **`sqrt_custom` is 117 % wrong at the fp32 extremes** (`1.17549e-38` and `3.40282e+38`), before and after
  identically. The bf16-magic seed runs out of range there. Same shape as the `rsqrt_compat` extremes in
  [MEASUREMENTS_52930_reciprocal_compat_pole.md](MEASUREMENTS_52930_reciprocal_compat_pole.md) §3, and it
  wants its own issue.
* **`sqrt_custom(-inf)` is now `-inf` where IEEE gives NaN.** The plan predicted this and deliberately excluded
  the negative-input guard from the minimal fix (§5). It is recorded, not hidden — see §4.

## 3. `Erfinv(±1)` — the reported symptom

`erfinv(±1)` now returns `∓inf`/`±inf` on **all 8** combinations, which is the plan's acceptance criterion:
the 6 that previously passed by having their NaN narrowed to infinity now pass for the right reason, and the
2 that could see the NaN stop failing.

| input → output | `dest_acc` | `erfinv(+1)` before | after |
|---|---|---|---|
| `Float16_b → Float32` | Yes | `0x7FC00001` NaN ❌ | `0x7F800000` ✅ |
| `Float32 → Float32` | Yes | `0x7FC00001` NaN ❌ | `0x7F800000` ✅ |
| the other six | — | `0x7F800000` (by narrowing) | `0x7F800000` (by construction) ✅ |

## 4. Test changes

| File | Change |
|---|---|
| `test_sfpu_unary.py` | `Erfinv` removed from `_EDGE_KNOWN_DIVERGENCES` and `_EDGE_DIVERGENCE_REASON`; its "STILL OPEN" paragraph rewritten as CLOSED, correcting the two claims the investigation showed were wrong (it returned NaN, not a saturated finite; it was wrong on all 8, not 2) |
| `sfpu_domains.py` | **`SqrtCustom` enrolled in `SPECIALS_READY_OPS`** — the plan's "real lesson": the op was absent, so `+inf` was never driven at it and the defect had to be found through a consumer |
| `test_sfpu_unary.py` | `SqrtCustom` added to the derived cat-B divergences for the `-inf` row, with the reason spelled out |

The `-inf` divergence is derived by the same rule as `Reciprocal`'s (*every combination carrying specials at
all* — 6 of 8), not hand-listed, so it stays correct if the format axis grows.

Worth stating plainly: **before this fix, `sqrt_custom(-inf)` returned `+inf`, which would have *agreed* with
the golden** — because the golden's NaN is itself narrowed to `inf` on a bf16 output. So enrolling `SqrtCustom`
without the fix would have passed on `-inf` by accident. The fix makes a real disagreement visible; it does not
introduce one.

## 5. Performance — `MATH_ISOLATE`, `perf_eltwise_unary_sfpu.py`

Per-tile `TILE_LOOP` cycles, `Float16_b → Float16_b`, `iterations = 32`, `loop_factor = 16`. Two runs each
side; run-to-run spread under 0.1 cycles.

| op | `dest_acc` | before | after | delta | % | `TEXT_SIZE` |
|---|---|---|---|---|---|---|
| `SqrtCustom` | No | 986.60 | 1082.23 | +95.62 | **+9.69 %** | 2825 → 2849 (+24) |
| `SqrtCustom` | Yes | 988.28 | 1084.34 | +96.05 | **+9.72 %** | 2817 → 2841 (+24) |
| `Erfinv` | No | 3264.02 | 3397.75 | +133.73 | **+4.10 %** | 3437 → 3453 (+16) |
| `Erfinv` | Yes | 3269.62 | 3398.86 | +129.24 | **+3.95 %** | 3437 → 3453 (+16) |
| `Asin` | No | 2394.78 | 2490.27 | +95.48 | **+3.99 %** | 3177 → 3201 (+24) |
| `Asin` | **Yes** | 1948.34 | 1948.26 | −0.08 | **0.00 %** | 3101 → 3101 (0) |
| `Acos` | No | 2426.48 | 2522.48 | +96.01 | **+3.96 %** | 3185 → 3209 (+24) |
| `Acos` | **Yes** | 2044.10 | 2044.10 | 0.00 | **0.00 %** | 3125 → 3125 (0) |
| `Acosh`, `Asinh` (controls, not consumers) | both | — | — | ±0.10 | **0.00 %** | unchanged |

**Reading the cost.** +96 cycles over 32 SFPU iterations is **3 cycles per iteration** — the `SFPEXEXP`, the
compare, and the predicate combine. `+24` bytes is 6 instructions; `Erfinv` gets `+16`/4 because it inlines
one call rather than two.

`Asin`/`Acos` at `dest_acc=Yes` are unchanged **and their text size is unchanged**, which says those variants
never instantiate `sqrt_custom` at all — they take a different path for an fp32 Dest. The controls `Acosh`
and `Asinh` moving 0.00 % confirms the measurement is picking up the guard and not drift.

**The honest trade.** 3 cycles is cheap in absolute terms but `sqrt_custom` is a 987-cycle kernel, so it lands
as ~10 % there. The plan estimated "one instruction's cost"; measured, it is three. Whether ~10 % on
`SqrtCustom` and ~4 % on `Erfinv` is worth an IEEE-correct `sqrt(+inf)` is a kernel-owner call. Option B
(retire `sqrt_custom` for `_calculate_sqrt_body_`, which handles both poles already) would avoid the added
predicate entirely — but it moves every consumer's numerics and needs the trig accuracy sweep the plan
scoped out.

## 6. Verification

```
test_sfpu_unary.py -k "edges and (SqrtCustom or Erfinv or Sqrt)"     31 passed, 9 xfailed, 0 xpassed
test_sfpu_unary.py -k "Erfinv or SqrtCustom or Asin or Acos or
                       Sqrt or Rsqrt"                              1061 passed, 2 skipped, 9 xfailed
test_sfpu_unary.py -k "edges"  (whole edge sweep)                   495 passed, 238 skipped, 19 xfailed,
                                                                    0 xpassed, 0 failed
Blackhole  test_sfpu_unary.py -k "Erfinv or SqrtCustom or
                                  Asin or Acos"                     compile only: 328 passed, 72 skipped
```

## 7. Still open

| # | Item | Why not closed here |
|---|---|---|
| 1 | `sqrt_custom(-inf)` = `-inf`, IEEE says NaN | The plan keeps the negative-input guard out of the minimal fix: returning NaN for negative input is a behaviour change for every consumer, and `asin`/`acos` reach `sqrt_custom` with a negative argument whenever `\|v\| > 1`. Recorded as a divergence with its reason. Plan §10 wants it filed as its own issue. |
| 2 | `sqrt_custom` is **117 % wrong at the fp32 extremes** | Pre-existing, unchanged, seed out of range (§2). Own issue. |
| 3 | Option B — one square root in the tree instead of two | Explicitly a follow-up; needs a trig accuracy sweep, and would remove the +3 cycles this fix adds. |
| 4 | `Erfinv` still not in `SPECIALS_READY_OPS` | Plan calls it optional and secondary. |
| 5 | **Blackhole is compile-verified only** | No BH hardware on this host. Plan §7 step 3 wants the edge sweep run there. |
| 6 | **Quasar carries the same defect, unfixed** | Surveyed per plan §4 and confirmed to have the identical `sqrt_custom` body, but Quasar is out of scope for this PR and is left unchanged. Wants its own change once someone owns that arch. |
