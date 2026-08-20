# Result — `0 ** 0`: fixed on both arches, and it cost 3.65 % on the pow path

**Plan:** [FIX_PLAN_52930_pow_zero_zero.md](FIX_PLAN_52930_pow_zero_zero.md) (finding 3 of
[ISSUE_52930_INVESTIGATION.md](ISSUE_52930_INVESTIGATION.md)).
**Question asked:** implement the plan, verify it is functionally correct, then measure the perf delta from
the local baseline using tt-llk's perf tests.

**Answer in one line:** the plan's fix works and is in the tree on both arches, but it needed the plan's own
risk-table fallback to be complete — the guard has to compare on `setsgn(pow, 0)`, because a bare
`pow == 0.0f` **does not fire for `pow == -0.0`**. Cost is **+132.3 `MATH_ISOLATE` cycles per tile, +3.65 %**,
uniform across all 60 measured pow variants. That is more than the plan's "immaterial" estimate, and the
final instruction — the one that buys `0 ** -0.0` — is 0.92 % of it.

| | Wormhole n300, silicon | |
|---|---|---|
| Tree | `tt-metal` @ `ldjurovic/wrong_sfpu_edge_cases`, `21557a33763` | |
| Runner | `tt_metal/tt-llk/.claude/scripts/run_test.sh` throughout | |
| Perf test | `perf_eltwise_binary_sfpu.py -k SfpuElwpow`, 3 runs per variant | |
| Date | 2026-08-17 | |

---

## 1. A second defect, found by the plan's own risk table

§7 of the plan asks for `(-2.0, 0.0)`, `(0.0, -0.0)` and `(2.0, -0.0)` to be added to `_POW_PAIRS` as
guards against a fix that overreaches. Added *before* the fix, they turned up a divergence the plan did not
know about:

| pair | golden | baseline hw | |
|---|---|---|---|
| `0 ** 0` | `0x3F800000` | `0x00000000` | ❌ the known finding |
| `0 ** -0.0` | `0x3F800000` | `0x7F800000` (`inf`) | ❌ **not in the plan** |
| `-2 ** 0` | `0x3F800000` | `0x3F800000` | ✅ |
| `2 ** -0.0` | `0x3F800000` | `0x3F800000` | ✅ |

This is direct confirmation of the plan's §2 root cause, from an angle the plan did not anticipate. Both
inputs form the same NaN at `val = pow * log_result` (`0 × -inf`), and both then reach the
`v_if(val < 0)` that `VectorUnit.md` leaves undefined on a NaN — and the two **resolve differently**:
`0 ** 0` falls through with `exp(NaN)`'s `+0`, while `0 ** -0.0` takes the reciprocal branch and turns that
`+0` into `inf`. One undefined predicate, two different wrong answers, selected by the sign the NaN happened
to inherit. The plan called the undefined predicate evaluation a second problem "on one line's worth of
code"; it is observable, not theoretical.

## 2. The fix, and the correction the plan's risk table predicted

The plan's §3 guard, applied verbatim as the last statement before `return`:

```c
    v_if(pow == 0.0f) { result = 1.0f; }
    v_endif;
```

fixed `0 ** 0`, left `(-2) ** 0`, `0 ** -1` and the `4 ** 0.5` accuracy figure alone — and left
`0 ** -0.0` at `inf`. That is exactly risk 2 of §7:

> `v_if(pow == 0.0f)` itself hits the SFPSETCC negative-zero caveat when `pow` is `-0.0` … If `-0.0`
> mis-predicates, compare on `setsgn(pow, 0)` instead.

So the shipped guard is the fallback, on both arches (the two hunks are byte-identical):

```c
    // IEEE 754: pow(x, 0) == 1 for every x, including 0, +/-inf and NaN. Without this the
    // composition above forms 0 * ln(0) = 0 * -inf = NaN at base == 0 (SFPMAD), exp(NaN)
    // collapses to +0, and the v_if(val < 0) is then evaluated on a NaN, which the ISA
    // leaves undefined (VectorUnit, SFPSETCC) -- measured as 0**0 = 0 but 0**-0.0 = inf.
    // Last, so the negative-base sign flip above cannot turn (-2)**0 into -1. Compared on
    // setsgn(pow, 0) because SFPSETCC's contract excludes negative zero: measured, a bare
    // pow == 0.0f does not fire for pow == -0.0 and leaves 0**-0.0 at inf.
    v_if(sfpi::setsgn(pow, 0) == 0.0f) { result = 1.0f; }
    v_endif;
```

Placement is the plan's: after the negative-base block, immediately before `return result;`. `(-2) ** 0`
staying `1.0` (not `-1.0`) is the measurement that confirms it, and it is now pinned by `_POW_PAIRS`.

## 3. Functional verification

**The probe, all 12 pairs** (`Float32→Float32`, `dest_acc=Yes` — the pipeline where nothing is narrowed):

| pair | golden | baseline | fixed |
|---|---|---|---|
| `0 ** 0` | `0x3F800000` | `0x00000000` | `0x3F800000` ✅ |
| `0 ** -0.0` | `0x3F800000` | `0x7F800000` | `0x3F800000` ✅ |
| `0 ** 1`, `0 ** 2` | `0x00000000` | `0x00000000` | unchanged ✅ |
| `1 ** 0`, `2 ** 0`, `4 ** 0`, `1e-30 ** 0`, `2 ** -0.0` | `0x3F800000` | `0x3F800000` | unchanged ✅ |
| `-2 ** 0` | `0x3F800000` | `0x3F800000` | unchanged ✅ |
| `0 ** -1` | `0x7F800000` | `0x7F800000` | unchanged ✅ |
| `4 ** 0.5` | `0x40000000` | `0x3FFFAAC3` | `0x3FFFAAC3` — unchanged, still out of scope |

The `4 ** 0.5` DIFF is the composition's own ~0.13 % error, which the plan explicitly does not address; it is
bit-identical before and after, which is the point.

**Shipped suites, Wormhole silicon:**

```
test_sfpu_binary.py -k "binary_edges and SfpuElwpow"   12 passed, 20 skipped, 0 xfail, 0 xpass
test_sfpu_binary.py -k "SfpuElwpow"                    79 passed, 81 skipped, 0 failed
test_sfpu_binary.py  (whole suite)                    871 passed, 392 skipped, 27 xfailed, 16 xpassed
```

The `binary_edges` run is the acceptance criterion: with both table entries removed, `both_zero` now
**asserts** `0**0 == 1` and passes on every driven combination, with no XPASS left over. The 16 XPASS in the
full suite are all `negative_zero_golden` on `SfpuElwdiv` / `SfpuXlogy` / `SfpuBinaryFmod` /
`SfpuBinaryRemainder` — pre-existing and untouched; pow contributes none, which the edges-only run
independently confirms (0 xpassed there).

**Blackhole is compile-verified only** — no Blackhole silicon on this host, same limitation as the
`RsqrtCompat` work. `test_sfpu_binary.py` compiles clean for the whole suite (709 passed compile).

## 4. Performance — `MATH_ISOLATE`, `perf_eltwise_binary_sfpu.py -k SfpuElwpow`

60 variants (5 input formats × 4 output formats × `dest_acc` × `approx_mode`), 3 runs each side,
`iterations = 32`, `loop_factor = 16`, `[128, 64]`. `SfpuElwpow` was already on this perf test, so no test
changes were needed.

Per-tile `TILE_LOOP` cycles, averaged over all 60 variants:

| variant | `TILE_LOOP` | delta | % | cycles/SFPU iteration | `TEXT_SIZE(MATH_ISOLATE)` |
|---|---|---|---|---|---|
| baseline | 3624.87 | — | — | — | 2935 |
| plain `pow == 0.0f` (incomplete — leaves `0**-0.0` wrong) | 3723.98 | +99.11 | **+2.73 %** | 3.10 | 2947 (+12 = 3 instr) |
| **shipped `setsgn(pow, 0) == 0.0f`** | 3757.20 | +132.32 | **+3.65 %** | 4.14 | 2951 (+16 = 4 instr) |

The delta is remarkably uniform: across all 60 variants it spans only **+130.48 … +133.76 cycles
(+3.60 … +3.71 %)**, and on **60/60 variants every fixed run is slower than every baseline run** — no
overlap at all, so this is not a noise artefact. (Worst per-row run-to-run spread was 23–44 cycles but the
*median* spread was 0.00, and the per-row separation check above does not depend on that spread anyway.)

Secondary counters, `TILE_LOOP`, median across variants:

| | delta |
|---|---|
| `mean(L1_TO_L1)` | **+3.46 %** — math is the bottleneck on pow, so the end-to-end figure tracks `MATH_ISOLATE` |
| `mean(UNPACK_ISOLATE)` | +0.000 % (range −0.065 … +0.038) — untouched, as expected |
| `mean(PACK_ISOLATE)` | +0.000 % (range −3.79 … +2.78) — noise, no trend |
| `INIT` | +0.00 median (−1.67 … +3.00) — no init work added |

**Reading the cost.** `+16` bytes is 4 SFPU instructions (`SFPSETSGN`, the compare's `SFPSETCC`, an
`SFPLOADI` for `1.0f`, and the predicated `SFPMOV`), and 4 instructions × 32 iterations = 128 cycles against
132.3 measured — the cost is exactly the instructions added, with no scheduling surprise.

**Two honest notes on this number:**

* The plan's §4 estimated "one `SFPSETCC` + one `SFPLOADI`/`SFPMOV` pair … ~2-3 cycles per iteration.
  Immaterial." The estimate was accurate *for the guard it specified* — the plain variant measures 3.10
  cycles/iteration. But that guard is functionally incomplete, and the `setsgn` needed to make `0 ** -0.0`
  correct adds a 4th instruction: **+33.2 cycles/tile, +0.92 %**, strictly slower on 60/60 variants. That is
  the price of the second defect, stated separately so it can be traded away deliberately rather than by
  accident.
* 3.65 % is not "immaterial" the way the plan expected, but it also is not measured against a cheap kernel —
  the pow path already runs a log polynomial, an exp and a 2-iteration reciprocal, 3625 cycles/tile. Whether
  3.65 % on binary `pow` is acceptable to buy IEEE-correct `pow(x, 0)` is a call for the kernel owners; it
  is not a call this measurement can make.

## 5. What is in the tree

| File | Change |
|---|---|
| `hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` | `calculate_sfpu_binary_power` ends in the `pow(x, 0) == 1` guard |
| `hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` | identical hunk, byte for byte |
| `tt-llk/tests/python_tests/test_sfpu_binary.py` | `SfpuElwpow` removed from `_BINARY_EDGE_COMBINATIONS` (all 6) and `_BINARY_EDGE_REASON`; the "STILL OPEN" block's `0**0` paragraph rewritten as CLOSED, now also recording `0**-0.0` and the SFPSETCC mechanism |
| `tt-llk/tests/python_tests/test_sfpu_wh_issue52930_probe.py` | `_POW_PAIRS` gains `(-2.0, 0.0)`, `(0.0, -0.0)`, `(2.0, -0.0)` |

The collection-time assertion `set(_BINARY_EDGE_REASON) == set(_BINARY_EDGE_COMBINATIONS)` still passes —
both entries were removed together, as §5 requires.

`test_sfpu_wh_issue52930_probe.py` remains **not for merge**, same status as before.

## 6. Still open, deliberately

* **A possible zero-cost variant, untested.** The 4 instructions exist to repair a NaN that the kernel
  creates one line earlier. Replacing `log_result = -inf` at `base == 0` with a large *finite* negative would
  make `pow == 0` yield an exact `0 × -huge = 0` — no NaN, no guard, no added instruction, and the `+3.65 %`
  disappears. The trade-off is real and is why it is not shipped here: with a finite stand-in, `0 ** y` for
  very small positive `y` no longer underflows to `0` (`1e-30 ** … ` style inputs land on `exp` of a small
  finite value instead), so it would replace one wrong answer at a point with wrong answers over a range.
  Worth a measurement if the 3.65 % matters; it needs its own numerics sweep, not just an edge probe.
* **`0 ** negative` still disagrees between the two implementations.** As §3 of the plan instructs, behaviour
  was **not** changed — `0 ** -1` returns `inf`, which torch and the golden both want. But
  `_sfpu_binary_power_f32_` in `ckernel_sfpu_binary_pow.h` still carries a `base == 0 && pow < 0 → NaN` guard
  and a docstring promising NaN, and `calculate_sfpu_binary_power` still has no such guard. That
  docstring/guard disagreement is unfiled; the plan asks for it to be raised separately so the two do not
  drift further.
* **`SfpuElwpow` is still not in `SPECIALS_READY_OPS`**, so `pow(inf, 0)` and `pow(nan, 0)` — which the new
  guard also makes correct, since the guard is unconditional in `pow` — are never driven. The plan calls this
  optional hardening and a follow-up rather than a blocker; it is not closed here.
* **Blackhole is compile-verified only.** The plan's §6 step 3 wants the edge sweep run on both arches; this
  host has Wormhole n300 only. The xfail tables are not arch-keyed, so if Blackhole's silicon behaves
  differently the removed entries would surface there as a failure rather than an XPASS — worth running
  before merge.
