# `sqrt_custom(+inf)` on Wormhole and Blackhole

Root cause, fix, and the measured accuracy and performance record for finding 5 of
[issue #52930](https://github.com/tenstorrent/tt-metal/issues/52930). The other four findings in
that issue are not addressed here.

| | |
|---|---|
| Kernels changed | `ckernel_sfpu_sqrt_custom.h` and `ckernel_sfpu_trigonometry.h`, Wormhole B0 and Blackhole |
| Silicon | Wormhole n300 and Blackhole p100a |
| Performance | measured on Wormhole n300 and Blackhole p100a |
| Quasar | unchanged — still carries the pre-fix kernel |
| SFPI | 7.69.0 (the pinned version) |

---

## 1. What was wrong

`sfpu_sqrt_custom` returned NaN for `+inf`.

The kernel seeds a fast inverse square root from the bf16 magic constant `0x5f37` and refines it
with Newton-Raphson. For `+inf` that seed is about `5.2e-20`. The first refinement squares it,
which underflows to a denormal; SFPMAD flushes the denormal to `+0`; and the next multiply is
`0 * -inf`, which is NaN.

Nothing in the kernel guarded against this, so every consumer inherited it. `erfinv(±1)` returned
NaN, which is how the defect was found — the issue originally scoped it as an `erfinv` tolerance
problem on two format combinations. It was neither: it was semantic, it lived in `sqrt_custom`,
and it was wrong on all eight combinations. The other six narrowed the NaN to `±inf` on the way
out and so agreed with the golden by accident.

`SqrtCustom` was absent from the `SPECIALS_READY_OPS` table, which is why `+inf` was never driven
at it directly. It is enrolled now.

## 2. The fix

Exclude non-finite input from the iteration, so `+inf` and NaN pass `val` straight through:

```cpp
v_if(val != 0.0f && sfpi::exexp(val, sfpi::ExponentMode::Biased) != 255) {
    out = sfpu_sqrt_custom_newton<NEWTON_ITERATIONS>(val);
}
v_endif;
```

Two details are deliberate.

**The test is on the biased exponent field, not a compare against infinity.** SFPSETCC's float
compare is specified only for inputs that are not NaN, and this predicate has to be evaluated on a
possible NaN. SFPI's `&&` is `SFPXBOOL(AND)` and does **not** short-circuit, so the `val != 0.0f`
compare is still emitted for NaN lanes. That is safe rather than merely tolerated: AND is monotone
and `exexp(NaN) == 255` makes the second conjunct false, so the conjunction is false for those
lanes whatever the unspecified compare returned.

**The guard is opt-out, via `GUARD_NON_FINITE`.** It defaults to on and is disabled only by
`asin`/`acos`. `sfpu_asin_range_reduced_bf16` is reached only from `sfpu_asin_bf16` /
`sfpu_acos_bf16` under `v_if(abs(val) <= 1.0f)`, so every lane whose result is committed hands
`sqrt_custom` an argument in `[0, 0.5]` and the exponent can never be 255. Lanes with `|v| > 1` do
execute it, but they keep the `quiet_NaN()` seed and their result is discarded. The conjunct is
therefore dead on that path while still costing +96 cycles/tile — see §4.

`erfinv` keeps the guard at both of its `sqrt_custom` call sites, and so does the direct
`SqrtCustom` op. Those are what the guard exists for.

## 3. Accuracy

Raw hardware bit patterns for a fixed stimulus list, driven through `SqrtCustom`, `Erfinv`,
`Asin` and `Acos` on all eight (format pair, dest_acc) combinations, and diffed byte for byte
against the branch point. 384 rows compared, **19 differ, and every one of them is a non-finite
input**. Every finite value is bit-identical, and `Asin`/`Acos` differ nowhere at all.

### Fixed

| Op | Input | Before | After | Golden |
|---|---|---|---|---|
| `SqrtCustom` | `+inf` | `0x7fc00001` (NaN) | `0x7f800000` (`+inf`) | `+inf` |
| `Erfinv` | `+1.0` | `0x7fc00001` (NaN) | `0x7f800000` (`+inf`) | `+inf` |
| `Erfinv` | `-1.0` | `0xffc00001` (NaN) | `0xff800000` (`-inf`) | `-inf` |
| `SqrtCustom` | `NaN` | `0x7fc00001` | `0x7fc00000` | NaN |

The `SqrtCustom` and `Erfinv` pole rows are visible on `Float16_b→Float32 dest_acc=Yes` and
`Float32→Float32 dest_acc=Yes`; on the other six the packer narrowed the NaN to `±inf` both before
and after. The NaN row is the input NaN now propagating instead of a manufactured one.

### Known residual — `sqrt_custom(-inf)`

`-inf` passes through as `-inf` (`0x7f800000` → `0xff800000`) on every combination. IEEE and the
golden give NaN, so **both the old and the new answer are wrong**. Before the fix this agreed with
the golden by accident on the bf16-output combinations, because the golden's NaN is itself narrowed
to inf there.

Synthesising NaN for negative input is deliberately not folded in. The constraint is `erfinv`, not
`asin`/`acos`: `asin`/`acos` seed `quiet_NaN()` and commit only under `abs(val) <= 1.0f`, so a NaN
out of `sqrt_custom` on their `|v| > 1` lanes is never observable. But `erfinv`'s NR iteration
systematically undershoots, driving `tmp + intermediate_result` non-positive for small in-domain
`x` — `erfinv(1e-6)` already reads `0x00000000` — so a negative-to-NaN guard would regress an
ordinary input to NaN. This is xfailed per combination rather than hidden in the golden, and needs
its own fix.

### Pre-existing errors, unchanged

Recorded so this is not read as a clean bill of health for `erfinv`: `erfinv(0.001)` has
`rel 1.025e-01`, and `erfinv(1e-6)` returns `0x00000000` against a non-zero golden. Both are
identical before and after, and neither is a `sqrt_custom` defect.

## 4. Performance

`perf_eltwise_unary_sfpu.py` with CI's flags, `MATH_ISOLATE` on the `TILE_LOOP`
marker, cycles per tile. Baseline and branch built from clean build roots so neither can serve the
other a stale ELF. `Acosh` and `Asinh` are carried as controls — same suite, same shape, no
`sqrt_custom` in them. `approx_mode` Yes and No measure identically for all six ops.

### Wormhole n300

| Op | dest_acc | Baseline | Branch | Δ | % |
|---|---|---|---|---|---|
| **SqrtCustom** | No | 983.0 | 1079.0 | +96.0 | **+9.76 %** |
| **SqrtCustom** | Yes | 985.6 | 1081.6 | +96.0 | **+9.71 %** |
| **Erfinv** | No | 3228.2 | 3362.6 | +134.4 | **+4.15 %** |
| **Erfinv** | Yes | 3232.0 | 3363.8 | +131.8 | **+4.08 %** |
| Asin | No | 2392.3 | 2392.3 | 0.0 | 0.00 % |
| Asin | Yes | 1945.6 | 1945.6 | 0.0 | 0.00 % |
| Acos | No | 2424.3 | 2424.3 | 0.0 | 0.00 % |
| Acos | Yes | 2041.6 | 2041.6 | 0.0 | 0.00 % |
| Acosh *(control)* | No, Yes | 3320.3, 3965.4 | 3320.3, 3965.4 | 0.0 | 0.00 % |
| Asinh *(control)* | No, Yes | 5122.6, 6540.8 | 5122.6, 6540.8 | 0.0 | 0.00 % |

### Blackhole p100a

Same method and the same six ops, measured on p100a. `approx_mode` Yes and No again measure
identically.

| Op | dest_acc | Baseline | Branch | Δ | % |
|---|---|---|---|---|---|
| **SqrtCustom** | No | 892.1 | 988.1 | +96.0 | **+10.76 %** |
| **SqrtCustom** | Yes | 892.1 | 988.1 | +96.0 | **+10.76 %** |
| **Erfinv** | No | 3010.1 | 3202.1 | +192.0 | **+6.38 %** |
| **Erfinv** | Yes | 3010.0 | 3202.0 | +192.0 | **+6.38 %** |
| Asin | No | 2300.1 | 2300.1 | 0.0 | 0.00 % |
| Asin | Yes | 1852.0 | 1852.0 | 0.0 | 0.00 % |
| Acos | No | 2332.1 | 2332.1 | 0.0 | 0.00 % |
| Acos | Yes | 1948.0 | 1948.0 | 0.0 | 0.00 % |
| Acosh *(control)* | No, Yes | 3228.0, 3900.0 | 3228.0, 3900.0 | 0.0 | 0.00 % |
| Asinh *(control)* | No, Yes | 4316.0, 5115.9 | 4316.0, 5115.9 | 0.0 | 0.00 % |

The direct op costs **+96.0 exactly** on Blackhole, the same absolute cost as Wormhole and the same
32 × 3 accounting below. The larger percentage is only a cheaper baseline (892.1 against 983.0),
not a more expensive guard.

`Erfinv` gains **+192.0 = 2 × 96** on Blackhole. `ckernel_sfpu_erfinv.h` calls `sqrt_custom` twice
(lines 39 and 43) on both architectures, and both sites are guarded on both — on Wormhole
`sfpu_sqrt_custom<false, 2>` leaves `GUARD_NON_FINITE` defaulted to true. Blackhole pays the full
serial cost of both guards where Wormhole measured +134.4, so about 58 cycles of the second guard
are absorbed into existing instruction slots there. A scheduling difference between the two
architectures, not a difference in what is guarded.

`Asin` and `Acos` are flat on Blackhole too, so the `GUARD_NON_FINITE=false` opt-out is confirmed
on BH silicon rather than only from the ELF.

**Where the SqrtCustom cycles go.** The guard adds three SFPU instructions per vector iteration —
`SFPEXEXP` to extract the biased exponent, `SFPIADD` to compare it against 255, `SFPSETCC` to
combine with the existing `!= 0.0f` predicate. A tile is 32 vector iterations, so 32 × 3 = 96
cycles per tile against +96.0 measured. The cost is exactly the instructions added, at one cycle
each, with nothing attributable to scheduling or register pressure.

**Why Asin and Acos are flat.** Without `GUARD_NON_FINITE` they measured +4.02 % and +3.96 %, the
same +96 cycles/tile landing on a larger kernel. With it they are flat to the last decimal, the
same as the controls. Confirmed independently from the ELF: `calculate_asin` and `calculate_acos`
are instruction-identical to the branch point on both Wormhole and Blackhole, while `erfinv` and
the direct op gain the guard.

`UNPACK_ISOLATE` is 0.00 % on every variant. `L1_TO_L1` tracks `MATH_ISOLATE` on the affected rows,
because math is the bottleneck for these ops.

## 5. Verification

### Wormhole n300

Suite results, `test_eltwise_unary_sfpu_edges` restricted to `SqrtCustom`, `Erfinv`, `Asin`, `Acos`:

| | Branch | Branch point |
|---|---|---|
| Result | 40 passed, 6 xfailed, **0 failed** | **2 failed**, 38 passed, 1 xfailed, 5 xpassed |

The two failures at the branch point are the `Erfinv(±1)` fp32-dest cells. The five XPASSes are the
`SqrtCustom` `-inf` entries, which agreed with the golden by accident before the fix.

`test_sqrt_custom_infinity_regression` asserts the repaired `+inf` strictly, and is deliberately
outside the edge sweep: that sweep's `SqrtCustom` invocation is non-strict xfailed for the `-inf`
divergence, and the marker would otherwise absorb a return to NaN. It runs on
`Float32 → Float32, dest_acc=Yes`, the only combination that both delivers `+inf` to the SFPU and
lets a NaN back out to L1. Confirmed non-vacuous: it passes on this branch and fails with
`sqrt_custom(+inf) returned nan` against the branch point.

### Blackhole p100a

The BH kernel is no longer compile-verified only. Full `test_sfpu_unary.py` on p100a silicon,
before and after dropping `_EDGE_BLACKHOLE_UNVERIFIED_DIVERGENCES`:

| `test_sfpu_unary.py` (whole file) | Result |
|---|---|
| With the BH-unverified table still in place | 5025 passed, 1601 skipped, 19 xfailed, **2 xpassed**, 0 failed |
| After dropping it (current) | **5027 passed**, 1601 skipped, 19 xfailed, 0 xpassed, 0 failed |

The two XPASSes in the first run are the `Erfinv(±1)` fp32-dest cells, which is the outcome
`_EDGE_BLACKHOLE_UNVERIFIED_REASON` was written to predict: the `sqrt_custom(+inf)` fix repairs
`erfinv(±1)` on Blackhole exactly as it does on Wormhole. That table and its arch-gated marker are
removed accordingly, so `Erfinv` now carries no divergence entry on either architecture and those
two cells are ordinary passes — the +2 between the rows above is exactly them.

Per-op, after the removal:

| | Selection | Result |
|---|---|---|
| `SqrtCustom` | `test_eltwise_unary_sfpu_edges` | 2 passed, 3 skipped, 3 xfailed (the `-inf` residual) |
| `Erfinv` | `test_eltwise_unary_sfpu_edges` | 5 passed, 3 skipped, 0 xfailed |
| `Asin`, `Acos`, `Asinh`, `Acosh` | whole file | 308 passed, 60 skipped, 0 failed |

The `SqrtCustom` `-inf` xfails behave as on Wormhole — same count, same reason, no unexpected
pass. The `Asin`/`Acos` row exercises the `GUARD_NON_FINITE=false` opt-out on BH silicon.

`test_sqrt_custom_infinity_regression` passes on BH silicon, and is non-vacuous there: with
`GUARD_NON_FINITE` defaulted to `false` in the BH kernel — which reduces it to the pre-fix
`v_if(val != 0.0f)` — the same test fails with `sqrt_custom(+inf) returned nan`. Blackhole
therefore exhibited the defect for the same reason Wormhole did, and the guard repairs it.
Confirmed against the include path the LLK suite actually builds with,
`-I../../hw/ckernels/blackhole/metal/llk_api/llk_sfpu`.

## 6. Scope and follow-ups

Worth filing separately:

* **`sqrt_custom` on negative input** — unguarded, and the `-inf` residual above is one face of it.
  Blocked on `erfinv`'s undershoot, per §3.
* **Two square-root implementations with different accuracy** — `sqrt` and `sqrt_custom` disagree,
  and nothing records which is authoritative.
