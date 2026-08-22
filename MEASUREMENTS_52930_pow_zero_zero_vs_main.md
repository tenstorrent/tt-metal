# Measurements — `pow(x, 0)`, accuracy and performance vs `main`

Accuracy and performance of the kernel this branch changes, measured against the branch point
rather than recalled from the original development run. Read-only record of the kernel delta.

Remeasured 2026-08-20 after the review comment that pinned exponent `-0.0` in the committed
edge sweep. That change is test-only — the kernel is unchanged — and the figures below
reproduced: 3 pairs fixed on every format combination, every other pair byte-identical;
`MATH_ISOLATE` **+3.50 % to +3.77 %** (median +3.57 %) on all 60 variants, 60/60 separated.

**Extended 2026-08-21 to Blackhole p100a silicon.** Blackhole is no longer compile-verified
only: the same probe, the same suites and the same perf harness now have cycle and bit-pattern
figures from hardware, and they close the "worth running before merge" caveat that
[RESULTS_52930_pow_zero_zero.md](RESULTS_52930_pow_zero_zero.md) §6 left open. Two findings are
specific to Blackhole and are new here — pre-fix Blackhole returns `+inf`, not `0`, for all three
pole rows (§3.2), and the guard costs **exactly +128.00 cycles per tile on all 60 variants**
(§5.2). The kernel was not edited for this; the Wormhole figures below are unchanged.

| | |
|---|---|
| Branch | `ldjurovic/sfpu_52930_pow_zero_zero` @ `4369664c9e2` |
| Baseline | `origin/main` @ `35ec0aba7a8` — the branch point, so the only kernel delta is this branch's |
| Silicon | Wormhole n300 (2026-08-20) and Blackhole p100a (2026-08-21), both UMD chip 0 |
| Date | 2026-08-20, Blackhole added 2026-08-21 |

Companion to [RESULTS_52930_pow_zero_zero.md](RESULTS_52930_pow_zero_zero.md), which records
the fix as it was developed. This one re-measures against the current branch point.

---

## Verdict

| | Regressed? | |
|---|---|---|
| **Accuracy** | **No, on both arches** | 3 input pairs fixed per format combination; every other pair byte-identical. Nothing got worse anywhere, on either arch. |
| **Performance** | **Yes — uniformly, on both arches** | `MATH_ISOLATE` **+3.50 % to +3.77 %** (median +3.57 %, ≈ +127 to +136 cycles per tile) on Wormhole; **+3.957 % to +3.981 %** (exactly +128.00 cycles per tile) on Blackhole. **All 60** variants on each arch. Fully accounted for by the 4 instructions the guard adds. |

Per arch, same experiment on each:

| | Wormhole n300 | Blackhole p100a |
|---|---|---|
| Probe pairs differing / total | 18 / 180 | 18 / 180 |
| Which pairs | the same 3 pole rows × 6 combinations | the same 3 pole rows × 6 combinations |
| Byte-identical | 162 / 180 | 162 / 180 |
| Pre-fix wrong answer | `0`, or `+inf` on the fp32 pipelines | **`+inf` on all 6 combinations** |
| `MATH_ISOLATE` median | +3.57 % (+129.6 cycles) | +3.958 % (+128.00 cycles) |
| Cycle delta spread | +126.9 … +136.5 | **+128.00 on every variant** |
| Variants slower / separated | 60 / 60, 60 separated | 60 / 60, 60 separated |
| Instructions added | +4 | +4 |

Blackhole's percentage is the larger of the two only because its baseline is cheaper
(3215–3234 vs Wormhole's 3620–3629 cycles per tile), not because the guard costs more there —
the absolute cost is 128 cycles on both, and on Blackhole it is 128 on the nose.

The trade is a correctness fix paid for at a flat ~3.6 % (Wormhole) / ~3.96 % (Blackhole) on
`SfpuElwpow` alone. The blast radius is exactly that one op: `calculate_sfpu_binary_power` has a
single caller, `BinaryOp::POW` in the same file, so nothing else in the SFPU is touched — which
Blackhole's flat `UNPACK_ISOLATE` and congestion counters confirm on silicon (§5.2).

---

## 1. Scope

| Item | Value |
|---|---|
| Kernel files changed | `ckernel_sfpu_binary.h` — Wormhole and Blackhole |
| Function changed | `calculate_sfpu_binary_power` |
| Callers of that function | 1 (`BinaryOp::POW`, same file) |
| Ops affected | `SfpuElwpow` only |
| Change | one trailing `v_if(setsgn(pow, 0) == 0.0f) { result = 1.0f; }` |
| Also in the Blackhole file | 5 spellings changed from `uint` to `std::uint32_t` in `calculate_sfpu_binary`, `_mul` and `_div`. **Applied automatically by the `fix-cstdint` pre-commit hook**, not by hand: `tt_metal/tt-llk/.pre-commit-config.yaml` is the installed config, and its hook matches `\.(cpp\|cc\|h\|hpp\|py)$` repo-wide, so any commit touching this file rewrites them. The Wormhole hunk has no counterpart because that file contained no bare `uint` to convert — which is the whole reason the two diffs are +18/−5 and +10/−0 rather than identical. Codegen-neutral: `uint` and `std::uint32_t` are the same type on this target, and the ELF comparison in §6 finds the guard's +4 SFPU instructions and **no other mnemonic changed**, in an object that contains all three of the touched functions. Not worth asking to revert — it would reappear on the next commit to this file. |

## 2. Method

Two checkouts of the same repo with isolated build roots (`RUNNER_TEMP`), so neither can serve
the other a stale ELF; baseline and branch run back to back on the same board.

Accuracy is raw hardware bit patterns from a fixed 30-pair list held in the probe rather than
read from `edge_pair_values()` — this branch edits the xfail table, so a pass/fail comparison
across the two trees would compare two different experiments rather than two kernels.
Performance is the CI harness with CI's flags (`--speed-of-light`, producer then consumer),
`MATH_ISOLATE` on the `TILE_LOOP` marker divided by `tile_cnt × loop_factor` (128), three
consumer runs per tree over 64 variants each (60 produce data; 4 skip).

The Blackhole pass (2026-08-21) is the same method against the same baseline SHA: a second
worktree at `35ec0aba7a8`, its own `RUNNER_TEMP` build root so neither tree can serve the other
a stale ELF, and the two trees measured back to back on the same p100a. The consumer ran serially
rather than under `-n`, so no two variants share the device while being timed. The probe is the
same fixed 30-pair list over the same 6 format combinations, which is the whole of
`_BINARY_EDGE_COMBINATIONS[SfpuElwpow]` and is fully supported on Blackhole — so the 180-pair
Blackhole table is directly comparable to the Wormhole one, pair for pair.

## 3. Accuracy — what changed

### 3.1 Wormhole n300

Three pairs change, identically on every format combination, and all three move from wrong to
correct:

| base | exponent | `main` | branch | IEEE / C / torch | Verdict |
|---|---|---|---|---|---|
| `0.0` | `0.0` | `0x00000000` = 0 | `0x3F800000` = 1 | 1 | **fixed** |
| `0.0` | `-0.0` | `0x00000000` = 0, or `0x7F800000` = `inf` on the fp32 pipelines | `0x3F800000` = 1 | 1 | **fixed** |
| `-0.0` | `0.0` | `0x00000000` = 0 | `0x3F800000` = 1 | 1 | **fixed** |

`0**-0.0` is the row that shows why the guard compares on `setsgn(pow, 0)`: on the two
fp32-carrying pipelines it read `+inf` on `main` where the other four read `0` — one undefined
`SFPSETCC` predicate on a NaN, two different wrong answers.

### 3.2 Blackhole p100a — the same three pairs, a different wrong answer

The same three pairs change, on all 6 combinations, and all three move to `1`. What differs is
what `main` returned: on Blackhole it is `+inf` **uniformly**, including for `0**0`, where
Wormhole returned `0`.

| base | exponent | `main` (WH) | `main` (BH) | branch (both) | IEEE / C / torch |
|---|---|---|---|---|---|
| `0.0` | `0.0` | `0x00000000` = 0 | **`0x7F800000` = `inf`** | `0x3F800000` = 1 | 1 |
| `0.0` | `-0.0` | `0` , or `inf` on the fp32 pipelines | **`0x7F800000` = `inf`** | `0x3F800000` = 1 | 1 |
| `-0.0` | `0.0` | `0x00000000` = 0 | **`0x7F800000` = `inf`** | `0x3F800000` = 1 | 1 |

This is the same root cause reading out differently, and it is direct evidence for it. All three
inputs form `0 * -inf = NaN` at `val = pow * log_result`; `exp(NaN)` collapses to `+0`; the
kernel's `v_if(val < 0)` is then evaluated on a NaN, which `SFPSETCC` leaves undefined. Wormhole
resolves that undefined predicate two ways depending on the sign the NaN inherited — falling
through to `+0` for `0**0`, taking the reciprocal branch for `0**-0.0`. **Blackhole takes the
reciprocal branch in all three cases**, turning the same `+0` into `+inf` every time. One
undefined predicate, three wrong answers on Wormhole and one on Blackhole, and the guard removes
the predicate's influence entirely on both.

Two consequences worth recording:

* The issue text and `main`'s own xfail reason string say "0**0 returns 0". That is **Wormhole-
  specific**; on Blackhole silicon it returns `+inf`. The branch deletes that reason string
  outright, so nothing needs re-wording — but a reader comparing the issue to Blackhole hardware
  would otherwise find the description wrong.
* `RESULTS_52930_pow_zero_zero.md` §6 flagged the risk that "if Blackhole's silicon behaves
  differently the removed entries would surface there as a failure rather than an XPASS".
  Blackhole *does* behave differently pre-fix — and the entries still pass, because the guard is
  unconditional and fixes both behaviours. The risk is retired, not merely untested (§7).

## 4. Accuracy — what did not change

Every other pair in the probe is **byte-identical** between the two trees, on all 6 measured
combinations — **162 of 180 on Wormhole and 162 of 180 on Blackhole**, the same 162 pairs. The
pairs chosen to catch an over-firing guard (values below are both trees, both arches):

| base | exponent | both trees | Why it is in the list |
|---|---|---|---|
| `-2.0` | `0.0` | `0x3F800000` = 1 | The guard is placed *after* the negative-base sign flip. Placed earlier, the odd-integer path would turn this into −1. |
| `2.0` | `0.0` | `0x3F800000` = 1 | ordinary `x**0` |
| `1.0` | `0.0` | `0x3F800000` = 1 | ordinary `x**0` |
| `0.0` | `1.0` | `0x00000000` = 0 | the guard must not fire on a zero *base* |
| `0.0` | `2.0` | `0x00000000` = 0 | as above |
| `0.0` | `-1.0` | `0x7F800000` = `inf` | deliberately left as-is; see §7 |
| `-2.0` | `2.0` | `+4` | even integer exponent, sign folded |
| `-2.0` | `3.0` | `-8` | odd integer exponent, sign folded |
| `2.0` `2.0` `2.0` `2.0` `2.0` | `1 2 3 0.5 -1 -2` | unchanged | the ordinary exponent range |
| `3.0` `9.0` `10.0` `0.5` `0.25` `1.5` `100.0` `7.0` `1.0` `-3.0` `1e-3` `1e3` `1.25` | various | unchanged | spread across the exponent range so a numerics regression cannot hide behind the pole rows |

Accuracy on ordinary input is therefore not merely "within tolerance" — it is bit-for-bit the
same kernel output, on Wormhole and on Blackhole. **No accuracy regression exists to report on
either arch.**

The zero-base controls are the ones that matter most on Blackhole: because pre-fix Blackhole
answered `+inf` for the three pole rows, a guard that keyed on the *base* rather than the
exponent would have looked correct on those three rows while silently changing `0**1` and `0**2`.
Both stay `0x00000000` on Blackhole, so the guard is firing on the exponent, as intended.

## 5. Performance — `MATH_ISOLATE` cycles per tile

### 5.1 Wormhole n300

Three runs per tree, 60 variants. "Separated" means the slowest branch run is still slower than
the fastest baseline run, so the delta is not inside the run-to-run spread.

| | Value |
|---|---|
| Variants measured | 60 |
| Variants slower | **60** |
| Variants faster | 0 |
| Variants with separated run ranges | **60 / 60** |
| `MATH_ISOLATE` change | **+3.50 % min, +3.57 % median, +3.77 % max** |
| Cycles per tile added | **+126.9 min, +129.6 median, +136.5 max** |

Grouped — the spread is driven by `dest_acc`, not by format or approximation mode:

| dest_acc | approx | n | min | median | max | median cycles |
|---|---|---|---|---|---|---|
| No | No | 14 | +3.59 % | +3.72 % | +3.77 % | +133.98 |
| No | Yes | 14 | +3.59 % | +3.72 % | +3.77 % | +133.99 |
| Yes | No | 16 | +3.50 % | +3.50 % | +3.57 % | +126.99 |
| Yes | Yes | 16 | +3.50 % | +3.50 % | +3.57 % | +126.98 |

`approx_mode` makes no difference at all — this kernel does not branch on it.

Representative absolute figures (`approx=No`):

| formats | dest_acc | `main` | branch | Δ cycles | Δ % |
|---|---|---|---|---|---|
| Float16_b/Float16_b→Float32 | Yes | 3629.37 | 3756.29 | +126.92 | **+3.50 %** *(smallest)* |
| Bfp8_b/Bfp8_b→Bfp8_b | Yes | 3629.30 | 3756.27 | +126.97 | +3.50 % |
| Float16/Float16→Float16 | Yes | 3626.73 | 3756.34 | +129.60 | +3.57 % |
| Bfp8_b/Bfp8_b→Bfp8_b | No | 3627.86 | 3757.92 | +130.06 | +3.59 % |
| Float16/Float16→Float16_b | No | 3620.25 | 3756.74 | +136.49 | **+3.77 %** *(largest)* |

Other run types:

| Run type | min | max | Reading |
|---|---|---|---|
| `L1_TO_L1` | +3.34 % | +3.52 % | tracks math — math is the bottleneck on this op |
| `UNPACK_ISOLATE` | −0.05 % | +0.02 % | flat, as it must be |
| `PACK_ISOLATE` | −2.21 % | +0.08 % | flat; the one outlier is a single variant's noise |

### 5.2 Blackhole p100a

Three runs per tree, 60 variants, same harness and same baseline SHA. Blackhole is the cleaner
of the two measurements: the delta is **the single value +128.00 cycles per tile on every one of
the 60 variants** — not a median of a spread, one number with no variance across formats,
`dest_acc`, or `approx_mode`.

| | Value |
|---|---|
| Variants measured | 60 |
| Variants slower | **60** |
| Variants faster | 0 |
| Variants with separated run ranges | **60 / 60** |
| `MATH_ISOLATE` change | **+3.957 % min, +3.958 % median, +3.981 % max** |
| Cycles per tile added | **+128.00 on all 60** (one distinct value) |

Grouped. The percentage spread is only the baseline moving underneath a constant +128:

| dest_acc | approx | n | min | median | max | median cycles |
|---|---|---|---|---|---|---|
| No | No | 14 | +3.957 % | +3.958 % | +3.981 % | +128.00 |
| No | Yes | 14 | +3.957 % | +3.958 % | +3.981 % | +128.00 |
| Yes | No | 16 | +3.958 % | +3.958 % | +3.958 % | +128.00 |
| Yes | Yes | 16 | +3.958 % | +3.958 % | +3.958 % | +128.00 |

`approx_mode` makes no difference, exactly as on Wormhole — this kernel does not branch on it.
The two `approx` rows are identical to the last decimal.

By input format, showing the constant delta against a slowly varying baseline:

| input format | n | `main` | branch | Δ cycles | Δ % |
|---|---|---|---|---|---|
| Bfp8_b | 14 | 3234.36 | 3362.36 | +128.00 | +3.958 % |
| Float16 | 16 | 3234.30 | 3362.30 | +128.00 | +3.958 % |
| Float16_b | 14 | 3234.03 | 3362.03 | +128.00 | +3.958 % |
| Float32 | 16 | 3224.94 | 3352.94 | +128.00 | +3.969 % |

The two extremes, which differ only in where the baseline sits:

| formats | dest_acc | `main` | branch | Δ cycles | Δ % |
|---|---|---|---|---|---|
| Bfp8_b/Bfp8_b→Bfp8_b | No | 3234.41 | 3362.41 | +128.00 | **+3.957 %** *(smallest)* |
| Float32/Float32→Float32 | No | 3215.52 | 3343.52 | +128.00 | **+3.981 %** *(largest)* |

Other run types:

| Run type | min | max | Reading |
|---|---|---|---|
| `L1_TO_L1` | +3.853 % | +3.958 % | +128.05 to +129.00 cycles — tracks math, which is the bottleneck on this op |
| `UNPACK_ISOLATE` | +0.000 % | +0.000 % | **exactly flat on all 60**, as it must be |
| `PACK_ISOLATE` | +0.000 % | +0.012 % | flat; 59 of 60 are 0.00, one variant is +0.01 cycles |
| `L1_CONGESTION[UNPACK]` | +0.000 % | +0.000 % | flat |
| `L1_CONGESTION[PACK]` | +0.000 % | +0.000 % | flat |

Blackhole's `UNPACK_ISOLATE` and both congestion counters are flat to the last decimal on every
variant, which is a stronger control than Wormhole's (−0.05 % to +0.02 %) and confirms the change
is confined to the math kernel.

## 6. Where the cycles go

Instruction counts of the `MATH_ISOLATE` `math.elf`, paired by build hash across the two trees:

| Arch | Variants compared | Instruction delta | Distribution |
|---|---|---|---|
| Wormhole B0 | 60 | **+4** | +4 on 60 of 60 |
| Blackhole | 60 | **+4** | +4 on 60 of 60 |

The four, identical on both arches:

| Mnemonic | `main` → branch (WH) | `main` → branch (BH) | Role in `v_if(setsgn(pow, 0) == 0.0f) { result = 1.0f; }` |
|---|---|---|---|
| `SFPSETSGN` | 5 → 6 | 3 → 4 | clear the sign of `pow`, so `-0.0` compares equal to `0.0` |
| `SFPSETCC` | 8 → 9 | 9 → 10 | the compare against zero |
| `SFPENCC` | 7 → 8 | 6 → 7 | close the predicate |
| `SFPMOV` | 5 → 6 | 2 → 3 | write `1.0f` into the predicated lanes |

The Blackhole column is re-confirmed on the ELFs built for the cycle measurement above: pairing
the two build roots by variant hash, **120 of 120** math-bearing `math.elf` pairs go from 75 to
79 SFPU instructions, **+4**, and the four are exactly the mnemonics tabulated. (The other 180 of
the 300 hash pairs are the `UNPACK`/`PACK`/`L1_CONGESTION` isolate ELFs, whose `math.elf` holds 3
SFPU instructions and is unchanged.)

The arithmetic closes on both arches: this loop is **not** unrolled, so 4 static instructions are
4 per SFPU vector iteration. A tile is 32 vector iterations, so **32 × 4 = 128 cycles per tile**:

| Arch | Predicted | Measured | Residual |
|---|---|---|---|
| Wormhole B0 | 128 | median 129.6, range 126.9 … 136.5 | +1.6 median, inside the format-to-format spread |
| Blackhole | 128 | **128.00 on all 60 variants** | **0.00 — exact** |

The cost is the instructions added, at one cycle each, with nothing left over for scheduling or
register pressure. Blackhole makes that point exactly rather than approximately: every variant
lands on the predicted number, so there is no room for a scheduling or register-pressure term at
all. This is the same result the sibling `sqrt_custom` fix measured on the same board, where a
3-instruction guard cost exactly +96.0 = 32 × 3.

That also prices the `SFPSETSGN` specifically: it is 1 of the 4, so **32 of the 128 cycles** —
roughly 0.9 % of the 3.6 % on Wormhole, 0.99 % of the 3.96 % on Blackhole — is what buys the
`0**-0.0` case. A bare `pow == 0.0f` would cost ~2.7 % / ~2.97 % and would leave `0**-0.0` at
`inf`, because `SFPSETCC`'s contract excludes negative zero. That is not an inference: the bare
compare was built and run on Blackhole silicon, and it does leave `0**-0.0` at `+inf` — see the
non-vacuity check in §7.

## 7. Suite results

### Wormhole n300

`test_sfpu_binary.py`:

| Sweep | `main` | branch | Δ |
|---|---|---|---|
| `-k SfpuElwpow` | 73 passed, 0 xfailed | **79 passed, 0 xfailed** | +6 passed |
| whole file | 1009 passed, 9 xfailed, 16 xpassed | **1015 passed, 3 xfailed, 16 xpassed** | +6 passed, −6 xfailed |

0 failed in all four runs. Both deltas are the same 6: `SfpuElwpow`'s `both_zero` xfails, which
the fix retires — those cells now *assert* `0**0 == 1` rather than pinning the old answer.

The **16 XPASS are identical on both trees** and are not this branch's: they are the
pre-existing `negative_zero_golden` cases on div / xlogy / fmod / remainder. `SfpuElwpow`
contributes none of them, before or after.

After the review comment on missing `-0.0` exponent coverage, the committed edge sweep
now cartesian-products Operand.B `(-0.0, 0.0, 1.0, 2.0)` with bases including `±2`, and
`test_pow_edge_pairs_include_negative_zero_exponent` pins that stimulus host-side. Re-run
on Wormhole: `binary_edges and SfpuElwpow` 12 passed / 0 xfailed / 0 xpassed (including
`Float32→Float32 dest_acc=Yes both_zero`, which is the pipeline that delivers `-0.0`);
`-k SfpuElwpow` still 79 passed. The kernel was not edited for that comment, and the
accuracy/perf figures above are the remeasure against the same `main` SHA. (That re-run also
reported "Blackhole compile 57 passed"; Blackhole is now run on silicon instead — below.)

### Blackhole p100a, on silicon

The same three selections, both trees, on hardware. Every delta is the same +6 passed / −6 xfailed
as Wormhole, and there are **0 failed and 0 xpassed in all six runs**:

| Sweep (Blackhole p100a) | `main` @ `35ec0aba7a8` | branch | Δ |
|---|---|---|---|
| `-k SfpuElwpow` | 57 passed, 105 skipped, 6 xfailed | **63 passed, 105 skipped, 0 xfailed** | +6 passed, −6 xfailed |
| whole file | 883 passed, 983 skipped, 12 xfailed | **889 passed, 983 skipped, 6 xfailed** | +6 passed, −6 xfailed |
| `binary_edges and SfpuElwpow` | 6 passed, 6 xfailed | **12 passed, 0 xfailed, 0 xpassed** | +6 passed, −6 xfailed |

`test_pow_edge_pairs_include_negative_zero_exponent` passes on Blackhole too (host-side, so this
is a consistency check rather than a silicon one).

The 105 and 983 skips are architecture and format constraints, none of them pow-specific:
`Bfp8_b is not supported for POW/XLOGY coverage` (32), `Float32 inputs with dest_acc=No are not
supported` (26), `Float16_a isn't supported for SFPU on Blackhole …` (16), row-broadcast with FP32
dest (13), and 18 `no <class> pair among its registered per-operand edges`. The same 6 format
combinations that `_BINARY_EDGE_COMBINATIONS[SfpuElwpow]` drives on Wormhole are driven on
Blackhole, which is why the two `binary_edges` rows are directly comparable.

Note the contrast with Wormhole on XPASS: Wormhole's whole-file run carries 16 pre-existing
`negative_zero_golden` XPASSes on div / xlogy / fmod / remainder. **Blackhole has none** — 0
xpassed on both trees. Those are a Wormhole-only phenomenon and, as on Wormhole, `SfpuElwpow`
contributes none of them.

### Blackhole: the committed sweep is non-vacuous

The reviewer's concern that motivated the `-0.0` exponent coverage was that the sweep could stay
green while `setsgn` was dropped. Checked directly on Blackhole silicon, in a throwaway worktree
at the branch commit with the guard weakened to a bare `v_if(pow == 0.0f)` and nothing else
changed:

| | Result on Blackhole p100a |
|---|---|
| Probe, `0**-0.0` | regresses to `0x7F800000` = `+inf` — the same failure mode as Wormhole |
| Probe, other 5 pole/control rows | still correct, so the bare compare is *almost* right, which is the trap |
| `binary_edges and SfpuElwpow` | **2 failed**, 10 passed |

The two failures are `Float32→Float32 dest_acc=Yes both_zero` and
`Float32→Float16_b dest_acc=Yes both_zero` — exactly the Float32-input pipelines that deliver a
signed zero to SFPU, which is what the committed `_OP_OPERAND_EDGE_POINTS` entry added. So on
Blackhole, as on Wormhole, `SFPSETSGN` is load-bearing and the committed sweep is what proves it:
without the `-0.0` exponent in the product, all 12 cells would have passed against a kernel that
answers `0**-0.0 = inf`. The guard was restored and the worktree deleted; the branch tree was
never modified for this check.

## 8. Deliberately unchanged

| Case | Behaviour | Why untouched |
|---|---|---|
| `0**-1` | `+inf`, both trees | torch and the golden both want `+inf` here. The sibling `_sfpu_binary_power_f32_` in `ckernel_sfpu_binary_pow.h` carries a `base == 0 && pow < 0 -> NaN` guard and a docstring promising NaN, so the two implementations disagree. That disagreement predates this branch and wants filing separately. |

## 9. What was not measured

| # | Item | Why |
|---|---|---|
| 1 | Quasar | Out of scope; no `BinaryOp::POW` in its `ckernel_sfpu_binary.h`. |
| 2 | End-to-end model impact | ~3.6 % (WH) / ~3.96 % (BH) on the `SfpuElwpow` MATH phase is an LLK-level figure. What it costs a model depends on how much of that model is `pow`. |
| 3 | A Wormhole re-run alongside the Blackhole pass | This host has Blackhole p100a only. The Wormhole figures are the 2026-08-20 measurement against the same baseline SHA and the same unchanged kernel, not a re-run on 2026-08-21. |
| 4 | `pow(inf, 0)` and `pow(nan, 0)` | `SfpuElwpow` is still not in `SPECIALS_READY_OPS`, so the sweep never drives a non-finite base. The guard is unconditional and makes both correct, but that is untested on either arch — the follow-up `RESULTS` §6 already records. |

Blackhole cycles were the previous edition's item 1; they are measured now (§5.2).
