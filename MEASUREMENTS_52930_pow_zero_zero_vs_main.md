# Measurements — `pow(x, 0)`, accuracy and performance vs `main`

Accuracy and performance of the kernel this branch changes, measured against the branch point
rather than recalled from the original development run. Read-only record of the kernel delta.

Remeasured 2026-08-20 after the review comment that pinned exponent `-0.0` in the committed
edge sweep. That change is test-only — the kernel is unchanged — and the figures below
reproduced: 3 pairs fixed on every format combination, every other pair byte-identical;
`MATH_ISOLATE` **+3.50 % to +3.77 %** (median +3.57 %) on all 60 variants, 60/60 separated.

| | |
|---|---|
| Branch | `ldjurovic/sfpu_52930_pow_zero_zero` |
| Baseline | `origin/main` @ `35ec0aba7a8` — the branch point, so the only kernel delta is this branch's |
| Silicon | Wormhole n300 (UMD chip 0) |
| Blackhole | compile-verified, plus an instruction-level comparison; no BH hardware for cycles |
| Date | 2026-08-20 |

Companion to [RESULTS_52930_pow_zero_zero.md](RESULTS_52930_pow_zero_zero.md), which records
the fix as it was developed. This one re-measures against the current branch point.

---

## Verdict

| | Regressed? | |
|---|---|---|
| **Accuracy** | **No** | 3 input pairs fixed per format combination; every other pair byte-identical. Nothing got worse anywhere. |
| **Performance** | **Yes — uniformly** | `MATH_ISOLATE` **+3.50 % to +3.77 %** (median +3.57 %, ≈ +127 to +136 cycles per tile) on **all 60** measured variants. Fully accounted for by the 4 instructions the guard adds. |

The trade is a correctness fix paid for at a flat ~3.6 % on `SfpuElwpow` alone. The blast
radius is exactly that one op: `calculate_sfpu_binary_power` has a single caller,
`BinaryOp::POW` in the same file, so nothing else in the SFPU is touched.

---

## 1. Scope

| Item | Value |
|---|---|
| Kernel files changed | `ckernel_sfpu_binary.h` — Wormhole and Blackhole |
| Function changed | `calculate_sfpu_binary_power` |
| Callers of that function | 1 (`BinaryOp::POW`, same file) |
| Ops affected | `SfpuElwpow` only |
| Change | one trailing `v_if(setsgn(pow, 0) == 0.0f) { result = 1.0f; }` |

## 2. Method

Two checkouts of the same repo with isolated build roots (`RUNNER_TEMP`), so neither can serve
the other a stale ELF; baseline and branch run back to back on the same board.

Accuracy is raw hardware bit patterns from a fixed 30-pair list held in the probe rather than
read from `edge_pair_values()` — this branch edits the xfail table, so a pass/fail comparison
across the two trees would compare two different experiments rather than two kernels.
Performance is the CI harness with CI's flags (`--speed-of-light`, producer then consumer),
`MATH_ISOLATE` on the `TILE_LOOP` marker divided by `tile_cnt × loop_factor` (128), three
consumer runs per tree over 64 variants each (60 produce data; 4 skip).

## 3. Accuracy — what changed

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

## 4. Accuracy — what did not change

Every other pair in the probe is **byte-identical** between the two trees, on all 6 measured
combinations. The pairs chosen to catch an over-firing guard:

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
same kernel output. **No accuracy regression exists to report.**

## 5. Performance — Wormhole n300, `MATH_ISOLATE` cycles per tile

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

The arithmetic closes: this loop is **not** unrolled, so 4 static instructions are 4 per SFPU
vector iteration. A tile is 32 vector iterations, so **32 × 4 = 128 cycles per tile** against a
measured median of **129.6** and a range of 126.9 to 136.5. The cost is the instructions added,
at one cycle each, with nothing left over for scheduling or register pressure.

That also prices the `SFPSETSGN` specifically: it is 1 of the 4, so roughly **0.9 % of the
3.6 %** is what buys the `0**-0.0` case. A bare `pow == 0.0f` would be ~2.7 % — and would leave
`0**-0.0` at `inf`, because `SFPSETCC`'s contract excludes negative zero.

## 7. Suite results

Wormhole n300, `test_sfpu_binary.py`:

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
`-k SfpuElwpow` still 79 passed; Blackhole compile 57 passed. The kernel was not edited
for that comment, and the accuracy/perf figures above are the remeasure against the same
`main` SHA.

Blackhole compiles clean — 57 correctness variants and 64 perf variants — and carries the
identical 4-instruction guard (§6). No BH silicon on this host, so no cycle figures.

## 8. Deliberately unchanged

| Case | Behaviour | Why untouched |
|---|---|---|
| `0**-1` | `+inf`, both trees | torch and the golden both want `+inf` here. The sibling `_sfpu_binary_power_f32_` in `ckernel_sfpu_binary_pow.h` carries a `base == 0 && pow < 0 -> NaN` guard and a docstring promising NaN, so the two implementations disagree. That disagreement predates this branch and wants filing separately. |

## 9. What was not measured

| # | Item | Why |
|---|---|---|
| 1 | Blackhole cycles | No BH silicon on this host. Compile-verified, and its instruction delta is measured and identical to Wormhole's. |
| 2 | Quasar | Out of scope; no `BinaryOp::POW` in its `ckernel_sfpu_binary.h`. |
| 3 | End-to-end model impact | +3.6 % on the `SfpuElwpow` MATH phase is an LLK-level figure. What it costs a model depends on how much of that model is `pow`. |
